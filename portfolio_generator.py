import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')


# ─── Configuration ───────────────────────────────────────────────
UNIVERSE_CSV   = "top_5000_us_by_marketcap.csv"
TOP_N          = 500          # how many stocks to consider
LOOKBACK_DAYS  = "2y"        # history to download for factor regression
BETA_WINDOW    = 250         # rolling OLS window for factor betas
VOL_WINDOW     = 20          # realized vol lookback (daily returns std)
NOTIONAL_PER_STOCK = 1000.0  # base notional (scaled later)


def _fetch_prices(tickers: list[str], period: str = "2y") -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance."""
    data = yf.download(
        tickers, period=period, group_by="column",
        threads=True, progress=False, auto_adjust=False,
    )
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(0):
            adj_close = data["Adj Close"]
        else:
            adj_close = data["Close"]
    else:
        adj_close = data.get("Adj Close", data["Close"])

    # Forward fill small gaps (halts), drop tickers with too much missing data
    adj_close = adj_close.ffill(limit=5)
    pct_missing = adj_close.isna().mean()
    good_tickers = pct_missing[pct_missing < 0.20].index
    adj_close = adj_close[good_tickers]
    return adj_close


def _rolling_beta(stock_ret: pd.Series, market_ret: pd.Series,
                  window: int = 250, min_periods: int = 120) -> pd.Series:
    """Compute rolling OLS beta with Vasicek shrinkage toward 1."""
    roll_cov = stock_ret.rolling(window, min_periods=min_periods).cov(market_ret)
    roll_var = market_ret.rolling(window, min_periods=min_periods).var()
    raw_beta = roll_cov / roll_var
    # Vasicek / Blume shrinkage: 0.2 + 0.8 * raw (pushes toward 1.0)
    return 0.2 + 0.8 * raw_beta


def generate_residual_portfolio(csv_path: str = UNIVERSE_CSV,
                                top_n: int = TOP_N,
                                notional_per_stock: float = NOTIONAL_PER_STOCK
                                ) -> pd.DataFrame:
    """
    Build a dollar-neutral portfolio by:
      1. Computing rolling beta for each stock vs SPY
      2. Computing each stock's raw alpha (recent return − beta × market return)
      3. Cross-sectionally regressing alpha on market to get residual signal
      4. Weighting the residual signal by inverse realized volatility
      5. Ranking the vol-weighted residual → dollar-neutral long/short weights
    """

    # ── 1. Load universe ─────────────────────────────────────────
    df = pd.read_csv(csv_path)
    symbols = (df.head(top_n)["symbol"]
               .astype(str).str.strip()
               .str.replace(".", "-", regex=False)
               .str.replace("/", "-", regex=False)
               .tolist())

    # ── 2. Fetch prices (stocks + SPY benchmark) ─────────────────
    print(f"Downloading price data for {len(symbols)} stocks + SPY ...")
    all_tickers = symbols + ["SPY"]
    adj_close = _fetch_prices(all_tickers, period=LOOKBACK_DAYS)

    # Separate SPY from the universe
    if "SPY" not in adj_close.columns:
        raise RuntimeError("SPY download failed – cannot regress alpha.")
    spy_close = adj_close["SPY"].dropna()
    stock_close = adj_close.drop(columns=["SPY"], errors="ignore")

    # Keep only tickers that survived the download
    valid_tickers = [t for t in symbols if t in stock_close.columns]
    stock_close = stock_close[valid_tickers]

    # ── 3. Returns ───────────────────────────────────────────────
    stock_ret = stock_close.pct_change()
    market_ret = spy_close.pct_change()

    # Align dates
    common_idx = stock_ret.index.intersection(market_ret.index).sort_values()
    stock_ret = stock_ret.loc[common_idx]
    market_ret = market_ret.loc[common_idx]
    stock_close = stock_close.loc[common_idx]

    # ── 4. Rolling betas ─────────────────────────────────────────
    print("Computing rolling betas ...")
    betas = pd.DataFrame(index=common_idx, columns=valid_tickers, dtype="float64")
    for ticker in valid_tickers:
        betas[ticker] = _rolling_beta(stock_ret[ticker], market_ret,
                                      window=BETA_WINDOW)

    # ── 5. Raw alpha: trailing 5-day stock return minus beta × market return
    H = 5  # horizon (days)
    trailing_stock_ret  = stock_ret.rolling(H).sum()
    trailing_market_ret = market_ret.rolling(H).sum()

    raw_alpha = trailing_stock_ret.sub(
        betas.mul(trailing_market_ret, axis=0)
    )

    # ── 6. Cross-sectional regression of alpha on market return ──
    #        to strip any residual market directionality.
    #        For each date:  alpha_i = gamma * market_ret + residual_i
    #        We keep only residual_i.
    print("Regressing out market exposure from alpha cross-section ...")
    
    # Use data from the day before yesterday
    from datetime import datetime, timedelta
    target_date = pd.Timestamp(datetime.now().date() - timedelta(days=2))
    valid_dates = common_idx[common_idx <= target_date]
    if len(valid_dates) == 0:
        raise ValueError("No data available up to the day before yesterday.")
    last_date = valid_dates[-1]
    
    alpha_today = raw_alpha.loc[last_date].dropna()

    if len(alpha_today) < 20:
        raise ValueError("Too few stocks with valid alpha on the last date.")

    mkt_today = trailing_market_ret.loc[last_date]

    # OLS: alpha = gamma * beta + epsilon  (cross-sectional, single regressor)
    beta_today = betas.loc[last_date, alpha_today.index].astype("float64")
    X = beta_today.values
    Y = alpha_today.values

    # Simple OLS:  gamma = cov(Y, X) / var(X)
    x_mean = np.nanmean(X)
    y_mean = np.nanmean(Y)
    gamma = np.nansum((X - x_mean) * (Y - y_mean)) / np.nansum((X - x_mean) ** 2)
    intercept = y_mean - gamma * x_mean
    residual = Y - (gamma * X + intercept)

    residual_signal = pd.Series(residual, index=alpha_today.index, name="residual")

    # ── 7. Volatility weighting ──────────────────────────────────
    #        Signal_i  =  residual_i / vol_i   (inverse vol: less noisy stocks
    #        get more weight; noisier stocks get attenuated)
    print("Applying inverse-volatility weighting ...")
    vol = stock_ret[alpha_today.index].iloc[-VOL_WINDOW:].std()
    vol = vol.replace(0, np.nan).dropna()

    # Intersection of stocks that have both residual and vol
    common = residual_signal.index.intersection(vol.index)
    residual_signal = residual_signal[common]
    vol = vol[common]

    vol_weighted_signal = residual_signal / vol

    # ── 8. Rank-based dollar-neutral portfolio ───────────────────
    print("Constructing rank-based dollar-neutral portfolio ...")
    ranks = vol_weighted_signal.rank()
    centered = ranks - ranks.mean()   # zero-mean → naturally dollar-neutral

    # Separate long / short
    longs  = centered[centered > 0]
    shorts = centered[centered < 0].abs()

    # Normalise each side to sum to 0.5 of total notional
    n_stocks = len(centered)
    total_notional = notional_per_stock * n_stocks
    half_notional  = total_notional / 2.0

    long_notionals  = (longs / longs.sum())  * half_notional
    short_notionals = (shorts / shorts.sum()) * half_notional

    # ── 9. Build targets DataFrame for QSec submission ───────────
    targets = pd.DataFrame()

    long_df = pd.DataFrame({
        "internal_code": long_notionals.index.astype(str) + " OQ",
        "currency": "USD",
        "target_notional": long_notionals.values.round(2),
    })

    short_df = pd.DataFrame({
        "internal_code": short_notionals.index.astype(str) + " OQ",
        "currency": "USD",
        "target_notional": -short_notionals.values.round(2),
    })

    targets = pd.concat([long_df, short_df], ignore_index=True)

    return targets


# ─── Quick sanity-check entry point ─────────────────────────────
if __name__ == "__main__":
    targets_df = generate_residual_portfolio()
    print("\n=== Generated Residual-Alpha Portfolio ===")
    print(f"Stocks: {len(targets_df)}")
    print(f"Total Long Notional:  ${targets_df[targets_df['target_notional'] > 0]['target_notional'].sum():,.2f}")
    print(f"Total Short Notional: ${targets_df[targets_df['target_notional'] < 0]['target_notional'].sum():,.2f}")
    print(f"Net Notional:         ${targets_df['target_notional'].sum():,.2f}")
    print(f"\nTop 10 Longs:")
    print(targets_df.nlargest(10, "target_notional")[["internal_code", "target_notional"]].to_string(index=False))
    print(f"\nTop 10 Shorts:")
    print(targets_df.nsmallest(10, "target_notional")[["internal_code", "target_notional"]].to_string(index=False))
