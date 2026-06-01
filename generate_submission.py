"""
generate_submission.py — Standalone Portfolio Generator for QRT Academy
=======================================================================
Fetches data from Yahoo Finance (via cached pickle), runs all strategies
at their correct rebalance frequencies, blends via inverse-volatility,
enforces all QRT constraints, and outputs a submission-ready CSV.

The reversal book is a super-sleeve: an inverse-vol blend of Alpha#11 (the
original mean-reversion sleeve) with three formulaic alphas from the 101-alpha
study (Alpha#24/#41/#100, traded with asymmetric hysteresis bands). The final
book is an equal-vol (inverse-vol) blend of {Reversal super-sleeve, Momentum,
Stat-Arb} — Momentum (weekly W-FRI) and Stat-Arb (Kalman warm-start) are
unchanged.

Usage:  python generate_submission.py
"""

import pandas as pd
import numpy as np
import os
import sys
import pickle
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# Formulaic-alpha library (101-alpha study). Guarded so the pipeline still runs
# (reversal = Alpha#11 only) if the package or its deps are unavailable.
try:
    from alpha101.data import Data as _AlphaData
    from alpha101.alphas import get_alpha as _get_alpha
    _ALPHA101_OK = True
except Exception as _e:  # pragma: no cover
    _ALPHA101_OK = False
    _ALPHA101_ERR = str(_e)

# ============================================================================
# CONFIGURATION
# ============================================================================
TARGET_RISK       = 470_000.0   # Target annualised risk (held under user's 475k ceiling with 5k safety margin)
MAX_WEIGHT        = 0.098       # Per-stock max weight (< 10%)
ADV_THRESHOLD     = 5_000_000   # 60-day ADV universe filter
ADV_LIMIT_PCT     = 0.025       # 2.5% of ADV position cap
MAX_POSITION_USD  = 2_000_000   # QRT hard cap per position
VOL_LOOKBACK      = 120         # Days for risk estimation
MR_SMOOTH_WINDOW  = 3           # Mean reversion signal smoothing
MOM_TARGET_VOL    = 0.40        # Momentum inverse-vol target
MOM_LONG_ENTER    = 0.90
MOM_LONG_EXIT     = 0.85
MOM_SHORT_ENTER   = 0.20
MOM_SHORT_EXIT    = 0.25
ALPHA_SECTORS = {
    'Technology', 'Energy', 'Consumer Discretionary', 'Consumer Staples',
    'Basic Materials', 'Industrials', 'Real Estate', 'Telecommunications',
}

# Formulaic-alpha reversal sleeves (101-alpha study). Each trades with asymmetric
# hysteresis bands: (long_enter, long_exit, short_enter, short_exit) in pct-rank.
NEW_ALPHAS   = [24, 41, 100]
ALPHA_SMOOTH = 3
ALPHA_DESIGNS = {
    24:  dict(le=0.70, lx=0.60, se=0.10, sx=0.20),   # long D8-D10 (buf D7), short D1 (buf D2)
    41:  dict(le=0.70, lx=0.60, se=0.10, sx=0.20),   # long D8-D10 (buf D7), short D1 (buf D2)
    100: dict(le=0.90, lx=0.80, se=0.50, sx=0.50),   # long D10 (buf D9), short D1-D5
}

# Data + state persistence
DATA_PICKLE       = "top_5000_yf_data.pkl"
KALMAN_STATE_PATH = "kalman_state.pkl"     # warm-start state for the Kalman filters
SA_WEIGHTS_CACHE  = "weights_sa_cache.pkl" # cached historical stat-arb weights
YF_FETCH_CHUNK    = 400                    # batch size for yfinance download
SKIP_REFRESH_ENV  = "SKIP_DATA_REFRESH"    # env var to skip the fetch step

# ============================================================================
# UTILITY FUNCTIONS (inlined from utils.py to be self-contained)
# ============================================================================

def scale_to_book_long_short(alpha: pd.Series) -> pd.Series:
    """Scale so longs sum to +0.5 and shorts sum to -0.5."""
    sum_pos = alpha[alpha > 0].sum()
    sum_neg = alpha[alpha < 0].sum()
    result = pd.Series(0.0, index=alpha.index)
    if sum_pos != 0:
        result[alpha > 0] = alpha[alpha > 0] / sum_pos * 0.5
    if sum_neg != 0:
        result[alpha < 0] = alpha[alpha < 0] / abs(sum_neg) * 0.5
    return result


def normalize_and_cap(w_matrix, target=0.50, cap=0.099):
    """Normalize each book to target GMV and cap individual weights."""
    row_sums = w_matrix.sum(axis=1) + 1e-10
    w = w_matrix.div(row_sums, axis=0) * target
    w = w.clip(upper=cap)
    row_sums_final = w.sum(axis=1) + 1e-10
    w = w.div(row_sums_final, axis=0) * target
    return w


def enforce_post_shift_strict_gmv(shifted_portfolio, universe_df, max_wt=0.098):
    """
    Water-fill normalizer: guarantees GMV=1.0 and max weight < 10%
    AFTER the universe mask has zeroed out invalid tickers.
    """
    portfolio = shifted_portfolio * universe_df
    abs_sum = portfolio.abs().sum(axis=1)

    for date, gmv in abs_sum.items():
        if gmv > 1e-8:
            row = portfolio.loc[date].copy()
            row = row / gmv
            for _ in range(20):
                if row.abs().max() <= max_wt + 1e-6:
                    break
                capped_mask = row.abs() > max_wt
                row[capped_mask] = np.sign(row[capped_mask]) * max_wt
                remaining = 1.0 - row[capped_mask].abs().sum()
                uncapped_sum = row[~capped_mask].abs().sum()
                if uncapped_sum > 1e-8:
                    row[~capped_mask] *= (remaining / uncapped_sum)
                else:
                    break
            if abs(row.abs().sum() - 1.0) > 0.01 or row.abs().max() > 0.1001:
                portfolio.loc[date] = 0.0
            else:
                portfolio.loc[date] = row
    return portfolio


def dedup(df):
    """Remove duplicate columns."""
    return df.loc[:, ~df.columns.duplicated()]


# ============================================================================
# STEP 0: DAILY DATA REFRESH FROM YAHOO FINANCE
# ============================================================================
def refresh_data_from_yfinance(pickle_path=DATA_PICKLE):
    """Append fresh OHLCV rows to the data pickle.

    - Reads the existing pickle to learn the universe of tickers and the last date
    - Fetches yfinance from (last_date + 1) to today, batched to avoid timeouts
    - Aligns/appends new rows to the existing column structure
    - Saves the updated pickle back

    Idempotent: if pickle is already up to date (last_date == today's business day)
    this is a no-op. Set env var SKIP_DATA_REFRESH=1 to bypass entirely.
    """
    print("=" * 60)
    print("STEP 0: Daily Data Refresh (Yahoo Finance)")
    print("=" * 60)

    if os.environ.get(SKIP_REFRESH_ENV):
        print("  SKIP_DATA_REFRESH set, using cached pickle as-is.")
        return

    import yfinance as yf  # local import so the rest of the script runs without yf

    if not os.path.exists(pickle_path):
        print(f"  ERROR: pickle {pickle_path} does not exist. Cold-fetch not implemented.")
        return

    df = pd.read_pickle(pickle_path)
    last_date = df.index.max().normalize()
    today = pd.Timestamp.now().normalize()
    print(f"  Pickle last date: {last_date.date()}")
    print(f"  Today:            {today.date()}")

    # Skip if last_date is today or yesterday (covers weekend/holiday no-op)
    business_days_behind = pd.bdate_range(last_date + pd.Timedelta(days=1), today).size
    if business_days_behind == 0:
        print("  Pickle already up-to-date through last business day. No fetch needed.")
        return

    # Universe of tickers from existing pickle's column index
    tickers = sorted(df["Adj Close"].columns.unique().tolist())
    print(f"  Need to fetch {business_days_behind} new business day(s) for {len(tickers)} tickers")

    fetch_start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    fetch_end = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")  # yf end is exclusive

    new_parts = []
    n_chunks = (len(tickers) + YF_FETCH_CHUNK - 1) // YF_FETCH_CHUNK
    for i in range(0, len(tickers), YF_FETCH_CHUNK):
        chunk = tickers[i:i + YF_FETCH_CHUNK]
        chunk_num = i // YF_FETCH_CHUNK + 1
        print(f"    Chunk {chunk_num}/{n_chunks}: downloading {len(chunk)} tickers...", flush=True)
        try:
            new = yf.download(
                tickers=chunk, start=fetch_start, end=fetch_end,
                auto_adjust=False, progress=False, threads=True, group_by="column",
            )
            if new is not None and not new.empty:
                # Strip timezone from index if present
                if new.index.tz is not None:
                    new.index = new.index.tz_localize(None)
                new_parts.append(new)
        except Exception as e:
            print(f"      WARN: chunk {chunk_num} failed: {e}")
            continue

    if not new_parts:
        print("  No new data returned by yfinance. Pickle unchanged.")
        return

    new_df = pd.concat(new_parts, axis=1)
    # Reindex columns to existing pickle structure; missing tickers stay NaN
    new_df = new_df.reindex(columns=df.columns)
    # Drop any dates we already have
    new_df = new_df[~new_df.index.normalize().isin(df.index.normalize())]

    if new_df.empty:
        print("  All fetched dates already in pickle. Pickle unchanged.")
        return

    combined = pd.concat([df, new_df], axis=0).sort_index()
    combined.to_pickle(pickle_path)
    print(f"  Appended {len(new_df)} new business day(s). "
          f"New range: {combined.index.min().date()} -> {combined.index.max().date()}")


# ============================================================================
# STEP 1: DATA LOADING & UNIVERSE
# ============================================================================
def load_data():
    print("=" * 60)
    print("STEP 1: Loading Data & Building Universe")
    print("=" * 60)

    print(f"  Loading historical YF data from {DATA_PICKLE}...")
    df_hist = pd.read_pickle(DATA_PICKLE)
    print(f"  Data range: {df_hist.index.min().date()} -> {df_hist.index.max().date()} ({len(df_hist)} days)")

    print("  Computing returns...")
    returns = df_hist["Adj Close"].pct_change(fill_method=None).fillna(0)
    returns = dedup(returns)

    print("  Computing 60-day ADV and universe mask...")
    daily_dollar_vol = df_hist["Close"].mul(df_hist["Volume"]).fillna(0)
    df_adv_60 = daily_dollar_vol.rolling(window=60, min_periods=60).mean()
    universe = (df_adv_60 >= ADV_THRESHOLD).astype(int)
    universe = dedup(universe)

    n_stocks = (universe.iloc[-1] == 1).sum()
    print(f"  Universe on last day: {n_stocks} stocks with ADV >= $5M")
    return df_hist, returns, universe, df_adv_60


# ============================================================================
# STEP 2: MEAN REVERSION (Daily rebalance)
# ============================================================================
def run_mean_reversion(df_hist, returns, universe, df_adv_60):
    """Mean reversion — Option 2 design (validated in cells 6d/6f):
       - Long: TOP 10% of (smoothed) alpha rank        (~+0.5 GMV, equal-weight)
       - Short: BOTTOM 50% of alpha rank               (~-0.5 GMV, equal-weight)
       - Middle 40% (D6-D9) untraded (noise per diagnostic)
       - T+1 execution lag
       - 3-day rolling smoothing of raw signal (cumulative IC peaks at h=3)
    """
    print("\n" + "=" * 60)
    print("STEP 2: Mean Reversion (Option 2: D10 long, D1-D5 short, daily)")
    print("=" * 60)

    vwap = (df_hist["High"] + df_hist["Low"] + df_hist["Close"]) / 3.0
    diff_vc = vwap - df_hist["Close"]
    vol_delta = df_hist["Volume"].diff(3)

    rank_max = diff_vc.rolling(window=3).max().rank(axis=1, pct=True)
    rank_min = diff_vc.rolling(window=3).min().rank(axis=1, pct=True)
    rank_vol = vol_delta.rank(axis=1, pct=True)

    alpha = (rank_max + rank_min) * rank_vol
    alpha = alpha.where(universe == 1, np.nan)
    alpha = alpha.rolling(window=MR_SMOOTH_WINDOW, min_periods=1).mean()

    # Percentile bucketing
    LONG_PCT, SHORT_PCT = 0.90, 0.50
    pct_rank = alpha.rank(axis=1, pct=True)
    long_mask  = (pct_rank >= LONG_PCT).astype(float)
    short_mask = (pct_rank <  SHORT_PCT).astype(float)

    n_longs  = long_mask.sum(axis=1).replace(0, np.nan)
    n_shorts = short_mask.sum(axis=1).replace(0, np.nan)
    w_longs  = long_mask.div(n_longs,  axis=0) *  0.5
    w_shorts = short_mask.div(n_shorts, axis=0) * -0.5
    weights = (w_longs.fillna(0) + w_shorts.fillna(0))

    avg_longs = long_mask.sum(axis=1).replace(0, np.nan).mean()
    avg_shorts = short_mask.sum(axis=1).replace(0, np.nan).mean()
    print(f"  Avg per-day positions:  longs={avg_longs:.0f}  shorts={avg_shorts:.0f}")

    # T+1 lag, dedup, GMV enforcement (per-name 2.5% ADV cap is applied in Step 6)
    weights = weights.shift(1).fillna(0)
    weights = dedup(weights)
    weights = enforce_post_shift_strict_gmv(weights, universe)
    print("  Mean Reversion complete.")
    return weights


# ============================================================================
# STEP 2b: FORMULAIC-ALPHA SLEEVES + REVERSAL SUPER-SLEEVE
# ============================================================================
def build_alpha_data(df_hist, returns):
    """Assemble the alpha101 Data panel from the (daily-refreshed) pickle."""
    o, h, l = dedup(df_hist["Open"]), dedup(df_hist["High"]), dedup(df_hist["Low"])
    c, v = dedup(df_hist["Close"]), dedup(df_hist["Volume"])
    vwap = (h + l + c) / 3.0
    meta = pd.read_csv("top_5000_us_by_marketcap.csv")
    meta["symbol"] = meta["symbol"].astype(str).str.replace("/", "-")
    meta = meta.drop_duplicates("symbol").set_index("symbol")
    cap = pd.to_numeric(meta["marketCap"], errors="coerce").reindex(c.columns)
    sector = meta["sector"].reindex(c.columns)
    industry = meta["industry"].reindex(c.columns)
    rets = returns.reindex(index=c.index, columns=c.columns).fillna(0)
    # subindustry falls back to industry (no finer classification available)
    return _AlphaData(o, h, l, c, v, vwap, rets, cap, sector, industry, industry)


def _alpha_bucket_rank(sig, universe):
    """3-day smooth -> cross-sectional demean -> pct rank (matches the study)."""
    s = sig.replace([np.inf, -np.inf], np.nan).rolling(ALPHA_SMOOTH, min_periods=1).mean()
    s = s.where(universe.astype(bool))
    s = s.sub(s.mean(axis=1), axis=0)
    return s.rank(axis=1, pct=True)


def _hysteresis_state(rp, le, lx, se, sx):
    """Stateful per-stock long(+1)/short(-1)/flat(0): enter on core, hold through buffer."""
    R = rp.values
    T, N = R.shape
    state = np.zeros(N)
    out = np.zeros((T, N))
    for t in range(T):
        r = R[t]
        valid = ~np.isnan(r)
        new = np.zeros(N)
        new[(state == 1) & valid & (r >= lx)] = 1     # hold long through buffer
        new[(state == -1) & valid & (r < sx)] = -1    # hold short through buffer
        new[valid & (r >= le)] = 1                    # fresh long entry
        new[valid & (r < se)] = -1                    # fresh short entry
        state = new
        out[t] = new
    return pd.DataFrame(out, index=rp.index, columns=rp.columns)


def run_alpha_sleeve(adata, universe, n):
    """One formulaic-alpha sleeve: signal -> hysteresis -> dollar-neutral T+1 weights."""
    sig = _get_alpha(n)(adata)
    rp = _alpha_bucket_rank(sig, universe)
    state = _hysteresis_state(rp, **ALPHA_DESIGNS[n]).astype(float)
    longs = (state > 0).astype(float)
    shorts = (state < 0).astype(float)
    w = (longs.div(longs.sum(axis=1).replace(0, np.nan), axis=0) * 0.5
         - shorts.div(shorts.sum(axis=1).replace(0, np.nan), axis=0) * 0.5).fillna(0)
    w = dedup(w.shift(1).fillna(0))
    return enforce_post_shift_strict_gmv(w, universe)


def inverse_vol_blend(sleeves, returns, universe, label="blend"):
    """Inverse-vol (equal-risk) blend of dollar-neutral sleeves -> GMV=1 book.

    Same construction as Step 5: 60-day rolling vol, inverse-vol allocation shifted
    one day (causal), combine weight matrices (internal netting), renormalize each
    book to 0.5 and water-fill to GMV=1 / max-weight<10%.
    """
    def rolling_vol(w):
        ar = returns.reindex(index=w.index, columns=w.columns).fillna(0)
        pnl = (w * ar).sum(axis=1)
        return (pnl.rolling(60, min_periods=20).std() * np.sqrt(252)).clip(lower=0.05)

    inv = {k: 1.0 / rolling_vol(w) for k, w in sleeves.items()}
    total = sum(inv.values())
    blended = None
    for k, w in sleeves.items():
        c = (inv[k] / total).shift(1).fillna(1.0 / len(sleeves))
        contrib = w.multiply(c, axis=0)
        blended = contrib if blended is None else blended.add(contrib, fill_value=0)

    el = blended.where(blended > 0, 0)
    es = blended.where(blended < 0, 0).abs()
    final = (normalize_and_cap(el) - normalize_and_cap(es)).fillna(0)
    final = enforce_post_shift_strict_gmv(final, universe)

    latest = {k: float((inv[k] / total).iloc[-1]) for k in sleeves}
    print(f"  {label} latest sleeve weights: "
          + ", ".join(f"{k}={latest[k]:.1%}" for k in sleeves))
    return final


def build_reversal_supersleeve(w_mr, df_hist, returns, universe):
    """Reversal super-sleeve = inverse-vol blend of Alpha#11 + Alpha#24/#41/#100.

    Falls back to Alpha#11 alone if the alpha101 package is unavailable, so the
    pipeline never breaks.
    """
    print("\n" + "=" * 60)
    print("STEP 2b: Reversal Super-Sleeve (Alpha#11 + Alpha#24/#41/#100)")
    print("=" * 60)

    if not _ALPHA101_OK:
        print(f"  WARNING: alpha101 unavailable ({_ALPHA101_ERR}); "
              f"reversal book = Alpha#11 only.")
        return w_mr

    adata = build_alpha_data(df_hist, returns)
    sleeves = {"A11": w_mr}
    for n in NEW_ALPHAS:
        print(f"  Building Alpha#{n} sleeve (hysteresis)...")
        sleeves[f"A{n}"] = run_alpha_sleeve(adata, universe, n)

    # align all reversal sleeves to a common grid
    cols, idx = w_mr.columns, w_mr.index
    sleeves = {k: dedup(w).reindex(index=idx, columns=cols).fillna(0) for k, w in sleeves.items()}
    uni = dedup(universe).reindex(index=idx, columns=cols).fillna(0)
    return inverse_vol_blend(sleeves, returns, uni, label="Reversal")


# ============================================================================
# STEP 3: MOMENTUM (Weekly rebalance)
# ============================================================================
def run_momentum(df_hist, returns, universe):
    print("\n" + "=" * 60)
    print("STEP 3: Sector-Neutral Momentum Strategy (Weekly)")
    print("=" * 60)

    import yfinance as yf

    # --- SPY residual returns ---
    print("  Fetching SPY for residual returns...")
    spy_df = yf.download("SPY", start=returns.index.min(),
                         end=returns.index.max() + pd.Timedelta(days=1),
                         progress=False)
    if isinstance(spy_df.columns, pd.MultiIndex):
        spy_df.columns = spy_df.columns.droplevel(1)
    spy_col = "Adj Close" if "Adj Close" in spy_df.columns else "Close"
    spy_adj = spy_df[spy_col].squeeze()
    if spy_adj.index.tz is not None:
        spy_adj.index = spy_adj.index.tz_localize(None)

    spy_rets = np.log(spy_adj / spy_adj.shift(1)).reindex(returns.index).fillna(0)
    spy_var = spy_rets.rolling(252, min_periods=63).var()
    rolling_cov = returns.rolling(252, min_periods=63).cov(spy_rets)
    rolling_beta = rolling_cov.div(spy_var, axis=0).shift(1)
    residual_returns = returns - rolling_beta.multiply(spy_rets, axis=0)

    # --- Garman-Klass volatility ---
    print("  Computing Garman-Klass volatility...")
    log_hl = np.log(df_hist["High"] / df_hist["Low"]) ** 2
    log_co = np.log(df_hist["Close"] / df_hist["Open"]) ** 2
    gk_var = 0.5 * log_hl - ((2 * np.log(2) - 1) * log_co)
    vol_floored = np.sqrt(gk_var.ewm(com=60, min_periods=60).mean() * 252).clip(lower=0.15)

    # --- Sector mapping + alpha-positive filter ---
    print("  Applying sector filter...")
    meta = pd.read_csv("top_5000_us_by_marketcap.csv")
    meta["symbol"] = meta["symbol"].str.replace("/", "-")
    sector_dict = meta.set_index("symbol")["sector"].to_dict()
    aligned_sectors = pd.Series(returns.columns).map(sector_dict).values

    # --- 6-month lagged residual return signal ---
    print("  Computing sector-neutral ranks...")
    signal = residual_returns.shift(21).rolling(105).sum()
    valid = signal * universe.replace(0, np.nan)

    ranks = valid.copy()
    ranks[:] = np.nan
    for sector in ALPHA_SECTORS:
        cols = valid.columns[aligned_sectors == sector]
        if len(cols) > 0:
            ranks[cols] = valid[cols].rank(axis=1, pct=True)

    # --- Hysteresis banding ---
    print("  Applying hysteresis bands...")
    universe_dropped = (universe.reindex_like(ranks).fillna(0) == 0)

    long_state = pd.DataFrame(np.nan, index=ranks.index, columns=ranks.columns)
    long_state[ranks >= MOM_LONG_ENTER] = 1
    long_state[ranks < MOM_LONG_EXIT] = -1
    long_state[universe_dropped] = -1
    long_signals = long_state.ffill().replace(-1, 0).fillna(0)

    short_state = pd.DataFrame(np.nan, index=ranks.index, columns=ranks.columns)
    short_state[ranks <= MOM_SHORT_ENTER] = 1
    short_state[ranks > MOM_SHORT_EXIT] = -1
    short_state[universe_dropped] = -1
    short_signals = short_state.ffill().replace(-1, 0).fillna(0) * -1

    tradable_signal = (long_signals + short_signals).fillna(0)

    # --- Risk-parity sizing + weekly resample ---
    print("  Constructing weekly dollar-neutral portfolio...")
    unconstrained = tradable_signal * (MOM_TARGET_VOL / vol_floored)
    raw = unconstrained * universe
    weekly = raw.resample("W-FRI").last()

    long_w = weekly.where(weekly > 0, 0)
    short_w = weekly.where(weekly < 0, 0).abs()
    norm_l = normalize_and_cap(long_w)
    norm_s = normalize_and_cap(short_w)
    portfolio_weekly = norm_l - norm_s

    # Forward-fill to daily and shift T+1
    portfolio_daily = portfolio_weekly.reindex(returns.index).ffill().shift(1)
    portfolio_daily = dedup(portfolio_daily)
    safe_universe = dedup(universe)
    portfolio_masked = portfolio_daily.reindex(
        columns=safe_universe.columns, fill_value=0) * safe_universe

    # Re-normalize post-mask
    fl = portfolio_masked.where(portfolio_masked > 0, 0)
    fs = portfolio_masked.where(portfolio_masked < 0, 0).abs()
    portfolio_final = (normalize_and_cap(fl) - normalize_and_cap(fs)).fillna(0)

    weights = enforce_post_shift_strict_gmv(portfolio_final, safe_universe)
    print("  Momentum complete.")
    return weights


# ============================================================================
# STEP 4: STAT-ARB KALMAN PAIRS (Daily)
# ============================================================================
class AnchoredKalmanFilter:
    def __init__(self, alpha, beta, spread_vol):
        self.theta = np.array([[alpha], [beta]])
        self.P = np.eye(2) * 0.01
        self.R = spread_vol ** 2
        self.Q = np.eye(2) * 1e-8
        self.anchor_beta = beta
        self.z_score = 0.0

    def step(self, log_y, log_x):
        H = np.array([[1.0, log_x]])
        P_pred = self.P + self.Q
        e = log_y - (H @ self.theta)[0, 0]
        S = H @ P_pred @ H.T + self.R
        self.z_score = e / np.sqrt(S[0, 0])
        K = (P_pred @ H.T) / S[0, 0]
        self.theta = self.theta + K * e
        self.P = (np.eye(2) - K @ H) @ P_pred
        return self.z_score, self.theta[1, 0]

    def to_dict(self):
        """Serializable filter state for daily persistence."""
        return {
            "theta": self.theta.copy(),
            "P": self.P.copy(),
            "R": self.R,
            "Q": self.Q.copy(),
            "anchor_beta": self.anchor_beta,
            "z_score": self.z_score,
        }

    @classmethod
    def from_dict(cls, d):
        obj = cls.__new__(cls)
        obj.theta = d["theta"]
        obj.P = d["P"]
        obj.R = d["R"]
        obj.Q = d["Q"]
        obj.anchor_beta = d["anchor_beta"]
        obj.z_score = d.get("z_score", 0.0)
        return obj


def load_kalman_state():
    """Returns (filters_dict, positions_dict, last_processed_date) or None for cold start."""
    if not os.path.exists(KALMAN_STATE_PATH):
        return None
    try:
        with open(KALMAN_STATE_PATH, "rb") as f:
            state = pickle.load(f)
        return state
    except Exception as e:
        print(f"  WARN: failed to load Kalman state: {e}. Cold-starting.")
        return None


def save_kalman_state(filters, positions, last_processed_date):
    state = {
        "filters":              {name: kf.to_dict() for name, kf in filters.items()},
        "positions":            dict(positions),
        "last_processed_date":  pd.Timestamp(last_processed_date),
        "saved_at":             pd.Timestamp.now(),
    }
    with open(KALMAN_STATE_PATH, "wb") as f:
        pickle.dump(state, f)


def run_stat_arb(df_hist, universe):
    print("\n" + "=" * 60)
    print("STEP 4: Stat-Arb Kalman Pairs (Daily, warm-start enabled)")
    print("=" * 60)

    config_path = "kalman_universe_config.csv"
    if not os.path.exists(config_path):
        print("  WARNING: kalman_universe_config.csv not found. Skipping Stat-Arb.")
        return None

    pairs_df = pd.read_csv(config_path)
    prices = dedup(df_hist["Adj Close"].copy())

    # ----- Warm-start logic -----
    # Try to resume from saved state. If state's last_processed_date is in the current
    # prices index, only step forward on new days and append to cached weights.
    saved = load_kalman_state()
    use_warm_start = False
    cached_weights = None
    warm_start_idx = 1
    portfolio = None

    if saved is not None and "last_processed_date" in saved:
        last_state_date = pd.Timestamp(saved["last_processed_date"])
        if last_state_date in prices.index:
            # Try to load cached weights too
            if os.path.exists(SA_WEIGHTS_CACHE):
                try:
                    cached_weights = pd.read_pickle(SA_WEIGHTS_CACHE)
                    # Sanity check: cached weights should cover dates up to last_state_date
                    if last_state_date in cached_weights.index:
                        use_warm_start = True
                    else:
                        print(f"  Cached weights don't cover {last_state_date.date()}, cold-starting.")
                except Exception as e:
                    print(f"  Failed to load cached weights: {e}, cold-starting.")
            else:
                print("  No cached weights file, cold-starting.")
        else:
            print(f"  Saved state date {last_state_date.date()} not in current prices index, cold-starting.")

    filters = {}
    positions = {}
    BETA_FLOOR = 0.5

    if use_warm_start:
        last_state_date = pd.Timestamp(saved["last_processed_date"])
        warm_start_idx = prices.index.get_loc(last_state_date) + 1
        n_new_days = len(prices.index) - warm_start_idx
        print(f"  WARM START: resuming from {last_state_date.date()} "
              f"({n_new_days} new business day(s) to process)")

        # Restore filters from saved state; cold-init any new pairs not in saved state
        n_restored = 0
        n_cold = 0
        for _, row in pairs_df.iterrows():
            name = f"{row['asset_y']}_{row['asset_x']}"
            if name in saved["filters"]:
                filters[name] = AnchoredKalmanFilter.from_dict(saved["filters"][name])
                n_restored += 1
            else:
                filters[name] = AnchoredKalmanFilter(
                    row["tls_alpha"], row["tls_beta"], row["spread_vol"])
                n_cold += 1
            positions[name] = saved["positions"].get(name, 0)
        print(f"  Filters restored: {n_restored} from state, {n_cold} cold-initialized (new pairs)")

        # Pre-fill portfolio with cached weights for old dates; new dates start zero
        portfolio = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        common_cols = portfolio.columns.intersection(cached_weights.columns)
        common_idx  = portfolio.index.intersection(cached_weights.index)
        portfolio.loc[common_idx, common_cols] = cached_weights.loc[common_idx, common_cols].values

        if n_new_days == 0:
            print("  No new days to process — submission will use cached weights as-is.")
            # Still save state to refresh timestamp
            save_kalman_state(filters, positions, prices.index[-1])
            weights = portfolio.shift(1).fillna(0)
            weights = enforce_post_shift_strict_gmv(weights, universe)
            return weights
    else:
        print(f"  COLD START: initializing {len(pairs_df)} Kalman filters from config")
        for _, row in pairs_df.iterrows():
            name = f"{row['asset_y']}_{row['asset_x']}"
            filters[name] = AnchoredKalmanFilter(
                row["tls_alpha"], row["tls_beta"], row["spread_vol"])
            positions[name] = 0
        portfolio = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        warm_start_idx = 1

    for i in range(warm_start_idx, len(prices.index)):
        today = prices.index[i]
        pair_vectors = []

        for pair_name, kf in filters.items():
            y_t, x_t = pair_name.split("_")
            p_y, p_x = prices.loc[today, y_t], prices.loc[today, x_t]

            if pd.isna(p_y) or pd.isna(p_x) or p_y == 0 or p_x == 0:
                positions[pair_name] = 0
                continue

            z, beta = kf.step(np.log(p_y), np.log(p_x))

            # Kill switch
            scale = max(abs(kf.anchor_beta), BETA_FLOOR)
            if abs(beta - kf.anchor_beta) > 0.40 * scale or abs(z) > 4.0:
                positions[pair_name] = 0
                continue

            # Entry/exit
            pos = positions[pair_name]
            if z > 2.0 and pos == 0:    pos = -1
            elif z < -2.0 and pos == 0: pos = 1
            elif pos == -1 and z < 0:   pos = 0
            elif pos == 1 and z > 0:    pos = 0
            positions[pair_name] = pos

            if pos != 0:
                raw_scale = 1.0 / np.sqrt(kf.R)
                denom = 1.0 + abs(beta)
                pair_vectors.append({
                    "y": y_t, "x": x_t,
                    "yw": pos * raw_scale / denom,
                    "xw": -pos * raw_scale * beta / denom,
                })

        if not pair_vectors:
            continue

        # Water-fill normalization
        c = np.ones(len(pair_vectors))
        for _ in range(20):
            temp = pd.Series(0.0, index=prices.columns)
            for idx, p in enumerate(pair_vectors):
                temp[p["y"]] += p["yw"] * c[idx]
                temp[p["x"]] += p["xw"] * c[idx]
            gmv = temp.abs().sum()
            if gmv < 1e-8:
                break
            c *= 1.0 / gmv

            temp = pd.Series(0.0, index=prices.columns)
            for idx, p in enumerate(pair_vectors):
                temp[p["y"]] += p["yw"] * c[idx]
                temp[p["x"]] += p["xw"] * c[idx]
            max_w = temp.abs().max()
            if max_w <= MAX_WEIGHT + 1e-6:
                break
            breached = temp.index[temp.abs() > MAX_WEIGHT]
            bp = [idx for idx, p in enumerate(pair_vectors)
                  if p["y"] in breached or p["x"] in breached]
            if len(bp) == len(pair_vectors):
                c *= MAX_WEIGHT / max_w
                break
            for idx in bp:
                c[idx] *= MAX_WEIGHT / max_w

        for idx, p in enumerate(pair_vectors):
            portfolio.loc[today, p["y"]] += p["yw"] * c[idx]
            portfolio.loc[today, p["x"]] += p["xw"] * c[idx]

    # Persist filter state + cached weights for tomorrow's warm-start
    last_processed = prices.index[-1]
    save_kalman_state(filters, positions, last_processed)
    portfolio.to_pickle(SA_WEIGHTS_CACHE)
    print(f"  Saved Kalman state to {KALMAN_STATE_PATH} (last_processed={last_processed.date()})")

    # T+1 shift + universe mask
    weights = portfolio.shift(1).fillna(0)
    weights = enforce_post_shift_strict_gmv(weights, universe)
    print("  Stat-Arb complete.")
    return weights


# ============================================================================
# STEP 5: INVERSE-VOLATILITY ENSEMBLE BLENDING
# ============================================================================
def blend_ensemble(w_rev, w_mom, w_sa, returns, universe):
    """Equal-vol (inverse-vol) blend across the three styles:
    {Reversal super-sleeve, Momentum, Stat-Arb}."""
    print("\n" + "=" * 60)
    print("STEP 5: Equal-Vol Ensemble Blending {Reversal, Mom, StatArb}")
    print("=" * 60)

    def rolling_vol(weights, rets, window=60):
        ar = rets.reindex(index=weights.index, columns=weights.columns).fillna(0)
        pnl = (weights * ar).sum(axis=1)
        return (pnl.rolling(window, min_periods=20).std() * np.sqrt(252)).clip(lower=0.05)

    vol_rev = rolling_vol(w_rev, returns)
    vol_mom = rolling_vol(w_mom, returns)
    vol_sa = rolling_vol(w_sa, returns)

    inv_rev, inv_mom, inv_sa = 1/vol_rev, 1/vol_mom, 1/vol_sa
    total = inv_rev + inv_mom + inv_sa

    # Shift by 1 to prevent lookahead
    c_rev = (inv_rev / total).shift(1).fillna(1/3)
    c_mom = (inv_mom / total).shift(1).fillna(1/3)
    c_sa  = (inv_sa / total).shift(1).fillna(1/3)

    print(f"  Latest allocations: Reversal={c_rev.iloc[-1]:.1%}, "
          f"Mom={c_mom.iloc[-1]:.1%}, SA={c_sa.iloc[-1]:.1%}")

    ensemble = (w_rev.multiply(c_rev, axis=0)
              + w_mom.multiply(c_mom, axis=0)
              + w_sa.multiply(c_sa, axis=0))

    # Normalize long/short separately
    el = ensemble.where(ensemble > 0, 0)
    es = ensemble.where(ensemble < 0, 0).abs()
    final = (normalize_and_cap(el) - normalize_and_cap(es)).fillna(0)
    final = enforce_post_shift_strict_gmv(final, universe)

    print("  Ensemble blending complete.")
    return final


# ============================================================================
# STEP 6: RISK SCALING & ADV ENFORCEMENT
# ============================================================================
def scale_and_enforce(ensemble_weights, returns, df_adv_60):
    print("\n" + "=" * 60)
    print("STEP 6: Risk Scaling & Constraint Enforcement")
    print("=" * 60)

    # Unit portfolio vol over last 120 days
    ar = returns.reindex(index=ensemble_weights.index,
                         columns=ensemble_weights.columns).fillna(0)
    pnl = (ensemble_weights * ar).sum(axis=1)
    unit_vol = pnl.tail(VOL_LOOKBACK).std() * np.sqrt(252)

    if unit_vol < 0.0001:
        print("  WARNING: Near-zero portfolio vol. Defaulting to $1M notional.")
        notional = 1_000_000.0
    else:
        notional = TARGET_RISK / unit_vol

    print(f"  Unit vol (120d): {unit_vol:.4f}")
    print(f"  Target risk: ${TARGET_RISK:,.0f} -> Gross notional: ${notional:,.0f}")

    latest = ensemble_weights.iloc[-1]
    active = latest[latest != 0]
    targets = active * notional

    # ADV hard cap: |target_i| <= 2.5% * ADV_60_i
    adv_dedup = dedup(df_adv_60)
    adv_latest = adv_dedup.iloc[-1].reindex(active.index).fillna(0)
    adv_cap = ADV_LIMIT_PCT * adv_latest
    breached = targets.abs() > adv_cap
    n_breached = breached.sum()
    if n_breached > 0:
        print(f"  Clipping {n_breached} positions to 2.5% ADV limit")
        targets[breached] = np.sign(targets[breached]) * adv_cap[breached]

    # Position cap: |target_i| <= $2M
    big = targets.abs() > MAX_POSITION_USD
    if big.sum() > 0:
        print(f"  Clipping {big.sum()} positions to ${MAX_POSITION_USD/1e6:.0f}M cap")
        targets[big] = np.sign(targets[big]) * MAX_POSITION_USD

    # Restore dollar neutrality after clipping
    net = targets.sum()
    longs = targets[targets > 0]
    shorts = targets[targets < 0]
    if abs(net) > 1.0 and len(longs) > 0 and len(shorts) > 0:
        if net > 0:
            longs *= (longs.sum() - net) / longs.sum()
        else:
            shorts *= (shorts.abs().sum() + net) / shorts.abs().sum()
        targets = pd.concat([longs, shorts])

    gmv = targets.abs().sum()
    net_final = targets.sum()
    print(f"  Final GMV: ${gmv:,.0f}")
    print(f"  Final Net: ${net_final:,.0f}")
    print(f"  Active positions: {len(targets)}")
    print(f"  Max position: ${targets.abs().max():,.0f}")
    return targets


# ============================================================================
# STEP 7: SUBMISSION CSV WITH CORRECT EXCHANGE SUFFIXES
# ============================================================================
def generate_csv(targets):
    print("\n" + "=" * 60)
    print("STEP 7: Generating Submission CSV")
    print("=" * 60)

    # Robust RIC resolution: map -> yfinance fallback (cached) -> exclude unresolved.
    # We never silently default to .OQ: a wrong suffix is silently rejected by QRT,
    # which previously dropped ~58% of the book (all NYSE/AMEX names sent as .OQ).
    from ric_resolver import resolve as resolve_rics
    codes_map, unresolved = resolve_rics(list(targets.index),
                                         ric_map_csv="ric_exchange_map.csv",
                                         allow_fetch=True)

    if unresolved:
        print(f"  EXCLUDING {len(unresolved)} unresolved ticker(s) from submission "
              f"(no valid RIC): {unresolved[:20]}")
        targets = targets.drop(index=unresolved)
        # Excluding names breaks dollar-neutrality slightly; rebalance long/short books.
        net = targets.sum()
        longs, shorts = targets[targets > 0], targets[targets < 0]
        if abs(net) > 1.0 and len(longs) > 0 and len(shorts) > 0:
            if net > 0:
                longs = longs * (longs.sum() - net) / longs.sum()
            else:
                shorts = shorts * (shorts.abs().sum() + net) / shorts.abs().sum()
            targets = pd.concat([longs, shorts])
            print(f"  Re-neutralized after exclusion: net=${targets.sum():,.2f}")

    codes = [codes_map[t] for t in targets.index]

    submission = pd.DataFrame({
        "internal_code": codes,
        "currency": "USD",
        "target_notional": targets.values.round(2),
    })

    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    filename = f"qrt_academy_IND22_{timestamp}.csv"
    submission.to_csv(filename, index=False)

    print(f"  Saved: {filename}")
    print(f"  Positions: {len(submission)}")
    print(f"  GMV: ${submission['target_notional'].abs().sum():,.0f}")
    print(f"  Net: ${submission['target_notional'].sum():,.0f}")
    return filename


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 60)
    print("QRT ACADEMY — ENSEMBLE PORTFOLIO GENERATOR")
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Step 0: refresh data from yfinance (no-op if pickle already current)
    refresh_data_from_yfinance()

    # Step 1
    df_hist, returns, universe, df_adv_60 = load_data()

    # Step 2: mean reversion (Alpha#11)
    w_mr = run_mean_reversion(df_hist, returns, universe, df_adv_60)

    # Step 2b: reversal super-sleeve (Alpha#11 + Alpha#24/#41/#100)
    w_rev = build_reversal_supersleeve(w_mr, df_hist, returns, universe)

    # Step 3: momentum (weekly W-FRI — UNCHANGED)
    w_mom = run_momentum(df_hist, returns, universe)

    # Step 4: stat-arb (Kalman warm-start — UNCHANGED)
    w_sa = run_stat_arb(df_hist, universe)
    if w_sa is None:
        print("  Creating zero Stat-Arb weights as fallback.")
        w_sa = pd.DataFrame(0.0, index=w_rev.index, columns=w_rev.columns)

    # Align all three styles
    common_cols = w_rev.columns.intersection(w_mom.columns).intersection(w_sa.columns)
    common_idx = w_rev.index.intersection(w_mom.index).intersection(w_sa.index)
    w_rev = w_rev.reindex(index=common_idx, columns=common_cols).fillna(0)
    w_mom = w_mom.reindex(index=common_idx, columns=common_cols).fillna(0)
    w_sa = w_sa.reindex(index=common_idx, columns=common_cols).fillna(0)
    universe_aligned = universe.reindex(index=common_idx, columns=common_cols).fillna(0)
    returns_aligned = returns.reindex(index=common_idx, columns=common_cols).fillna(0)

    # Step 5: equal-vol blend {Reversal, Mom, StatArb}
    ensemble = blend_ensemble(w_rev, w_mom, w_sa, returns_aligned, universe_aligned)

    # Step 6
    targets = scale_and_enforce(ensemble, returns_aligned, df_adv_60)

    # Step 7
    filename = generate_csv(targets)

    print("\n" + "=" * 60)
    print(f"DONE — Submit '{filename}' via submit_portfolio.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
