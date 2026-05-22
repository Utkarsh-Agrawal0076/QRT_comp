import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath('phase2_qrt_challenge/scripts'))
import utils
from ensemble_pipeline import AnchoredKalmanFilter, execute_pro_portfolio, enforce_post_shift_strict_gmv, normalize_and_cap

DATA_DIR = "stores"

def extract_top_positions():
    print("Loading data...")
    df_historical = pd.read_pickle('top_5000_yf_data.pkl')
    df_historical = df_historical.loc[:, ~df_historical.columns.duplicated()]
    universe_5m = pd.read_parquet(os.path.join(DATA_DIR, 'universe_5m.parquet'))
    universe_5m = universe_5m.loc[:, ~universe_5m.columns.duplicated()]
    returns = pd.read_parquet(os.path.join(DATA_DIR, 'returns.parquet'))
    returns = returns.loc[:, ~returns.columns.duplicated()]
    alpha_weights = pd.read_parquet(os.path.join(DATA_DIR, 't2_weights.parquet'))
    alpha_weights = alpha_weights.loc[:, ~alpha_weights.columns.duplicated()]
    
    max_date = df_historical.index.max()
    oos_start = max_date - pd.DateOffset(months=6)
    warmup_start = max_date - pd.DateOffset(months=18)
    prices = df_historical['Adj Close'].loc[:, ~df_historical['Adj Close'].columns.duplicated()]
    
    # --- 1. Mean Reversion (Alpha) ---
    print("Extracting Mean Reversion...")
    alpha_portfolio = alpha_weights.reindex(index=returns.index, columns=returns.columns).fillna(0)
    alpha_masked = alpha_portfolio * universe_5m
    alpha_latest = alpha_masked.loc[(alpha_masked != 0).any(axis=1)].iloc[-1].astype(float)
    
    # --- 2. Momentum ---
    print("Extracting Momentum...")
    meta_df = pd.read_csv('top_5000_us_by_marketcap.csv')
    meta_df['symbol'] = meta_df['symbol'].str.replace('/', '-')
    sector_dict = meta_df.set_index('symbol')['sector'].to_dict()
    aligned_sectors = pd.Series(returns.columns).map(sector_dict).values
    
    spy_adj_close = prices['SPY'] if 'SPY' in prices.columns else prices.mean(axis=1)
    spy_rets_aligned = spy_adj_close.pct_change(fill_method=None).fillna(0)
    spy_var = spy_rets_aligned.rolling(window=252, min_periods=63).var()
    
    rolling_cov = returns.rolling(window=252, min_periods=63).cov(spy_rets_aligned)
    rolling_beta = rolling_cov.div(spy_var, axis=0).fillna(1.0)
    market_component = rolling_beta.multiply(spy_rets_aligned, axis=0)
    residual_returns = returns - market_component
    
    res_ret_6m_lagged = residual_returns.shift(21).rolling(window=105).sum()
    valid_returns = res_ret_6m_lagged * universe_5m.replace(0, np.nan)
    
    ranks = valid_returns.copy()
    ranks[:] = np.nan
    unique_sectors = [s for s in pd.Series(aligned_sectors).unique() if pd.notna(s)]
    for sector in unique_sectors:
        sector_cols = valid_returns.columns[aligned_sectors == sector]
        if len(sector_cols) > 0:
            ranks[sector_cols] = valid_returns[sector_cols].rank(axis=1, pct=True)
            
    LONG_ENTER, LONG_EXIT = 0.90, 0.80
    SHORT_ENTER, SHORT_EXIT = 0.20, 0.30
    
    long_state = pd.DataFrame(index=ranks.index, columns=ranks.columns, data=np.nan)
    long_state[ranks >= LONG_ENTER] = 1
    long_state[ranks < LONG_EXIT] = -1
    long_signals = long_state.ffill().replace(-1, 0).fillna(0)
    
    short_state = pd.DataFrame(index=ranks.index, columns=ranks.columns, data=np.nan)
    short_state[ranks <= SHORT_ENTER] = 1
    short_state[ranks > SHORT_EXIT] = -1
    short_signals = short_state.ffill().replace(-1, 0).fillna(0) * -1
    
    df_tradable_signal = (long_signals + short_signals).fillna(0)

    log_hl = np.log(df_historical['High'] / df_historical['Low']) ** 2
    log_co = np.log(df_historical['Close'] / df_historical['Open']) ** 2
    gk_daily_var = 0.5 * log_hl - ((2 * np.log(2) - 1) * log_co)
    gk_ewma_var = gk_daily_var.ewm(com=60, min_periods=60).mean()
    df_annualized_vol = np.sqrt(gk_ewma_var * 252)
    df_vol_floored = df_annualized_vol.clip(lower=0.15)

    TARGET_VOL = 0.40
    df_unconstrained_weights = df_tradable_signal * (TARGET_VOL / df_vol_floored)
    raw_weights = df_unconstrained_weights * universe_5m

    weekly_raw_weights = raw_weights.resample('W-FRI').last()
    long_w = weekly_raw_weights.where(weekly_raw_weights > 0, 0)
    short_w = weekly_raw_weights.where(weekly_raw_weights < 0, 0).abs()

    normalized_longs = normalize_and_cap(long_w)
    normalized_shorts = normalize_and_cap(short_w)

    portfolio_weekly = normalized_longs - normalized_shorts
    portfolio_daily = portfolio_weekly.reindex(returns.index).ffill()
    portfolio_shifted = portfolio_daily.shift(1)
    portfolio_aligned = portfolio_shifted.reindex(columns=returns.columns, fill_value=0)
    momentum_portfolio = portfolio_aligned.fillna(0)

    spy_sma_200 = spy_adj_close.rolling(window=200).mean()
    trend_is_positive = spy_adj_close > spy_sma_200
    spy_daily_rets = np.log(spy_adj_close / spy_adj_close.shift(1))
    spy_ann_vol = spy_daily_rets.rolling(window=20).std() * np.sqrt(252)
    volatility_is_safe = spy_ann_vol < 0.35
    regime_mask_daily = (trend_is_positive & volatility_is_safe).shift(1).fillna(False)
    regime_mask_weekly = regime_mask_daily.resample('W-FRI').last().reindex(returns.index).ffill()
    momentum_portfolio = momentum_portfolio.multiply(regime_mask_weekly, axis=0)
    
    momentum_masked = momentum_portfolio * universe_5m
    momentum_latest = momentum_masked.loc[(momentum_masked != 0).any(axis=1)].iloc[-1].astype(float)

    # --- 3. Stat Arb ---
    print("Extracting Stat Arb...")
    df_final_pairs = pd.read_csv('kalman_universe_config.csv')
    oos_prices = prices.loc[warmup_start:max_date]
    universe_df_oos = universe_5m.loc[warmup_start:max_date]
    qrt_portfolio_weights = execute_pro_portfolio(df_final_pairs, oos_prices, universe_df_oos)
    qrt_portfolio_weights = qrt_portfolio_weights.shift(1).fillna(0)
    
    stat_arb_portfolio = qrt_portfolio_weights.reindex(index=returns.index, columns=returns.columns).fillna(0)
    for _, row in df_final_pairs.iterrows():
        y_t, x_t = row['asset_y'], row['asset_x']
        invalid_mask = (universe_5m[y_t] == 0) | (universe_5m[x_t] == 0)
        stat_arb_portfolio.loc[invalid_mask, y_t] = 0
        stat_arb_portfolio.loc[invalid_mask, x_t] = 0

    sa_latest = stat_arb_portfolio.loc[(stat_arb_portfolio != 0).any(axis=1)].iloc[-1].astype(float)
    
    # Note: We display raw component weights (which represent the fractional portfolio allocation
    # BEFORE the inverse-volatility parity weighting is applied).
    
    def print_top(series, name):
        print(f"\n==========================================")
        print(f"=== Top 5 Positions: {name} ===")
        print(f"==========================================")
        print("--- Largest Longs ---")
        longs = series[series > 0].nlargest(5)
        for ticker, weight in longs.items():
            print(f"{ticker:<10}: {weight:.4f}")
            
        print("\n--- Largest Shorts ---")
        shorts = series[series < 0].nsmallest(5)
        for ticker, weight in shorts.items():
            print(f"{ticker:<10}: {weight:.4f}")

    print_top(alpha_latest, "Mean Reversion (Alpha)")
    print_top(momentum_latest, "Sector-Neutral Momentum")
    print_top(sa_latest, "Statistical Arbitrage (Kalman Pairs)")

if __name__ == '__main__':
    extract_top_positions()
