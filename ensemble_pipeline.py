import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Import utils
sys.path.append(os.path.abspath('phase2_qrt_challenge/scripts'))
import utils

DATA_DIR = "stores"

class AnchoredKalmanFilter:
    def __init__(self, initial_alpha, initial_beta, spread_vol):
        self.theta = np.array([[initial_alpha], [initial_beta]])
        self.P = np.eye(2) * 0.01
        self.R = spread_vol ** 2
        self.Q = np.eye(2) * 1e-8 
        self.anchor_beta = initial_beta
        self.z_score = 0.0

    def step(self, log_y, log_x):
        H = np.array([[1.0, log_x]])
        theta_pred = self.theta
        P_pred = self.P + self.Q
        e = log_y - (H @ theta_pred)[0, 0]
        S = H @ P_pred @ H.T + self.R
        self.z_score = e / np.sqrt(S[0,0])
        K = (P_pred @ H.T) / S[0,0]
        self.theta = theta_pred + K * e
        self.P = (np.eye(2) - K @ H) @ P_pred
        return self.z_score, self.theta[1, 0]

def execute_pro_portfolio(df_final_pairs, prices, universe_df, max_weight=0.098):
    print(f"Initializing {len(df_final_pairs)} Pro Filters...")
    active_filters = {}
    for _, row in df_final_pairs.iterrows():
        pair_name = f"{row['asset_y']}_{row['asset_x']}"
        active_filters[pair_name] = AnchoredKalmanFilter(
            initial_alpha=row['tls_alpha'], 
            initial_beta=row['tls_beta'], 
            spread_vol=row['spread_vol']
        )
        
    portfolio_weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    positions = {pair: 0 for pair in active_filters.keys()}
    
    for i in range(1, len(prices.index)):
        today = prices.index[i]
        daily_pair_vectors = [] 
        
        for pair_name, kf in active_filters.items():
            y_t, x_t = pair_name.split('_')
            p_y, p_x = prices.loc[today, y_t], prices.loc[today, x_t]
            
            if pd.isna(p_y) or pd.isna(p_x) or p_y == 0 or p_x == 0:
                positions[pair_name] = 0
                continue
            if universe_df.loc[today, y_t] == 0 or universe_df.loc[today, x_t] == 0:
                positions[pair_name] = 0
                continue
                
            z_score, current_beta = kf.step(np.log(p_y), np.log(p_x))
            
            if abs((current_beta - kf.anchor_beta) / kf.anchor_beta) > 0.40 or abs(z_score) > 4.0:
                positions[pair_name] = 0
                continue
            
            pos = positions[pair_name]
            if z_score > 2.0 and pos == 0: pos = -1 
            elif z_score < -2.0 and pos == 0: pos = 1  
            elif pos == -1 and z_score < 0: pos = 0  
            elif pos == 1 and z_score > 0: pos = 0  
            positions[pair_name] = pos
            
            if pos != 0:
                raw_scale = 1.0 / np.sqrt(kf.R)
                daily_pair_vectors.append({
                    'y': y_t, 'x': x_t, 
                    'yw': pos * raw_scale, 
                    'xw': -pos * raw_scale * current_beta
                })

        if not daily_pair_vectors: continue

        c = np.ones(len(daily_pair_vectors))
        for _ in range(20):
            temp_weights = pd.Series(0.0, index=prices.columns)
            for idx, p in enumerate(daily_pair_vectors):
                temp_weights[p['y']] += p['yw'] * c[idx]
                temp_weights[p['x']] += p['xw'] * c[idx]
                
            current_gmv = temp_weights.abs().sum()
            if current_gmv < 1e-8: break
                
            c *= (1.0 / current_gmv)
            
            temp_weights = pd.Series(0.0, index=prices.columns)
            for idx, p in enumerate(daily_pair_vectors):
                temp_weights[p['y']] += p['yw'] * c[idx]
                temp_weights[p['x']] += p['xw'] * c[idx]
                
            max_w = temp_weights.abs().max()
            if max_w <= max_weight + 1e-6: break
                
            breached_stocks = temp_weights.index[temp_weights.abs() > max_weight]
            breached_pairs = [idx for idx, p in enumerate(daily_pair_vectors) if p['y'] in breached_stocks or p['x'] in breached_stocks]
            
            if len(breached_pairs) == len(daily_pair_vectors):
                c *= (max_weight / max_w)
                break
                
            for idx in breached_pairs:
                c[idx] *= (max_weight / max_w)
                
        for idx, p in enumerate(daily_pair_vectors):
            portfolio_weights.loc[today, p['y']] += p['yw'] * c[idx]
            portfolio_weights.loc[today, p['x']] += p['xw'] * c[idx]

    return portfolio_weights

def enforce_post_shift_strict_gmv(shifted_portfolio, universe_df):
    # Just mask with universe, do not re-normalize daily to avoid massive turnover.
    # Since strategies already enforce GMV=1.0 internally (or weekly), missing stocks
    # just slightly reduce GMV (e.g. to 0.99) which is fine and avoids full portfolio rebalancing.
    return shifted_portfolio * universe_df

def normalize_and_cap(w_matrix, target=0.50, cap=0.099):
    row_sums = w_matrix.sum(axis=1) + 1e-10
    w = w_matrix.div(row_sums, axis=0) * target
    w = w.clip(upper=cap)
    row_sums_final = w.sum(axis=1) + 1e-10
    w = w.div(row_sums_final, axis=0) * target
    return w

def main():
    print("1. Loading required data...")
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
    # Warmup Kalman filters for 1 year prior to OOS
    warmup_start = max_date - pd.DateOffset(months=18)

    prices = df_historical['Adj Close']
    prices = prices.loc[:, ~prices.columns.duplicated()]
    
    # --- 2. Momentum Strategy ---
    print("\n2. Generating Momentum Portfolio (Actual Ranked Hysteresis)...")
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

    # Regime filter - apply weekly to avoid daily flickering trades
    spy_adj_close = prices['SPY'] if 'SPY' in prices.columns else prices.mean(axis=1)
    spy_sma_200 = spy_adj_close.rolling(window=200).mean()
    trend_is_positive = spy_adj_close > spy_sma_200
    spy_daily_rets = np.log(spy_adj_close / spy_adj_close.shift(1))
    spy_ann_vol = spy_daily_rets.rolling(window=20).std() * np.sqrt(252)
    volatility_is_safe = spy_ann_vol < 0.35
    regime_mask_daily = (trend_is_positive & volatility_is_safe).shift(1).fillna(False)
    regime_mask_weekly = regime_mask_daily.resample('W-FRI').last().reindex(returns.index).ffill()
    momentum_portfolio = momentum_portfolio.multiply(regime_mask_weekly, axis=0)

    # --- 3. Stat Arb Strategy ---
    print("\n3. Generating Stat Arb Portfolio (heating up Kalman Filter)...")
    df_final_pairs = pd.read_csv('kalman_universe_config.csv')
    oos_prices = prices.loc[warmup_start:max_date]
    universe_df_oos = universe_5m.loc[warmup_start:max_date]
    qrt_portfolio_weights = execute_pro_portfolio(df_final_pairs, oos_prices, universe_df_oos)
    qrt_portfolio_weights = qrt_portfolio_weights.shift(1).fillna(0)
    
    # Strictly maintain pairwise weights post-shift
    stat_arb_portfolio = qrt_portfolio_weights.reindex(index=returns.index, columns=returns.columns).fillna(0)
    
    for _, row in df_final_pairs.iterrows():
        y_t = row['asset_y']
        x_t = row['asset_x']
        # If either leg drops out of universe_5m on day t, zero out BOTH legs for StatArb
        invalid_mask = (universe_5m[y_t] == 0) | (universe_5m[x_t] == 0)
        stat_arb_portfolio.loc[invalid_mask, y_t] = 0
        stat_arb_portfolio.loc[invalid_mask, x_t] = 0

    # Align Alpha
    alpha_portfolio = alpha_weights.reindex(index=returns.index, columns=returns.columns).fillna(0)
    
    # --- 4. Inverse Volatility Weighting ---
    print("\n4. Combining strategies via Inverse Volatility Weighting...")
    # Calculate PnL for each strategy
    alpha_pnl = (alpha_portfolio * returns).sum(axis=1)
    mom_pnl = (momentum_portfolio * returns).sum(axis=1)
    sa_pnl = (stat_arb_portfolio * returns).sum(axis=1)

    # Calculate 60-day rolling volatility
    alpha_vol = alpha_pnl.rolling(window=60, min_periods=30).std()
    mom_vol = mom_pnl.rolling(window=60, min_periods=30).std()
    sa_vol = sa_pnl.rolling(window=60, min_periods=30).std()

    inv_alpha_vol = 1.0 / alpha_vol.clip(lower=1e-6)
    inv_mom_vol = 1.0 / mom_vol.clip(lower=1e-6)
    inv_sa_vol = 1.0 / sa_vol.clip(lower=1e-6)

    total_inv_vol = inv_alpha_vol + inv_mom_vol + inv_sa_vol

    w_alpha = inv_alpha_vol / total_inv_vol
    w_mom = inv_mom_vol / total_inv_vol
    w_sa = inv_sa_vol / total_inv_vol

    # Apply universe masks independently
    # For Alpha and Momentum, if a stock drops out, just drop that stock.
    alpha_masked = alpha_portfolio * universe_5m
    momentum_masked = momentum_portfolio * universe_5m
    # For StatArb, stat_arb_portfolio is ALREADY pairwise masked, so it drops both legs cleanly.
    
    ensemble_raw = w_alpha.values[:, None] * alpha_masked + w_mom.values[:, None] * momentum_masked + w_sa.values[:, None] * stat_arb_portfolio
    
    # The strategies can bet against each other. Let's pick a random date and stock to show exact weights.
    check_date = ensemble_raw.index[-1]
    example_stock = ensemble_raw.loc[check_date].abs().idxmax()
    print(f"\n--- Checking Conflicting Bets for {example_stock} on {check_date.date()} ---")
    print(f"Alpha weight:    {alpha_masked.loc[check_date, example_stock]:.4f}  (Strategy Weight: {w_alpha.loc[check_date]:.2f})")
    print(f"Momentum weight: {momentum_masked.loc[check_date, example_stock]:.4f}  (Strategy Weight: {w_mom.loc[check_date]:.2f})")
    print(f"StatArb weight:  {stat_arb_portfolio.loc[check_date, example_stock]:.4f}  (Strategy Weight: {w_sa.loc[check_date]:.2f})")
    print(f"Ensemble raw:    {ensemble_raw.loc[check_date, example_stock]:.4f}")
    
    # No daily re-normalization to avoid massive turnover. The GMV will naturally be <= 1.0.
    ensemble_portfolio = ensemble_raw.fillna(0)

    # --- 5. Backtest Ensemble ---
    print("\n5. Testing the ensemble portfolio for the last 6 months...")
    df_pnl = pd.DataFrame({'Alpha': alpha_pnl, 'Momentum': mom_pnl, 'StatArb': sa_pnl, 'Ensemble': (ensemble_portfolio * returns).sum(axis=1)}).loc[oos_start:]
    
    print("\nCorrelation matrix between strategies (Last 6 Months):")
    print(df_pnl.corr().round(3))

    oos_ensemble = ensemble_portfolio.loc[oos_start:max_date]
    oos_returns = returns.loc[oos_start:max_date]
    oos_universe = universe_5m.loc[oos_start:max_date]
    
    print("\nExecuting OOS Backtest...")
    net_sharpe, daily_pnl = utils.backtest_portfolio(
        portfolio=oos_ensemble, 
        returns=oos_returns, 
        universe=oos_universe, 
        plot_=False, 
        print_=True
    )
    print(f"Ensemble Total Net PnL: {daily_pnl.sum():.2%}")
    
    # Also backtesting individual strategies for comparison
    print("\n--- Benchmark: Alpha only ---")
    _, alpha_pnl = utils.backtest_portfolio(portfolio=alpha_masked.loc[oos_start:max_date], returns=oos_returns, universe=oos_universe, plot_=False, print_=True)
    print(f"Alpha Total Net PnL: {alpha_pnl.sum():.2%}")
    
    print("\n--- Benchmark: Momentum only ---")
    _, mom_pnl = utils.backtest_portfolio(portfolio=momentum_masked.loc[oos_start:max_date], returns=oos_returns, universe=oos_universe, plot_=False, print_=True)
    print(f"Momentum Total Net PnL: {mom_pnl.sum():.2%}")
    
    print("\n--- Benchmark: StatArb only ---")
    _, sa_pnl = utils.backtest_portfolio(portfolio=stat_arb_portfolio.loc[oos_start:max_date], returns=oos_returns, universe=oos_universe, plot_=False, print_=True)
    print(f"StatArb Total Net PnL: {sa_pnl.sum():.2%}")

if __name__ == '__main__':
    main()
