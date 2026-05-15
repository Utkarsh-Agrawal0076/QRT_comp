import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Import utils
sys.path.append(os.path.abspath('phase2_qrt_challenge/scripts'))
import utils

DATA_DIR = "stores"

def main():
    print("1. Loading required data...")
    df_historical = pd.read_pickle('top_5000_yf_data.pkl')
    universe = pd.read_parquet(os.path.join(DATA_DIR, 'universe_5m.parquet'))
    returns = pd.read_parquet(os.path.join(DATA_DIR, 'returns.parquet'))
    
    # 1. Compute Base Features
    print("2. Computing Base Features (VWAP, diffs, volume delta)...")
    high = df_historical['High']
    low = df_historical['Low']
    close = df_historical['Close']
    volume = df_historical['Volume']
    
    # Surrogate VWAP
    vwap = (high + low + close) / 3.0
    
    # ADV for constraints
    adv_60 = (close * volume).rolling(window=60, min_periods=60).mean()
    
    # Features for rank
    diff_vwap_close = vwap - close
    max_diff = diff_vwap_close.rolling(window=3).max()
    min_diff = diff_vwap_close.rolling(window=3).min()
    vol_delta = volume.diff(3)
    
    # 2. Compute Alpha & Cross-Sectional Ranking
    print("3. Computing Alpha...")
    rank_max = max_diff.rank(axis=1, pct=True)
    rank_min = min_diff.rank(axis=1, pct=True)
    rank_vol_delta = vol_delta.rank(axis=1, pct=True)
    
    alpha = (rank_max + rank_min) * rank_vol_delta
    
    # Filter alpha by universe early
    alpha = alpha.where(universe == 1, np.nan)
    
    # 3. Portfolio Neutralization & Constraints
    print("4. Applying Portfolio Constraints Iteratively...")
    
    # Demean the alpha cross-sectionally to achieve dollar neutrality BEFORE scaling
    alpha = alpha.sub(alpha.mean(axis=1), axis=0)
    
    # Initial Neutralization
    weights = alpha.apply(utils.scale_to_book_long_short, axis=1)
    
    # We risk $500k. Limit = 0.025 * ADV / AUM
    AUM = 500000.0
    adv_limit_df = (0.025 * adv_60) / AUM
    max_weight = 0.10
    
    # Iterative clipping
    max_iter = 10
    for i in range(max_iter):
        w_sign = np.sign(weights)
        w_abs = np.abs(weights)
        
        limit_matrix = np.minimum(adv_limit_df, max_weight)
        
        weights_clipped_abs = np.minimum(w_abs, limit_matrix)
        weights_clipped = w_sign * weights_clipped_abs
        
        # Check convergence
        diff = np.abs(weights.fillna(0) - weights_clipped.fillna(0)).max().max()
        if diff < 1e-5:
            print(f"   Converged after {i} iterations.")
            break
            
        weights = weights_clipped
        # Re-scale positive to 0.5 and negative to -0.5
        weights = weights.apply(utils.scale_to_book_long_short, axis=1)
        
    final_weights = weights.fillna(0)
    
    # 4. T+2 Alignment
    print("5. Shifting Portfolio T+2...")
    t2_weights = final_weights.shift(2).fillna(0)
    
    # Make sure we only hold universal assets after shift
    t2_weights = t2_weights.where(universe == 1, 0)
    
    # After applying the universe mask on day T+2, some stocks may have dropped out,
    # lowering our gross exposure. We must re-normalize to ensure Unit Capital (Total sum |weight| = 1.0)
    t2_weights = t2_weights.apply(utils.scale_to_book_long_short, axis=1)
    
    # Quick sanity check for unit capital constraint
    abs_sums = t2_weights.abs().sum(axis=1)
    invalid_rows = ((abs_sums - 1.0).abs() > 0.01) & (abs_sums > 1e-10)
    if invalid_rows.sum() > 0:
        print(f"Warning: {invalid_rows.sum()} rows failed unit capital constraint (likely bad NAs). Cleaning up...")
        t2_weights.loc[invalid_rows] = 0.0
        
    print("Saving T+2 weights to stores/t2_weights.parquet...")
    t2_weights.to_parquet(os.path.join(DATA_DIR, 't2_weights.parquet'), engine='pyarrow')

    # 5. Backtesting
    print("\n================== BACKTEST RESULTS ==================")
    years = range(2020, datetime.today().year + 1)
    
    for y in years:
        start = f"{y}-01-01"
        end = f"{y}-12-31"
        
        w_y = t2_weights.loc[start:end]
        r_y = returns.loc[start:end]
        u_y = universe.loc[start:end]
        
        if w_y.empty or w_y.abs().sum().sum() == 0:
            continue
            
        print(f"\n--- Year {y} ---")
        try:
            # We wrap it in a try-catch to ensure one bad year doesn't block the rest
            utils.backtest_portfolio(w_y, r_y, u_y, plot_=False, print_=True)
        except Exception as e:
            print(f"Could not backtest {y}: {e}")
            
    print("\n--- Entire Post-2020 Period ---")
    start = "2020-01-01"
    w_all = t2_weights.loc[start:]
    r_all = returns.loc[start:]
    u_all = universe.loc[start:]
    
    try:
        sr_all, pnl_all = utils.backtest_portfolio(w_all, r_all, u_all, plot_=False, print_=True)
        
        # Calculate Rolling 120 Day Sharpe
        print("\n--- Recent 120-Day Sharpe ---")
        rolling_120_pnl = pnl_all.tail(120)
        
        if len(rolling_120_pnl) == 120 and rolling_120_pnl.std() != 0:
            rolling_sr_120 = (rolling_120_pnl.mean() / rolling_120_pnl.std()) * np.sqrt(252)
            print(f"120-Day Gross Sharpe Ratio: {rolling_sr_120:.3f}")
        else:
            print("Not enough days or zero variance in the last 120 days.")
    except Exception as e:
        print(f"Could not complete final backtest: {e}")

if __name__ == '__main__':
    main()