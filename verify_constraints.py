import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def verify_constraints():
    print("Loading data...")
    # 1. Load Portfolio
    df_port = pd.read_csv("qrt_academy_IND22_20260518-1756.csv")
    
    # Process internal_code to remove .OQ for matching with historical data
    # In submission, internal_code has .OQ e.g. "AAPL.OQ". In YF data, it's just "AAPL"
    df_port['symbol'] = df_port['internal_code'].str.replace('.OQ', '', regex=False)
    
    # 2. Load Historical Data
    df_historical = pd.read_pickle("top_5000_yf_data.pkl")
    df_historical = df_historical.loc[:, ~df_historical.columns.duplicated()]
    
    returns = pd.read_parquet("stores/returns.parquet")
    returns = returns.loc[:, ~returns.columns.duplicated()]
    
    universe_5m = pd.read_parquet("stores/universe_5m.parquet")
    universe_5m = universe_5m.loc[:, ~universe_5m.columns.duplicated()]
    
    # Last date of the data
    last_date = returns.index.max()
    print(f"Data Date: {last_date.date()}")
    
    # Compute 60-day ADV in USD
    df_daily_volume = df_historical['Close'].mul(df_historical['Volume']).fillna(0)
    adv_60 = df_daily_volume.rolling(window=60, min_periods=60).mean().loc[last_date]
    
    # Prepare positions series aligned with returns
    positions = pd.Series(0.0, index=returns.columns)
    for _, row in df_port.iterrows():
        sym = row['symbol']
        notional = row['target_notional']
        if sym in positions.index:
            positions[sym] = notional
            
    # Filter out zero positions
    active_positions = positions[positions != 0]
    
    # Validation 1: Absolute Max Position <= $2M
    max_abs_pos = active_positions.abs().max()
    print(f"Constraint 1 (Max Position <= $2M USD): {'PASS' if max_abs_pos <= 2000000 else 'FAIL'} (Max: ${max_abs_pos:,.2f})")
    
    # Validation 2: Max Position <= 2.5% of ADV
    # Validation 3: Universe 5M ADV Mask
    pass_adv_limit = True
    pass_universe = True
    max_adv_usage = 0.0
    
    for sym, pos in active_positions.items():
        sym_adv = adv_60.get(sym, 0)
        limit = sym_adv * 0.025
        abs_pos = abs(pos)
        
        if sym_adv > 0:
            usage = abs_pos / sym_adv
            max_adv_usage = max(max_adv_usage, usage)
            
        if abs_pos > limit + 1e-4:  # Small floating point tolerance
            pass_adv_limit = False
            print(f"   -> FAIL ADV Limit: {sym} Position: ${abs_pos:,.2f}, Limit: ${limit:,.2f} (ADV: ${sym_adv:,.2f})")
            
        if sym_adv < 5_000_000 - 1: # Small tolerance
            pass_universe = False
            print(f"   -> FAIL Universe Limit: {sym} ADV: ${sym_adv:,.2f} < 5M")

    print(f"Constraint 2 (Max Position <= 2.5% ADV): {'PASS' if pass_adv_limit else 'FAIL'} (Max Used: {max_adv_usage*100:.2f}%)")
    print(f"Constraint 3 (ADV >= 5M USD): {'PASS' if pass_universe else 'FAIL'}")
    
    # Validation 4: Ex-ante Risk <= $500k USD
    # Risk calculation:
    # 1. Select the last 60 daily returns for all instruments
    # 2. Compute the book's daily P&L series that you would have had with today's position
    # 3. Compute standard deviation of these daily P&L
    # 4. Multiply by sqrt(252) to annualise
    
    # Get last 60 daily returns
    returns_60 = returns.loc[:last_date].iloc[-60:]
    
    # Align positions with returns_60 columns
    pos_aligned = positions.reindex(returns_60.columns).fillna(0)
    
    # Compute daily simulated PnL
    sim_daily_pnl = returns_60.dot(pos_aligned)
    
    # Standard deviation of daily PnL
    daily_std = sim_daily_pnl.std()
    
    # Annualised Risk
    annualised_risk = daily_std * np.sqrt(252)
    
    print(f"Constraint 4 (Risk <= $500k USD): {'PASS' if annualised_risk <= 500000 else 'FAIL'} (Risk: ${annualised_risk:,.2f})")
    
if __name__ == '__main__':
    verify_constraints()
