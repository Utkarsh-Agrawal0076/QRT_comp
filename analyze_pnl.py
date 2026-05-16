import pandas as pd
import numpy as np
import datetime
import os
import sys

# Add the scripts directory to path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), 'phase2_qrt_challenge', 'scripts'))
from utils import backtest_portfolio

class PairsKalmanFilter:
    def __init__(self, initial_beta, spread_vol):
        self.theta = np.array([[0.0], [initial_beta]])
        self.P = np.zeros((2, 2))
        np.fill_diagonal(self.P, 1.0)
        self.R = spread_vol ** 2
        self.Q = np.zeros((2, 2))
        np.fill_diagonal(self.Q, 1e-5)
        self.current_spread = 0.0
        self.z_score = 0.0

    def step(self, log_price_y, log_price_x):
        H = np.array([[1.0, log_price_x]])
        theta_pred = self.theta
        P_pred = self.P + self.Q
        y_pred = H @ theta_pred
        e = log_price_y - y_pred[0, 0]
        self.current_spread = e
        S = H @ P_pred @ H.T + self.R
        prediction_variance = S[0, 0]
        self.z_score = e / np.sqrt(prediction_variance)
        K = (P_pred @ H.T) / prediction_variance
        self.theta = theta_pred + K * e
        self.P = (np.eye(2) - K @ H) @ P_pred
        current_beta = self.theta[1, 0]
        return self.z_score, current_beta, self.current_spread

def generate_kalman_portfolio():
    print("Loading data...")
    df_historical = pd.read_pickle('top_5000_yf_data.pkl')
    log_prices = np.log(df_historical['Adj Close'].replace(0, np.nan))
    
    universe = pd.read_parquet('stores/universe_5m.parquet')
    returns = pd.read_parquet('stores/returns.parquet')
    config = pd.read_csv('kalman_universe_config.csv')
    
    last_date = universe.index[-1]
    start_date = last_date - pd.DateOffset(months=6)
    burn_in_start = start_date - pd.DateOffset(months=12)
    
    print(f"Backtest period: {start_date.strftime('%Y-%m-%d')} to {last_date.strftime('%Y-%m-%d')}")
    
    sim_dates = log_prices.loc[burn_in_start:last_date].index
    oos_dates = universe.loc[start_date:last_date].index
    portfolio = pd.DataFrame(0.0, index=oos_dates, columns=universe.columns)
    
    active_filters = {}
    positions = {}
    
    for _, row in config.iterrows():
        pair_name = f"{row['asset_y']}_{row['asset_x']}"
        active_filters[pair_name] = PairsKalmanFilter(initial_beta=row['tls_beta'], spread_vol=row['spread_vol'])
        positions[pair_name] = 0

    entry_threshold = 2.0
    exit_threshold = 0.5
    alloc_per_leg = 0.02 
    
    print("Running Kalman Filter simulation and generating signals...")
    for t in range(len(sim_dates)-1):
        today = sim_dates[t]
        tomorrow = sim_dates[t+1]
        
        record_weights = tomorrow >= start_date
        daily_weights = {} if record_weights else None
        
        for _, row in config.iterrows():
            y_ticker = row['asset_y']
            x_ticker = row['asset_x']
            pair_name = f"{y_ticker}_{x_ticker}"
            
            kf = active_filters[pair_name]
            
            if today in log_prices.index and pd.notna(log_prices.loc[today, y_ticker]) and pd.notna(log_prices.loc[today, x_ticker]):
                log_y = log_prices.loc[today, y_ticker]
                log_x = log_prices.loc[today, x_ticker]
                
                z_score, current_beta, spread = kf.step(log_y, log_x)
                
                current_pos = positions[pair_name]
                if current_pos == 0:
                    if z_score > entry_threshold:
                        positions[pair_name] = -1 # Short spread
                    elif z_score < -entry_threshold:
                        positions[pair_name] = 1 # Long spread
                elif current_pos == 1 and z_score > -exit_threshold:
                    positions[pair_name] = 0
                elif current_pos == -1 and z_score < exit_threshold:
                    positions[pair_name] = 0
                        
            if record_weights:
                pos = positions[pair_name]
                if pos != 0:
                    # Using 'today' for universe check to avoid lookahead bias
                    if today in universe.index and universe.loc[today, y_ticker] == 1 and universe.loc[today, x_ticker] == 1:
                        daily_weights[y_ticker] = daily_weights.get(y_ticker, 0.0) + pos * alloc_per_leg
                        daily_weights[x_ticker] = daily_weights.get(x_ticker, 0.0) - pos * alloc_per_leg
                    else:
                        positions[pair_name] = 0
                        
        if record_weights and daily_weights:
            w_series = pd.Series(daily_weights)
            w_series = w_series.clip(-0.1, 0.1)
            
            if abs(w_series.sum()) > 1e-4:
                w_series -= w_series.sum() / len(w_series)
                
            for ticker, w in w_series.items():
                portfolio.loc[tomorrow, ticker] = w

    return portfolio, returns.loc[start_date:last_date]

def analyze_day_to_day_pnl():
    portfolio, rets = generate_kalman_portfolio()
    
    print("Analyzing day-to-day stock PnL...")
    portfolio = portfolio.fillna(0)
    rets = rets.fillna(0)
    
    # Calculate difference for execution costs
    portfolio_diff = portfolio.diff(1).fillna(portfolio.iloc[0])
    
    analysis_records = []
    
    dates = portfolio.index
    for i, date in enumerate(dates):
        # Find tickers with non-zero weights or non-zero trades on this day
        active_tickers = portfolio.columns[(portfolio.loc[date] != 0) | (portfolio_diff.loc[date] != 0)]
        
        for ticker in active_tickers:
            weight = portfolio.loc[date, ticker]
            trade_size = abs(portfolio_diff.loc[date, ticker])
            ret = rets.loc[date, ticker]
            
            gross_pnl = weight * ret
            execution_cost = trade_size * 2e-4
            financing_cost = abs(weight) * (0.005 / 252)
            net_pnl = gross_pnl - execution_cost - financing_cost
            
            analysis_records.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Ticker': ticker,
                'Weight': round(weight, 4),
                'Trade_Size': round(trade_size, 4),
                'Return': round(ret, 4),
                'Gross_PnL': round(gross_pnl, 6),
                'Execution_Cost': round(execution_cost, 6),
                'Financing_Cost': round(financing_cost, 6),
                'Net_PnL': round(net_pnl, 6)
            })
            
    df_analysis = pd.DataFrame(analysis_records)
    output_file = 'detailed_trade_pnl.csv'
    df_analysis.to_csv(output_file, index=False)
    
    print(f"Analysis complete! Detailed PnL saved to {output_file}")
    
    # Print summary
    print("\n--- Strategy Summary ---")
    print(f"Total Gross PnL: {df_analysis['Gross_PnL'].sum():.4f}")
    print(f"Total Execution Cost: {df_analysis['Execution_Cost'].sum():.4f}")
    print(f"Total Financing Cost: {df_analysis['Financing_Cost'].sum():.4f}")
    print(f"Total Net PnL: {df_analysis['Net_PnL'].sum():.4f}")
    
    # Top 5 most profitable stocks
    ticker_pnl = df_analysis.groupby('Ticker')['Net_PnL'].sum().sort_values(ascending=False)
    print("\nTop 5 Profitable Stocks:")
    print(ticker_pnl.head(5))
    print("\nBottom 5 Losing Stocks:")
    print(ticker_pnl.tail(5))

if __name__ == "__main__":
    analyze_day_to_day_pnl()
