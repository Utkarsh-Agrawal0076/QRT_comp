import pandas as pd
import yfinance as yf
import os
import sys

# Ensure phase2_qrt_challenge/scripts is in path
sys.path.append(os.path.abspath('phase2_qrt_challenge/scripts'))
from technical_indicators import calculate_all_indicators_parallel

import warnings
warnings.filterwarnings("ignore")

def get_latest_data(tickers):
    """
    Downloads the last 250 trading days (1 year) of data from Yahoo Finance.
    We need this window to properly warm up the EWMA-based technical indicators.
    """
    # Using period="1y" fetches approximately 252 trading days
    df = yf.download(
        tickers=tickers,
        period="1y",
        group_by='column',
        threads=True,
        progress=False,
        auto_adjust=False,
    )
    
    if df.empty:
        raise ValueError("No data fetched from Yahoo Finance.")
        
    if not isinstance(df.columns, pd.MultiIndex) and len(tickers) == 1:
        df.columns = pd.MultiIndex.from_product([df.columns, tickers])
        
    df = df.sort_index(axis=1)
    
    # Handle missing values (forward fill up to 5 days for halts)
    idx = pd.IndexSlice
    features = df.columns.levels[0]
    price_vol_features = [f for f in features if f not in ['Dividends', 'Stock Splits']]
    if price_vol_features:
        df.loc[:, idx[price_vol_features, :]] = df.loc[:, idx[price_vol_features, :]].ffill(limit=5).bfill(limit=1)
        
    return df

def main():
    features_path = "stores/features.parquet"
    if not os.path.exists(features_path):
        print("Historical features.parquet not found. Please run round2_2_create_features.ipynb first.")
        return
    
    print("Loading universe...")
    universe_df = pd.read_csv("top_5000_us_by_marketcap.csv")
    tickers = universe_df['symbol'].astype(str).str.strip().str.replace('.', '-', regex=False).str.replace('/', '-', regex=False).dropna().unique().tolist()
    
    print(f"Fetching latest 1-year data for {len(tickers)} tickers...")
    recent_data = get_latest_data(tickers)
    
    print("Calculating technical indicators on the recent window...")
    # Compute indicators over the 250-day window
    indicators_dict = calculate_all_indicators_parallel(recent_data, n_jobs=-1, verbose=0)
    
    # Combine dictionary of DataFrames into a single MultiIndex DataFrame
    indicators_df = pd.concat(indicators_dict, axis=1).astype("float32")
    
    # Extract only the very last row (today's features)
    latest_date = indicators_df.index[-1]
    latest_features = indicators_df.loc[[latest_date]]
    
    print(f"Appending features for {latest_date.date()}...")
    
    # Load historical database
    historical_features = pd.read_parquet(features_path)
    
    # Check if we're running it twice on the same day
    if latest_date in historical_features.index:
        print(f"Data for {latest_date.date()} already exists. Overwriting to update...")
        historical_features = historical_features.drop(latest_date)
        
    # Append the new row to the historical dataset
    updated_features = pd.concat([historical_features, latest_features])
    
    # Save back to disk
    updated_features.to_parquet(features_path, compression="zstd", engine="pyarrow")
    print("Incremental feature update complete!")

if __name__ == "__main__":
    main()
