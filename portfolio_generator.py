import pandas as pd

def generate_dummy_portfolio(csv_path: str, top_n: int = 500, notional_per_stock: float = 1000.0) -> pd.DataFrame:
    """
    Generates a dummy portfolio taking the top N stocks by market cap.
    We split it into 50% Long and 50% Short to maintain Dollar Neutrality 
    as required by standard QRT rules, while mimicking a large cap (S&P500) universe.
    """
    df = pd.read_csv(csv_path)
    
    # Get top N stocks
    top_stocks = df.head(top_n).copy()
    
    # We will do a 50/50 long/short split
    half = top_n // 2
    
    # Initialize the targets
    targets = pd.DataFrame()
    targets['internal_code'] = top_stocks['symbol'].astype(str).str.strip().str.replace('.', '-', regex=False).str.replace('/', '-', regex=False)
    targets['currency'] = 'USD'
    
    # Assign notionals: Positive for first half, Negative for second half
    notionals = [notional_per_stock] * half + [-notional_per_stock] * (top_n - half)
    targets['target_notional'] = notionals
    
    return targets

if __name__ == "__main__":
    targets_df = generate_dummy_portfolio("top_5000_us_by_marketcap.csv")
    print("Generated Dummy Portfolio:")
    print(targets_df.head())
    print(f"Total Long: {targets_df[targets_df['target_notional'] > 0]['target_notional'].sum()}")
    print(f"Total Short: {targets_df[targets_df['target_notional'] < 0]['target_notional'].sum()}")
