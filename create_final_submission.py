import pandas as pd
from pathlib import Path
import sys

# Import qsec_client
sys.path.append(str(Path('.').resolve() / 'qsec-client'))
from qsec_client.sample_code import Region, prepare_targets_file

def main():
    GROUP_ID = "IND22"
    REGION = Region.AMER
    
    # Read the master ensemble submission
    df = pd.read_csv("submission.csv")
    
    # Rename column if necessary
    if "target_notionals" in df.columns:
        df = df.rename(columns={"target_notionals": "target_notional"})
        
    # Ensure internal_code has .OQ
    df["internal_code"] = df["internal_code"].astype(str)
    # If some don't have .OQ and don't have OQ at all, add .OQ
    # If they have space OQ, replace with .OQ
    df["internal_code"] = df["internal_code"].str.replace(" OQ", ".OQ", regex=False)
    # If they don't end with .OQ, append .OQ
    df["internal_code"] = df.apply(lambda row: row["internal_code"] if row["internal_code"].endswith(".OQ") else row["internal_code"] + ".OQ", axis=1)
    
    # Add optional columns so qsec_client doesn't fail
    df["ric"] = df["internal_code"]
    df["ticker"] = df["internal_code"]
    df["target_contracts"] = 0
    df["ref_price"] = 0.0
    
    # Call prepare_targets_file
    output_dir = Path(".")
    csv_filepath = prepare_targets_file(
        targets=df,
        group_id=GROUP_ID,
        region=REGION,
        output_dir=output_dir
    )
    print(f"Final submission file created at: {csv_filepath}")

if __name__ == "__main__":
    main()
