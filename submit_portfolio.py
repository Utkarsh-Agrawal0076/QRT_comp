import os
from pathlib import Path
from qsec_client.sample_code import Region, prepare_targets_file, upload_targets_file
from portfolio_generator import generate_dummy_portfolio

def main():
    # Credentials and Configuration
    USERNAME = "q4615"
    GROUP_ID = "IND22"
    PRIVATE_KEY_PATH = Path(r"C:\Users\utkar\OneDrive\Desktop\QRT\dehli_keys\ind22_id_rsa")
    SFTP_HOST = "sftp.qrt.cloud"
    REGION = Region.AMER  # Assuming AMER for US stocks
    
    # 1. Generate the Portfolio
    print("Generating S&P 500 Dummy Portfolio...")
    targets_df = generate_dummy_portfolio("top_5000_us_by_marketcap.csv", top_n=500, notional_per_stock=1000.0)
    
    # 2. Prepare the Targets CSV using QSec Client
    print("Preparing targets file...")
    output_dir = Path(".")
    csv_filepath = prepare_targets_file(
        targets=targets_df,
        group_id=GROUP_ID,
        region=REGION,
        output_dir=output_dir
    )
    print(f"File prepared at: {csv_filepath}")
    
    # 3. Upload the Targets File
    print(f"Uploading file to {SFTP_HOST} as {USERNAME}...")
    try:
        upload_targets_file(
            targets_csv_path=csv_filepath,
            region=REGION,
            sftp_username=USERNAME,
            private_key_path=PRIVATE_KEY_PATH,
            sftp_host=SFTP_HOST
        )
        print("Upload successful!")
    except Exception as e:
        print(f"Upload failed: {e}")

if __name__ == "__main__":
    main()
