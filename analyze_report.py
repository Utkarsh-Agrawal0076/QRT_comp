import pandas as pd
import numpy as np

# Load submitted targets
df_sub = pd.read_csv('qrt_academy_IND22_20260518-1756.csv')
df_sub['internal_code'] = df_sub['internal_code'].str.strip()

# Load received report
df_rep = pd.read_excel(r'reports_received\QSec_Detailed_IND22_2026-05-18.xlsx', sheet_name='Sheet1')
df_rep['Instrument'] = df_rep['Instrument'].astype(str).str.strip()

# Merge them on ticker
merged = pd.merge(df_sub, df_rep, left_on='internal_code', right_on='Instrument', how='outer', indicator=True)

# 1. Missing stocks (in submitted but not in report)
missing_in_report = merged[merged['_merge'] == 'left_only']
print("==================================================")
print(f"Stocks submitted but missing in report: {len(missing_in_report)}")
print("==================================================")
if len(missing_in_report) > 0:
    for _, row in missing_in_report.nlargest(10, 'target_notional').iterrows():
        print(f"{row['internal_code']}: Submitted Target = ${row['target_notional']:,.2f}")
    if len(missing_in_report) > 10:
        print(f"... and {len(missing_in_report)-10} more.")

# 2. Check for anomalies between submitted targets and report targets
print("\n==================================================")
print("Anomalies: Submitted Target != Report Target")
print("==================================================")
both = merged[merged['_merge'] == 'both']

# Allow a small float tolerance (e.g., $1.00)
anomalies = both[np.abs(both['target_notional'] - both['Target USD'].fillna(0)) > 1.0]

if len(anomalies) > 0:
    print(f"Found {len(anomalies)} anomalies in target amounts:")
    # Sort by absolute difference
    anomalies['diff'] = np.abs(anomalies['target_notional'] - anomalies['Target USD'])
    anomalies = anomalies.sort_values('diff', ascending=False)
    for _, row in anomalies.head(10).iterrows():
        print(f"{row['internal_code']}: Submitted = ${row['target_notional']:,.2f} | Report Target = ${row['Target USD']:,.2f} | Diff = ${row['diff']:,.2f} | In Universe = {row['In Universe']}")
    if len(anomalies) > 10:
        print(f"... and {len(anomalies)-10} more anomalies.")
else:
    print("No anomalies found in target amounts! QRT's system correctly parsed your target sizes.")

# 3. Check if actual Position EOD differs from Target USD (Constraints applied)
print("\n==================================================")
print("Constraints Applied: Target USD != Position EOD USD")
print("==================================================")
# Find where Position EOD does not equal Target USD (ignoring small rounding differences)
constrained = both[np.abs(both['Target USD'].fillna(0) - both['Position EOD USD'].fillna(0)) > 1.0]
if len(constrained) > 0:
    print(f"Found {len(constrained)} positions where final EOD position did not reach the target.")
    constrained['diff'] = np.abs(constrained['Target USD'] - constrained['Position EOD USD'])
    constrained = constrained.sort_values('diff', ascending=False)
    for _, row in constrained.head(10).iterrows():
        print(f"{row['internal_code']}: Target = ${row['Target USD']:,.2f} | EOD Pos = ${row['Position EOD USD']:,.2f} | Traded = ${row['Traded USD']:,.2f} | ADV = ${row['ADV USD']:,.2f}")
else:
    print("All positions successfully reached their targets! No liquidity constraints hit.")

# 4. Check extra stocks in report (not in submitted)
extra_in_report = merged[merged['_merge'] == 'right_only']
print("\n==================================================")
print(f"Stocks in report but not in submitted targets: {len(extra_in_report)}")
print("==================================================")
if len(extra_in_report) > 0:
    # Most likely .SPX (the auto-hedge)
    for _, row in extra_in_report.nlargest(5, 'Position EOD USD').iterrows():
        print(f"{row['Instrument']}: EOD Pos = ${row['Position EOD USD']:,.2f}")
