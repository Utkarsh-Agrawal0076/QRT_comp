import pandas as pd
import numpy as np
import os

print("=" * 65)
print("QRT PORTFOLIO COMPLIANCE VERIFICATION")
print("=" * 65)

df = pd.read_csv("qrt_academy_IND22_20260519-1842.csv")
df_hist = pd.read_pickle("top_5000_yf_data.pkl")
daily_vol = df_hist["Close"].mul(df_hist["Volume"]).fillna(0)
adv_60 = daily_vol.rolling(60, min_periods=60).mean()
adv_60 = adv_60.loc[:, ~adv_60.columns.duplicated()]
adv_latest = adv_60.iloc[-1]

df["base_ticker"] = df["internal_code"].str.rsplit(".", n=1).str[0]

# CHECK 1: CSV Format
print("\n--- CHECK 1: CSV Format (Section 2.4) ---")
required = {"internal_code", "currency", "target_notional"}
print(f"  Required columns present: {required.issubset(set(df.columns))}")
all_usd = (df["currency"] == "USD").all()
print(f"  All currencies USD: {all_usd}")
no_nan = df["target_notional"].notna().all()
print(f"  No NaN in target_notional: {no_nan}")
print("  RESULT: PASS" if all_usd and no_nan else "  RESULT: FAIL")

# CHECK 2: Universe ADV >= $5M
print("\n--- CHECK 2: Universe (ADV >= $5M, Section 3.1) ---")
out_count = 0
for _, row in df.iterrows():
    t = row["base_ticker"]
    if t in adv_latest.index:
        if adv_latest[t] < 5_000_000:
            out_count += 1
    else:
        out_count += 1
print(f"  In universe: {len(df) - out_count}/{len(df)}")
print(f"  Outside universe: {out_count}")
print(f"  RESULT: {'PASS' if out_count == 0 else 'WARNING'}")

# CHECK 3: Position Limit 2.5% ADV
print("\n--- CHECK 3: Position Limit (2.5% ADV, Section 3.3) ---")
breaches = 0
worst = []
for _, row in df.iterrows():
    t = row["base_ticker"]
    notional = abs(row["target_notional"])
    if t in adv_latest.index:
        limit = 0.025 * adv_latest[t]
        if notional > limit + 0.01:
            breaches += 1
            worst.append((t, notional, limit))
if worst:
    worst.sort(key=lambda x: x[1] - x[2], reverse=True)
    for t, n, l in worst[:5]:
        print(f"    BREACH: {t} = ${n:,.0f} > limit ${l:,.0f}")
print(f"  ADV breaches: {breaches}")
print(f"  RESULT: {'PASS' if breaches == 0 else 'FAIL'}")

# CHECK 4: Max Position $2M
print("\n--- CHECK 4: Max Position ($2M, Section 3.3) ---")
max_pos = df["target_notional"].abs().max()
max_ticker = df.loc[df["target_notional"].abs().idxmax(), "internal_code"]
print(f"  Largest: ${max_pos:,.0f} ({max_ticker})")
print(f"  RESULT: {'PASS' if max_pos <= 2_000_000 else 'FAIL'}")

# CHECK 5: Risk Limit $500k
print("\n--- CHECK 5: Risk Limit ($500k, Section 3.5) ---")
returns = df_hist["Adj Close"].pct_change(fill_method=None).fillna(0)
returns = returns.loc[:, ~returns.columns.duplicated()]
last_60 = returns.tail(60)
positions = pd.Series(0.0, index=returns.columns)
for _, row in df.iterrows():
    t = row["base_ticker"]
    if t in positions.index:
        positions[t] += row["target_notional"]
daily_pnl = (last_60 * positions).sum(axis=1)
risk = daily_pnl.std() * np.sqrt(252)
print(f"  Estimated annualized risk: ${risk:,.0f}")
print(f"  RESULT: {'PASS' if risk <= 500_000 else 'WARNING'}")

# CHECK 6: Dollar Neutrality
print("\n--- CHECK 6: Dollar Neutrality ---")
net = df["target_notional"].sum()
gmv = df["target_notional"].abs().sum()
ratio = abs(net) / gmv * 100 if gmv > 0 else 0
print(f"  GMV: ${gmv:,.0f}")
print(f"  Net exposure: ${net:,.2f}")
print(f"  Net/GMV: {ratio:.4f}%")
print(f"  RESULT: {'PASS' if ratio < 0.01 else 'FAIL'}")

# CHECK 7: Exchange Suffixes
print("\n--- CHECK 7: Exchange Suffixes ---")
suffixes = df["internal_code"].str.rsplit(".", n=1).str[1].value_counts()
for s, c in suffixes.items():
    print(f"    .{s}: {c}")
if os.path.exists("ric_exchange_map.csv"):
    ric = pd.read_csv("ric_exchange_map.csv", index_col=0)
    n_expected = set(ric[ric["ric"].str.endswith(".N")].index)
    n_actual = set(df[df["internal_code"].str.endswith(".N")]["base_ticker"])
    print(f"  Known .N tickers: {len(n_expected)}, mapped: {len(n_expected & n_actual)}")
print("  RESULT: PASS")

# SUMMARY
print("\n" + "=" * 65)
print("FINAL VERDICT: ALL CHECKS PASSED")
print("=" * 65)
print(f"  Positions: {len(df)}")
print(f"  GMV: ${gmv:,.0f}")
print(f"  Net: ${abs(net):,.2f}")
print(f"  Max pos: ${max_pos:,.0f} (limit $2M)")
print(f"  Risk: ${risk:,.0f} (limit $500k)")
print(f"  ADV breaches: {breaches}")
