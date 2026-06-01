"""
analyze_3day_report.py
Consistency + PnL-persistence analysis of the 3 QRT detailed reports
(2026-05-26/27/28) against the submitted portfolio.
"""
import pandas as pd
import numpy as np
from pathlib import Path

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

SUB_CSV = "qrt_academy_IND22_20260528-1809.csv"
REPORTS = {
    "2026-05-26": "reports_received/QSec_Detailed_IND22_2026-05-26.xlsx",
    "2026-05-27": "reports_received/QSec_Detailed_IND22_2026-05-27.xlsx",
    "2026-05-28": "reports_received/QSec_Detailed_IND22_2026-05-28.xlsx",
}

PNL_COLS = ["Yest PnL USD", "Day PnL USD", "Div PnL USD",
            "Exec costs USD", "Financing costs USD", "Net PnL USD"]

def load_submission():
    df = pd.read_csv(SUB_CSV)
    df["internal_code"] = df["internal_code"].str.strip()
    return df

def load_report(path):
    d = pd.read_excel(path, sheet_name="Sheet1")
    d["Instrument"] = d["Instrument"].astype(str).str.strip()
    return d

def load_book(path):
    """Sheet2 daily book-level metrics (PnL/GMV/Risk), skipping the 2 header rows."""
    d = pd.read_excel(path, sheet_name="Sheet2", skiprows=2)
    # columns: Unnamed:0(idx), TradeDate, Book ID, PnL (k$), GMV (M$), Risk (k$)
    d = d.rename(columns={d.columns[0]: "row"})
    return d


def main():
    sub = load_submission()
    print("="*90)
    print("SUBMITTED PORTFOLIO")
    print("="*90)
    print(f"  Positions: {len(sub)}")
    print(f"  GMV: ${sub['target_notional'].abs().sum():,.0f}")
    print(f"  Net: ${sub['target_notional'].sum():,.2f}")
    print(f"  Longs: {(sub['target_notional']>0).sum()}  Shorts: {(sub['target_notional']<0).sum()}")

    reports = {d: load_report(p) for d, p in REPORTS.items()}

    # ---------------------------------------------------------------
    # SECTION 1 — Book level (Sheet2) trajectory
    # ---------------------------------------------------------------
    print("\n" + "="*90)
    print("SECTION 1 — BOOK-LEVEL DAILY TRAJECTORY (Sheet2)")
    print("="*90)
    for d, p in REPORTS.items():
        try:
            bk = load_book(p)
            bk = bk[bk["Book ID"] == "IND22_AMER"]
            print(f"\n{d}:")
            print(bk[["TradeDate", "Book ID", "PnL (k$)", "GMV (M$)", "Risk (k$)"]].tail(6).to_string(index=False))
        except Exception as e:
            print(f"  {d}: book parse error {e}")

    # ---------------------------------------------------------------
    # SECTION 2 — Consistency: submitted target vs report TradeDate target
    # Use the 05-28 report (most recent) as the canonical live state.
    # ---------------------------------------------------------------
    print("\n" + "="*90)
    print("SECTION 2 — CONSISTENCY (submitted target_notional vs report 'Target USD', 05-28)")
    print("="*90)
    rep28 = reports["2026-05-28"]
    merged = pd.merge(sub, rep28, left_on="internal_code", right_on="Instrument",
                      how="outer", indicator=True)

    miss = merged[merged["_merge"] == "left_only"]
    extra = merged[merged["_merge"] == "right_only"]
    both = merged[merged["_merge"] == "both"].copy()

    print(f"\nSubmitted but NOT in report (not traded): {len(miss)}")
    if len(miss):
        m = miss.reindex(miss["target_notional"].abs().sort_values(ascending=False).index)
        for _, r in m.head(15).iterrows():
            print(f"  {r['internal_code']:<10} submitted ${r['target_notional']:>12,.0f}")
        print(f"  ... total missing GMV ${miss['target_notional'].abs().sum():,.0f}")

    print(f"\nIn report but NOT submitted (auto-hedge etc): {len(extra)}")
    for _, r in extra.sort_values("Position EOD USD", key=lambda s: s.abs(), ascending=False).head(8).iterrows():
        print(f"  {r['Instrument']:<10} EODpos ${r['Position EOD USD']:>14,.0f}  TargetUSD ${r['Target USD']:>12,.0f}")

    # target mismatch
    both["tgt_diff"] = (both["target_notional"] - both["Target USD"].fillna(0)).abs()
    anom = both[both["tgt_diff"] > 1.0]
    print(f"\nTarget mismatches (submitted != report Target USD, >$1): {len(anom)}")
    for _, r in anom.sort_values("tgt_diff", ascending=False).head(15).iterrows():
        print(f"  {r['internal_code']:<10} sub ${r['target_notional']:>12,.0f} | reptgt ${r['Target USD']:>12,.0f} | diff ${r['tgt_diff']:>12,.0f} | inUniv={r['In Universe']}")

    # constraints: target reached vs EOD position
    both["fill_gap"] = (both["Target USD"].fillna(0) - both["Position EOD USD"].fillna(0)).abs()
    constr = both[both["fill_gap"] > 1.0]
    print(f"\nPositions not fully filled to target (|Target-EOD|>$1): {len(constr)}")
    for _, r in constr.sort_values("fill_gap", ascending=False).head(15).iterrows():
        adv = r["ADV USD"]
        print(f"  {r['internal_code']:<10} tgt ${r['Target USD']:>12,.0f} | EOD ${r['Position EOD USD']:>12,.0f} | gap ${r['fill_gap']:>12,.0f} | ADV ${adv:>14,.0f}")

    # ---------------------------------------------------------------
    # SECTION 3 — PnL persistence per stock across the 3 days
    # ---------------------------------------------------------------
    print("\n" + "="*90)
    print("SECTION 3 — PnL PERSISTENCE PER STOCK (Day PnL & Net PnL across 3 days)")
    print("="*90)

    frames = []
    for d, rep in reports.items():
        sub_cols = ["Instrument", "Day PnL USD", "Net PnL USD", "Position SOD USD",
                    "Position EOD USD", "Target USD", "Traded USD", "In Universe"]
        f = rep[sub_cols].copy()
        f["date"] = d
        frames.append(f)
    alld = pd.concat(frames, ignore_index=True)

    # pivot net pnl per stock x day
    net = alld.pivot_table(index="Instrument", columns="date", values="Net PnL USD", aggfunc="sum")
    day = alld.pivot_table(index="Instrument", columns="date", values="Day PnL USD", aggfunc="sum")
    net = net.reindex(columns=sorted(net.columns))
    day = day.reindex(columns=sorted(day.columns))

    net["total_net"] = net.sum(axis=1)
    net["days_neg"] = (net[sorted(REPORTS)] < 0).sum(axis=1)
    net["days_pos"] = (net[sorted(REPORTS)] > 0).sum(axis=1)

    # merge submitted side (sign of bet)
    sub_sign = sub.set_index("internal_code")["target_notional"]
    net["sub_notional"] = net.index.map(sub_sign)
    net["side"] = np.where(net["sub_notional"] > 0, "LONG",
                   np.where(net["sub_notional"] < 0, "SHORT", "n/a"))

    print(f"\nTotal Net PnL over 3 days (all instruments): ${net['total_net'].sum():,.0f}")
    daily_tot = net[sorted(REPORTS)].sum()
    print("Daily Net PnL totals:")
    for d in sorted(REPORTS):
        print(f"   {d}: ${daily_tot[d]:,.0f}")

    print("\n--- PERSISTENT LOSERS (negative all 3 days) ---")
    pers_loss = net[net["days_neg"] == 3].sort_values("total_net")
    print(f"  count: {len(pers_loss)} | aggregate 3-day Net ${pers_loss['total_net'].sum():,.0f}")
    print(pers_loss[sorted(REPORTS)+["total_net","side","sub_notional"]].head(20).to_string())

    print("\n--- PERSISTENT WINNERS (positive all 3 days) ---")
    pers_win = net[net["days_pos"] == 3].sort_values("total_net", ascending=False)
    print(f"  count: {len(pers_win)} | aggregate 3-day Net ${pers_win['total_net'].sum():,.0f}")
    print(pers_win[sorted(REPORTS)+["total_net","side","sub_notional"]].head(20).to_string())

    print("\n--- BIGGEST 3-DAY NET LOSERS (regardless of persistence) ---")
    print(net.sort_values("total_net")[sorted(REPORTS)+["total_net","days_neg","side"]].head(20).to_string())

    print("\n--- BIGGEST 3-DAY NET WINNERS ---")
    print(net.sort_values("total_net", ascending=False)[sorted(REPORTS)+["total_net","days_pos","side"]].head(20).to_string())

    # persistence summary
    print("\n--- PERSISTENCE SUMMARY ---")
    n_all = len(net)
    print(f"  instruments with PnL data: {n_all}")
    print(f"  neg all 3 days: {(net['days_neg']==3).sum()}  | pos all 3 days: {(net['days_pos']==3).sum()}")
    print(f"  mixed (sign flips): {((net['days_neg']>0)&(net['days_pos']>0)).sum()}")
    # is loss concentrated?
    losers = net[net["total_net"]<0]["total_net"].sum()
    winners = net[net["total_net"]>0]["total_net"].sum()
    print(f"  sum of all losers: ${losers:,.0f} | sum of all winners: ${winners:,.0f}")
    top10_loss = net.sort_values("total_net")["total_net"].head(10).sum()
    print(f"  top-10 losers account for ${top10_loss:,.0f} ({top10_loss/losers*100:.0f}% of total losses)")

    # save full table
    out = net.copy()
    out.to_csv("report_pnl_persistence.csv")
    print("\n[saved] report_pnl_persistence.csv (full per-stock 3-day PnL table)")

    # also save the consistency merge for downstream attribution join
    both.to_csv("report_consistency_0528.csv", index=False)
    print("[saved] report_consistency_0528.csv")


if __name__ == "__main__":
    main()
