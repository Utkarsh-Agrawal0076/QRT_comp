"""
attribute_strategies.py
Recompute the 3 production sleeves (Mean Reversion, Momentum, Stat-Arb) exactly
as generate_submission.py does, decompose each submitted stock's final ensemble
weight into per-sleeve contributions, and attribute it to the dominant strategy.

Also: stat-arb pair-integrity check vs the RIC dropout, and join with the
3-day report PnL + traded/not-traded status.
"""
import os, sys
os.environ["SKIP_DATA_REFRESH"] = "1"
import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)

import generate_submission as gs

RIC_MAP_CSV = "ric_exchange_map.csv"
PAIRS_CSV = "kalman_universe_config.csv"
SUB_CSV = "qrt_academy_IND22_20260528-1809.csv"
REP_CSV = "report_pnl_persistence.csv"   # produced by analyze_3day_report.py


def blend_coeffs(w_mr, w_mom, w_sa, returns):
    """Replicate generate_submission.blend_ensemble's coefficient calc, return last-day c_*."""
    def rolling_vol(weights, rets, window=60):
        ar = rets.reindex(index=weights.index, columns=weights.columns).fillna(0)
        pnl = (weights * ar).sum(axis=1)
        return (pnl.rolling(window, min_periods=20).std() * np.sqrt(252)).clip(lower=0.05)
    vol_mr, vol_mom, vol_sa = rolling_vol(w_mr, returns), rolling_vol(w_mom, returns), rolling_vol(w_sa, returns)
    inv_mr, inv_mom, inv_sa = 1/vol_mr, 1/vol_mom, 1/vol_sa
    total = inv_mr + inv_mom + inv_sa
    c_mr = (inv_mr/total).shift(1).fillna(1/3)
    c_mom = (inv_mom/total).shift(1).fillna(1/3)
    c_sa = (inv_sa/total).shift(1).fillna(1/3)
    return c_mr.iloc[-1], c_mom.iloc[-1], c_sa.iloc[-1]


def main():
    df_hist, returns, universe, df_adv_60 = gs.load_data()

    w_mr = gs.run_mean_reversion(df_hist, returns, universe, df_adv_60)
    try:
        w_mom = gs.run_momentum(df_hist, returns, universe)
        mom_ok = True
    except Exception as e:
        print(f"  !! Momentum recompute failed ({e}); using zero momentum sleeve.")
        w_mom = pd.DataFrame(0.0, index=w_mr.index, columns=w_mr.columns)
        mom_ok = False
    w_sa = gs.run_stat_arb(df_hist, universe)
    if w_sa is None:
        w_sa = pd.DataFrame(0.0, index=w_mr.index, columns=w_mr.columns)

    # Align (same as generate_submission.main)
    cols = w_mr.columns.intersection(w_mom.columns).intersection(w_sa.columns)
    idx = w_mr.index.intersection(w_mom.index).intersection(w_sa.index)
    w_mr = w_mr.reindex(index=idx, columns=cols).fillna(0)
    w_mom = w_mom.reindex(index=idx, columns=cols).fillna(0)
    w_sa = w_sa.reindex(index=idx, columns=cols).fillna(0)
    returns_a = returns.reindex(index=idx, columns=cols).fillna(0)

    c_mr, c_mom, c_sa = blend_coeffs(w_mr, w_mom, w_sa, returns_a)
    print(f"\n  Blend coefficients (last day): MR={c_mr:.3f}  Mom={c_mom:.3f}  SA={c_sa:.3f}  (mom_ok={mom_ok})")

    # Last-day sleeve weights and contributions
    last = idx[-1]
    mr_l, mom_l, sa_l = w_mr.loc[last], w_mom.loc[last], w_sa.loc[last]
    contrib = pd.DataFrame({
        "w_mr": mr_l, "w_mom": mom_l, "w_sa": sa_l,
        "c_mr": c_mr*mr_l, "c_mom": c_mom*mom_l, "c_sa": c_sa*sa_l,
    })
    contrib["ens"] = contrib[["c_mr", "c_mom", "c_sa"]].sum(axis=1)
    # dominant strategy by |contribution|
    abscon = contrib[["c_mr", "c_mom", "c_sa"]].abs()
    contrib["dominant"] = abscon.idxmax(axis=1).map({"c_mr": "MR", "c_mom": "Mom", "c_sa": "SA"})
    # which sleeves are active (nonzero) for the name
    def tags(r):
        t = []
        if abs(r["w_mr"]) > 1e-12: t.append("MR")
        if abs(r["w_mom"]) > 1e-12: t.append("Mom")
        if abs(r["w_sa"]) > 1e-12: t.append("SA")
        return "+".join(t) if t else "none"
    contrib["sleeves"] = contrib.apply(tags, axis=1)
    contrib = contrib[contrib["ens"].abs() > 1e-12].copy()
    print(f"\n  Names with nonzero ensemble weight: {len(contrib)}")
    print(f"  Sleeve-combo counts:\n{contrib['sleeves'].value_counts().to_string()}")
    print(f"\n  Dominant-strategy counts:\n{contrib['dominant'].value_counts().to_string()}")

    # Map ticker -> RIC (same default-.OQ logic as generate_csv)
    ric = pd.read_csv(RIC_MAP_CSV, index_col=0)["ric"].to_dict()
    contrib["internal_code"] = [ric.get(t, f"{t}.OQ") for t in contrib.index]
    contrib["in_ric_map"] = [t in ric for t in contrib.index]
    contrib = contrib.reset_index().rename(columns={"index": "ticker"})

    # Join submission notional + report PnL/traded status
    sub = pd.read_csv(SUB_CSV); sub["internal_code"] = sub["internal_code"].str.strip()
    rep = pd.read_csv(REP_CSV)  # index col = Instrument
    rep = rep.rename(columns={rep.columns[0]: "Instrument"})
    traded_set = set(rep["Instrument"])

    m = contrib.merge(sub[["internal_code", "target_notional"]], on="internal_code", how="left")
    m["traded"] = m["internal_code"].isin(traded_set)
    m = m.merge(rep[["Instrument", "total_net", "days_neg", "days_pos"]],
                left_on="internal_code", right_on="Instrument", how="left")

    m.to_csv("strategy_attribution.csv", index=False)
    print("\n[saved] strategy_attribution.csv")

    # ----- Dropout by strategy -----
    print("\n" + "="*90)
    print("RIC DROPOUT BY DOMINANT STRATEGY (which sleeves are being silently gutted)")
    print("="*90)
    tab = m.groupby("dominant").agg(
        n=("ticker", "size"),
        n_dropped=("traded", lambda s: (~s).sum()),
        gmv=("target_notional", lambda s: s.abs().sum()),
        gmv_dropped=("target_notional", lambda s: s[~m.loc[s.index, "traded"]].abs().sum()),
    )
    tab["pct_dropped_names"] = (tab["n_dropped"]/tab["n"]*100).round(1)
    tab["pct_dropped_gmv"] = (tab["gmv_dropped"]/tab["gmv"]*100).round(1)
    print(tab.to_string())

    # ----- 3-day PnL by strategy (traded names only) -----
    print("\n" + "="*90)
    print("3-DAY NET PnL BY DOMINANT STRATEGY (traded names only)")
    print("="*90)
    tr = m[m["traded"]]
    pnl_tab = tr.groupby("dominant").agg(
        n_traded=("ticker", "size"),
        net_3d=("total_net", "sum"),
        persistent_losers=("days_neg", lambda s: (s == 3).sum()),
        persistent_winners=("days_pos", lambda s: (s == 3).sum()),
    )
    print(pnl_tab.to_string())

    # ----- Stat-arb pair integrity -----
    print("\n" + "="*90)
    print("STAT-ARB PAIR INTEGRITY (both legs traded vs one leg dropped = naked directional risk)")
    print("="*90)
    pairs = pd.read_csv(PAIRS_CSV)
    sa_names = set(contrib[contrib["w_sa"].abs() > 1e-12]["ticker"])
    rows = []
    for _, p in pairs.iterrows():
        y, x = p["asset_y"], p["asset_x"]
        y_active = y in sa_names
        x_active = x in sa_names
        if not (y_active or x_active):
            continue  # pair not on today
        y_ric = ric.get(y, f"{y}.OQ"); x_ric = ric.get(x, f"{x}.OQ")
        rows.append({
            "pair": f"{y}/{x}", "industry": p["industry"],
            "y_active": y_active, "x_active": x_active,
            "y_traded": y_ric in traded_set, "x_traded": x_ric in traded_set,
        })
    pdf = pd.DataFrame(rows)
    if len(pdf):
        pdf["both_traded"] = pdf["y_traded"] & pdf["x_traded"]
        pdf["one_leg_only"] = pdf["y_traded"] ^ pdf["x_traded"]
        pdf["both_dropped"] = ~pdf["y_traded"] & ~pdf["x_traded"]
        print(f"  Active pairs today: {len(pdf)}")
        print(f"    both legs traded (intact): {pdf['both_traded'].sum()}")
        print(f"    ONE leg only (BROKEN -> naked directional): {pdf['one_leg_only'].sum()}")
        print(f"    both legs dropped (pair fully gone): {pdf['both_dropped'].sum()}")
        print("\n  Broken pairs (one leg only):")
        for _, r in pdf[pdf["one_leg_only"]].iterrows():
            kept = r["pair"].split("/")[0] if r["y_traded"] else r["pair"].split("/")[1]
            print(f"    {r['pair']:<16} ({r['industry']}) -> only {kept} trades")
        pdf.to_csv("pair_integrity.csv", index=False)
        print("\n[saved] pair_integrity.csv")
    else:
        print("  No active stat-arb pairs on the last day.")


if __name__ == "__main__":
    main()
