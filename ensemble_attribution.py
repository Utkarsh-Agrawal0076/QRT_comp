"""Per-year attribution: where do the gains/losses in MVO 4-sleeve come from?

Three questions:
  1. Per-year per-sleeve standalone Sharpe — which sleeves had a good / bad year?
  2. Per-year mean allocation under IV (A) vs MVO (D) — did MVO mis-size?
  3. Per-sleeve PnL contribution to each ensemble per year — actual attribution.

Plus counterfactuals:
  E1: MVO-blend over only the 3 EXISTING sleeves     -> isolates "smart blender" effect
  E2: D with ResMR allocation forced to ZERO         -> isolates "new sleeve" effect
"""
from __future__ import annotations
import sys, warnings, time, json
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.append(str(Path("phase2_qrt_challenge/scripts").resolve()))
import utils  # noqa

from squeeze_breakout_pipeline import load_data, build_universe, build_returns, enforce_post_shift_strict_gmv
from ensemble_with_residual_mr import inverse_vol_blend, normalize_and_cap
from ensemble_sharpe_blender import sharpe_aware_blend

ART = Path("stores/sharpe_blender")
SLEEVE_CACHE = ART / "sleeves.pkl"
OOS_YEARS = 4

def sleeve_pnl(weights, returns):
    a = returns.reindex(index=weights.index, columns=weights.columns).fillna(0)
    return (weights * a).sum(axis=1)

def yearly_sharpe(s):
    if s.std() == 0: return 0.0
    return s.mean() / s.std() * np.sqrt(252)

def main():
    t0 = time.time()
    ohlcv = load_data()
    universe = build_universe(ohlcv)
    returns = build_returns(ohlcv)
    universe = universe.reindex(columns=returns.columns).fillna(0).astype(int)

    sleeves_all = pd.read_pickle(SLEEVE_CACHE)
    w_mr, w_mom, w_sa, w_res = (sleeves_all["MR"], sleeves_all["Mom"],
                                 sleeves_all["SA"], sleeves_all["ResMR"])
    sleeves_3 = {"MR": w_mr, "Mom": w_mom, "SA": w_sa}
    sleeves_4 = {"MR": w_mr, "Mom": w_mom, "SA": w_sa, "ResMR": w_res}

    oos_start = returns.index.max() - pd.DateOffset(years=OOS_YEARS)
    oos_idx = returns.index[returns.index >= oos_start]

    base_kw = dict(window=252, mu_shrink=0.5, sigma_shrink=0.5,
                   rebal='W-FRI', w_min=0.05, w_max=0.60)

    # ---- Build all ensembles + capture allocations ----
    print("[build] IV-3 (A)...")
    ens_A, alloc_A = inverse_vol_blend(sleeves_3, returns, universe)
    alloc_A = pd.DataFrame(alloc_A)
    print("[build] IV-4 (B)...")
    ens_B, alloc_B = inverse_vol_blend(sleeves_4, returns, universe)
    alloc_B = pd.DataFrame(alloc_B)
    print("[build] MVO-3 (C)...")
    ens_C, alloc_C = sharpe_aware_blend(sleeves_3, returns, universe, **base_kw)
    print("[build] MVO-4 (D)...")
    ens_D, alloc_D = sharpe_aware_blend(sleeves_4, returns, universe, **base_kw)

    # ---- Standalone sleeve PnLs ----
    pnl_mr  = sleeve_pnl(w_mr, returns)
    pnl_mom = sleeve_pnl(w_mom, returns)
    pnl_sa  = sleeve_pnl(w_sa, returns)
    pnl_res = sleeve_pnl(w_res, returns)
    pnl_sleeves = pd.DataFrame({"MR": pnl_mr, "Mom": pnl_mom, "SA": pnl_sa, "ResMR": pnl_res})

    # ---- Ensemble PnLs ----
    pnl_A = sleeve_pnl(ens_A, returns)
    pnl_D = sleeve_pnl(ens_D, returns)

    # ============================================================
    # Q1: per-year standalone sleeve Sharpe
    # ============================================================
    print("\n" + "="*78)
    print("Q1: per-year standalone Sharpe per sleeve (OOS only)")
    print("="*78)
    years = sorted(set(oos_idx.year))
    rows = []
    for y in years:
        idx = oos_idx[oos_idx.year == y]
        row = {"year": y, "n": len(idx)}
        for n in ["MR", "Mom", "SA", "ResMR"]:
            row[n] = yearly_sharpe(pnl_sleeves[n].loc[idx])
        rows.append(row)
    sleeve_sr = pd.DataFrame(rows)
    print(sleeve_sr.round(2).to_string(index=False))

    # ============================================================
    # Q2: per-year mean allocation A vs D
    # ============================================================
    print("\n" + "="*78)
    print("Q2: per-year mean allocation under IV-3 (A) vs MVO-4 (D)")
    print("="*78)
    rows = []
    for y in years:
        idx = oos_idx[oos_idx.year == y]
        row = {"year": y}
        for n in ["MR", "Mom", "SA"]:
            row[f"A_{n}"] = alloc_A[n].loc[idx].mean() * 100
        for n in ["MR", "Mom", "SA", "ResMR"]:
            row[f"D_{n}"] = alloc_D[n].loc[idx].mean() * 100
        rows.append(row)
    alloc_df = pd.DataFrame(rows)
    print(alloc_df.round(1).to_string(index=False))

    # ============================================================
    # Q3: per-sleeve contribution to ensemble PnL by year
    # Contribution_s = mean(alloc_s * sleeve_PnL_s) over the year, scaled to annual
    # ============================================================
    print("\n" + "="*78)
    print("Q3: per-sleeve PnL contribution per year (annualized %)")
    print("="*78)
    def contrib_table(alloc, sleeve_pnls, label):
        rows = []
        for y in years:
            idx = oos_idx[oos_idx.year == y]
            row = {"year": y}
            tot = 0.0
            for n in alloc.columns:
                c = (alloc[n].loc[idx] * sleeve_pnls[n].loc[idx]).mean() * 252 * 100
                row[n] = c; tot += c
            row["total"] = tot
            rows.append(row)
        df = pd.DataFrame(rows)
        print(f"\n[{label}] per-sleeve contribution (ann %)")
        print(df.round(2).to_string(index=False))
        return df

    contrib_A = contrib_table(alloc_A, pnl_sleeves[["MR","Mom","SA"]], "A) IV 3-sleeve")
    contrib_D = contrib_table(alloc_D, pnl_sleeves, "D) MVO 4-sleeve")

    # Side-by-side delta
    print("\n[delta D - A] (which sleeve drove the change each year)")
    rows = []
    for y in years:
        idx = oos_idx[oos_idx.year == y]
        row = {"year": y}
        for n in ["MR", "Mom", "SA"]:
            cA = (alloc_A[n].loc[idx] * pnl_sleeves[n].loc[idx]).mean() * 252 * 100
            cD = (alloc_D[n].loc[idx] * pnl_sleeves[n].loc[idx]).mean() * 252 * 100
            row[n] = cD - cA
        cD_res = (alloc_D["ResMR"].loc[idx] * pnl_sleeves["ResMR"].loc[idx]).mean() * 252 * 100
        row["ResMR (new)"] = cD_res
        row["total_delta"] = sum(row[k] for k in ["MR","Mom","SA","ResMR (new)"])
        rows.append(row)
    delta_df = pd.DataFrame(rows)
    print(delta_df.round(2).to_string(index=False))

    # ============================================================
    # Counterfactuals
    # ============================================================
    print("\n" + "="*78)
    print("COUNTERFACTUALS")
    print("="*78)

    # E1: MVO over the 3 existing sleeves (isolates "smart blender" effect)
    pnl_C = sleeve_pnl(ens_C, returns)

    # E2: D-style allocations but ResMR forced to 0 then renormalized
    alloc_D_nores = alloc_D[["MR","Mom","SA"]].copy()
    alloc_D_nores = alloc_D_nores.div(alloc_D_nores.sum(axis=1), axis=0).fillna(1.0/3.0)
    ens_D_noresmr_raw = (sleeves_3["MR"].multiply(alloc_D_nores["MR"], axis=0)
                         + sleeves_3["Mom"].multiply(alloc_D_nores["Mom"], axis=0)
                         + sleeves_3["SA"].multiply(alloc_D_nores["SA"], axis=0))
    longs = ens_D_noresmr_raw.where(ens_D_noresmr_raw > 0, 0)
    shorts = ens_D_noresmr_raw.where(ens_D_noresmr_raw < 0, 0).abs()
    ens_D_noresmr = (normalize_and_cap(longs) - normalize_and_cap(shorts)).fillna(0)
    ens_D_noresmr = enforce_post_shift_strict_gmv(ens_D_noresmr, universe)
    pnl_D_noresmr = sleeve_pnl(ens_D_noresmr, returns)

    # E3: A blender (IV) but with 4 sleeves -> already have (this is ensemble B)
    pnl_B = sleeve_pnl(ens_B, returns)

    print("\n[per-year Sharpe — head-to-head with counterfactuals]")
    rows = []
    for y in years:
        idx = oos_idx[oos_idx.year == y]
        rows.append({
            "year": y,
            "A (IV3)":            yearly_sharpe(pnl_A.loc[idx]),
            "B (IV4 naive)":      yearly_sharpe(pnl_B.loc[idx]),
            "C (MVO3 no-newsig)": yearly_sharpe(pnl_C.loc[idx]),
            "D (MVO4 full)":      yearly_sharpe(pnl_D.loc[idx]),
            "D w/o ResMR (cf)":   yearly_sharpe(pnl_D_noresmr.loc[idx]),
        })
    cf = pd.DataFrame(rows)
    print(cf.round(2).to_string(index=False))

    # Decompose D vs A into two effects per year:
    #   Blender effect  =  D_no_resmr  -  A
    #   New sleeve effect = D - D_no_resmr
    print("\n[decomposition: D - A  =  (blender effect) + (new sleeve effect)]")
    rows = []
    for y in years:
        idx = oos_idx[oos_idx.year == y]
        srA = yearly_sharpe(pnl_A.loc[idx])
        srD = yearly_sharpe(pnl_D.loc[idx])
        srD_no = yearly_sharpe(pnl_D_noresmr.loc[idx])
        rows.append({
            "year": y,
            "A SR": srA, "D SR": srD,
            "blender_effect": srD_no - srA,
            "new_sleeve_effect": srD - srD_no,
            "total_delta": srD - srA,
        })
    decomp = pd.DataFrame(rows)
    print(decomp.round(3).to_string(index=False))

    # Aggregate across OOS
    print("\n[aggregate over full OOS]")
    srA_full = yearly_sharpe(pnl_A.loc[oos_idx])
    srD_full = yearly_sharpe(pnl_D.loc[oos_idx])
    srD_no_full = yearly_sharpe(pnl_D_noresmr.loc[oos_idx])
    print(f"  A net SR (full OOS) : {srA_full:.3f}")
    print(f"  D w/o ResMR         : {srD_no_full:.3f}   (blender effect alone: {srD_no_full-srA_full:+.3f})")
    print(f"  D full              : {srD_full:.3f}   (full delta: {srD_full-srA_full:+.3f})")
    print(f"  new sleeve effect (D - Dwo): {srD_full-srD_no_full:+.3f}")

    # ============================================================
    # 2024 / 2026 deep dive — why did D underperform?
    # ============================================================
    print("\n" + "="*78)
    print("WHY 2024 / 2026 HURT UNDER MVO")
    print("="*78)
    for y in [2024, 2026]:
        idx = oos_idx[oos_idx.year == y]
        if len(idx) == 0: continue
        print(f"\n--- {y} ---")
        print(f"  N days: {len(idx)}")
        # standalone sleeve SR
        print(f"  Standalone sleeve SR:")
        for n in ["MR", "Mom", "SA", "ResMR"]:
            print(f"    {n:6s}: {yearly_sharpe(pnl_sleeves[n].loc[idx]):+.3f}")
        # mean allocations
        print(f"  Mean allocations (% of book):")
        print(f"    A (IV-3) : MR={alloc_A['MR'].loc[idx].mean()*100:.1f}  "
              f"Mom={alloc_A['Mom'].loc[idx].mean()*100:.1f}  "
              f"SA={alloc_A['SA'].loc[idx].mean()*100:.1f}")
        print(f"    D (MVO-4): MR={alloc_D['MR'].loc[idx].mean()*100:.1f}  "
              f"Mom={alloc_D['Mom'].loc[idx].mean()*100:.1f}  "
              f"SA={alloc_D['SA'].loc[idx].mean()*100:.1f}  "
              f"ResMR={alloc_D['ResMR'].loc[idx].mean()*100:.1f}")
        # contribution to PnL
        print(f"  Contribution to ensemble (ann %):")
        for n in ["MR", "Mom", "SA"]:
            cA = (alloc_A[n].loc[idx] * pnl_sleeves[n].loc[idx]).mean() * 252 * 100
            cD = (alloc_D[n].loc[idx] * pnl_sleeves[n].loc[idx]).mean() * 252 * 100
            print(f"    {n:6s}:  A={cA:+.2f}%   D={cD:+.2f}%   (D-A: {cD-cA:+.2f}%)")
        cR = (alloc_D["ResMR"].loc[idx] * pnl_sleeves["ResMR"].loc[idx]).mean() * 252 * 100
        print(f"    ResMR :   D={cR:+.2f}%   (new sleeve drag in {y}: {cR:+.2f}%)")

    # ============================================================
    # 2022 / 2023 / 2025 deep dive — where did gains come from?
    # ============================================================
    print("\n" + "="*78)
    print("WHY 2022 / 2023 / 2025 GAINED UNDER MVO")
    print("="*78)
    for y in [2022, 2023, 2025]:
        idx = oos_idx[oos_idx.year == y]
        if len(idx) == 0: continue
        print(f"\n--- {y} ---")
        print(f"  Standalone sleeve SR:  ", end="")
        print("  ".join([f"{n}={yearly_sharpe(pnl_sleeves[n].loc[idx]):+.2f}" for n in ["MR","Mom","SA","ResMR"]]))
        print(f"  Mean alloc D: MR={alloc_D['MR'].loc[idx].mean()*100:.1f}  "
              f"Mom={alloc_D['Mom'].loc[idx].mean()*100:.1f}  "
              f"SA={alloc_D['SA'].loc[idx].mean()*100:.1f}  "
              f"ResMR={alloc_D['ResMR'].loc[idx].mean()*100:.1f}")
        # Decompose this year's lift
        srA = yearly_sharpe(pnl_A.loc[idx])
        srD = yearly_sharpe(pnl_D.loc[idx])
        srD_no = yearly_sharpe(pnl_D_noresmr.loc[idx])
        print(f"  Decomposition: A={srA:+.2f}  ->  D={srD:+.2f}  (delta {srD-srA:+.2f})")
        print(f"    blender effect (better allocation of existing): {srD_no-srA:+.2f}")
        print(f"    new sleeve effect (ResMR alpha)               : {srD-srD_no:+.2f}")

    # Save
    out = {
        "sleeve_sharpe_by_year": sleeve_sr.to_dict("records"),
        "alloc_by_year": alloc_df.to_dict("records"),
        "contrib_A": contrib_A.to_dict("records"),
        "contrib_D": contrib_D.to_dict("records"),
        "decomp_per_year": decomp.to_dict("records"),
        "aggregate": {"A": srA_full, "D_no_resmr": srD_no_full, "D_full": srD_full},
    }
    (ART / "attribution.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[saved] {ART}/attribution.json")
    print(f"[total runtime: {time.time()-t0:.1f}s]")

if __name__ == "__main__":
    main()
