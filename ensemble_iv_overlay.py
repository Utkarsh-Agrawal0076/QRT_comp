"""IV-3 baseline + fixed-allocation ResMR overlay (no MVO).

Simplest possible integration of the new sleeve:
  ens = (1-x) * IV_3_sleeve_ensemble  +  x * ResMR_sleeve
  ens = normalize_and_cap (longs/shorts separately), enforce GMV=1, max_weight cap

Sweeps x in {0.00, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20} and compares to
  A) IV 3-sleeve baseline   (= x=0)
  B) IV 4-sleeve naive      (lets IV pick allocation, ~16% to ResMR)
  D) MVO 4-sleeve           (MVO picks allocation, ~5% to ResMR)

Reports: full-OOS net SR / DD, per-year SR, first-half / second-half SR.
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

def overlay(ens_iv3, w_resmr, x, universe):
    """(1-x) * IV_3-sleeve  +  x * ResMR, then renormalize and cap."""
    raw = (1 - x) * ens_iv3 + x * w_resmr
    longs = raw.where(raw > 0, 0)
    shorts = raw.where(raw < 0, 0).abs()
    final = (normalize_and_cap(longs) - normalize_and_cap(shorts)).fillna(0)
    return enforce_post_shift_strict_gmv(final, universe)

def quick_eval(weights, returns, universe, oos_idx, label=""):
    import io, contextlib
    w = weights.loc[oos_idx].reindex(columns=returns.columns, fill_value=0)
    r = returns.loc[oos_idx]
    u = universe.loc[oos_idx].reindex(columns=returns.columns, fill_value=0)
    with contextlib.redirect_stdout(io.StringIO()):
        sr, pnl = utils.backtest_portfolio(w, r, u, plot_=False, print_=True)
    eq = (1+pnl).cumprod(); dd = eq/eq.cummax() - 1
    turn = w.diff(1).abs().sum(axis=1).mean() / w.abs().sum(axis=1).mean() * 100
    return {"label": label, "net_sr": float(sr), "max_dd": float(dd.min()*100),
            "ann_vol": float(pnl.std()*np.sqrt(252)*100),
            "hit_days": float((pnl>0).mean()*100),
            "turnover": float(turn), "pnl": pnl}

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
    print(f"[OOS] {oos_idx.min().date()}..{oos_idx.max().date()}  ({len(oos_idx)} days)")

    # Reference ensembles
    print("\n[ref] building reference ensembles...")
    ens_A, _ = inverse_vol_blend(sleeves_3, returns, universe)
    ens_B, _ = inverse_vol_blend(sleeves_4, returns, universe)
    ens_D, _ = sharpe_aware_blend(sleeves_4, returns, universe,
                                   window=252, mu_shrink=0.5, sigma_shrink=0.5,
                                   rebal='W-FRI', w_min=0.05, w_max=0.60)
    res_A = quick_eval(ens_A, returns, universe, oos_idx, "A IV3 base")
    res_B = quick_eval(ens_B, returns, universe, oos_idx, "B IV4 naive")
    res_D = quick_eval(ens_D, returns, universe, oos_idx, "D MVO4 base")

    # Build overlay sweep
    print("\n[overlay] sweeping ResMR allocation x = 0..0.20")
    results = []
    for x in [0.00, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]:
        ens_x = overlay(ens_A, w_res, x, universe)
        r = quick_eval(ens_x, returns, universe, oos_idx, f"IV3+{x*100:.0f}%ResMR")
        r["x"] = x
        # per-year
        pnl = r["pnl"]
        yr_sr = {y: (pnl[pnl.index.year==y].mean()/pnl[pnl.index.year==y].std()*np.sqrt(252))
                 if pnl[pnl.index.year==y].std() > 0 else 0
                 for y in sorted(set(pnl.index.year))}
        for y, sr in yr_sr.items():
            r[f"SR_{y}"] = sr
        # subperiods
        mid = oos_idx[len(oos_idx)//2]
        first = pnl[pnl.index < mid]; second = pnl[pnl.index >= mid]
        r["SR_first_half"] = first.mean()/first.std()*np.sqrt(252) if first.std() > 0 else 0
        r["SR_second_half"] = second.mean()/second.std()*np.sqrt(252) if second.std() > 0 else 0
        results.append(r)

    # ---- Display ----
    print("\n" + "="*90)
    print("OVERLAY SWEEP: net Sharpe vs ResMR share")
    print("="*90)
    df = pd.DataFrame(results)
    cols = ["x", "net_sr", "max_dd", "ann_vol", "turnover", "hit_days",
            "SR_first_half", "SR_second_half"]
    print(df[cols].round(3).to_string(index=False))

    yr_cols = [c for c in df.columns if c.startswith("SR_2")]
    print("\n[per-year breakdown]")
    print(df[["x"] + yr_cols].round(3).to_string(index=False))

    # Reference comparison
    print("\n" + "="*90)
    print("VS REFERENCE ENSEMBLES")
    print("="*90)
    def row(r):
        return {"label": r["label"], "net_sr": r["net_sr"], "max_dd": r["max_dd"],
                "ann_vol": r["ann_vol"], "turnover": r["turnover"]}
    refs = pd.DataFrame([row(res_A), row(res_B), row(res_D)] +
                        [{"label": f"IV3+{r['x']*100:.0f}%ResMR", **{k: r[k] for k in ["net_sr","max_dd","ann_vol","turnover"]}}
                         for r in results])
    print(refs.round(3).to_string(index=False))

    # Best overlay
    best = max(results, key=lambda r: r["net_sr"])
    print(f"\n[best overlay] x={best['x']*100:.0f}%  net_SR={best['net_sr']:.3f}  "
          f"max_DD={best['max_dd']:.2f}%   delta vs A: {best['net_sr']-res_A['net_sr']:+.3f}")
    print(f"[vs MVO D]    : D net_SR={res_D['net_sr']:.3f}  (best overlay {'beats' if best['net_sr'] > res_D['net_sr'] else 'loses to'} D)")

    # Subperiod stability across overlays
    print("\n" + "="*90)
    print("SUBPERIOD STABILITY (does x's effect stay positive in both halves?)")
    print("="*90)
    A_first_pnl = res_A["pnl"][res_A["pnl"].index < oos_idx[len(oos_idx)//2]]
    A_second_pnl = res_A["pnl"][res_A["pnl"].index >= oos_idx[len(oos_idx)//2]]
    SR_A_first = A_first_pnl.mean()/A_first_pnl.std()*np.sqrt(252)
    SR_A_second = A_second_pnl.mean()/A_second_pnl.std()*np.sqrt(252)
    print(f"  Baseline A: first_half_SR={SR_A_first:.3f}  second_half_SR={SR_A_second:.3f}")
    rows = []
    for r in results:
        rows.append({
            "x": r["x"],
            "first_SR": r["SR_first_half"],
            "second_SR": r["SR_second_half"],
            "first_delta": r["SR_first_half"] - SR_A_first,
            "second_delta": r["SR_second_half"] - SR_A_second,
            "min_delta": min(r["SR_first_half"] - SR_A_first, r["SR_second_half"] - SR_A_second),
        })
    sub_df = pd.DataFrame(rows)
    print(sub_df.round(3).to_string(index=False))

    print("\n[Pareto check]: any x where both halves improve over A?")
    pareto = sub_df[(sub_df["first_delta"] > 0) & (sub_df["second_delta"] > 0)]
    if len(pareto):
        print(pareto.round(3).to_string(index=False))
    else:
        print("  None — every x trades first-half gain for second-half loss (or vice versa)")

    # Save
    out = {
        "baseline_A": {k: v for k, v in res_A.items() if k != "pnl"},
        "iv_naive_B": {k: v for k, v in res_B.items() if k != "pnl"},
        "mvo_D":      {k: v for k, v in res_D.items() if k != "pnl"},
        "overlay_sweep": [{k: v for k, v in r.items() if k != "pnl"} for r in results],
        "subperiod_stability": sub_df.to_dict("records"),
    }
    (ART / "iv_overlay.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[saved] {ART}/iv_overlay.json")
    print(f"[total runtime: {time.time()-t0:.1f}s]")

if __name__ == "__main__":
    main()
