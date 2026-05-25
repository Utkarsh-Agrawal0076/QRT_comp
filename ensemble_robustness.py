"""Robustness sweep for the MVO 4-sleeve ensemble (Ensemble D).

Base config: w_min=0.05, w_max=0.60, lookback=252, mu_shrink=0.5, sigma_shrink=0.5,
             rebalance=W-FRI.

Sweeps (one at a time around the base):
  1) w_min in {0.00, 0.02, 0.05, 0.10}    -> does new sleeve survive lower floor?
  2) w_max in {0.50, 0.60, 0.70, 0.80}    -> does relaxing SA cap help?
  3) lookback in {126, 252, 504}          -> sensitivity to estimation window
  4) shrinkage in (mu, sigma) variations  -> what if estimates are less regularized?
  5) rebalance freq in {W-FRI, M, BMS}    -> turnover/adaptation tradeoff

Also: subperiod split (first half / second half of OOS) for both A and D
to check that the lift isn't driven by a single regime.
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

def quick_eval(weights, returns, universe, oos_idx, label=""):
    """Backtest on OOS and return key metrics + pnl series (suppress prints)."""
    w = weights.loc[oos_idx].reindex(columns=returns.columns, fill_value=0)
    r = returns.loc[oos_idx]
    u = universe.loc[oos_idx].reindex(columns=returns.columns, fill_value=0)
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        sr, pnl = utils.backtest_portfolio(w, r, u, plot_=False, print_=True)
    eq = (1+pnl).cumprod(); dd = eq/eq.cummax() - 1
    turnover = w.diff(1).abs().sum(axis=1).mean() / w.abs().sum(axis=1).mean() * 100
    return {
        "label": label, "net_sr": float(sr),
        "gross_sr": float((pnl.mean()/pnl.std()*np.sqrt(252))) if pnl.std() > 0 else 0.0,
        "max_dd": float(dd.min()*100),
        "ann_vol": float(pnl.std()*np.sqrt(252)*100),
        "turnover": float(turnover),
        "hit_days": float((pnl > 0).mean()*100),
        "pnl": pnl,
    }

def run_config(sleeves, returns, universe, oos_idx, label, **kwargs):
    w, alloc = sharpe_aware_blend(sleeves, returns, universe, **kwargs)
    r = quick_eval(w, returns, universe, oos_idx, label)
    if "ResMR" in alloc.columns:
        a_oos = alloc.loc[oos_idx].dropna()
        r["resmr_alloc_mean"] = float(a_oos["ResMR"].mean() * 100)
        r["sa_alloc_mean"] = float(a_oos["SA"].mean() * 100)
    return r

def print_sweep(rows, title, vary_col):
    print(f"\n{'='*78}\n{title}\n{'='*78}")
    df = pd.DataFrame(rows)
    keep = [vary_col, "net_sr", "gross_sr", "ann_vol", "max_dd", "turnover",
            "hit_days", "resmr_alloc_mean", "sa_alloc_mean"]
    keep = [c for c in keep if c in df.columns]
    print(df[keep].round(3).to_string(index=False))

def main():
    t0 = time.time()
    ohlcv = load_data()
    universe = build_universe(ohlcv)
    returns = build_returns(ohlcv)
    universe = universe.reindex(columns=returns.columns).fillna(0).astype(int)

    if not SLEEVE_CACHE.exists():
        print(f"[err] cache missing: {SLEEVE_CACHE} — run ensemble_sharpe_blender.py first")
        return
    sleeves_all = pd.read_pickle(SLEEVE_CACHE)
    sleeves_3 = {"MR": sleeves_all["MR"], "Mom": sleeves_all["Mom"], "SA": sleeves_all["SA"]}
    sleeves_4 = {**sleeves_3, "ResMR": sleeves_all["ResMR"]}

    oos_start = returns.index.max() - pd.DateOffset(years=OOS_YEARS)
    oos_idx = returns.index[returns.index >= oos_start]
    print(f"[OOS] {oos_idx.min().date()}..{oos_idx.max().date()}  ({len(oos_idx)} days)")

    base_kw = dict(window=252, mu_shrink=0.5, sigma_shrink=0.5,
                   rebal='W-FRI', w_min=0.05, w_max=0.60)

    # Reference points
    print("\n[ref] computing IV-3 (baseline A) and MVO-4 base (D)")
    ens_A, _ = inverse_vol_blend(sleeves_3, returns, universe)
    res_A = quick_eval(ens_A, returns, universe, oos_idx, "A) IV3 baseline")
    res_D = run_config(sleeves_4, returns, universe, oos_idx, "D) MVO4 base", **base_kw)
    print(f"  A (IV3) net_SR={res_A['net_sr']:.3f}  max_DD={res_A['max_dd']:.2f}%")
    print(f"  D (MVO4 base) net_SR={res_D['net_sr']:.3f}  max_DD={res_D['max_dd']:.2f}%  "
          f"resmr_alloc={res_D['resmr_alloc_mean']:.1f}%")

    # ---- Sweep 1: w_min ----
    print("\n\n[sweep] w_min ...")
    rows = []
    for w_min in [0.00, 0.02, 0.05, 0.10]:
        kw = {**base_kw, "w_min": w_min}
        r = run_config(sleeves_4, returns, universe, oos_idx, f"w_min={w_min}", **kw)
        r["w_min"] = w_min
        rows.append(r)
    print_sweep(rows, "SWEEP 1: w_min (sleeve allocation floor)", "w_min")
    sweep1 = rows

    # ---- Sweep 2: w_max ----
    print("\n\n[sweep] w_max ...")
    rows = []
    for w_max in [0.50, 0.60, 0.70, 0.80]:
        kw = {**base_kw, "w_max": w_max}
        r = run_config(sleeves_4, returns, universe, oos_idx, f"w_max={w_max}", **kw)
        r["w_max"] = w_max
        rows.append(r)
    print_sweep(rows, "SWEEP 2: w_max (sleeve allocation cap)", "w_max")
    sweep2 = rows

    # ---- Sweep 3: lookback window ----
    print("\n\n[sweep] window ...")
    rows = []
    for window in [126, 252, 504, 756]:
        kw = {**base_kw, "window": window}
        r = run_config(sleeves_4, returns, universe, oos_idx, f"window={window}", **kw)
        r["window"] = window
        rows.append(r)
    print_sweep(rows, "SWEEP 3: lookback window (days)", "window")
    sweep3 = rows

    # ---- Sweep 4: shrinkage ----
    print("\n\n[sweep] shrinkage ...")
    rows = []
    for (mu_s, sig_s) in [(0.0, 0.0), (0.2, 0.2), (0.5, 0.5), (0.8, 0.8), (0.0, 0.5), (0.5, 0.0)]:
        kw = {**base_kw, "mu_shrink": mu_s, "sigma_shrink": sig_s}
        r = run_config(sleeves_4, returns, universe, oos_idx, f"mu={mu_s} sig={sig_s}", **kw)
        r["mu_shrink"] = mu_s; r["sigma_shrink"] = sig_s
        rows.append(r)
    print_sweep(rows, "SWEEP 4: shrinkage (mu_shrink, sigma_shrink)", "label")
    sweep4 = rows

    # ---- Sweep 5: rebalance frequency ----
    print("\n\n[sweep] rebalance ...")
    rows = []
    for reb in ['W-FRI', 'BM', 'BMS', '2W-FRI']:
        kw = {**base_kw, "rebal": reb}
        r = run_config(sleeves_4, returns, universe, oos_idx, f"rebal={reb}", **kw)
        r["rebal"] = reb
        rows.append(r)
    print_sweep(rows, "SWEEP 5: rebalance frequency", "rebal")
    sweep5 = rows

    # ---- Subperiod split for A vs D ----
    print("\n\n" + "="*78); print("SUBPERIOD ROBUSTNESS — does Ensemble D beat A in both halves?"); print("="*78)
    mid = oos_idx[len(oos_idx)//2]
    first_half = oos_idx[oos_idx <  mid]
    second_half = oos_idx[oos_idx >= mid]
    print(f"  first half : {first_half.min().date()}..{first_half.max().date()}  ({len(first_half)} d)")
    print(f"  second half: {second_half.min().date()}..{second_half.max().date()}  ({len(second_half)} d)")

    ens_D_w, _ = sharpe_aware_blend(sleeves_4, returns, universe, **base_kw)
    sub_rows = []
    for sub_idx, name in [(first_half, "first half"), (second_half, "second half")]:
        rA = quick_eval(ens_A, returns, universe, sub_idx, f"A {name}")
        rD = quick_eval(ens_D_w, returns, universe, sub_idx, f"D {name}")
        sub_rows.append({"period": name, "A_net_SR": rA["net_sr"], "D_net_SR": rD["net_sr"],
                         "A_DD": rA["max_dd"], "D_DD": rD["max_dd"],
                         "delta_SR": rD["net_sr"] - rA["net_sr"]})
    print(pd.DataFrame(sub_rows).round(3).to_string(index=False))

    # ---- Per-year head-to-head ----
    print("\n\n" + "="*78); print("PER-YEAR HEAD-TO-HEAD: A vs D"); print("="*78)
    pnl_A = res_A["pnl"]; pnl_D = res_D["pnl"]
    yrs = sorted(set(pnl_A.index.year))
    rows = []
    for y in yrs:
        pa = pnl_A[pnl_A.index.year == y]; pd_ = pnl_D[pnl_D.index.year == y]
        rows.append({
            "year": y, "n": len(pa),
            "A_SR": (pa.mean()/pa.std()*np.sqrt(252)) if pa.std()>0 else 0,
            "D_SR": (pd_.mean()/pd_.std()*np.sqrt(252)) if pd_.std()>0 else 0,
        })
    yr_df = pd.DataFrame(rows)
    yr_df["delta"] = yr_df["D_SR"] - yr_df["A_SR"]
    print(yr_df.round(3).to_string(index=False))

    # ---- Headline summary ----
    print("\n\n" + "="*78); print("ROBUSTNESS HEADLINE"); print("="*78)
    print(f"  Baseline A (IV 3-sleeve) net_SR: {res_A['net_sr']:.3f}")
    print(f"  Base D (MVO 4-sleeve)    net_SR: {res_D['net_sr']:.3f}   delta: {res_D['net_sr']-res_A['net_sr']:+.3f}")

    def best_worst(rows, label):
        sr = [r["net_sr"] for r in rows]
        return min(sr), max(sr), label
    for label, rows in [("w_min", sweep1), ("w_max", sweep2), ("window", sweep3),
                        ("shrinkage", sweep4), ("rebalance", sweep5)]:
        lo, hi, _ = best_worst(rows, label)
        bw = "OK " if lo > res_A["net_sr"] else "MIX" if hi > res_A["net_sr"] else "BAD"
        print(f"  sweep {label:10s} min={lo:.3f}  max={hi:.3f}   {bw}  (vs baseline A {res_A['net_sr']:.3f})")

    # Save
    def clean(rs): return [{k: v for k, v in r.items() if k != "pnl"} for r in rs]
    out = {
        "baseline_A": {k: v for k, v in res_A.items() if k != "pnl"},
        "base_D":     {k: v for k, v in res_D.items() if k != "pnl"},
        "sweep_w_min":    clean(sweep1),
        "sweep_w_max":    clean(sweep2),
        "sweep_window":   clean(sweep3),
        "sweep_shrinkage":clean(sweep4),
        "sweep_rebalance":clean(sweep5),
        "subperiod_A_vs_D": sub_rows,
        "per_year_A_vs_D": yr_df.to_dict("records"),
    }
    (ART / "robustness.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[saved] {ART}/robustness.json")
    print(f"[total runtime: {time.time()-t0:.1f}s]")

if __name__ == "__main__":
    main()
