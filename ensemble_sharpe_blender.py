"""Sharpe-aware ensemble blender with shrinkage MVO.

Compares four ensembles on the same 4-year OOS window:
  A) Inverse-vol blend, 3 sleeves  (current production baseline)
  B) Inverse-vol blend, 4 sleeves  (naive addition — last run showed -0.025 SR)
  C) Sharpe-aware MVO, 3 sleeves   (does MVO help the existing ensemble?)
  D) Sharpe-aware MVO, 4 sleeves   (does MVO unlock the new sleeve?)

Blender uses 252-day rolling mu + Sigma, shrinkage 0.5 on both, monthly rebalance,
constrained max-Sharpe with w in [0.05, 0.60], sum(w)=1.
"""
from __future__ import annotations
import sys, warnings, time, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
sys.path.append(str(Path("phase2_qrt_challenge/scripts").resolve()))
import utils  # noqa

from squeeze_breakout_pipeline import load_data, build_universe, build_returns, enforce_post_shift_strict_gmv
from ensemble_with_residual_mr import (
    build_mr_sleeve, build_mom_sleeve, load_sa_sleeve, build_residual_mr_sleeve,
    inverse_vol_blend, normalize_and_cap, report_backtest,
)

ART = Path("stores/sharpe_blender"); ART.mkdir(exist_ok=True, parents=True)
SLEEVE_CACHE = ART / "sleeves.pkl"
OOS_YEARS = 4

# -------- Sharpe-aware MVO blender --------
def solve_max_sharpe(mu, Sigma, w_min=0.05, w_max=0.60):
    n = len(mu)
    Sigma = Sigma + np.eye(n) * 1e-10  # regularize
    def neg_sr(w):
        pr = float(w @ mu)
        pv = float(w @ Sigma @ w)
        if pv <= 0: return 0.0
        return -pr / np.sqrt(pv)
    bounds = [(w_min, w_max)] * n
    cons = [{'type': 'eq', 'fun': lambda w: w.sum() - 1.0}]
    x0 = np.ones(n) / n
    res = minimize(neg_sr, x0, method='SLSQP', bounds=bounds, constraints=cons,
                   options={'maxiter': 200, 'ftol': 1e-9})
    return res.x if res.success else x0

def sharpe_aware_blend(sleeves, returns, universe,
                       window=252, mu_shrink=0.5, sigma_shrink=0.5,
                       rebal='W-FRI', w_min=0.05, w_max=0.60):
    names = list(sleeves.keys())
    # Daily PnL per sleeve (causal — weights are already T+1 shifted upstream)
    pnls = {}
    for n in names:
        w = sleeves[n]
        a = returns.reindex(index=w.index, columns=w.columns).fillna(0)
        pnls[n] = (w * a).sum(axis=1)
    pnl_df = pd.DataFrame(pnls)

    rebal_dates = pnl_df.resample(rebal).last().index
    rebal_dates = [d for d in rebal_dates if d in pnl_df.index]

    allocs = pd.DataFrame(np.nan, index=pnl_df.index, columns=names)
    for dt in rebal_dates:
        end_iloc = pnl_df.index.get_loc(dt)
        if end_iloc < window:
            continue
        hist = pnl_df.iloc[end_iloc - window:end_iloc]   # strictly past
        mu_raw = hist.mean().values
        Sig_raw = hist.cov().values
        # Shrinkage
        mu = (1 - mu_shrink) * mu_raw + mu_shrink * mu_raw.mean()
        Sig = (1 - sigma_shrink) * Sig_raw + sigma_shrink * np.diag(np.diag(Sig_raw))
        w_opt = solve_max_sharpe(mu, Sig, w_min, w_max)
        allocs.loc[dt] = w_opt

    allocs = allocs.ffill().shift(1)            # next-day apply
    allocs = allocs.fillna(1.0 / len(names))    # warmup

    blended = sum(sleeves[n].multiply(allocs[n], axis=0) for n in names)
    longs = blended.where(blended > 0, 0)
    shorts = blended.where(blended < 0, 0).abs()
    final = (normalize_and_cap(longs) - normalize_and_cap(shorts)).fillna(0)
    return enforce_post_shift_strict_gmv(final, universe), allocs

# -------- Allocation summary --------
def alloc_summary(allocs, oos_idx, label):
    a = allocs.loc[oos_idx].dropna()
    print(f"\n[{label} allocations OOS]  N rebalance days: {a.dropna(how='all').shape[0]}")
    stat = pd.DataFrame({
        "mean": a.mean()*100, "min": a.min()*100, "max": a.max()*100,
        "std": a.std()*100
    })
    print(stat.round(2).to_string())

# -------- Main --------
def main():
    t0 = time.time()
    ohlcv = load_data()
    universe = build_universe(ohlcv)
    returns = build_returns(ohlcv)
    universe = universe.reindex(columns=returns.columns).fillna(0).astype(int)
    print(f"[data] {returns.index.min().date()}..{returns.index.max().date()}  stocks={returns.shape[1]}")

    # ---- Build (or load cached) sleeves ----
    if SLEEVE_CACHE.exists():
        print(f"[sleeves] loading cached sleeves from {SLEEVE_CACHE}")
        sleeves_all = pd.read_pickle(SLEEVE_CACHE)
        w_mr, w_mom, w_sa, w_res = sleeves_all["MR"], sleeves_all["Mom"], sleeves_all["SA"], sleeves_all["ResMR"]
    else:
        print("[sleeve 1] MR (3d vwap)...");     t1=time.time(); w_mr  = build_mr_sleeve(ohlcv, universe);          print(f"  {time.time()-t1:.1f}s")
        print("[sleeve 2] sector-neutral mom..."); t1=time.time(); w_mom = build_mom_sleeve(ohlcv, returns, universe); print(f"  {time.time()-t1:.1f}s")
        print("[sleeve 3] stat-arb cache...");    t1=time.time(); w_sa  = load_sa_sleeve(returns, universe);          print(f"  {time.time()-t1:.1f}s")
        print("[sleeve 4 NEW] residual MR k=5..."); t1=time.time(); w_res = build_residual_mr_sleeve(returns, universe, k=5); print(f"  {time.time()-t1:.1f}s")
        pd.to_pickle({"MR": w_mr, "Mom": w_mom, "SA": w_sa, "ResMR": w_res}, SLEEVE_CACHE)
        print(f"[cached] {SLEEVE_CACHE}")

    oos_start = returns.index.max() - pd.DateOffset(years=OOS_YEARS)
    oos_idx = returns.index[returns.index >= oos_start]
    print(f"\n[OOS] {oos_idx.min().date()}..{oos_idx.max().date()}  ({len(oos_idx)} days)")

    # ---- A) Inverse-vol 3-sleeve ----
    print("\n\n" + "#"*70); print("# A) INVERSE-VOL  3-SLEEVE  (current baseline)"); print("#"*70)
    sleeves_3 = {"MR": w_mr, "Mom": w_mom, "SA": w_sa}
    ens_A, alloc_A = inverse_vol_blend(sleeves_3, returns, universe)
    res_A = report_backtest(ens_A, returns, universe, "A) IV 3-sleeve", oos_idx)

    # ---- B) Inverse-vol 4-sleeve ----
    print("\n\n" + "#"*70); print("# B) INVERSE-VOL  4-SLEEVE  (naive add)"); print("#"*70)
    sleeves_4 = {"MR": w_mr, "Mom": w_mom, "SA": w_sa, "ResMR": w_res}
    ens_B, alloc_B = inverse_vol_blend(sleeves_4, returns, universe)
    res_B = report_backtest(ens_B, returns, universe, "B) IV 4-sleeve", oos_idx)

    # ---- C) Sharpe-aware 3-sleeve ----
    print("\n\n" + "#"*70); print("# C) SHARPE-AWARE MVO  3-SLEEVE"); print("#"*70)
    ens_C, alloc_C = sharpe_aware_blend(sleeves_3, returns, universe)
    res_C = report_backtest(ens_C, returns, universe, "C) MVO 3-sleeve", oos_idx)
    alloc_summary(alloc_C, oos_idx, "C MVO 3-sleeve")

    # ---- D) Sharpe-aware 4-sleeve ----
    print("\n\n" + "#"*70); print("# D) SHARPE-AWARE MVO  4-SLEEVE"); print("#"*70)
    ens_D, alloc_D = sharpe_aware_blend(sleeves_4, returns, universe)
    res_D = report_backtest(ens_D, returns, universe, "D) MVO 4-sleeve", oos_idx)
    alloc_summary(alloc_D, oos_idx, "D MVO 4-sleeve")

    # ---- Head-to-head matrix ----
    print("\n\n" + "="*70); print("HEAD-TO-HEAD"); print("="*70)
    rows = [
        ("A) IV 3-sleeve  (current baseline)", res_A),
        ("B) IV 4-sleeve  (naive add)        ", res_B),
        ("C) MVO 3-sleeve (smarter blender)  ", res_C),
        ("D) MVO 4-sleeve (smarter + new)    ", res_D),
    ]
    print(f"  {'label':40s}  {'net_SR':>7s}  {'max_DD%':>7s}  {'hit%':>6s}")
    for lbl, r in rows:
        print(f"  {lbl:40s}  {r['net_sr']:7.3f}  {r['max_dd']:7.2f}  {r['hit_days']:6.1f}")

    print("\n[Deltas vs current baseline (A)]")
    base = res_A["net_sr"]
    for lbl, r in rows[1:]:
        d_sr = r["net_sr"] - base
        d_dd = r["max_dd"] - res_A["max_dd"]
        print(f"  {lbl}: dSR {d_sr:+.3f}  dDD {d_dd:+.2f}pp")

    # PnL correlations between the four
    print("\n[PnL correlations between ensembles]")
    pnl_4 = pd.DataFrame({
        "A_IV3": res_A["pnl"], "B_IV4": res_B["pnl"],
        "C_MVO3": res_C["pnl"], "D_MVO4": res_D["pnl"],
    })
    print(pnl_4.corr().round(3).to_string())

    # IR of (D - C) and (D - B): does adding new sleeve under MVO add value?
    print("\n[Information ratios — does the new sleeve help under each blender?]")
    diff_BA = res_B["pnl"] - res_A["pnl"]
    diff_DC = res_D["pnl"] - res_C["pnl"]
    diff_CA = res_C["pnl"] - res_A["pnl"]
    print(f"  (B-A) IV blender, new sleeve added : ann={diff_BA.mean()*252*100:+.2f}%  IR={diff_BA.mean()/diff_BA.std()*np.sqrt(252):+.2f}")
    print(f"  (D-C) MVO blender, new sleeve added: ann={diff_DC.mean()*252*100:+.2f}%  IR={diff_DC.mean()/diff_DC.std()*np.sqrt(252):+.2f}")
    print(f"  (C-A) blender swap, 3 sleeves     : ann={diff_CA.mean()*252*100:+.2f}%  IR={diff_CA.mean()/diff_CA.std()*np.sqrt(252):+.2f}")

    # ---- Save ----
    out = {
        "oos": [str(oos_idx.min().date()), str(oos_idx.max().date())],
        "A_iv3":  {k:v for k,v in res_A.items() if k!="pnl"},
        "B_iv4":  {k:v for k,v in res_B.items() if k!="pnl"},
        "C_mvo3": {k:v for k,v in res_C.items() if k!="pnl"},
        "D_mvo4": {k:v for k,v in res_D.items() if k!="pnl"},
        "ensemble_corr": pnl_4.corr().round(3).to_dict(),
    }
    (ART / "summary.json").write_text(json.dumps(out, indent=2, default=str))
    pnl_4.to_csv(ART / "ensemble_pnl.csv")
    alloc_D.to_csv(ART / "alloc_D_mvo4.csv")
    print(f"\n[saved] {ART}/")
    print(f"[total runtime: {time.time()-t0:.1f}s]")

if __name__ == "__main__":
    main()
