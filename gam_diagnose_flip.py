"""Sign-flipped GAM portfolios + best-pair + orthogonality vs existing 3d MR.

Uses the cached GAM predictions.

  (A) flip continuous tilt
  (B) flip bucket 10/10
  (C) best-pair: long D1, short D9 (the empirically best pair from decile audit)
  (D) flip bucket 20/20
  (E) flip bucket 30/30

For each: net SR / max DD / per-year SR / PnL correlation vs existing 3d MR sleeve.
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
from gam_diagnose_fast import portfolio_bucket, portfolio_continuous, report

ART = Path("stores/gam")
CACHE_PRED = ART / "predictions.pkl"
CACHE_TARGET = ART / "target.pkl"
SLEEVE_CACHE = Path("stores/sharpe_blender/sleeves.pkl")
OOS_START_YEAR = 2021

def portfolio_pair(alpha, universe, long_pct_lo, long_pct_hi, short_pct_lo, short_pct_hi):
    """Long the decile in [long_pct_lo, long_pct_hi); short the decile in [short_pct_lo, short_pct_hi).

    e.g. long D1: lo=0.0, hi=0.1
         short D9: lo=0.8, hi=0.9
    """
    rk = alpha.where(universe == 1, np.nan).rank(axis=1, pct=True)
    lm = ((rk >= long_pct_lo) & (rk < long_pct_hi)).astype(float)
    sm = ((rk >= short_pct_lo) & (rk < short_pct_hi)).astype(float)
    nL = lm.sum(axis=1).replace(0, np.nan)
    nS = sm.sum(axis=1).replace(0, np.nan)
    w = lm.div(nL, axis=0) * 0.5 + sm.div(nS, axis=0) * -0.5
    return enforce_post_shift_strict_gmv(w.fillna(0).shift(1), universe)

def main():
    t0 = time.time()
    print("[load] cache ...")
    pred_df = pd.read_pickle(CACHE_PRED)
    tgt = pd.read_pickle(CACHE_TARGET)
    log_ret = tgt["log_ret"]

    ohlcv = load_data()
    universe = build_universe(ohlcv)
    returns = build_returns(ohlcv)
    universe = universe.reindex(columns=returns.columns).fillna(0).astype(int)
    pred_df = pred_df.reindex(columns=returns.columns)

    oos_idx = returns.index[returns.index.year >= OOS_START_YEAR]
    print(f"[OOS] {oos_idx.min().date()}..{oos_idx.max().date()}  ({len(oos_idx)} days)")
    rets_oos = returns.loc[oos_idx]; univ_oos = universe.loc[oos_idx]
    pred_oos = pred_df.reindex(oos_idx)

    pred_flip = -pred_oos

    print("\n" + "#"*70); print("# SIGN-FLIPPED PORTFOLIOS + BEST-PAIR"); print("#"*70)

    print("\n--- (A) flip continuous tilt + inv-vol ---")
    w_A = portfolio_continuous(pred_flip, univ_oos, log_ret.loc[oos_idx])
    rA = report(w_A, rets_oos, univ_oos, "(A) flip continuous")

    print("\n--- (B) flip bucket 10/10 EW ---")
    w_B = portfolio_bucket(pred_flip, univ_oos)
    rB = report(w_B, rets_oos, univ_oos, "(B) flip bucket 10/10")

    print("\n--- (C) BEST PAIR: long D1 of original pred, short D9 ---")
    # In original prediction space: D1 = lowest 10%, D9 = 80-90%
    w_C = portfolio_pair(pred_oos, univ_oos,
                         long_pct_lo=0.00, long_pct_hi=0.10,
                         short_pct_lo=0.80, short_pct_hi=0.90)
    rC = report(w_C, rets_oos, univ_oos, "(C) long D1 / short D9")

    print("\n--- (D) flip bucket 20/20 EW ---")
    w_D = portfolio_bucket(pred_flip, univ_oos, long_pct=0.80, short_pct=0.20)
    rD = report(w_D, rets_oos, univ_oos, "(D) flip bucket 20/20")

    print("\n--- (E) flip bucket 30/30 EW ---")
    w_E = portfolio_bucket(pred_flip, univ_oos, long_pct=0.70, short_pct=0.30)
    rE = report(w_E, rets_oos, univ_oos, "(E) flip bucket 30/30")

    # ---- Orthogonality vs existing 3d MR sleeve ----
    print("\n\n" + "#"*70); print("# ORTHOGONALITY vs EXISTING 3d MR + LOW-VOL FACTOR"); print("#"*70)
    if SLEEVE_CACHE.exists():
        sleeves = pd.read_pickle(SLEEVE_CACHE)
        w_mr = sleeves["MR"]
        pnl_mr_oos = ((w_mr.loc[oos_idx] * rets_oos.fillna(0)).sum(axis=1))
        for label, r in [("(A) flip continuous", rA), ("(B) flip bucket 10/10", rB),
                         ("(C) D1/D9 best pair", rC), ("(D) flip bucket 20/20", rD),
                         ("(E) flip bucket 30/30", rE)]:
            corr = r["pnl"].corr(pnl_mr_oos)
            print(f"  PnL corr  {label:25s} vs existing 3d MR: {corr:+.3f}")

    # ---- 2026 sanity: does the signal still work or did it die? ----
    print("\n\n" + "#"*70); print("# 2026 SANITY: did the signal die?"); print("#"*70)
    print("\n[per-year breakdown of the best portfolio]")
    rows = [
        ("(A) flip continuous", rA),
        ("(B) flip bucket 10/10", rB),
        ("(C) D1/D9 best pair", rC),
        ("(D) flip bucket 20/20", rD),
        ("(E) flip bucket 30/30", rE),
    ]
    yr_data = {}
    for lbl, r in rows:
        pnl = r["pnl"]
        yr_sr = pnl.groupby(pnl.index.year).apply(lambda s: (s.mean()/s.std()*np.sqrt(252)) if s.std() > 0 else 0)
        yr_data[lbl] = yr_sr
    yr_df = pd.DataFrame(yr_data)
    print(yr_df.round(2).to_string())

    # ---- Headline ----
    print("\n\n" + "="*70); print("HEADLINE"); print("="*70)
    print(f"  {'label':30s}  {'net_SR':>7s}  {'max_DD%':>7s}  {'active_d':>8s}")
    for lbl, r in rows:
        print(f"  {lbl:30s}  {r['net_sr']:7.3f}  {r['max_dd']:7.2f}  {r['active_days']:8d}")

    best = max(rows, key=lambda x: x[1]["net_sr"])
    print(f"\n[best]: {best[0]} — net_SR {best[1]['net_sr']:.3f}, max_DD {best[1]['max_dd']:.2f}%")

    # Save
    out = {
        "oos": [str(oos_idx.min().date()), str(oos_idx.max().date())],
        "results": {lbl: {k: v for k, v in r.items() if k != "pnl"} for lbl, r in rows},
        "best": best[0],
    }
    (ART / "summary_flip.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[saved] {ART}/summary_flip.json")
    print(f"[total runtime: {time.time()-t0:.1f}s]")

if __name__ == "__main__":
    main()
