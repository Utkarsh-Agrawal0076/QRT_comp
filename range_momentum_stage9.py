"""Stage 9 — HMM regime-gated sleeve (point-in-time).

Uses the frozen 3-state Gaussian HMM (hidden_markov_model/) to get causal daily
P(bull)/P(chop)/P(bear) on SPY. Stage 8 showed the sleeve thrives in stress and
dies in calm bull, so the gate is ANTI-bull: scale GMV up in bear/chop, down in
bull. Regime lagged 1 day (no look-ahead beyond the frozen params).

Tests, on the full-history walk-forward sleeve (2013-2026):
  ungated | hard: skip bull-dominant days | hard: trade bear-dominant only
        | soft: GMV x [P(bear)+0.5 P(chop)] | soft: GMV x P(bear)
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.append(str(Path("phase2_qrt_challenge/scripts").resolve()))
sys.path.append(str(Path("hidden_markov_model").resolve()))
import range_momentum_pipeline as s1
import hmm_model

SQRT = np.sqrt(252); EXEC = 2e-4; FIN = 0.005/252
WF_FULL = Path("stores/range_mom/scores_wf_full.pkl")


def build_weights(score, universe, returns):
    """Quintile D9-10 long / D1-2 short, smooth-3, 5pt hysteresis. Returns shifted, univ-masked, GMV=1 weights."""
    idx = score.index; univ = universe.reindex(idx); ub = (univ == 1)
    ubr = univ.reindex(columns=returns.columns, fill_value=0)
    sc = score.rolling(3, min_periods=1).mean()
    r = sc.where(ub, np.nan).rank(axis=1, pct=True)
    inL = pd.DataFrame(False, index=r.index, columns=r.columns); inS = inL.copy()
    pL = pd.Series(False, index=r.columns); pS = pL.copy()
    for dt in r.index:
        x = r.loc[dt]
        cL = ((pL & (x >= 0.75)) | (x >= 0.80)) & x.notna()
        cS = ((pS & (x <= 0.25)) | (x <= 0.20)) & x.notna()
        inL.loc[dt] = cL.values; inS.loc[dt] = cS.values; pL, pS = cL.fillna(False), cS.fillna(False)
    L = (inL & ub).astype(float); S = (inS & ub).astype(float)
    nL = L.sum(1).replace(0, np.nan); nS = S.sum(1).replace(0, np.nan)
    w = (L.div(nL, axis=0).fillna(0)*0.5 - S.div(nS, axis=0).fillna(0)*0.5).shift(1).fillna(0)
    return w.reindex(columns=returns.columns, fill_value=0.0) * ubr.values


def net_pnl(w, returns, mult=None):
    if mult is not None:
        w = w.mul(mult.reindex(w.index).fillna(0.0), axis=0)
    rr = returns.reindex(index=w.index, columns=w.columns).fillna(0)
    g = (w * rr).sum(1); traded = w.diff().abs().sum(1).fillna(0); bv = w.abs().sum(1)
    return g - traded*EXEC - bv*FIN


def sr(x):
    x = x[x.abs() > 0] if (x.abs() > 0).any() else x
    return x.mean()/x.std()*SQRT if x.std() else 0.0


def main():
    ohlcv = s1.load_data(); universe = s1.build_universe(ohlcv); returns = s1.build_returns(ohlcv)
    universe = universe.reindex(columns=returns.columns).fillna(0).astype(int)
    wf = pd.read_pickle(WF_FULL)
    w = build_weights(wf, universe, returns)
    idx = w.index

    # HMM regime probs (causal filter, frozen params), lagged 1 day
    spy = pd.read_parquet("hidden_markov_model/spy_cache.parquet")
    spx = spy["Adj Close"] if "Adj Close" in spy.columns else spy.iloc[:, 0]
    if getattr(spx.index, "tz", None) is not None:
        spx.index = spx.index.tz_localize(None)
    probs = hmm_model.compute_regime_probs(spx).reindex(idx).ffill()
    pb, pc, pr = probs["bull"], probs["chop"], probs["bear"]
    print(f"[hmm] regime day-share over {idx.min().date()}..{idx.max().date()}: "
          f"bull={ (probs.idxmax(1)=='bull').mean()*100:.0f}%  chop={(probs.idxmax(1)=='chop').mean()*100:.0f}%  "
          f"bear={(probs.idxmax(1)=='bear').mean()*100:.0f}%")

    bull_dom = (probs.idxmax(1) == "bull").astype(float).shift(1).fillna(0)
    bear_dom = (probs.idxmax(1) == "bear").astype(float).shift(1).fillna(0)
    soft_antibull = (pr + 0.5*pc).shift(1).fillna(0.0)
    soft_bear = pr.shift(1).fillna(0.0)

    variants = {
        "ungated":                 None,
        "hard: skip bull-dominant": (1.0 - bull_dom),
        "hard: bear-dominant only": bear_dom,
        "soft: P(bear)+0.5P(chop)": soft_antibull,
        "soft: P(bear)":            soft_bear,
    }
    print("\n=== HMM-gated sleeve (full-history walk-forward, point-in-time) ===")
    rows = []
    for name, mult in variants.items():
        pnl = net_pnl(w, returns, mult)
        active = 100.0 if mult is None else (mult > 0.01).mean()*100
        avg_gmv = 1.0 if mult is None else mult.reindex(idx).fillna(0).mean()
        full = sr(pnl)
        yr = pnl.groupby(pnl.index.year).apply(lambda x: sr(x))
        rows.append((name, full, active, avg_gmv, yr))
        print(f"  {name:26s}: net_SR={full:+.3f}  active_days={active:3.0f}%  avgGMV={avg_gmv:.2f}")

    print("\n=== per-year net Sharpe ===")
    yt = pd.DataFrame({r[0]: r[4] for r in rows}).round(2)
    print(yt.to_string())


if __name__ == "__main__":
    main()
