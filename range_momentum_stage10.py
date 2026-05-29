"""Stage 10 — cross-sectional DISPERSION gate (the theoretically-correct regime var).

Stage 9 showed the SPY HMM is the wrong detector: this is a cross-sectional
dispersion sleeve, but the HMM measures index-level regime. Here the gate is the
strategy's actual driver: daily cross-sectional std of returns across the
universe (dispersion). High dispersion -> spikes/dislocations to harvest -> trade;
low dispersion -> flatten.

Point-in-time: dispersion smoothed (21d), ranked within a TRAILING 252d window
(no look-ahead), lagged 1 day. Compared to ungated + the best HMM gate.
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
import range_momentum_stage9 as s9
import hmm_model

SQRT = np.sqrt(252)
WF_FULL = Path("stores/range_mom/scores_wf_full.pkl")


def trailing_pctile(s, win=252):
    """Point-in-time percentile rank of each value within its trailing `win` window."""
    return s.rolling(win, min_periods=60).apply(lambda x: (x[-1] >= x).mean(), raw=True)


def main():
    ohlcv = s1.load_data(); universe = s1.build_universe(ohlcv); returns = s1.build_returns(ohlcv)
    universe = universe.reindex(columns=returns.columns).fillna(0).astype(int)
    wf = pd.read_pickle(WF_FULL)
    w = s9.build_weights(wf, universe, returns)
    idx = w.index

    # cross-sectional dispersion = daily XS std of returns across the universe
    rmask = returns.where(universe == 1)
    xs_disp = rmask.std(axis=1)
    disp_smooth = xs_disp.rolling(21, min_periods=10).mean()
    disp_pct = trailing_pctile(disp_smooth, 252).reindex(idx).shift(1)   # point-in-time, lagged

    print(f"[disp] cross-sectional dispersion gate, trailing-252 percentile, lagged 1d")
    print(f"  mean daily XS std (ann) ~ {xs_disp.reindex(idx).mean()*100:.2f}%  "
          f"pctile coverage {disp_pct.notna().mean()*100:.0f}%")

    # HMM best gate for reference
    spy = pd.read_parquet("hidden_markov_model/spy_cache.parquet")
    spx = spy["Adj Close"] if "Adj Close" in spy.columns else spy.iloc[:, 0]
    if getattr(spx.index, "tz", None) is not None:
        spx.index = spx.index.tz_localize(None)
    probs = hmm_model.compute_regime_probs(spx).reindex(idx).ffill()
    hmm_skipbull = (1.0 - (probs.idxmax(1) == "bull").astype(float).shift(1).fillna(0))

    variants = {
        "ungated":                       None,
        "HMM skip bull-dominant (ref)":  hmm_skipbull,
        "disp: skip bottom tercile":     (disp_pct > 0.33).astype(float),
        "disp: above-median only":       (disp_pct > 0.50).astype(float),
        "disp: top tercile only":        (disp_pct > 0.67).astype(float),
        "disp: soft GMV x pctile":       disp_pct.clip(0, 1),
    }
    print("\n=== cross-sectional dispersion gate (full-history walk-forward, point-in-time) ===")
    rows = []
    for name, mult in variants.items():
        m = None if mult is None else mult.reindex(idx).fillna(0.0)
        pnl = s9.net_pnl(w, returns, m)
        active = 100.0 if m is None else (m > 0.01).mean()*100
        avg = 1.0 if m is None else m.mean()
        full = s9.sr(pnl)
        yr = pnl.groupby(pnl.index.year).apply(lambda x: s9.sr(x))
        rows.append((name, yr));
        print(f"  {name:30s}: net_SR={full:+.3f}  active={active:3.0f}%  avgGMV={avg:.2f}")

    print("\n=== per-year net Sharpe ===")
    print(pd.DataFrame({n: y for n, y in rows}).round(2).to_string())

    # confirm the conditional thesis: strategy SR vs dispersion-pctile buckets
    print("\n=== strategy net SR by realized dispersion bucket (sanity, in-sample slicing) ===")
    pnl0 = s9.net_pnl(w, returns, None)
    dp = disp_pct.reindex(pnl0.index)
    for lo, hi, lab in [(0.0, 0.33, "low disp"), (0.33, 0.67, "mid disp"), (0.67, 1.01, "high disp")]:
        mask = (dp >= lo) & (dp < hi)
        p = pnl0[mask.values]
        print(f"  {lab:9s}: n={len(p):4d}  net_SR={s9.sr(p):+.2f}  ann={p.mean()*252*100:+.1f}%")


if __name__ == "__main__":
    main()
