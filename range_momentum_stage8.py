"""Stage 8 — full-history walk-forward + market-regime conditioning.

Hypothesis (user): the strategy may die in some market regimes and excel in
others. If so, portfolio generation should be a function of (prediction, regime)
-- gated by the existing HMM.

Protocol:
  - Walk-forward from the EARLIEST possible date: train trailing 3yr, predict the
    next 126d, roll through 2026. Gives point-in-time scores ~2013-2026.
  - Portfolio: principled quintile design (D9-10 long / D1-2 short, smooth-3,
    5pt hysteresis) -- NOT tuned to results.
  - Slice the strategy's net Sharpe by market regime (SPY trend x realized vol)
    to see whether performance is regime-conditional.
"""
from __future__ import annotations
import sys, warnings, time, json
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.append(str(Path("phase2_qrt_challenge/scripts").resolve()))
import range_momentum_pipeline as s1
import range_momentum_stage2 as s2
from sklearn.ensemble import HistGradientBoostingClassifier

ART = Path("stores/range_mom")
SQRT = np.sqrt(252); EXEC = 2e-4; FIN = 0.005/252
FEATS = ["f1_mom", "f2_vol", "f3_rng"]
EMBARGO = s2.LABEL_T + 1
PANEL_CACHE = ART / "panel.pkl"
WF_FULL = ART / "scores_wf_full.pkl"


def new_model():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4,
                                          l2_regularization=1.0, random_state=s2.SEED,
                                          validation_fraction=0.1, early_stopping=True)


def build_wf_full(ohlcv, universe, returns):
    if WF_FULL.exists():
        print("[wf] loading cached full-history walk-forward scores")
        return pd.read_pickle(WF_FULL)
    t0 = time.time()
    if PANEL_CACHE.exists():
        X = pd.read_pickle(PANEL_CACHE)
    else:
        fac = s1.build_factors(ohlcv, universe); label = s2.make_label(returns, s2.LABEL_T)
        X = pd.DataFrame({n: fac[n].stack(dropna=False) for n in FEATS}); X["y"] = label.stack(dropna=False)
        X = X.dropna(); pd.to_pickle(X, PANEL_CACHE)
    dates = X.index.get_level_values(0)
    all_dates = returns.index
    first_pred = all_dates[all_dates >= all_dates.min() + pd.DateOffset(years=3)][0]
    pred_dates = all_dates[all_dates >= first_pred]
    print(f"[wf] panel rows={len(X):,}; predicting {pred_dates.min().date()}..{pred_dates.max().date()} [{time.time()-t0:.0f}s]", flush=True)
    step = 126; segs = []
    for i in range(0, len(pred_dates), step):
        seg = pred_dates[i:i+step]; cutoff = seg[0]
        mask = (dates >= cutoff - pd.DateOffset(years=3)) & (dates <= cutoff - pd.Timedelta(days=EMBARGO*2))
        tr = X[mask]
        if len(tr) > s2.TRAIN_SUBSAMPLE:
            tr = tr.sample(s2.TRAIN_SUBSAMPLE, random_state=s2.SEED)
        m = new_model(); m.fit(tr[FEATS].values, tr["y"].values)
        pr = X[dates.isin(seg)]
        p = pd.Series(m.predict_proba(pr[FEATS].values)[:, 1], index=pr.index)
        segs.append(p.unstack().reindex(index=seg, columns=returns.columns))
    wf = pd.concat(segs)
    pd.to_pickle(wf, WF_FULL)
    print(f"[wf] built {wf.shape} [{time.time()-t0:.0f}s]", flush=True)
    return wf


def portfolio_netpnl(score, universe, returns):
    idx = score.index; univ = universe.reindex(idx); ub = (univ == 1); ubr = univ.reindex(columns=returns.columns, fill_value=0)
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
    w = w.reindex(columns=returns.columns, fill_value=0.0) * ubr.values
    rr = returns.reindex(index=idx, columns=w.columns).fillna(0)
    g = (w * rr).sum(1); traded = w.diff().abs().sum(1).fillna(0); bv = w.abs().sum(1)
    return g - traded*EXEC - bv*FIN


def market_regime(returns, idx):
    """SPY trend (126d) x realized vol (21d) regime labels on idx."""
    spy = pd.read_parquet("stores/spy_adj_close.parquet")
    spy = spy.iloc[:, 0] if isinstance(spy, pd.DataFrame) else spy
    sret = spy.reindex(returns.index).pct_change(fill_method=None)
    trend = np.where(spy.pct_change(126).reindex(idx) > 0, "bull", "bear")
    vol = (sret.rolling(21).std()*SQRT).reindex(idx)
    vlab = pd.qcut(vol, 3, labels=["lowvol", "midvol", "highvol"])
    return pd.Series(trend, index=idx, name="trend"), pd.Series(vlab, index=idx, name="vol"), sret.reindex(idx)


def sr(x): return x.mean()/x.std()*SQRT if x.std() else 0.0


def main():
    ohlcv = s1.load_data(); universe = s1.build_universe(ohlcv); returns = s1.build_returns(ohlcv)
    universe = universe.reindex(columns=returns.columns).fillna(0).astype(int)
    wf = build_wf_full(ohlcv, universe, returns)
    pnl = portfolio_netpnl(wf, universe, returns).dropna()
    idx = pnl.index

    print(f"\n=== full-history walk-forward, principled quintile portfolio ===")
    print(f"  period {idx.min().date()}..{idx.max().date()} ({len(idx)} days)  FULL net SR={sr(pnl):+.3f}  ann={pnl.mean()*252*100:+.1f}%")

    print("\n=== per-year NET Sharpe ===")
    yr = pnl.groupby(pnl.index.year).agg(SR=lambda x: sr(x), ann_pct=lambda x: x.mean()*252*100, n="count")
    print(yr.round(2).to_string())

    trend, vol, sret = market_regime(returns, idx)
    print("\n=== net Sharpe by MARKET REGIME (SPY trend x realized vol) ===")
    rows = []
    for t in ["bull", "bear"]:
        for v in ["lowvol", "midvol", "highvol"]:
            m = (trend == t) & (vol == v)
            if m.sum() < 30: continue
            p = pnl[m.values]
            rows.append({"trend": t, "vol": v, "n_days": int(m.sum()), "net_SR": sr(p), "ann_%": p.mean()*252*100})
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    print("\n=== net Sharpe by trend only / vol only ===")
    for lab, ser in [("trend", trend), ("vol", vol)]:
        for g in ser.dropna().unique():
            p = pnl[(ser == g).values]
            print(f"  {lab}={str(g):8s}: n={len(p):4d}  net_SR={sr(p):+.2f}  ann={p.mean()*252*100:+.1f}%")

    # correlation of strategy daily pnl with market return + market vol
    print(f"\n  corr(strategy pnl, SPY ret) = {pnl.corr(sret):+.3f}")
    print(f"  rolling-63d strategy Sharpe vs SPY vol corr = "
          f"{(pnl.rolling(63).mean()/pnl.rolling(63).std()*SQRT).corr((sret.rolling(21).std()*SQRT)):+.3f}")

    out = {"full_net_SR": float(sr(pnl)), "per_year": {int(k): float(v) for k, v in yr['SR'].items()},
           "regime": rows}
    (ART / "stage8_regime.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[saved] {ART}/stage8_regime.json")


if __name__ == "__main__":
    main()
