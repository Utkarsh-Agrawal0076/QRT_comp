"""Stage 6 — regime-staleness test (user hypothesis).

Hypothesis: the static model trained on 2010-2021 learned a mean-reversion-era
mapping and predicts stale winners/losers in the 2023-2026 regime. If true, a
walk-forward model that always trains on the RECENT regime should beat it.

Tests:
  A. Walk-forward: retrain every 126d on trailing 3yr, predict next 126d.
     Compare per-year IC + net SR vs the static model.
  B. Train-era split: model trained on 2010-2015 vs 2016-2021, scored on the
     same OOS. If OOS predictions/IC are similar, the f->return relationship is
     regime-stable (hypothesis false). If recent-era model wins, it's real.
Portfolio eval uses the chosen design: D7-D10 long / D1-D4 short, 5pt hysteresis,
3-day score smoothing, raw cross-sectional rank.
"""
from __future__ import annotations
import sys, warnings, time, json
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.append(str(Path("phase2_qrt_challenge/scripts").resolve()))
import utils  # noqa
import range_momentum_pipeline as s1
import range_momentum_stage2 as s2
from sklearn.ensemble import HistGradientBoostingClassifier

ART = Path("stores/range_mom")
SQRT = np.sqrt(252); EXEC = 2e-4; FIN = 0.005 / 252
FEATS = ["f1_mom", "f2_vol", "f3_rng"]
EMBARGO = s2.LABEL_T + 1   # days between train feature date and cutoff (label leakage guard)


def new_model():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4,
                                          l2_regularization=1.0, random_state=s2.SEED,
                                          validation_fraction=0.1, early_stopping=True)


def chosen_design_netpnl(score, univ, returns, oos_idx):
    """D7-D10 long / D1-D4 short, 5pt hysteresis, smooth-3, raw rank. Returns net pnl."""
    ub = (univ == 1)
    sc = score.rolling(3, min_periods=1).mean()
    r = sc.where(ub, np.nan).rank(axis=1, pct=True)
    inL = pd.DataFrame(False, index=r.index, columns=r.columns); inS = inL.copy()
    pL = pd.Series(False, index=r.columns); pS = pL.copy()
    for dt in r.index:
        x = r.loc[dt]
        cL = ((pL & (x >= 0.55)) | (x >= 0.60)) & x.notna()
        cS = ((pS & (x <= 0.45)) | (x <= 0.40)) & x.notna()
        inL.loc[dt] = cL.values; inS.loc[dt] = cS.values
        pL, pS = cL.fillna(False), cS.fillna(False)
    L = (inL & ub).astype(float); S = (inS & ub).astype(float)
    nL = L.sum(1).replace(0, np.nan); nS = S.sum(1).replace(0, np.nan)
    w = (L.div(nL, axis=0).fillna(0)*0.5 - S.div(nS, axis=0).fillna(0)*0.5).shift(1).fillna(0)
    w = w.reindex(columns=returns.columns, fill_value=0.0) * univ.reindex(columns=returns.columns, fill_value=0).values
    rr = returns.reindex(columns=w.columns).loc[oos_idx].fillna(0)
    g = (w.loc[oos_idx] * rr).sum(1)
    traded = w.loc[oos_idx].diff().abs().sum(1).fillna(0); bv = w.loc[oos_idx].abs().sum(1)
    return g - traded*EXEC - bv*FIN


def daily_ic(sig, fwd, mn=200):
    common = sig.dropna(how="all").index.intersection(fwd.dropna(how="all").index)
    out, dts = [], []
    for dt in common:
        s = sig.loc[dt].dropna(); f = fwd.loc[dt].reindex(s.index).dropna()
        if len(f) < mn: continue
        cs = s.index.intersection(f.index); out.append(s[cs].rank().corr(f[cs].rank())); dts.append(dt)
    return pd.Series(out, index=pd.DatetimeIndex(dts))


def per_year_sr(pnl):
    return pnl.groupby(pnl.index.year).apply(lambda x: (x.mean()/x.std()*SQRT) if x.std() else 0)


def main():
    t0 = time.time()
    ohlcv = s1.load_data(); universe = s1.build_universe(ohlcv); returns = s1.build_returns(ohlcv)
    universe = universe.reindex(columns=returns.columns).fillna(0).astype(int)
    oos_start = returns.index.max() - pd.DateOffset(years=s2.OOS_YEARS)
    oos_idx = returns.index[returns.index >= oos_start]
    fac = s1.build_factors(ohlcv, universe)
    label = s2.make_label(returns, s2.LABEL_T)

    # ---- build full stacked panel once (date,stock)->X,y ----
    print(f"[panel] stacking full sample ... [{time.time()-t0:.0f}s]", flush=True)
    cols = {n: fac[n].stack(dropna=False) for n in FEATS}
    X = pd.DataFrame(cols); X["y"] = label.stack(dropna=False)
    X = X.dropna()
    dates = X.index.get_level_values(0)
    print(f"  panel rows={len(X):,}  [{time.time()-t0:.0f}s]", flush=True)

    def fit_predict(train_mask, pred_dates):
        tr = X[train_mask]
        if len(tr) > s2.TRAIN_SUBSAMPLE:
            tr = tr.sample(s2.TRAIN_SUBSAMPLE, random_state=s2.SEED)
        m = new_model(); m.fit(tr[FEATS].values, tr["y"].values)
        pr = X[dates.isin(pred_dates)]
        p = pd.Series(m.predict_proba(pr[FEATS].values)[:, 1], index=pr.index)
        return p.unstack().reindex(index=pred_dates, columns=returns.columns)

    # ---- A. Walk-forward (trailing 3yr, retrain every 126d) ----
    print(f"\n[A] walk-forward retraining (trailing 3yr, step 126d)", flush=True)
    step = 126; trail = pd.DateOffset(years=3); wf_scores = []
    starts = list(range(0, len(oos_idx), step))
    for i in starts:
        seg = oos_idx[i:i+step]
        cutoff = seg[0]
        tr_lo = cutoff - trail; tr_hi = cutoff - pd.Timedelta(days=EMBARGO*2)
        mask = (dates >= tr_lo) & (dates <= tr_hi)
        wf_scores.append(fit_predict(mask, seg))
        print(f"  seg {seg[0].date()}..{seg[-1].date()} train {tr_lo.date()}..{tr_hi.date()} "
              f"n={int(mask.sum()):,} [{time.time()-t0:.0f}s]", flush=True)
    wf = pd.concat(wf_scores).reindex(index=oos_idx, columns=returns.columns)

    # ---- B. Train-era split ----
    print(f"\n[B] train-era split (2010-2015 vs 2016-2021), scored on OOS", flush=True)
    old = fit_predict((dates >= "2010-01-01") & (dates <= "2015-12-31"), oos_idx)
    rec = fit_predict((dates >= "2016-01-01") & (dates <= oos_start - pd.Timedelta(days=EMBARGO*2)), oos_idx)

    # ---- static model (cached) ----
    import range_momentum_stage3 as s3
    is_idx = returns.index[returns.index < oos_start]
    static = s3.build_score(ohlcv, universe, returns, is_idx, oos_idx)

    # ---- compare ----
    rets = returns.loc[oos_idx]; univ = universe.loc[oos_idx]; fwd1 = rets.shift(-1)
    print("\n=== per-year IC ===")
    icrows = {}
    for nm, sc in [("static", static), ("walkfwd", wf), ("era2010-15", old), ("era2016-21", rec)]:
        ic = daily_ic(sc.reindex(oos_idx), fwd1)
        icrows[nm] = ic.groupby(ic.index.year).mean()
    print(pd.DataFrame(icrows).round(4).to_string())

    print("\n=== chosen-design net Sharpe (per year + full) ===")
    srrows = {}
    for nm, sc in [("static", static), ("walkfwd", wf), ("era2010-15", old), ("era2016-21", rec)]:
        pnl = chosen_design_netpnl(sc, univ, returns, oos_idx)
        srrows[nm] = per_year_sr(pnl)
        srrows[nm]["FULL"] = pnl.mean()/pnl.std()*SQRT
    print(pd.DataFrame(srrows).round(3).to_string())

    out = {nm: {str(k): float(v) for k, v in srrows[nm].items()} for nm in srrows}
    (ART / "stage6_regime.json").write_text(json.dumps(out, indent=2))
    print(f"\n[saved] {ART}/stage6_regime.json  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
