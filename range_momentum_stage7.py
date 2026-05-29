"""Stage 7 — construction-agnostic SIGNAL QUALITY battery across training schemes.

The fixed D7-D10/D1-D4 portfolio was reverse-engineered from the STATIC signal's
decile structure, so imposing it on other signals is unfair. This evaluates each
training scheme's SIGNAL directly (independent of portfolio construction):

  schemes: static (2010-2021) | walk-forward (trailing 3yr) | era 2010-2015 | era 2016-2021
  tests  : mean IC / IR / t-stat / %>0 (T+1), IC decay (lags 1-10),
           demeaned decile spread + monotonicity, IC by year.

Score matrices are cached to stores/range_mom/scores_*.pkl.
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
import range_momentum_stage3 as s3
from sklearn.ensemble import HistGradientBoostingClassifier

ART = Path("stores/range_mom")
SQRT = np.sqrt(252)
FEATS = ["f1_mom", "f2_vol", "f3_rng"]
EMBARGO = s2.LABEL_T + 1
PANEL_CACHE = ART / "panel.pkl"


def new_model():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4,
                                          l2_regularization=1.0, random_state=s2.SEED,
                                          validation_fraction=0.1, early_stopping=True)


def build_scores(ohlcv, universe, returns, oos_idx, oos_start, is_idx):
    """Return dict of 4 score matrices, cached."""
    want = ["static", "walkfwd", "era2010-15", "era2016-21"]
    if all((ART / f"scores_{n}.pkl").exists() for n in want):
        print("[scores] loading cached")
        return {n: pd.read_pickle(ART / f"scores_{n}.pkl") for n in want}

    t0 = time.time()
    fac = s1.build_factors(ohlcv, universe)
    label = s2.make_label(returns, s2.LABEL_T)
    if PANEL_CACHE.exists():
        X = pd.read_pickle(PANEL_CACHE)
    else:
        print(f"[panel] stacking ... [{time.time()-t0:.0f}s]", flush=True)
        X = pd.DataFrame({n: fac[n].stack(dropna=False) for n in FEATS})
        X["y"] = label.stack(dropna=False); X = X.dropna()
        pd.to_pickle(X, PANEL_CACHE)
    dates = X.index.get_level_values(0)
    print(f"[panel] rows={len(X):,} [{time.time()-t0:.0f}s]", flush=True)

    def fit_predict(mask, pred_dates):
        tr = X[mask]
        if len(tr) > s2.TRAIN_SUBSAMPLE:
            tr = tr.sample(s2.TRAIN_SUBSAMPLE, random_state=s2.SEED)
        m = new_model(); m.fit(tr[FEATS].values, tr["y"].values)
        pr = X[dates.isin(pred_dates)]
        p = pd.Series(m.predict_proba(pr[FEATS].values)[:, 1], index=pr.index)
        return p.unstack().reindex(index=pred_dates, columns=returns.columns)

    scores = {}
    scores["static"] = s3.build_score(ohlcv, universe, returns, is_idx, oos_idx)
    # walk-forward
    print("[walkfwd] retraining trailing-3yr step-126d", flush=True)
    step = 126; segs = []
    for i in range(0, len(oos_idx), step):
        seg = oos_idx[i:i+step]; cutoff = seg[0]
        mask = (dates >= cutoff - pd.DateOffset(years=3)) & (dates <= cutoff - pd.Timedelta(days=EMBARGO*2))
        segs.append(fit_predict(mask, seg))
    scores["walkfwd"] = pd.concat(segs).reindex(index=oos_idx, columns=returns.columns)
    scores["era2010-15"] = fit_predict((dates >= "2010-01-01") & (dates <= "2015-12-31"), oos_idx)
    scores["era2016-21"] = fit_predict((dates >= "2016-01-01") & (dates <= oos_start - pd.Timedelta(days=EMBARGO*2)), oos_idx)
    for n, sc in scores.items():
        pd.to_pickle(sc, ART / f"scores_{n}.pkl")
    print(f"[scores] built+cached [{time.time()-t0:.0f}s]", flush=True)
    return scores


def daily_ic(sig, fwd, mn=200):
    common = sig.dropna(how="all").index.intersection(fwd.dropna(how="all").index)
    out, dts = [], []
    for dt in common:
        s = sig.loc[dt].dropna(); f = fwd.loc[dt].reindex(s.index).dropna()
        if len(f) < mn: continue
        cs = s.index.intersection(f.index); out.append(s[cs].rank().corr(f[cs].rank())); dts.append(dt)
    return pd.Series(out, index=pd.DatetimeIndex(dts))


def main():
    ohlcv = s1.load_data(); universe = s1.build_universe(ohlcv); returns = s1.build_returns(ohlcv)
    universe = universe.reindex(columns=returns.columns).fillna(0).astype(int)
    oos_start = returns.index.max() - pd.DateOffset(years=s2.OOS_YEARS)
    is_idx = returns.index[returns.index < oos_start]
    oos_idx = returns.index[returns.index >= oos_start]

    scores = build_scores(ohlcv, universe, returns, oos_idx, oos_start, is_idx)
    rets = returns.loc[oos_idx]; univ = universe.loc[oos_idx]
    fwd1 = rets.shift(-1).replace([np.inf, -np.inf], np.nan)

    # ---- headline IC stats ----
    print("\n=== SIGNAL QUALITY (OOS T+1) ===")
    hdr = []
    ics = {}
    for n, sc in scores.items():
        ic = daily_ic(sc.reindex(oos_idx), fwd1); ics[n] = ic
        hdr.append({"scheme": n, "mean_IC": ic.mean(), "IR_ann": ic.mean()/ic.std()*SQRT,
                    "t_stat": ic.mean()/(ic.std()/np.sqrt(len(ic))), "pct_pos": (ic > 0).mean(),
                    "n_days": len(ic)})
    print(pd.DataFrame(hdr).set_index("scheme").round(4).to_string())

    # ---- IC decay ----
    print("\n=== IC DECAY (mean IC by forward lag) ===")
    dec = {}
    for n, sc in scores.items():
        s_oos = sc.reindex(oos_idx)
        dec[n] = {L: daily_ic(s_oos, rets.shift(-L)).mean() for L in (1, 2, 3, 5, 10)}
    print(pd.DataFrame(dec).round(4).to_string())

    # ---- demeaned decile spread + monotonicity ----
    print("\n=== DEMEANED DECILE next-day ret (bps) — where the L/S alpha lives ===")
    rows = {}
    for n, sc in scores.items():
        rk = sc.where(univ == 1, np.nan).rank(axis=1, pct=True)
        tr = (univ == 1) & rk.notna()
        bench = fwd1.where(tr).mean(axis=1); dm = fwd1.sub(bench, axis=0)
        vals = []
        for d in range(1, 11):
            lo, hi = (d-1)/10, d/10
            m = ((rk >= lo) & (rk < hi)) if d < 10 else (rk >= lo)
            vals.append(dm.where(m & tr).stack().mean()*1e4)
        rows[n] = vals
    dtab = pd.DataFrame(rows, index=[f"D{i}" for i in range(1, 11)])
    print(dtab.round(2).to_string())
    print("\n  D10-D1 spread (bps/d) and decile monotonicity:")
    for n in scores:
        v = dtab[n].values
        mono = pd.Series(range(1, 11)).corr(pd.Series(v))
        print(f"    {n:11s}: spread={v[-1]-v[0]:+.2f}  best_long=D{int(np.argmax(v))+1}({v.max():+.2f})  "
              f"best_short=D{int(np.argmin(v))+1}({v.min():+.2f})  monotonicity={mono:+.2f}")

    # ---- IC by year ----
    print("\n=== IC BY YEAR ===")
    iy = {n: ics[n].groupby(ics[n].index.year).mean() for n in scores}
    print(pd.DataFrame(iy).round(4).to_string())

    out = {"headline": {r["scheme"]: {k: float(v) for k, v in r.items() if k != "scheme"} for r in hdr},
           "decile_demean_bps": {n: [float(x) for x in dtab[n].values] for n in scores}}
    (ART / "stage7_signal_quality.json").write_text(json.dumps(out, indent=2))
    print(f"\n[saved] {ART}/stage7_signal_quality.json")


if __name__ == "__main__":
    main()
