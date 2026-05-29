"""Stage 3 — micro-decomposition of the spike-classifier sleeve.

Question: the source strategy was LONG-ONLY and reportedly made ~25%/yr. Our
Stage-2 dollar-neutral L/S sleeve went net −0.53. Is the short book the drag?

Decomposes the OOS classifier score into:
  - Long book only (D10)           : the source's actual form
  - Short book only (D1)
  - Per-decile raw vs demeaned ret, and per-decile turnover (what churns)
  - Market-drift attribution (what a dollar-neutral short fights)
  - Long-only GMV=1, raw vs beta-hedged (competition auto-hedge proxy)

Reuses the Stage-2 trained model; caches the OOS score matrix.
"""
from __future__ import annotations
import sys, warnings, json
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.append(str(Path("phase2_qrt_challenge/scripts").resolve()))
import range_momentum_pipeline as s1
import range_momentum_stage2 as s2
from sklearn.ensemble import HistGradientBoostingClassifier

ART = Path("stores/range_mom"); ART.mkdir(exist_ok=True, parents=True)
SCORE_CACHE = ART / "oos_score.pkl"
SQRT252 = np.sqrt(252)
EXEC_BPS = 2e-4
FIN_DAILY = 0.005 / 252


def build_score(ohlcv, universe, returns, is_idx, oos_idx):
    if SCORE_CACHE.exists():
        print(f"[score] loading cached {SCORE_CACHE}")
        return pd.read_pickle(SCORE_CACHE)
    print("[score] training Stage-2 model + scoring OOS")
    fac = s1.build_factors(ohlcv, universe)
    label = s2.make_label(returns, s2.LABEL_T)
    Xtr, ytr = s2.stack_panel(fac, label, is_idx)
    samp = Xtr.sample(s2.TRAIN_SUBSAMPLE, random_state=s2.SEED).index
    Xtr, ytr = Xtr.loc[samp], ytr.loc[samp]
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4,
                                         l2_regularization=1.0, random_state=s2.SEED,
                                         validation_fraction=0.1, early_stopping=True)
    clf.fit(Xtr.values, ytr.values)
    Xall, _ = s2.stack_panel(fac, label.reindex(oos_idx).fillna(0.0), oos_idx)
    proba = pd.Series(clf.predict_proba(Xall.values)[:, 1], index=Xall.index)
    score = proba.unstack().reindex(index=oos_idx, columns=returns.columns)
    pd.to_pickle(score, SCORE_CACHE)
    return score


def pnl_stats(w, rets, shift=True):
    """Cost model matches utils.backtest_portfolio (2bps exec + 0.5%/yr financing)."""
    wt = (w.shift(1) if shift else w).fillna(0)
    gross = (wt * rets).sum(axis=1)
    traded = wt.diff().abs().sum(axis=1).fillna(0)
    bv = wt.abs().sum(axis=1).replace(0, np.nan)
    net = gross - traded * EXEC_BPS - bv.fillna(0) * FIN_DAILY
    eq = (1 + net).cumprod(); dd = (eq / eq.cummax() - 1).min()
    turn = (traded.mean() / bv.mean()) * 100
    return {
        "ann_gross_%": gross.mean() * 252 * 100,
        "ann_net_%": net.mean() * 252 * 100,
        "gross_SR": gross.mean() / gross.std() * SQRT252 if gross.std() else 0,
        "net_SR": net.mean() / net.std() * SQRT252 if net.std() else 0,
        "turnover_%": turn,
        "cost_drag_%/yr": (gross.mean() - net.mean()) * 252 * 100,
        "max_dd_%": dd * 100,
    }, net, gross


def decile_weights(score, universe, d, gmv_abs, sign):
    """Equal-weight bucket d (1..10) to `sign*gmv_abs` total notional."""
    rk = score.where(universe == 1, np.nan).rank(axis=1, pct=True)
    lo, hi = (d - 1) / 10, d / 10
    m = ((rk >= lo) & (rk < hi)) if d < 10 else (rk >= lo)
    m = m.astype(float)
    n = m.sum(axis=1).replace(0, np.nan)
    return m.div(n, axis=0).fillna(0) * (sign * gmv_abs)


def fmt(d):
    return "  ".join(f"{k}={v:.2f}" for k, v in d.items())


def main():
    ohlcv = s1.load_data()
    universe = s1.build_universe(ohlcv)
    returns = s1.build_returns(ohlcv)
    universe = universe.reindex(columns=returns.columns).fillna(0).astype(int)
    oos_start = returns.index.max() - pd.DateOffset(years=s2.OOS_YEARS)
    is_idx = returns.index[returns.index < oos_start]
    oos_idx = returns.index[returns.index >= oos_start]

    score = build_score(ohlcv, universe, returns, is_idx, oos_idx)
    rets = returns.loc[oos_idx]
    univ = universe.loc[oos_idx]

    # ---- market drift the short book fights ----
    mkt = rets.where(univ == 1).mean(axis=1)
    print(f"\n[market] equal-weight universe drift: {mkt.mean()*252*100:+.1f}%/yr  "
          f"SR={mkt.mean()/mkt.std()*SQRT252:.2f}")

    # ---- per-decile raw vs demeaned fwd ret + turnover ----
    print("\n[per-decile] raw & demeaned next-day ret (bps), and bucket turnover %")
    fwd1 = rets.shift(-1)
    csmean = fwd1.mean(axis=1)
    rk = score.where(univ == 1, np.nan).rank(axis=1, pct=True)
    rows = []
    for d in range(1, 11):
        lo, hi = (d - 1) / 10, d / 10
        m = ((rk >= lo) & (rk < hi)) if d < 10 else (rk >= lo)
        raw = fwd1.where(m).stack().mean() * 1e4
        dem = fwd1.sub(csmean, axis=0).where(m).stack().mean() * 1e4
        w = decile_weights(score, univ, d, 1.0, 1.0)
        traded = w.shift(1).diff().abs().sum(axis=1).fillna(0)
        bv = w.abs().sum(axis=1).replace(0, np.nan)
        turn = (traded.mean() / bv.mean()) * 100
        rows.append({"decile": d, "raw_bps": raw, "demean_bps": dem, "turnover_%": turn})
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    # ---- long book only (D10) at 0.5 GMV (its share of the L/S sleeve) ----
    wL = decile_weights(score, univ, 10, 0.5, +1.0)
    wS = decile_weights(score, univ, 1, 0.5, -1.0)
    sL, netL, _ = pnl_stats(wL, rets)
    sS, netS, _ = pnl_stats(wS, rets)
    print("\n[L/S sleeve halves @ 0.5 GMV each]")
    print(f"  LONG  (D10): {fmt(sL)}")
    print(f"  SHORT (D1) : {fmt(sS)}")

    # ---- the source's actual form: long-only D10 at full GMV=1 ----
    wLO = decile_weights(score, univ, 10, 1.0, +1.0)
    sLO, netLO, grossLO = pnl_stats(wLO, rets)
    print("\n[long-only D10 @ GMV=1  (source's form)]")
    print(f"  {fmt(sLO)}")
    beta = grossLO.cov(mkt) / mkt.var()
    print(f"  market beta={beta:.2f}  -> auto-hedge shorts {beta:.2f} of SPX")

    # beta-hedged long-only (competition auto-hedge proxy: subtract beta*market)
    hedged = grossLO - beta * mkt
    traded = wLO.shift(1).diff().abs().sum(axis=1).fillna(0)
    bv = wLO.abs().sum(axis=1)
    hedged_net = hedged - traded * EXEC_BPS - bv * FIN_DAILY
    print(f"  beta-hedged: ann_gross={hedged.mean()*252*100:+.1f}%  "
          f"ann_net={hedged_net.mean()*252*100:+.1f}%  "
          f"net_SR={hedged_net.mean()/hedged_net.std()*SQRT252:.2f}")

    # ---- top-bucket concentration sweep (source may use a tighter top) ----
    print("\n[long-only concentration sweep, GMV=1, net of costs]")
    for pct in [0.99, 0.98, 0.95, 0.90, 0.80]:
        m = (rk >= pct).astype(float)
        n = m.sum(axis=1).replace(0, np.nan)
        w = m.div(n, axis=0).fillna(0)
        s, net, gross = pnl_stats(w, rets)
        print(f"  top {(1-pct)*100:4.0f}%: ann_net={s['ann_net_%']:+6.1f}%  "
              f"net_SR={s['net_SR']:+.2f}  turn={s['turnover_%']:.0f}%  "
              f"gross_SR={s['gross_SR']:+.2f}  dd={s['max_dd_%']:.1f}%")

    out = {"market_drift_ann_%": float(mkt.mean()*252*100),
           "long_only_D10": sLO, "long_half": sL, "short_half": sS,
           "long_only_beta": float(beta)}
    (ART / "stage3_decomp.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[saved] {ART}/stage3_decomp.json")


if __name__ == "__main__":
    main()
