"""Stage 2 — Spike-classifier L/S sleeve on f1/f2/f3.

Faithful to the original proposal: train a gradient-boosted classifier on the
three factors to predict "sustained upward drift", rank the predicted probability
cross-sectionally, trade D10 long / D1 short, T+1, water-fill GMV=1, and let the
net-Sharpe backtest be the judge (accepting the documented GAM-repeat risk).

Label: y = 1 if forward T-day cumulative return > cross-sectional median (relative
       outperformance) — balanced, and aligned with an L/S ranking objective.
Model: HistGradientBoostingClassifier (XGBoost-equivalent; xgboost not installed).
"""
from __future__ import annotations
import sys, warnings, json
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.append(str(Path("phase2_qrt_challenge/scripts").resolve()))
import utils  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier

import range_momentum_pipeline as s1  # reuse data/factor/diagnostic helpers

ART_DIR = Path("stores/range_mom"); ART_DIR.mkdir(exist_ok=True, parents=True)
OOS_YEARS = 5
LABEL_T = 5            # forward horizon for the classification label
TRAIN_SUBSAMPLE = 700_000
SEED = 0


def make_label(returns, T):
    """1 if forward T-day cumulative return beats the cross-sectional median."""
    fwd = (1 + returns).rolling(T).apply(np.prod, raw=True).shift(-T) - 1
    med = fwd.median(axis=1)
    return (fwd.gt(med, axis=0)).astype(float).where(fwd.notna())


def stack_panel(fac, label, idx):
    """Long-format (date,stock) -> X[f1,f2,f3], y. Rows with any NaN dropped."""
    names = ["f1_mom", "f2_vol", "f3_rng"]
    cols = {n: fac[n].reindex(idx).stack(dropna=False) for n in names}
    X = pd.DataFrame(cols)
    y = label.reindex(idx).stack(dropna=False)
    df = X.join(y.rename("y")).dropna()
    return df[names], df["y"]


def score_to_weights(score, universe, long_pct=0.90, short_pct=0.10, gmv=1.0):
    masked = score.where(universe == 1, np.nan)
    rk = masked.rank(axis=1, pct=True)
    long_m = (rk >= long_pct).astype(float)
    short_m = (rk <= short_pct).astype(float)
    nL = long_m.sum(axis=1).replace(0, np.nan)
    nS = short_m.sum(axis=1).replace(0, np.nan)
    wL = long_m.div(nL, axis=0) * (gmv / 2)
    wS = short_m.div(nS, axis=0) * (-gmv / 2)
    return wL.fillna(0) + wS.fillna(0)


def main():
    ohlcv = s1.load_data()
    universe = s1.build_universe(ohlcv)
    returns = s1.build_returns(ohlcv)
    universe = universe.reindex(columns=returns.columns).fillna(0).astype(int)

    oos_start = returns.index.max() - pd.DateOffset(years=OOS_YEARS)
    is_idx = returns.index[returns.index < oos_start]
    oos_idx = returns.index[returns.index >= oos_start]
    print(f"[split] IS {is_idx.min().date()}..{is_idx.max().date()} ({len(is_idx)})  "
          f"OOS {oos_idx.min().date()}..{oos_idx.max().date()} ({len(oos_idx)})")

    print("[factors] building f1/f2/f3 (winsorized 2/98)")
    fac = s1.build_factors(ohlcv, universe)

    print(f"[label] T={LABEL_T}d relative-outperformance binary label")
    label = make_label(returns, LABEL_T)

    print("[train] stacking IS panel")
    Xtr, ytr = stack_panel(fac, label, is_idx)
    print(f"  IS rows={len(Xtr):,}  pos_rate={ytr.mean():.3f}")
    if len(Xtr) > TRAIN_SUBSAMPLE:
        samp = Xtr.sample(TRAIN_SUBSAMPLE, random_state=SEED).index
        Xtr, ytr = Xtr.loc[samp], ytr.loc[samp]
        print(f"  subsampled to {len(Xtr):,}")

    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=4,
        l2_regularization=1.0, random_state=SEED, validation_fraction=0.1,
        early_stopping=True)
    clf.fit(Xtr.values, ytr.values)
    print(f"  trained ({clf.n_iter_} iters)")
    print(f"  feature importances (perm-free split-gain proxy unavailable; "
          f"using mean |contribution| not computed)")

    # ---- score the full OOS panel ----
    print("[score] predicting OOS probabilities")
    Xall, _ = stack_panel(fac, label.reindex(oos_idx).fillna(0.0), oos_idx)
    proba = pd.Series(clf.predict_proba(Xall.values)[:, 1], index=Xall.index)
    score = proba.unstack().reindex(index=oos_idx, columns=returns.columns)

    # ---- diagnostics ----
    fwd1 = returns.loc[oos_idx].shift(-1)
    ic = s1.daily_ic(score, fwd1)
    dec = s1.decile_spread(score, fwd1)
    ls_bps = (dec.iloc[-1] - dec.iloc[0]) * 1e4
    dec_corr = pd.Series(range(1, 11)).corr(pd.Series(dec.values))
    print(f"\n[OOS classifier signal]")
    print(f"  mean_IC={ic.mean():.4f}  IR={ic.mean()/ic.std()*np.sqrt(252):.2f}  "
          f"t={ic.mean()/(ic.std()/np.sqrt(len(ic))):.2f}  %>0={(ic>0).mean()*100:.1f}%")
    print(f"  decile spread D10-D1={ls_bps:.2f} bps/d  decile_corr={dec_corr:.2f}")
    print("  decile fwd ret (bps): " + " ".join(f"{v*1e4:+.1f}" for v in dec.values))
    print("\n[OOS IC by year]")
    print(s1.ic_by_year(ic).round(4).to_string())

    # ---- backtest ----
    print("\n[backtest] D10 long / D1 short, T+1, water-fill GMV=1")
    w = score_to_weights(score, universe.loc[oos_idx])
    w_final = _waterfill(w.shift(1), universe.loc[oos_idx])
    oos_rets = returns.loc[oos_idx]
    w_final = w_final.reindex(columns=oos_rets.columns, fill_value=0.0)
    univ_b = universe.loc[oos_idx].reindex(columns=oos_rets.columns, fill_value=0)
    sr, pnl = utils.backtest_portfolio(w_final, oos_rets, univ_b, plot_=False, print_=True)

    yr = pnl.groupby(pnl.index.year).apply(
        lambda s: (s.mean()/s.std()*np.sqrt(252)) if s.std() > 0 else 0.0)
    print("\n[OOS net PnL Sharpe by year]")
    print(yr.round(3).to_string())
    eq = (1 + pnl).cumprod(); dd = (eq/eq.cummax() - 1).min()
    print(f"\n[OOS] net SR={sr}  max DD={dd*100:.1f}%")

    # ---- PnL correlation vs existing sleeves ----
    try:
        sleeves = pd.read_pickle("stores/sharpe_blender/sleeves.pkl")
        print("\n[PnL corr vs existing sleeves (OOS)]")
        for nm, sw in sleeves.items():
            sw_o = sw.reindex(index=oos_idx, columns=oos_rets.columns).fillna(0)
            spnl = (sw_o.shift(1).fillna(0) * oos_rets).sum(axis=1)
            c = pnl.reindex(spnl.index).corr(spnl)
            print(f"  {nm:6s}: {c:+.3f}")
    except Exception as e:
        print(f"  (sleeve corr skipped: {e})")

    out = {"label_T": LABEL_T, "oos_mean_IC": float(ic.mean()),
           "oos_decile_spread_bps": float(ls_bps), "oos_decile_corr": float(dec_corr),
           "oos_net_SR": float(sr), "oos_max_dd_pct": float(dd*100)}
    (ART_DIR / "stage2_classifier.json").write_text(json.dumps(out, indent=2))
    print(f"\n[saved] {ART_DIR}/stage2_classifier.json")


def _waterfill(shifted_portfolio, universe_df, max_weight=0.098):
    portfolio = shifted_portfolio * universe_df
    abs_sum = portfolio.abs().sum(axis=1)
    for date, gmv in abs_sum.items():
        if gmv <= 1e-8:
            continue
        row = (portfolio.loc[date] / gmv).copy()
        for _ in range(20):
            if row.abs().max() <= max_weight + 1e-6:
                break
            cap = row.abs() > max_weight
            row[cap] = np.sign(row[cap]) * max_weight
            rem = 1.0 - row[cap].abs().sum()
            us = row[~cap].abs().sum()
            if us > 1e-8:
                row[~cap] *= rem / us
            else:
                break
        if abs(row.abs().sum() - 1.0) > 0.01 or row.abs().max() > 0.1001:
            portfolio.loc[date] = 0.0
        else:
            portfolio.loc[date] = row
    return portfolio


if __name__ == "__main__":
    main()
