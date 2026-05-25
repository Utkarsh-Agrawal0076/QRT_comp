"""Two follow-up experiments on the squeeze sleeve.

  Exp 1: Rogers-Satchell variance estimator (intraday, drift-independent),
         everything else identical to the original.
  Exp 2: Sign-flipped alpha at T+3 forward horizon (the IC-decay peak),
         with the RS estimator.

Both use the same IS/OOS split (last 5 years = OOS) and report IC, decile
spread, IC by year, and a portfolio backtest via utils.backtest_portfolio.
"""
from __future__ import annotations
import sys, warnings, itertools, time, json
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.append(str(Path("phase2_qrt_challenge/scripts").resolve()))
import utils  # noqa

from squeeze_breakout_pipeline import (
    load_data, build_universe, build_returns,
    clv, signed_volume, squeeze_z,
    daily_ic, ic_decay, decile_spread, ic_by_year,
    alpha_to_weights, enforce_post_shift_strict_gmv,
    MIN_STOCKS, OOS_YEARS,
)

ART = Path("stores/squeeze"); ART.mkdir(exist_ok=True, parents=True)

# ---- Rogers-Satchell variance ----
def rs_var(ohlcv, k=20):
    O, H, L, C = ohlcv["Open"], ohlcv["High"], ohlcv["Low"], ohlcv["Close"]
    rs_daily = (np.log(H / C) * np.log(H / O)
                + np.log(L / C) * np.log(L / O))
    # rolling mean (daily RS is already a variance estimator per bar)
    return rs_daily.rolling(k, min_periods=k // 2).mean()

def build_alpha_rs(ohlcv, universe, k=20, m=3, tau=1.0, agree=True):
    v = rs_var(ohlcv, k=k)
    z = squeeze_z(v, k=k)              # reuse log-var z-score from main module
    c = clv(ohlcv, m=m).shift(1)
    sv = signed_volume(ohlcv).shift(1)
    w = np.exp(-np.clip(z + tau, 0, None))
    if agree:
        ok = (np.sign(c) == np.sign(sv)) & (sv.abs() > 0) & (c.abs() > 0)
        anchor = c.where(ok, 0.0)
    else:
        anchor = c
    return (w * anchor).where(universe == 1, np.nan)

def run_diagnostics_and_backtest(alpha, returns, universe, oos_idx, label, horizon=1, flip=False):
    """Slice alpha to OOS and report everything."""
    a = alpha.reindex(oos_idx)
    if flip:
        a = -a
        label += " [SIGN-FLIPPED]"
    rets = returns.loc[oos_idx]
    univ = universe.loc[oos_idx]
    fwd = rets.shift(-horizon)

    print(f"\n{'='*70}\n{label}  (horizon=T+{horizon}, flip={flip})\n{'='*70}")

    ic = daily_ic(a, fwd)
    print(f"\n[OOS IC]  N={len(ic)}  mean={ic.mean():.4f}  median={ic.median():.4f}  "
          f"IR_ann={ic.mean()/ic.std()*np.sqrt(252):.2f}  "
          f"%>0={(ic>0).mean()*100:.1f}%  t={ic.mean()/(ic.std()/np.sqrt(len(ic))):.2f}")

    print("\n[OOS IC by year]")
    print(ic_by_year(ic).round(4).to_string())

    print("\n[OOS IC decay]")
    print(ic_decay(a, rets).round(4).to_string(index=False))

    print(f"\n[OOS decile spread (avg fwd T+{horizon} ret, bps)]")
    dec = decile_spread(a, fwd) * 1e4
    print(dec.round(2).to_string())
    ls = dec.iloc[-1] - dec.iloc[0]
    print(f"  L-S spread (D10-D1): {ls:.2f} bps/day  ~  {ls*252/100:.1f}%/yr gross")
    monotonic = dec.is_monotonic_increasing or dec.is_monotonic_decreasing
    print(f"  monotonic: {monotonic}   "
          f"corr(decile, fwd_ret): {pd.Series(range(1,11)).corr(dec.reset_index(drop=True)):.3f}")

    # Hit rates
    rk = a.rank(axis=1, pct=True)
    csmean = fwd.mean(axis=1)
    hits = {}
    for d in range(1, 11):
        lo, hi = (d-1)/10, d/10
        mask = (rk >= lo) & (rk < hi) if d < 10 else (rk >= lo)
        hits[d] = (fwd.where(mask).mean(axis=1) > csmean).mean()
    print("\n[Hit rate per decile vs cross-sectional mean]")
    print(pd.Series(hits, name="hit").round(3).to_string())

    # Backtest
    w_raw = alpha_to_weights(a, univ)
    w = enforce_post_shift_strict_gmv(w_raw.shift(1), univ)
    w = w.reindex(columns=rets.columns, fill_value=0.0)
    univ_b = univ.reindex(columns=rets.columns, fill_value=0)
    print("\n[OOS BACKTEST]")
    sr, pnl = utils.backtest_portfolio(w, rets, univ_b, plot_=False, print_=True)
    yr = pnl.groupby(pnl.index.year).agg(
        ann_ret=lambda s: s.mean()*252,
        ann_vol=lambda s: s.std()*np.sqrt(252),
        sharpe=lambda s: (s.mean()/s.std()*np.sqrt(252)) if s.std()>0 else 0,
        n="count",
    )
    print("\n[Per-year PnL]")
    print(yr.round(3).to_string())
    eq = (1+pnl).cumprod(); dd = eq/eq.cummax() - 1
    print(f"\n[summary] gross_SR={sr}  max_DD={dd.min()*100:.1f}%  hit_days={(pnl>0).mean()*100:.1f}%")
    return {"label": label, "horizon": horizon, "flip": flip,
            "mean_IC": float(ic.mean()), "IR_ann": float(ic.mean()/ic.std()*np.sqrt(252)),
            "LS_bps": float(ls), "monotonic": bool(monotonic),
            "net_sharpe": float(sr), "max_dd_pct": float(dd.min()*100)}

# ---- IS grid for a given alpha-builder ----
def is_grid_search(alpha_builder, ohlcv, universe, returns, is_idx, horizon, flip, label):
    grid = list(itertools.product([15, 20, 30], [3, 5], [0.5, 1.0, 1.5], [True, False]))
    is_rets = returns.loc[is_idx]
    fwd_is = is_rets.shift(-horizon)
    rows = []
    for (k, m, tau, agr) in grid:
        a = alpha_builder(ohlcv, universe, k=k, m=m, tau=tau, agree=agr)
        a_is = a.reindex(is_idx)
        if flip: a_is = -a_is
        ic = daily_ic(a_is, fwd_is)
        if len(ic) < 100: continue
        rows.append({"k": k, "m": m, "tau": tau, "agree": agr, "n": len(ic),
                     "mean_IC": ic.mean(),
                     "IR_ann": ic.mean()/ic.std()*np.sqrt(252),
                     "t_stat": ic.mean()/(ic.std()/np.sqrt(len(ic)))})
    g = pd.DataFrame(rows).sort_values("mean_IC", ascending=False)
    print(f"\n[IS grid — {label}] top 8 by mean IC (horizon=T+{horizon}, flip={flip})")
    print(g.head(8).round(4).to_string(index=False))
    g.to_csv(ART / f"is_grid_{label.replace(' ','_')}.csv", index=False)
    best = g.iloc[0]
    return int(best.k), int(best.m), float(best.tau), bool(best.agree), float(best.mean_IC)

def main():
    ohlcv = load_data()
    universe = build_universe(ohlcv)
    returns = build_returns(ohlcv)
    universe = universe.reindex(columns=returns.columns).fillna(0).astype(int)
    oos_start = returns.index.max() - pd.DateOffset(years=OOS_YEARS)
    is_idx = returns.index[returns.index < oos_start]
    oos_idx = returns.index[returns.index >= oos_start]
    print(f"[split] IS {is_idx.min().date()}..{is_idx.max().date()}  ({len(is_idx)})")
    print(f"[split] OOS {oos_idx.min().date()}..{oos_idx.max().date()}  ({len(oos_idx)})")

    results = []

    # ============ EXPERIMENT 1: RS estimator, original sign, T+1 ============
    print("\n\n#### EXPERIMENT 1: Rogers-Satchell estimator, T+1, original sign ####")
    K, M, TAU, AGR, is_ic = is_grid_search(
        build_alpha_rs, ohlcv, universe, returns, is_idx,
        horizon=1, flip=False, label="exp1_RS_T1")
    print(f"[pick] k={K} m={M} tau={TAU} agree={AGR}  IS_IC={is_ic:.4f}")
    alpha = build_alpha_rs(ohlcv, universe, k=K, m=M, tau=TAU, agree=AGR)
    r1 = run_diagnostics_and_backtest(alpha, returns, universe, oos_idx,
                                       label="EXP1  RS  T+1  original-sign",
                                       horizon=1, flip=False)
    results.append({**r1, "exp": "1_RS_T1", "is_IC": is_ic,
                    "config": {"k": K, "m": M, "tau": TAU, "agree": AGR}})

    # ============ EXPERIMENT 2: RS estimator, sign-flip, T+3 ============
    print("\n\n#### EXPERIMENT 2: Rogers-Satchell estimator, T+3, SIGN-FLIPPED ####")
    K, M, TAU, AGR, is_ic = is_grid_search(
        build_alpha_rs, ohlcv, universe, returns, is_idx,
        horizon=3, flip=True, label="exp2_RS_T3_flip")
    print(f"[pick] k={K} m={M} tau={TAU} agree={AGR}  IS_IC={is_ic:.4f}")
    alpha = build_alpha_rs(ohlcv, universe, k=K, m=M, tau=TAU, agree=AGR)
    r2 = run_diagnostics_and_backtest(alpha, returns, universe, oos_idx,
                                       label="EXP2  RS  T+3  SIGN-FLIPPED",
                                       horizon=3, flip=True)
    results.append({**r2, "exp": "2_RS_T3_flip", "is_IC": is_ic,
                    "config": {"k": K, "m": M, "tau": TAU, "agree": AGR}})

    print("\n\n" + "="*70)
    print("SUMMARY OF EXPERIMENTS")
    print("="*70)
    summary = pd.DataFrame(results)[["exp","is_IC","mean_IC","IR_ann","LS_bps","monotonic","net_sharpe","max_dd_pct"]]
    print(summary.round(4).to_string(index=False))
    (ART / "experiments_summary.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[saved] {ART}/experiments_summary.json")

if __name__ == "__main__":
    main()
