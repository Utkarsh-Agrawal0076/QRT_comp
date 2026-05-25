"""Walk-forward GAM with ONLY the RSI x rel_vol tensor (no vol spline).

Same train/test windows as gam_walkforward.py:
  - 3-year trailing train, 6-month test, 6-month step.
  - Subsample train to 800k.
  - 4 lambdas in gridsearch.

Drops the volatility feature entirely. If the tensor interaction has stable
cross-sectional structure across regimes, we'll see:
  - Positive IC every year
  - Positive (or at least non-negative) decile-return correlation every year
  - Monotonic-ish full-OOS decile spread
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
from gam_diagnose_fast import (
    fast_ic, fast_decile, portfolio_bucket, portfolio_continuous, report
)
from gam_free_vol import build_features
from gam_walkforward import build_windows

ART = Path("stores/gam"); ART.mkdir(exist_ok=True, parents=True)
CACHE_TARGET = ART / "target.pkl"
CACHE_NO_VOL = ART / "predictions_no_vol.pkl"

MAX_TRAIN = 800_000
N_SPLINES_TE = (8, 8)
LAM_SPACE = np.logspace(2, 4, 4)

def main():
    t0 = time.time()
    ohlcv = load_data()
    universe = build_universe(ohlcv)
    returns = build_returns(ohlcv)
    universe = universe.reindex(columns=returns.columns).fillna(0).astype(int)

    print("[features] building ...")
    vol_z, rsi_rank, rel_vol, log_ret = build_features(ohlcv, universe)
    tgt = pd.read_pickle(CACHE_TARGET)
    Y = tgt["Y"]

    print("[stack] full panel ...")
    parts = {"rsi": rsi_rank, "rel": rel_vol, "y": Y}
    df = pd.concat({k: v.stack() for k, v in parts.items()}, axis=1).dropna()
    df.index.names = ["date", "ticker"]
    dates_all = df.index.get_level_values("date")
    print(f"[stack] panel rows: {len(df):,}")

    oos_start = returns.index[returns.index.year >= 2021][0]
    oos_end = returns.index[-1]
    windows = build_windows(oos_start, oos_end)
    print(f"[walk-forward] {len(windows)} steps")

    if CACHE_NO_VOL.exists():
        print(f"[cache] loading no-vol predictions from {CACHE_NO_VOL}")
        pred_df = pd.read_pickle(CACHE_NO_VOL)
    else:
        from pygam import LinearGAM, te
        pred_pieces = []
        for i, (ts, te_dt, vs, ve) in enumerate(windows):
            print(f"\n[step {i+1}/{len(windows)}] train [{ts.date()}..{te_dt.date()}], test [{vs.date()}..{ve.date()}]")
            train_mask = (dates_all >= ts) & (dates_all < te_dt)
            test_mask = (dates_all >= vs) & (dates_all < ve)
            train_df = df[train_mask]; test_df = df[test_mask]
            if len(train_df) > MAX_TRAIN:
                rs = np.random.default_rng(42 + i)
                idx = rs.choice(len(train_df), MAX_TRAIN, replace=False)
                train_df = train_df.iloc[idx]
            if len(test_df) == 0:
                continue
            X_train = train_df[["rsi", "rel"]].values
            y_train = train_df["y"].values
            X_test = test_df[["rsi", "rel"]].values

            # ONLY the tensor interaction — no vol spline
            gam = LinearGAM(te(0, 1, n_splines=N_SPLINES_TE))
            t_fit = time.time()
            gam.gridsearch(X_train, y_train, lam=LAM_SPACE, progress=False)
            print(f"  fit {time.time()-t_fit:.1f}s   lam={gam.lam}   pseudo_r2={gam.statistics_['pseudo_r2']['explained_deviance']:.5f}")
            pred = gam.predict(X_test)
            pred_pieces.append(pd.Series(pred, index=test_df.index, name="pred"))

            # Sanity probe of the learned surface — corners
            XX = gam.generate_X_grid(term=0, n=10)
            pdep = gam.partial_dependence(term=0, X=XX)
            # XX has 100 rows (10x10 grid). Show 4 corners and center.
            Z = pdep.reshape(10, 10)
            print(f"  surface  corners:  "
                  f"(low_rsi,low_rel)={Z[0,0]:+.3f}  "
                  f"(low_rsi,high_rel)={Z[0,-1]:+.3f}  "
                  f"(high_rsi,low_rel)={Z[-1,0]:+.3f}  "
                  f"(high_rsi,high_rel)={Z[-1,-1]:+.3f}")

        all_pred = pd.concat(pred_pieces, axis=0)
        pred_df = all_pred.unstack().reindex(columns=universe.columns).where(universe == 1)
        pd.to_pickle(pred_df, CACHE_NO_VOL)
        print(f"\n[saved] predictions {pred_df.shape} -> {CACHE_NO_VOL}")

    # ---- Diagnostics ----
    oos_idx = pred_df.dropna(how='all').index
    rets_oos = returns.reindex(oos_idx)
    univ_oos = universe.reindex(oos_idx)
    pred_oos = pred_df.reindex(oos_idx)

    print(f"\n[OOS] {oos_idx.min().date()}..{oos_idx.max().date()}  ({len(oos_idx)} days)")
    print("\n" + "="*70); print("NO-VOL GAM DIAGNOSTICS (RSI x rel_vol only)"); print("="*70)

    ic_by_lag = fast_ic(pred_oos, rets_oos, lags=(1, 3, 5, 10))
    print("\n[IC decay]")
    for lag, ic in ic_by_lag.items():
        if len(ic) == 0: continue
        ir = ic.mean() / ic.std() * np.sqrt(252)
        t = ic.mean() / (ic.std() / np.sqrt(len(ic)))
        print(f"  T+{lag:>2}:  mean={ic.mean():+.4f}  IR_ann={ir:+.2f}  %>0={(ic>0).mean()*100:.1f}%  t={t:.2f}  n={len(ic)}")

    ic1 = ic_by_lag[1]
    print("\n[IC by year (T+1)]")
    yr = ic1.groupby(ic1.index.year).agg(
        mean_IC=lambda s: s.mean(), std_IC=lambda s: s.std(), n=lambda s: len(s),
    )
    yr["IR_ann"] = yr["mean_IC"] / yr["std_IC"] * np.sqrt(252)
    yr["t_stat"] = yr["mean_IC"] / (yr["std_IC"] / np.sqrt(yr["n"]))
    print(yr.round(4).to_string())

    print("\n[Decile spread, T+1, bps - full OOS]")
    fwd1 = rets_oos.shift(-1)
    dec = fast_decile(pred_oos, fwd1) * 1e4
    print(dec.round(2).to_string())
    ls = dec.iloc[-1] - dec.iloc[0]
    mono = dec.is_monotonic_increasing or dec.is_monotonic_decreasing
    corr_r = pd.Series(range(1,11)).corr(dec.reset_index(drop=True))
    print(f"  L-S (D10-D1): {ls:.2f} bps/d  ~  {ls*252/100:.1f}%/yr gross")
    print(f"  monotonic: {mono}   corr(decile, ret): {corr_r:.3f}")

    print("\n[Decile spread by year — the regime test]")
    yrs = sorted(set(oos_idx.year))
    for y in yrs:
        sub = oos_idx[oos_idx.year == y]
        d = fast_decile(pred_oos.loc[sub], fwd1.loc[sub]) * 1e4
        c = pd.Series(range(1,11)).corr(d.reset_index(drop=True))
        print(f"  {y}: D1={d.iloc[0]:.2f}  D5={d.iloc[4]:.2f}  D10={d.iloc[-1]:.2f}  "
              f"L-S(D10-D1)={d.iloc[-1]-d.iloc[0]:+.2f}  corr(d,r)={c:+.2f}")

    # ---- Portfolios ----
    print("\n\n" + "#"*70); print("# PORTFOLIO CONSTRUCTIONS (no-vol GAM)"); print("#"*70)
    print("\n--- (b) continuous tilt * inv-vol ---")
    w_b = portfolio_continuous(pred_oos, univ_oos, log_ret.reindex(oos_idx))
    res_b = report(w_b, rets_oos, univ_oos, "(b) continuous")

    print("\n--- (c) bucket 10/10 EW ---")
    w_c = portfolio_bucket(pred_oos, univ_oos)
    res_c = report(w_c, rets_oos, univ_oos, "(c) bucket 10/10")

    print("\n--- (c2) bucket 20/20 EW ---")
    w_c2 = portfolio_bucket(pred_oos, univ_oos, long_pct=0.80, short_pct=0.20)
    res_c2 = report(w_c2, rets_oos, univ_oos, "(c2) bucket 20/20")

    print("\n--- (b-) flip continuous ---")
    w_bn = portfolio_continuous(-pred_oos, univ_oos, log_ret.reindex(oos_idx))
    res_bn = report(w_bn, rets_oos, univ_oos, "(b-) flip continuous")

    print("\n--- (c-) flip bucket 10/10 ---")
    w_cn = portfolio_bucket(-pred_oos, univ_oos)
    res_cn = report(w_cn, rets_oos, univ_oos, "(c-) flip bucket 10/10")

    # ---- Orthogonality vs existing MR sleeve ----
    SLEEVE_CACHE = Path("stores/sharpe_blender/sleeves.pkl")
    if SLEEVE_CACHE.exists():
        sleeves = pd.read_pickle(SLEEVE_CACHE)
        pnl_mr = ((sleeves["MR"].loc[oos_idx] * rets_oos.fillna(0)).sum(axis=1))
        print("\n[PnL corr vs existing 3d MR]")
        for lbl, r in [("(b) continuous", res_b), ("(c) bucket 10/10", res_c),
                       ("(c2) bucket 20/20", res_c2),
                       ("(b-) flip cont", res_bn), ("(c-) flip 10/10", res_cn)]:
            print(f"  {lbl:25s}: {r['pnl'].corr(pnl_mr):+.3f}")

    # Headline
    print("\n\n" + "="*70); print("HEADLINE — no-vol GAM"); print("="*70)
    rows = [
        ("(b)  continuous tilt + inv-vol  ", res_b),
        ("(c)  bucket 10/10 EW            ", res_c),
        ("(c2) bucket 20/20 EW            ", res_c2),
        ("(b-) flip continuous             ", res_bn),
        ("(c-) flip bucket 10/10           ", res_cn),
    ]
    print(f"  {'label':38s}  {'net_SR':>7s}  {'max_DD%':>7s}")
    for lbl, r in rows:
        print(f"  {lbl}  {r['net_sr']:7.3f}  {r['max_dd']:7.2f}")

    best = max(rows, key=lambda x: x[1]["net_sr"])
    print(f"\n[best]: {best[0]}  net_SR={best[1]['net_sr']:.3f}  max_DD={best[1]['max_dd']:.2f}%")

    out = {
        "oos": [str(oos_idx.min().date()), str(oos_idx.max().date())],
        "ic_decay": {f"T+{lag}": float(ic.mean()) for lag, ic in ic_by_lag.items()},
        "decile_spread_bps": dec.round(2).to_dict(),
        "monotonic": bool(mono), "decile_corr": float(corr_r),
        "portfolios": {lbl: {k: v for k, v in r.items() if k != "pnl"} for lbl, r in rows},
    }
    (ART / "summary_no_vol.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[saved] {ART}/summary_no_vol.json")
    print(f"[total runtime: {time.time()-t0:.1f}s]")

if __name__ == "__main__":
    main()
