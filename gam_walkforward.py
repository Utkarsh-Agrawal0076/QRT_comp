"""Walk-forward GAM: re-train every 6 months on trailing 3-year window.

Same features (vol_z, rsi_rank, rel_vol), same target (5d beta-residual fwd log
return z-scored). Free vol spline (no monotonic constraint).

For each step:
  - Train window: trailing TRAIN_YEARS ending at step start
  - Test window:  TEST_MONTHS following step start
  - Subsample train to MAX_TRAIN if larger
  - Fit GAM, predict on test window, store predictions

Then concatenate test-window predictions across all steps → one OOS prediction
matrix → same diagnostic + portfolio harness.
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

ART = Path("stores/gam"); ART.mkdir(exist_ok=True, parents=True)
CACHE_TARGET = ART / "target.pkl"
CACHE_WF = ART / "predictions_walkforward.pkl"

TRAIN_YEARS = 3
TEST_MONTHS = 6
STEP_MONTHS = 6
OOS_START_YEAR = 2021
MAX_TRAIN = 800_000
N_SPLINES_VOL = 8
N_SPLINES_TE = (6, 6)
LAM_SPACE = np.logspace(2, 4, 4)   # 4 lambdas (was 10 in original)

def build_windows(oos_start, oos_end):
    """Generate (train_start, train_end, test_start, test_end) tuples."""
    windows = []
    cur = pd.Timestamp(oos_start)
    end = pd.Timestamp(oos_end)
    while cur < end:
        train_end = cur
        train_start = train_end - pd.DateOffset(years=TRAIN_YEARS)
        test_start = cur
        test_end = min(cur + pd.DateOffset(months=TEST_MONTHS), end)
        windows.append((train_start, train_end, test_start, test_end))
        cur = test_end
    return windows

def main():
    t0 = time.time()
    ohlcv = load_data()
    universe = build_universe(ohlcv)
    returns = build_returns(ohlcv)
    universe = universe.reindex(columns=returns.columns).fillna(0).astype(int)

    print("[features] building ...")
    vol_z, rsi_rank, rel_vol, log_ret = build_features(ohlcv, universe)
    print("[target] from cache ...")
    if not CACHE_TARGET.exists():
        print("[err] target.pkl missing — run gam_diagnose.py first")
        return
    tgt = pd.read_pickle(CACHE_TARGET)
    Y = tgt["Y"]

    print("[stack] full panel ...")
    parts = {"vol": vol_z, "rsi": rsi_rank, "rel": rel_vol, "y": Y}
    df = pd.concat({k: v.stack() for k, v in parts.items()}, axis=1).dropna()
    df.index.names = ["date", "ticker"]
    dates_all = df.index.get_level_values("date")
    print(f"[stack] full panel rows: {len(df):,}")

    oos_start = returns.index[returns.index.year >= OOS_START_YEAR][0]
    oos_end = returns.index[-1]
    windows = build_windows(oos_start, oos_end)
    print(f"\n[walk-forward] {len(windows)} steps")
    for i, (ts, te_, vs, ve) in enumerate(windows):
        print(f"  step {i+1}: train [{ts.date()}..{te_.date()}]  test [{vs.date()}..{ve.date()}]")

    if CACHE_WF.exists():
        print(f"\n[cache] loading walk-forward predictions from {CACHE_WF}")
        pred_df = pd.read_pickle(CACHE_WF)
    else:
        from pygam import LinearGAM, s, te
        pred_pieces = []
        for i, (ts, te_, vs, ve) in enumerate(windows):
            print(f"\n[step {i+1}/{len(windows)}] train [{ts.date()}..{te_.date()}], test [{vs.date()}..{ve.date()}]")

            train_mask = (dates_all >= ts) & (dates_all < te_)
            test_mask = (dates_all >= vs) & (dates_all < ve)
            train_df = df[train_mask]
            test_df = df[test_mask]
            print(f"  train rows: {len(train_df):,}   test rows: {len(test_df):,}")

            if len(train_df) > MAX_TRAIN:
                rs = np.random.default_rng(42 + i)
                idx = rs.choice(len(train_df), MAX_TRAIN, replace=False)
                train_df = train_df.iloc[idx]
                print(f"  subsampled train to {MAX_TRAIN:,}")
            if len(test_df) == 0:
                print("  empty test window — skipping"); continue

            X_train = train_df[["vol", "rsi", "rel"]].values
            y_train = train_df["y"].values
            X_test = test_df[["vol", "rsi", "rel"]].values

            gam = LinearGAM(
                s(0, n_splines=N_SPLINES_VOL)
              + te(1, 2, n_splines=N_SPLINES_TE)
            )
            t_fit = time.time()
            gam.gridsearch(X_train, y_train, lam=LAM_SPACE, progress=False)
            print(f"  fit done {time.time()-t_fit:.1f}s   lam={gam.lam}   pseudo_r2={gam.statistics_['pseudo_r2']['explained_deviance']:.5f}")

            pred = gam.predict(X_test)
            pred_pieces.append(pd.Series(pred, index=test_df.index, name="pred"))

            # Print learned vol curve briefly
            XX = gam.generate_X_grid(term=0)
            pdep, _ = gam.partial_dependence(term=0, X=XX, width=0.95)
            sign = "DEC" if pdep[0] > pdep[-1] else "INC" if pdep[0] < pdep[-1] else "FLAT"
            print(f"  vol curve: pdep[low_vol]={pdep[0]:+.3f}  pdep[high_vol]={pdep[-1]:+.3f}  shape={sign}")

        all_pred = pd.concat(pred_pieces, axis=0)
        pred_df = all_pred.unstack().reindex(columns=universe.columns).where(universe == 1)
        pd.to_pickle(pred_df, CACHE_WF)
        print(f"\n[saved] predictions {pred_df.shape} -> {CACHE_WF}")

    # ---- Diagnostics ----
    oos_idx = pred_df.dropna(how='all').index
    rets_oos = returns.reindex(oos_idx)
    univ_oos = universe.reindex(oos_idx)
    pred_oos = pred_df.reindex(oos_idx)

    print(f"\n[OOS] {oos_idx.min().date()}..{oos_idx.max().date()}  ({len(oos_idx)} days)")
    print("\n" + "="*70); print("WALK-FORWARD GAM DIAGNOSTICS"); print("="*70)

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

    print("\n[Decile spread, T+1, bps — full OOS]")
    fwd1 = rets_oos.shift(-1)
    dec = fast_decile(pred_oos, fwd1) * 1e4
    print(dec.round(2).to_string())
    ls = dec.iloc[-1] - dec.iloc[0]
    mono = dec.is_monotonic_increasing or dec.is_monotonic_decreasing
    corr_r = pd.Series(range(1,11)).corr(dec.reset_index(drop=True))
    print(f"  L-S (D10-D1): {ls:.2f} bps/d  ~  {ls*252/100:.1f}%/yr gross")
    print(f"  monotonic: {mono}   corr(decile, ret): {corr_r:.3f}")

    print("\n[Decile spread by year (the critical regime test)]")
    yrs = sorted(set(oos_idx.year))
    for y in yrs:
        sub = oos_idx[oos_idx.year == y]
        d = fast_decile(pred_oos.loc[sub], fwd1.loc[sub]) * 1e4
        print(f"  {y}: D1={d.iloc[0]:.2f}  D5={d.iloc[4]:.2f}  D10={d.iloc[-1]:.2f}  "
              f"L-S(D10-D1)={d.iloc[-1]-d.iloc[0]:+.2f}  corr(d,r)={pd.Series(range(1,11)).corr(d.reset_index(drop=True)):+.2f}")

    # ---- Portfolios ----
    print("\n\n" + "#"*70); print("# PORTFOLIO CONSTRUCTIONS (walk-forward GAM)"); print("#"*70)

    print("\n--- (b) continuous tilt * inv-vol ---")
    w_b = portfolio_continuous(pred_oos, univ_oos, log_ret.reindex(oos_idx))
    res_b = report(w_b, rets_oos, univ_oos, "(b) continuous")

    print("\n--- (c) bucket 10/10 EW ---")
    w_c = portfolio_bucket(pred_oos, univ_oos)
    res_c = report(w_c, rets_oos, univ_oos, "(c) bucket 10/10")

    print("\n--- (c2) bucket 20/20 EW ---")
    w_c2 = portfolio_bucket(pred_oos, univ_oos, long_pct=0.80, short_pct=0.20)
    res_c2 = report(w_c2, rets_oos, univ_oos, "(c2) bucket 20/20")

    # Also test sign-flipped
    print("\n--- (b-) flip continuous ---")
    w_bn = portfolio_continuous(-pred_oos, univ_oos, log_ret.reindex(oos_idx))
    res_bn = report(w_bn, rets_oos, univ_oos, "(b-) flip continuous")

    print("\n--- (c-) flip bucket 10/10 ---")
    w_cn = portfolio_bucket(-pred_oos, univ_oos)
    res_cn = report(w_cn, rets_oos, univ_oos, "(c-) flip bucket 10/10")

    # Headline
    print("\n\n" + "="*70); print("HEADLINE — walk-forward GAM"); print("="*70)
    rows = [
        ("(b)  continuous tilt + inv-vol  ", res_b),
        ("(c)  bucket 10/10 EW            ", res_c),
        ("(c2) bucket 20/20 EW            ", res_c2),
        ("(b-) flip continuous             ", res_bn),
        ("(c-) flip bucket 10/10           ", res_cn),
    ]
    print(f"  {'label':38s}  {'net_SR':>7s}  {'max_DD%':>7s}  {'active_d':>8s}")
    for lbl, r in rows:
        print(f"  {lbl}  {r['net_sr']:7.3f}  {r['max_dd']:7.2f}  {r['active_days']:8d}")

    best = max(rows, key=lambda x: x[1]["net_sr"])
    print(f"\n[best]: {best[0]} — net_SR {best[1]['net_sr']:.3f}, max_DD {best[1]['max_dd']:.2f}%")

    out = {
        "oos": [str(oos_idx.min().date()), str(oos_idx.max().date())],
        "config": {"TRAIN_YEARS": TRAIN_YEARS, "TEST_MONTHS": TEST_MONTHS,
                   "STEP_MONTHS": STEP_MONTHS, "MAX_TRAIN": MAX_TRAIN},
        "ic_decay": {f"T+{lag}": float(ic.mean()) for lag, ic in ic_by_lag.items()},
        "decile_spread_bps": dec.round(2).to_dict(),
        "monotonic": bool(mono), "decile_corr": float(corr_r),
        "portfolios": {lbl: {k: v for k, v in r.items() if k != "pnl"} for lbl, r in rows},
    }
    (ART / "summary_walkforward.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[saved] {ART}/summary_walkforward.json")
    print(f"[total runtime: {time.time()-t0:.1f}s]")

if __name__ == "__main__":
    main()
