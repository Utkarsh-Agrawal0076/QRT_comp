"""Refit GAM without the monotonic_dec constraint on volatility.

Lets the vol spline learn its own shape — should produce a non-monotonic curve
if low-vol and high-vol regimes have opposite signs.

Trains on 2010-2020, predicts on 2021+ (≥5yr OOS), then runs the same fast
diagnostic + 4-portfolio comparison as gam_diagnose_fast.py.
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
from gam_diagnose_flip import portfolio_pair

ART = Path("stores/gam"); ART.mkdir(exist_ok=True, parents=True)
CACHE_TARGET = ART / "target.pkl"
CACHE_FREE = ART / "predictions_free_vol.pkl"

TRAIN_END_YEAR = 2020
OOS_START_YEAR = 2021
ZSCORE_CLIP = 3.0
MAX_SAMPLES = 300_000  # smaller than original 500k for speed

def build_features(ohlcv, universe):
    print("[features] building vol_z, rsi_rank, rel_vol ...")
    C, V = ohlcv["Adj Close"], ohlcv["Volume"]
    log_ret = np.log(C / C.shift(1))
    univ_mask = (universe == 1)

    vol_20 = log_ret.rolling(20).std()
    vol_20m = vol_20.where(univ_mask)
    vol_z = vol_20m.sub(vol_20m.mean(axis=1), axis=0).div(vol_20m.std(axis=1), axis=0)
    vol_z = vol_z.replace([np.inf, -np.inf], np.nan).clip(-ZSCORE_CLIP, ZSCORE_CLIP)

    delta = C.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    rg = gain.ewm(com=13, min_periods=14).mean()
    rl = loss.ewm(com=13, min_periods=14).mean()
    rs = rg / rl.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(100)
    rsi_m = rsi.where(univ_mask)
    rsi_rank = rsi_m.rank(axis=1, pct=True)

    rel_vol = V / V.rolling(20).mean()
    rel_vol = rel_vol.where(univ_mask)
    rel_vol = rel_vol.replace([np.inf, -np.inf], np.nan).clip(0, 10)
    return vol_z, rsi_rank, rel_vol, log_ret

def main():
    t0 = time.time()
    ohlcv = load_data()
    universe = build_universe(ohlcv)
    returns = build_returns(ohlcv)
    universe = universe.reindex(columns=returns.columns).fillna(0).astype(int)

    if CACHE_FREE.exists():
        print(f"[cache] loading free-vol predictions from {CACHE_FREE}")
        pred_df = pd.read_pickle(CACHE_FREE)
        tgt = pd.read_pickle(CACHE_TARGET)
        log_ret = tgt["log_ret"]
    else:
        # Need target from cache + features rebuilt
        if not CACHE_TARGET.exists():
            print("[err] target.pkl missing — run gam_diagnose.py first")
            return
        tgt = pd.read_pickle(CACHE_TARGET)
        Y = tgt["Y"]; log_ret = tgt["log_ret"]

        vol_z, rsi_rank, rel_vol, _ = build_features(ohlcv, universe)

        print("[GAM] stacking train data ...")
        parts = {"vol": vol_z, "rsi": rsi_rank, "rel": rel_vol, "y": Y}
        df = pd.concat({k: v.stack() for k, v in parts.items()}, axis=1).dropna()
        df.index.names = ["date", "ticker"]
        train_mask = df.index.get_level_values("date").year <= TRAIN_END_YEAR
        train_df = df[train_mask]
        print(f"[GAM] train samples raw: {len(train_df)}")
        if len(train_df) > MAX_SAMPLES:
            rs = np.random.default_rng(42)
            idx = rs.choice(len(train_df), MAX_SAMPLES, replace=False)
            train_df = train_df.iloc[idx]
            print(f"[GAM] subsampled to {MAX_SAMPLES}")

        X = train_df[["vol", "rsi", "rel"]].values
        y = train_df["y"].values
        from pygam import LinearGAM, s, te
        # KEY CHANGE: drop constraints='monotonic_dec' on vol spline
        gam = LinearGAM(
            s(0, n_splines=10)              # FREE vol spline (was monotonic_dec)
          + te(1, 2, n_splines=(8, 8))      # RSI x Rel_Volume interaction (same)
        )
        lam_space = np.logspace(1, 5, 5)    # 5 lambdas (was 10) for speed
        print(f"[GAM] gridsearch over {len(lam_space)} lambdas ...")
        t1 = time.time()
        gam.gridsearch(X, y, lam=lam_space, progress=False)
        print(f"[GAM] fit done in {time.time()-t1:.1f}s   lam={gam.lam}   "
              f"pseudo_r2={gam.statistics_['pseudo_r2']['explained_deviance']:.5f}")

        # Predict everywhere
        print("[GAM] predicting full panel ...")
        Xp = df[["vol", "rsi", "rel"]].values
        pred = gam.predict(Xp)
        s_ = pd.Series(pred, index=df.index, name="pred")
        pred_df = s_.unstack().reindex(columns=universe.columns).where(universe == 1)
        pd.to_pickle(pred_df, CACHE_FREE)

        # Quick look at the learned vol curve
        print("\n[learned vol curve — partial dependence]")
        XX = gam.generate_X_grid(term=0)
        pdep, _ = gam.partial_dependence(term=0, X=XX, width=0.95)
        # Show 5 evenly-spaced points
        idxs = np.linspace(0, len(XX)-1, 9).astype(int)
        for i in idxs:
            print(f"  vol_z={XX[i,0]:+.2f}   partial alpha={pdep[i]:+.4f}")

    oos_idx = returns.index[returns.index.year >= OOS_START_YEAR]
    print(f"\n[OOS] {oos_idx.min().date()}..{oos_idx.max().date()}  ({len(oos_idx)} days)")
    pred_df = pred_df.reindex(columns=returns.columns)
    rets_oos = returns.loc[oos_idx]; univ_oos = universe.loc[oos_idx]
    pred_oos = pred_df.reindex(oos_idx)

    # ---- Diagnostics ----
    print("\n" + "="*70); print("FREE-VOL GAM SIGNAL DIAGNOSTICS"); print("="*70)
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

    print("\n[Decile spread, T+1, bps]")
    fwd1 = rets_oos.shift(-1)
    dec = fast_decile(pred_oos, fwd1) * 1e4
    print(dec.round(2).to_string())
    ls = dec.iloc[-1] - dec.iloc[0]
    mono = dec.is_monotonic_increasing or dec.is_monotonic_decreasing
    corr_r = pd.Series(range(1,11)).corr(dec.reset_index(drop=True))
    print(f"  L-S (D10-D1): {ls:.2f} bps/d  ~  {ls*252/100:.1f}%/yr gross")
    print(f"  monotonic: {mono}   corr(decile, ret): {corr_r:.3f}")

    # Also decile-by-year to see if regime story improves
    print("\n[Decile spread by year (D10-D1, bps/d)]")
    yrs = sorted(set(oos_idx.year))
    for y in yrs:
        sub = oos_idx[oos_idx.year == y]
        d = fast_decile(pred_oos.loc[sub], fwd1.loc[sub]) * 1e4
        print(f"  {y}: D1={d.iloc[0]:.2f}  D5={d.iloc[4]:.2f}  D10={d.iloc[-1]:.2f}  "
              f"L-S(D10-D1)={d.iloc[-1]-d.iloc[0]:+.2f}  corr(d,r)={pd.Series(range(1,11)).corr(d.reset_index(drop=True)):+.2f}")

    # ---- Portfolios ----
    print("\n\n" + "#"*70); print("# PORTFOLIO CONSTRUCTIONS"); print("#"*70)

    print("\n--- (b) continuous tilt * inv-vol ---")
    w_b = portfolio_continuous(pred_oos, univ_oos, log_ret.loc[oos_idx])
    res_b = report(w_b, rets_oos, univ_oos, "(b) continuous")

    print("\n--- (c) bucket 10/10 EW ---")
    w_c = portfolio_bucket(pred_oos, univ_oos)
    res_c = report(w_c, rets_oos, univ_oos, "(c) bucket 10/10")

    print("\n--- (c2) bucket 20/20 EW ---")
    w_c2 = portfolio_bucket(pred_oos, univ_oos, long_pct=0.80, short_pct=0.20)
    res_c2 = report(w_c2, rets_oos, univ_oos, "(c2) bucket 20/20")

    # Sign-flipped variants too
    print("\n--- (b-) flip continuous ---")
    w_bn = portfolio_continuous(-pred_oos, univ_oos, log_ret.loc[oos_idx])
    res_bn = report(w_bn, rets_oos, univ_oos, "(b-) flip continuous")

    print("\n--- (c-) flip bucket 10/10 ---")
    w_cn = portfolio_bucket(-pred_oos, univ_oos)
    res_cn = report(w_cn, rets_oos, univ_oos, "(c-) flip bucket 10/10")

    # ---- Headline ----
    print("\n\n" + "="*70); print("HEADLINE — free-vol GAM, six portfolios"); print("="*70)
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

    # Per-year head-to-head for the best portfolio
    best = max(rows, key=lambda x: x[1]["net_sr"])
    print(f"\n[best]: {best[0]} — net_SR {best[1]['net_sr']:.3f}, max_DD {best[1]['max_dd']:.2f}%")
    pnl_best = best[1]["pnl"]
    yr_sr = pnl_best.groupby(pnl_best.index.year).apply(lambda s: (s.mean()/s.std()*np.sqrt(252)) if s.std()>0 else 0)
    print("[best per-year SR]")
    print(yr_sr.round(2).to_string())

    # Save
    out = {
        "oos": [str(oos_idx.min().date()), str(oos_idx.max().date())],
        "ic_decay": {f"T+{lag}": float(ic.mean()) for lag, ic in ic_by_lag.items()},
        "decile_spread_bps": dec.round(2).to_dict(),
        "monotonic": bool(mono), "decile_corr": float(corr_r),
        "portfolios": {lbl: {k: v for k, v in r.items() if k != "pnl"} for lbl, r in rows},
    }
    (ART / "summary_free_vol.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[saved] {ART}/summary_free_vol.json")
    print(f"[total runtime: {time.time()-t0:.1f}s]")

if __name__ == "__main__":
    main()
