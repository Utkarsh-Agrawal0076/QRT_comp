"""Fast GAM diagnostics — uses cached predictions, skips slow hysteresis loops.

Keeps:
  - Signal quality: IC at T+1 only, IC-by-year, decile spread (one go).
  - Continuous tilt + inverse-vol portfolio.
  - Percentile-bucket portfolios (10/10 and 20/20).
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

ART = Path("stores/gam"); ART.mkdir(exist_ok=True, parents=True)
CACHE_PRED = ART / "predictions.pkl"
CACHE_TARGET = ART / "target.pkl"

OOS_START_YEAR = 2021

# ---------- Vectorized diagnostics (fast) ----------
def fast_ic(signal, returns, lags=(1, 3, 5, 10), min_stocks=200):
    """Vectorized cross-sectional Spearman IC for several lags at once."""
    sig = signal.dropna(how='all')
    out = {}
    for lag in lags:
        fwd = returns.shift(-lag).reindex(sig.index)
        # Rank cross-sectionally
        s_r = sig.rank(axis=1)
        f_r = fwd.rank(axis=1)
        # Mask: both finite
        valid = sig.notna() & fwd.notna()
        s_r = s_r.where(valid)
        f_r = f_r.where(valid)
        # Daily cross-section means/stds
        mu_s = s_r.mean(axis=1); mu_f = f_r.mean(axis=1)
        sd_s = s_r.std(axis=1); sd_f = f_r.std(axis=1)
        cov = ((s_r.sub(mu_s, axis=0)) * (f_r.sub(mu_f, axis=0))).sum(axis=1) / (valid.sum(axis=1) - 1)
        n_per_day = valid.sum(axis=1)
        ic = (cov / (sd_s * sd_f)).where(n_per_day >= min_stocks)
        ic = ic.dropna()
        out[lag] = ic
    return out

def fast_decile(signal, fwd, n_bins=10):
    rk = signal.rank(axis=1, pct=True)
    rows = []
    for d in range(1, n_bins + 1):
        lo, hi = (d-1)/n_bins, d/n_bins
        m = (rk >= lo) & (rk < hi) if d < n_bins else (rk >= lo)
        rows.append(fwd.where(m).stack().mean())
    return pd.Series(rows, index=range(1, n_bins+1))

# ---------- Portfolio constructions ----------
def portfolio_bucket(alpha, universe, long_pct=0.90, short_pct=0.10):
    rk = alpha.where(universe == 1, np.nan).rank(axis=1, pct=True)
    lm = (rk >= long_pct).astype(float)
    sm = (rk <= short_pct).astype(float)
    nL = lm.sum(axis=1).replace(0, np.nan)
    nS = sm.sum(axis=1).replace(0, np.nan)
    w = lm.div(nL, axis=0) * 0.5 + sm.div(nS, axis=0) * -0.5
    return enforce_post_shift_strict_gmv(w.fillna(0).shift(1), universe)

def portfolio_continuous(alpha, universe, log_ret, vol_floor=0.005):
    a = alpha.where(universe == 1, np.nan)
    a_z = a.sub(a.mean(axis=1), axis=0).div(a.std(axis=1), axis=0).clip(-3, 3)
    vol_20 = log_ret.rolling(20).std().clip(lower=vol_floor)
    a_z = a_z.div(vol_20)
    longs = a_z.where(a_z > 0, 0)
    shorts = a_z.where(a_z < 0, 0).abs()
    def norm_cap(w, target=0.5, cap=0.099):
        s = w.sum(axis=1) + 1e-10
        w = w.div(s, axis=0) * target
        w = w.clip(upper=cap)
        s2 = w.sum(axis=1) + 1e-10
        w = w.div(s2, axis=0) * target
        return w
    final = (norm_cap(longs) - norm_cap(shorts)).fillna(0)
    return enforce_post_shift_strict_gmv(final.shift(1), universe)

def report(weights, returns, universe, label):
    w = weights.reindex(columns=returns.columns, fill_value=0.0)
    u = universe.reindex(columns=returns.columns, fill_value=0)
    sr, pnl = utils.backtest_portfolio(w, returns, u, plot_=False, print_=True)
    eq = (1+pnl).cumprod(); dd = eq/eq.cummax()-1
    print(f"  max_DD={dd.min()*100:.2f}%  hit_days={(pnl>0).mean()*100:.1f}%")
    yr = pnl.groupby(pnl.index.year).agg(
        ann_ret=lambda s: s.mean()*252,
        sharpe=lambda s: (s.mean()/s.std()*np.sqrt(252)) if s.std()>0 else 0,
        n="count",
    )
    print("  per-year:")
    print(yr.round(3).to_string(index=True))
    n_active = (w.abs().sum(axis=1) > 0.01).sum()
    print(f"  active trading days: {n_active}/{len(w)}")
    return {"net_sr": float(sr), "max_dd": float(dd.min()*100),
            "active_days": int(n_active), "pnl": pnl}

# ---------- Main ----------
def main():
    t0 = time.time()
    if not (CACHE_PRED.exists() and CACHE_TARGET.exists()):
        print("[err] cache files missing — run gam_diagnose.py first")
        return

    print("[load] cache + market data ...")
    pred_df = pd.read_pickle(CACHE_PRED)
    tgt = pd.read_pickle(CACHE_TARGET)
    log_ret = tgt["log_ret"]

    ohlcv = load_data()
    universe = build_universe(ohlcv)
    returns = build_returns(ohlcv)
    universe = universe.reindex(columns=returns.columns).fillna(0).astype(int)
    pred_df = pred_df.reindex(columns=returns.columns)

    oos_idx = returns.index[returns.index.year >= OOS_START_YEAR]
    print(f"[OOS] {oos_idx.min().date()}..{oos_idx.max().date()}  ({len(oos_idx)} days)")

    rets_oos = returns.loc[oos_idx]
    univ_oos = universe.loc[oos_idx]
    pred_oos = pred_df.reindex(oos_idx)

    # ---- Signal diagnostics ----
    print("\n" + "="*70); print("GAM SIGNAL DIAGNOSTICS (vectorized)"); print("="*70)
    t1 = time.time()
    ic_by_lag = fast_ic(pred_oos, rets_oos, lags=(1, 3, 5, 10))
    print(f"  IC computed in {time.time()-t1:.1f}s")

    print("\n[IC decay]")
    for lag, ic in ic_by_lag.items():
        if len(ic) == 0: continue
        ir = ic.mean() / ic.std() * np.sqrt(252)
        t = ic.mean() / (ic.std() / np.sqrt(len(ic)))
        print(f"  T+{lag:>2}:  mean={ic.mean():+.4f}  median={ic.median():+.4f}  "
              f"IR_ann={ir:+.2f}  %>0={(ic>0).mean()*100:.1f}%  t={t:.2f}  n={len(ic)}")

    ic1 = ic_by_lag[1]
    print("\n[IC by year (T+1)]")
    yr = ic1.groupby(ic1.index.year).agg(
        mean_IC=lambda s: s.mean(),
        std_IC=lambda s: s.std(),
        n=lambda s: len(s),
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

    # ---- Portfolios ----
    print("\n\n" + "#"*70); print("# PORTFOLIO CONSTRUCTIONS (signal mapping comparison)"); print("#"*70)

    print("\n--- (b) continuous tilt * inv-vol ---")
    w_b = portfolio_continuous(pred_oos, univ_oos, log_ret.loc[oos_idx])
    res_b = report(w_b, rets_oos, univ_oos, "(b) continuous")

    print("\n--- (c) bucket 10/10 EW ---")
    w_c = portfolio_bucket(pred_oos, univ_oos)
    res_c = report(w_c, rets_oos, univ_oos, "(c) bucket 10/10")

    print("\n--- (c2) bucket 20/20 EW ---")
    w_c2 = portfolio_bucket(pred_oos, univ_oos, long_pct=0.80, short_pct=0.20)
    res_c2 = report(w_c2, rets_oos, univ_oos, "(c2) bucket 20/20")

    print("\n--- (c3) bucket 30/30 EW (wider still) ---")
    w_c3 = portfolio_bucket(pred_oos, univ_oos, long_pct=0.70, short_pct=0.30)
    res_c3 = report(w_c3, rets_oos, univ_oos, "(c3) bucket 30/30")

    print("\n\n" + "="*70); print("HEADLINE — same signal, four portfolios"); print("="*70)
    rows = [
        ("(b)  continuous tilt + inv-vol  ", res_b),
        ("(c)  bucket 10/10 EW            ", res_c),
        ("(c2) bucket 20/20 EW            ", res_c2),
        ("(c3) bucket 30/30 EW            ", res_c3),
    ]
    print(f"  {'label':38s}  {'net_SR':>7s}  {'max_DD%':>7s}  {'active_d':>8s}")
    for lbl, r in rows:
        print(f"  {lbl}  {r['net_sr']:7.3f}  {r['max_dd']:7.2f}  {r['active_days']:8d}")

    # Save
    out = {
        "oos": [str(oos_idx.min().date()), str(oos_idx.max().date())],
        "ic_decay": {f"T+{lag}": float(ic.mean()) for lag, ic in ic_by_lag.items()},
        "decile_spread_bps": dec.round(2).to_dict(),
        "monotonic": bool(mono), "decile_corr": float(corr_r),
        "portfolios": {lbl: {k: v for k, v in r.items() if k != "pnl"} for lbl, r in rows},
    }
    (ART / "summary_fast.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[saved] {ART}/summary_fast.json")
    print(f"[total runtime: {time.time()-t0:.1f}s]")

if __name__ == "__main__":
    main()
