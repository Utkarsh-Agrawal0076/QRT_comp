"""FFT/Welch as a horizon detector for mean-reversion.

Pipeline:
  1. Build residual returns vs cross-sectional market proxy (no yfinance dep).
  2. Rolling Welch PSD per stock; extract dominant period T_hat in [3,30] days.
  3. KILL-TEST diagnostics:
       (a) within-stock T_hat stability
       (b) cross-sectional distribution of T_hat
       (c) peak prominence (peak PSD / band-median PSD)
  4. Horizon-matched MR signal: per-stock k = T_hat(t), -1 * trailing-k residual return
     Baseline uniform MR: same construction with fixed k = 5.
  5. OOS = 4 years; report IC, decile, backtest both side-by-side.
  6. Alpha correlation against existing 3-day MR sleeve construction.
"""
from __future__ import annotations
import sys, warnings, time, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import welch

warnings.filterwarnings("ignore")
sys.path.append(str(Path("phase2_qrt_challenge/scripts").resolve()))
import utils  # noqa

from squeeze_breakout_pipeline import (
    load_data, build_universe, build_returns,
    daily_ic, ic_by_year,
    alpha_to_weights, enforce_post_shift_strict_gmv,
    MIN_STOCKS,
)

ART = Path("stores/fft_horizon"); ART.mkdir(exist_ok=True, parents=True)

OOS_YEARS = 4
SPEC_LOOKBACK = 252       # spectrum estimation window
CHECKPOINT_STRIDE = 21    # update T_hat monthly
WELCH_NPERSEG = 64        # Welch sub-window
PERIOD_BAND = (3, 30)     # search dominant period in this range (days)
K_GRID = np.array([3, 5, 7, 10, 14, 21])

# ---------- residual returns (cross-sectional, no external data) ----------
def residualize(returns, universe):
    """r_resid = r - cross-sectional mean within universe."""
    mask = universe.astype(bool)
    mkt = returns.where(mask).mean(axis=1)
    r_resid = returns.sub(mkt, axis=0).where(mask)
    return r_resid

# ---------- rolling Welch peak period ----------
def rolling_peak_period(r_resid, lookback=SPEC_LOOKBACK, stride=CHECKPOINT_STRIDE,
                        nperseg=WELCH_NPERSEG, band=PERIOD_BAND):
    """Compute dominant period and peak-prominence at every `stride` days.

    Returns (peak_period_df, prominence_df) — both indexed at checkpoint dates,
    columns = stocks.
    """
    arr = r_resid.values.astype(np.float64)
    arr = np.nan_to_num(arr, nan=0.0)
    n_t, n_s = arr.shape

    checkpoint_iloc = list(range(lookback, n_t, stride))
    checkpoint_dates = r_resid.index[checkpoint_iloc]

    period_rows, prom_rows = [], []
    for j, iloc in enumerate(checkpoint_iloc):
        window = arr[iloc - lookback:iloc, :]  # (lookback, n_s)
        # Welch along time axis
        freqs, psd = welch(window, fs=1.0, nperseg=nperseg, axis=0,
                            scaling="density", detrend="linear")
        # band of interest: period in [3,30] -> freq in [1/30, 1/3]
        f_lo, f_hi = 1.0 / band[1], 1.0 / band[0]
        m = (freqs >= f_lo) & (freqs <= f_hi)
        psd_b = psd[m, :]            # (Fb, n_s)
        freqs_b = freqs[m]
        if psd_b.shape[0] == 0:
            continue
        peak_idx = np.argmax(psd_b, axis=0)
        peak_psd = psd_b[peak_idx, np.arange(n_s)]
        # prominence: peak / band median (1 = no peak, >1.5 = real peak)
        band_median = np.median(psd_b, axis=0) + 1e-30
        prom = peak_psd / band_median
        period = 1.0 / freqs_b[peak_idx]
        period_rows.append(period)
        prom_rows.append(prom)

    period_df = pd.DataFrame(period_rows, index=checkpoint_dates, columns=r_resid.columns)
    prom_df = pd.DataFrame(prom_rows, index=checkpoint_dates, columns=r_resid.columns)
    return period_df, prom_df

# ---------- kill-test diagnostics ----------
def kill_test(period_df, prom_df, universe):
    print("\n" + "="*70)
    print("KILL-TEST DIAGNOSTICS")
    print("="*70)

    # Restrict to stocks/dates where universe was active at that checkpoint
    univ_ck = universe.reindex(period_df.index).fillna(0).astype(bool)
    period_masked = period_df.where(univ_ck)
    prom_masked = prom_df.where(univ_ck)

    # (a) within-stock stability
    per_stock_std = period_masked.std(axis=0)
    per_stock_count = period_masked.count(axis=0)
    valid = per_stock_count >= 10
    print(f"\n[a] Within-stock T_hat stability (std across {period_df.shape[0]} checkpoints)")
    print(f"   n stocks with >=10 checkpoints: {valid.sum()}")
    print(f"   median per-stock std of T_hat: {per_stock_std[valid].median():.2f} days")
    print(f"   p25 / p75: {per_stock_std[valid].quantile(0.25):.2f} / {per_stock_std[valid].quantile(0.75):.2f}")
    print(f"   fraction stocks with std < 3 days (stable): {(per_stock_std[valid] < 3).mean()*100:.1f}%")
    print(f"   fraction stocks with std > 7 days (wandering): {(per_stock_std[valid] > 7).mean()*100:.1f}%")

    # (b) cross-sectional distribution
    print("\n[b] Cross-sectional distribution of T_hat (pooled across stocks x dates)")
    pooled = period_masked.values.flatten()
    pooled = pooled[~np.isnan(pooled)]
    bins = [3, 5, 7, 10, 14, 21, 30]
    hist, _ = np.histogram(pooled, bins=bins)
    pct = hist / hist.sum() * 100
    for b_lo, b_hi, p in zip(bins[:-1], bins[1:], pct):
        bar = "#" * int(p)
        print(f"   T in [{b_lo:2d}, {b_hi:2d}): {p:5.1f}%  {bar}")

    # (c) peak prominence
    print("\n[c] Peak prominence (peak PSD / band median PSD)")
    prom_pooled = prom_masked.values.flatten()
    prom_pooled = prom_pooled[~np.isnan(prom_pooled)]
    print(f"   median prominence: {np.median(prom_pooled):.2f}x")
    print(f"   p25 / p75: {np.percentile(prom_pooled, 25):.2f}x / {np.percentile(prom_pooled, 75):.2f}x")
    print(f"   fraction obs with prominence > 1.5x (real peak): {(prom_pooled > 1.5).mean()*100:.1f}%")
    print(f"   fraction obs with prominence > 2.0x (strong peak): {(prom_pooled > 2.0).mean()*100:.1f}%")

    # Verdict
    print("\n[verdict]")
    stable_frac = (per_stock_std[valid] < 3).mean()
    strong_frac = (prom_pooled > 1.5).mean()
    multimodal_check = pct.std() / pct.mean()  # higher = less uniform
    if stable_frac > 0.3 and strong_frac > 0.3 and multimodal_check > 0.3:
        print("   PROCEED - spectral structure looks tradable")
        verdict = "proceed"
    elif stable_frac > 0.15 or strong_frac > 0.2:
        print("   MARGINAL - some structure but weak; results likely break even with baseline")
        verdict = "marginal"
    else:
        print("   FAIL - spectra are flat/wandering; horizon segmentation will not help")
        verdict = "fail"
    return verdict, per_stock_std, prom_pooled

# ---------- horizon-matched MR signal ----------
def build_horizon_mr(r_resid, peak_period_df, prom_df, universe,
                     min_prom=1.5, k_grid=K_GRID):
    """Per-stock k = nearest in K_GRID to T_hat(t); fallback to uniform if low prominence.

    Returns horizon_mr and the per-day chosen-k matrix.
    """
    # Forward-fill peak period and prominence to daily, no future leak (use as-of t-1)
    period_daily = peak_period_df.reindex(r_resid.index).ffill().shift(1)
    prom_daily = prom_df.reindex(r_resid.index).ffill().shift(1)

    # Snap to nearest k in K_GRID
    # build per-cell choice
    k_arr = k_grid.astype(float)
    def snap(x):
        return k_arr[np.argmin(np.abs(k_arr - x))] if not np.isnan(x) else np.nan
    # Vectorized snap via broadcasting
    p_vals = period_daily.values  # (T, N)
    diffs = np.abs(p_vals[..., None] - k_arr[None, None, :])  # (T, N, K)
    k_idx = np.argmin(diffs, axis=2)
    k_chosen = k_arr[k_idx]  # (T, N)
    k_chosen = np.where(np.isnan(p_vals), np.nan, k_chosen)
    # If prominence too low, fall back to NaN (will be filled with median k)
    if min_prom > 0:
        low_prom = (prom_daily.values < min_prom) | np.isnan(prom_daily.values)
        k_chosen = np.where(low_prom, np.nan, k_chosen)

    # Precompute trailing-k residual return for each k in grid
    trailing = {int(k): r_resid.rolling(int(k), min_periods=max(2, int(k)//2)).sum().shift(1) for k in k_grid}

    # For each (t,i), pick trailing[k_chosen]
    arr_t = {int(k): trailing[int(k)].values for k in k_grid}
    out = np.full_like(p_vals, np.nan, dtype=np.float64)
    for kk in k_grid:
        kk_int = int(kk)
        mask = (k_chosen == kk)
        out[mask] = arr_t[kk_int][mask]

    # MR signal = -trailing_return (large positive = sell, large negative = buy)
    mr = -pd.DataFrame(out, index=r_resid.index, columns=r_resid.columns)
    mr = mr.where(universe == 1, np.nan)
    return mr, pd.DataFrame(k_chosen, index=r_resid.index, columns=r_resid.columns)

def build_uniform_mr(r_resid, universe, k=5):
    trailing = r_resid.rolling(k, min_periods=max(2, k//2)).sum().shift(1)
    return (-trailing).where(universe == 1, np.nan)

# ---------- diagnostics + backtest ----------
def evaluate(alpha, returns, universe, oos_idx, label):
    a = alpha.reindex(oos_idx)
    rets = returns.loc[oos_idx]
    univ = universe.loc[oos_idx]
    fwd = rets.shift(-1)

    print(f"\n{'='*70}\n{label}\n{'='*70}")
    ic = daily_ic(a, fwd)
    print(f"\n[IC, T+1]  N={len(ic)}  mean={ic.mean():.4f}  median={ic.median():.4f}  "
          f"IR_ann={ic.mean()/ic.std()*np.sqrt(252):.2f}  "
          f"%>0={(ic>0).mean()*100:.1f}%  t={ic.mean()/(ic.std()/np.sqrt(len(ic))):.2f}")

    print("\n[IC by year]")
    print(ic_by_year(ic).round(4).to_string())

    # decile spread
    rk = a.rank(axis=1, pct=True)
    dec = {}
    for d in range(1, 11):
        lo, hi = (d-1)/10, d/10
        m = (rk >= lo) & (rk < hi) if d < 10 else (rk >= lo)
        dec[d] = fwd.where(m).stack().mean()
    dec_s = pd.Series(dec) * 1e4
    print("\n[Decile spread (bps fwd T+1)]")
    print(dec_s.round(2).to_string())
    ls = dec_s.iloc[-1] - dec_s.iloc[0]
    mono = dec_s.is_monotonic_increasing or dec_s.is_monotonic_decreasing
    corr = pd.Series(range(1,11)).corr(dec_s.reset_index(drop=True))
    print(f"  L-S (D10-D1): {ls:.2f} bps/d  ~  {ls*252/100:.1f}%/yr gross")
    print(f"  monotonic: {mono}   corr(decile, ret): {corr:.3f}")

    # backtest
    w = alpha_to_weights(a, univ)
    w = enforce_post_shift_strict_gmv(w.shift(1), univ)
    w = w.reindex(columns=rets.columns, fill_value=0.0)
    univ_b = univ.reindex(columns=rets.columns, fill_value=0)
    print("\n[BACKTEST]")
    sr, pnl = utils.backtest_portfolio(w, rets, univ_b, plot_=False, print_=True)
    yr = pnl.groupby(pnl.index.year).agg(
        ann_ret=lambda s: s.mean()*252,
        ann_vol=lambda s: s.std()*np.sqrt(252),
        sharpe=lambda s: (s.mean()/s.std()*np.sqrt(252)) if s.std()>0 else 0,
        n="count",
    )
    print("\n[Per-year PnL]")
    print(yr.round(3).to_string())
    eq = (1+pnl).cumprod(); dd = eq/eq.cummax()-1
    print(f"\n[summary] net_SR={sr}  max_DD={dd.min()*100:.1f}%  hit_days={(pnl>0).mean()*100:.1f}%")
    return {"label": label, "ic_mean": float(ic.mean()),
            "ic_ir": float(ic.mean()/ic.std()*np.sqrt(252)),
            "ls_bps": float(ls), "monotonic": bool(mono), "decile_corr": float(corr),
            "net_sr": float(sr), "max_dd": float(dd.min()*100),
            "pnl": pnl}

def alpha_corr(a1, a2, oos_idx, label1, label2):
    """Daily cross-sectional correlation between two alpha matrices, then averaged."""
    a1o = a1.reindex(oos_idx); a2o = a2.reindex(oos_idx)
    corrs = []
    for dt in oos_idx:
        s1 = a1o.loc[dt].dropna(); s2 = a2o.loc[dt].dropna()
        common = s1.index.intersection(s2.index)
        if len(common) < 100: continue
        corrs.append(s1[common].rank().corr(s2[common].rank()))
    s = pd.Series(corrs)
    print(f"\n[Alpha rank-corr]  {label1}  vs  {label2}")
    print(f"  mean rank-corr (daily, OOS): {s.mean():.3f}   median: {s.median():.3f}")
    return float(s.mean())

# ---------- existing 3d MR alpha for orthogonality check ----------
def build_existing_mr(ohlcv, universe):
    """Mirror of master pipeline's mean-reversion alpha."""
    vwap = (ohlcv['High'] + ohlcv['Low'] + ohlcv['Close']) / 3.0
    diff = vwap - ohlcv['Close']
    vd = ohlcv['Volume'].diff(3)
    rmax = diff.rolling(3).max().rank(axis=1, pct=True)
    rmin = diff.rolling(3).min().rank(axis=1, pct=True)
    rvd = vd.rank(axis=1, pct=True)
    a = (rmax + rmin) * rvd
    a = a.where(universe == 1, np.nan)
    a = a.rolling(3, min_periods=1).mean()
    # demean cross-sectionally for sign convention
    return a.sub(a.mean(axis=1), axis=0)

# ---------- main ----------
def main():
    t0 = time.time()
    ohlcv = load_data()
    universe = build_universe(ohlcv)
    returns = build_returns(ohlcv)
    universe = universe.reindex(columns=returns.columns).fillna(0).astype(int)
    print(f"[data] dates {returns.index.min().date()}..{returns.index.max().date()}  stocks={returns.shape[1]}")

    print("[residualize] cross-sectional market proxy")
    r_resid = residualize(returns, universe)

    print("[welch] rolling PSD ...")
    t1 = time.time()
    period_df, prom_df = rolling_peak_period(r_resid)
    print(f"  done in {time.time()-t1:.1f}s   shape={period_df.shape}")

    # KILL TEST
    verdict, _, _ = kill_test(period_df, prom_df, universe)

    # Split
    oos_start = returns.index.max() - pd.DateOffset(years=OOS_YEARS)
    is_idx = returns.index[returns.index < oos_start]
    oos_idx = returns.index[returns.index >= oos_start]
    print(f"\n[split] IS {is_idx.min().date()}..{is_idx.max().date()}  ({len(is_idx)})")
    print(f"[split] OOS {oos_idx.min().date()}..{oos_idx.max().date()}  ({len(oos_idx)})")

    # IS quick check: do horizon-matched and uniform differ?
    print("\n[IS check] horizon-matched vs uniform (k=5) mean IC at T+1")
    a_match = build_horizon_mr(r_resid, period_df, prom_df, universe, min_prom=1.5)[0]
    a_unif5 = build_uniform_mr(r_resid, universe, k=5)
    a_unif10 = build_uniform_mr(r_resid, universe, k=10)
    fwd_is = returns.loc[is_idx].shift(-1)
    for name, a in [("horizon-matched (min_prom=1.5)", a_match),
                    ("uniform k=5", a_unif5),
                    ("uniform k=10", a_unif10)]:
        ic = daily_ic(a.reindex(is_idx), fwd_is)
        if len(ic) == 0: print(f"  {name}: no IC data"); continue
        print(f"  {name:35s}  IS mean_IC={ic.mean():+.4f}  IR_ann={ic.mean()/ic.std()*np.sqrt(252):+.2f}  n={len(ic)}")

    # ---- OOS evaluation ----
    print("\n\n" + "#"*70)
    print("# OOS EVALUATION (4-year out-of-sample window)")
    print("#"*70)

    res_match = evaluate(a_match, returns, universe, oos_idx, "HORIZON-MATCHED MR (Welch-selected k)")
    res_unif5 = evaluate(a_unif5, returns, universe, oos_idx, "BASELINE UNIFORM MR (k=5)")
    res_unif10 = evaluate(a_unif10, returns, universe, oos_idx, "BASELINE UNIFORM MR (k=10)")

    # ---- Orthogonality vs existing 3d MR sleeve ----
    print("\n\n" + "#"*70)
    print("# ORTHOGONALITY vs EXISTING 3-DAY MR SLEEVE")
    print("#"*70)
    a_exist = build_existing_mr(ohlcv, universe)
    c_match = alpha_corr(a_match, a_exist, oos_idx, "horizon-matched", "existing 3d MR")
    c_u5 = alpha_corr(a_unif5, a_exist, oos_idx, "uniform k=5", "existing 3d MR")
    c_u10 = alpha_corr(a_unif10, a_exist, oos_idx, "uniform k=10", "existing 3d MR")

    # PnL correlation between sleeves
    pnl_exist_w = enforce_post_shift_strict_gmv(alpha_to_weights(a_exist, universe).shift(1), universe)
    pnl_exist_w = pnl_exist_w.reindex(columns=returns.columns, fill_value=0.0)
    pnl_exist = (pnl_exist_w * returns.fillna(0)).sum(axis=1)
    pnl_exist_oos = pnl_exist.loc[oos_idx]
    for name, r in [("horizon-matched", res_match), ("uniform k=5", res_unif5), ("uniform k=10", res_unif10)]:
        cor = r["pnl"].corr(pnl_exist_oos)
        print(f"\n[PnL daily corr]  {name} vs existing 3d MR: {cor:.3f}")

    # ---- Summary ----
    print("\n\n" + "="*70)
    print("HEADLINE SUMMARY")
    print("="*70)
    summary = pd.DataFrame([
        {"strategy": r["label"], "ic_mean": r["ic_mean"], "ic_ir": r["ic_ir"],
         "ls_bps": r["ls_bps"], "monotonic": r["monotonic"],
         "decile_corr": r["decile_corr"], "net_sr": r["net_sr"], "max_dd": r["max_dd"]}
        for r in [res_match, res_unif5, res_unif10]
    ])
    print(summary.round(4).to_string(index=False))

    out = {
        "verdict": verdict,
        "oos_window": [str(oos_idx.min().date()), str(oos_idx.max().date())],
        "results": [{k: v for k, v in r.items() if k != "pnl"} for r in [res_match, res_unif5, res_unif10]],
        "alpha_corr_vs_existing_3dMR": {
            "horizon-matched": c_match, "uniform_5": c_u5, "uniform_10": c_u10
        },
    }
    (ART / "summary.json").write_text(json.dumps(out, indent=2))
    res_match["pnl"].to_csv(ART / "pnl_horizon_matched.csv")
    res_unif5["pnl"].to_csv(ART / "pnl_uniform5.csv")
    res_unif10["pnl"].to_csv(ART / "pnl_uniform10.csv")
    period_df.to_pickle(ART / "peak_period.pkl")
    prom_df.to_pickle(ART / "peak_prominence.pkl")
    print(f"\n[saved] {ART}/")
    print(f"[total runtime: {time.time()-t0:.1f}s]")

if __name__ == "__main__":
    main()
