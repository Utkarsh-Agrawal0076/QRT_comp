"""Build new uniform-k=5 trailing-residual MR sleeve and integrate into the ensemble.

Compares:
  Baseline 3-sleeve:  MR (vwap-close)  +  Sector-Neutral Momentum  +  Stat-Arb Kalman
  New 4-sleeve     :  Baseline + Trailing-Residual MR (k=5)

Both use the master pipeline's inverse-vol blending; both evaluated on the same
4-year OOS window (matches the FFT-horizon study).
"""
from __future__ import annotations
import sys, warnings, time, json, os
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.append(str(Path("phase2_qrt_challenge/scripts").resolve()))
import utils  # noqa

from squeeze_breakout_pipeline import (
    load_data, build_universe, build_returns,
    alpha_to_weights, enforce_post_shift_strict_gmv,
)

ART = Path("stores/ensemble_with_resmr"); ART.mkdir(exist_ok=True, parents=True)
OOS_YEARS = 4

# -------- shared helpers from master pipeline --------
def normalize_and_cap(w_matrix, target=0.50, cap=0.099):
    row_sums = w_matrix.sum(axis=1) + 1e-10
    w = w_matrix.div(row_sums, axis=0) * target
    w = w.clip(upper=cap)
    row_sums_final = w.sum(axis=1) + 1e-10
    w = w.div(row_sums_final, axis=0) * target
    return w

def residualize_xs(returns, universe):
    mask = universe.astype(bool)
    mkt = returns.where(mask).mean(axis=1)
    return returns.sub(mkt, axis=0).where(mask)

# -------- Sleeve 1: existing 3-day MR (vwap-close) --------
def build_mr_sleeve(ohlcv, universe):
    vwap = (ohlcv['High'] + ohlcv['Low'] + ohlcv['Close']) / 3.0
    diff = vwap - ohlcv['Close']
    vd = ohlcv['Volume'].diff(3)
    rmax = diff.rolling(3).max().rank(axis=1, pct=True)
    rmin = diff.rolling(3).min().rank(axis=1, pct=True)
    rvd = vd.rank(axis=1, pct=True)
    alpha = (rmax + rmin) * rvd
    alpha = alpha.where(universe == 1, np.nan)
    alpha = alpha.rolling(3, min_periods=1).mean()

    pct_rank = alpha.rank(axis=1, pct=True)
    long_m = (pct_rank >= 0.90).astype(float)
    short_m = (pct_rank < 0.50).astype(float)
    nL = long_m.sum(axis=1).replace(0, np.nan)
    nS = short_m.sum(axis=1).replace(0, np.nan)
    w = long_m.div(nL, axis=0) * 0.5 + short_m.div(nS, axis=0) * -0.5
    w = w.fillna(0).shift(1)
    return enforce_post_shift_strict_gmv(w, universe)

# -------- Sleeve 2: sector-neutral 6m residual momentum --------
def build_mom_sleeve(ohlcv, returns, universe):
    import yfinance as yf
    spy_cache = Path("stores/spy_adj_close.parquet")
    if spy_cache.exists():
        spy = pd.read_parquet(spy_cache).iloc[:, 0]
    else:
        df = yf.download('SPY', start=returns.index.min(), end=returns.index.max() + pd.Timedelta(days=1), progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        spy = df['Adj Close'].squeeze() if 'Adj Close' in df.columns else df['Close'].squeeze()
        if spy.index.tz is not None: spy.index = spy.index.tz_localize(None)
        spy.to_frame("SPY").to_parquet(spy_cache)

    spy_rets = np.log(spy / spy.shift(1)).reindex(returns.index).fillna(0)
    spy_var = spy_rets.rolling(252, min_periods=63).var()
    cov = returns.rolling(252, min_periods=63).cov(spy_rets)
    beta = cov.div(spy_var, axis=0).shift(1)
    resid_ret = returns - beta.multiply(spy_rets, axis=0)

    # GK vol floor
    log_hl = np.log(ohlcv['High'] / ohlcv['Low']) ** 2
    log_co = np.log(ohlcv['Close'] / ohlcv['Open']) ** 2
    gk = 0.5 * log_hl - ((2*np.log(2)-1) * log_co)
    vol_floor = np.sqrt(gk.ewm(com=60, min_periods=60).mean() * 252).clip(lower=0.15)

    meta = pd.read_csv('top_5000_us_by_marketcap.csv')
    meta['symbol'] = meta['symbol'].str.replace('/', '-')
    sec_map = meta.set_index('symbol')['sector'].to_dict()
    sectors = pd.Series(returns.columns).map(sec_map).values
    ALPHA_SEC = {'Technology', 'Energy', 'Consumer Discretionary', 'Consumer Staples',
                 'Basic Materials', 'Industrials', 'Real Estate', 'Telecommunications'}

    sig_6m = resid_ret.shift(21).rolling(105).sum()
    sig_6m = sig_6m * universe.replace(0, np.nan)

    ranks = pd.DataFrame(np.nan, index=sig_6m.index, columns=sig_6m.columns)
    for sec in ALPHA_SEC:
        cols = sig_6m.columns[sectors == sec]
        if len(cols): ranks[cols] = sig_6m[cols].rank(axis=1, pct=True)

    LE, LX, SE, SX = 0.90, 0.85, 0.20, 0.25
    univ_drop = (universe.reindex_like(ranks).fillna(0) == 0)

    ls_state = pd.DataFrame(np.nan, index=ranks.index, columns=ranks.columns)
    ls_state[ranks >= LE] = 1; ls_state[ranks < LX] = -1; ls_state[univ_drop] = -1
    long_sig = ls_state.ffill().replace(-1, 0).fillna(0)

    ss_state = pd.DataFrame(np.nan, index=ranks.index, columns=ranks.columns)
    ss_state[ranks <= SE] = 1; ss_state[ranks > SX] = -1; ss_state[univ_drop] = -1
    short_sig = ss_state.ffill().replace(-1, 0).fillna(0) * -1

    sig = (long_sig + short_sig).fillna(0)
    raw = sig * (0.40 / vol_floor) * universe
    weekly = raw.resample('W-FRI').last()

    longs = normalize_and_cap(weekly.where(weekly > 0, 0))
    shorts = normalize_and_cap(weekly.where(weekly < 0, 0).abs())
    port = (longs - shorts).reindex(returns.index).ffill().shift(1)
    port = port.reindex(columns=returns.columns, fill_value=0).fillna(0)

    longs_f = normalize_and_cap(port.where(port > 0, 0))
    shorts_f = normalize_and_cap(port.where(port < 0, 0).abs())
    port_f = (longs_f - shorts_f).fillna(0)
    return enforce_post_shift_strict_gmv(port_f, universe)

# -------- Sleeve 3: Stat-Arb (from cache) --------
def load_sa_sleeve(returns, universe):
    cache = Path("weights_sa_cache.pkl")
    if not cache.exists():
        print(f"  [warn] {cache} missing — stat-arb sleeve will be zeros")
        return pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    w = pd.read_pickle(cache)
    w = w.reindex(index=returns.index, columns=returns.columns, fill_value=0).fillna(0)
    return enforce_post_shift_strict_gmv(w, universe)

# -------- Sleeve 4 (NEW): uniform k=5 trailing residual MR --------
def build_residual_mr_sleeve(returns, universe, k=5):
    r_resid = residualize_xs(returns, universe)
    trailing = r_resid.rolling(k, min_periods=max(2, k//2)).sum().shift(1)
    alpha = (-trailing).where(universe == 1, np.nan)
    w_raw = alpha_to_weights(alpha, universe)  # top-decile long, bottom-decile short
    w = w_raw.shift(1)
    return enforce_post_shift_strict_gmv(w, universe)

# -------- Inverse-vol blender (matches master pipeline) --------
def inverse_vol_blend(sleeves, returns, universe, window=60):
    """sleeves: dict of name -> weights DataFrame."""
    rolling_vols = {}
    for name, w in sleeves.items():
        aligned = returns.reindex(index=w.index, columns=w.columns).fillna(0)
        pnl = (w * aligned).sum(axis=1)
        rv = pnl.rolling(window, min_periods=20).std() * np.sqrt(252)
        rolling_vols[name] = rv.clip(lower=0.05)

    inv = {n: 1.0 / v for n, v in rolling_vols.items()}
    tot = sum(inv.values())
    alloc = {n: (inv[n] / tot).shift(1).fillna(1.0/len(sleeves)) for n in sleeves}

    blended = sum(sleeves[n].multiply(alloc[n], axis=0) for n in sleeves)

    longs = blended.where(blended > 0, 0)
    shorts = blended.where(blended < 0, 0).abs()
    longs_n = normalize_and_cap(longs)
    shorts_n = normalize_and_cap(shorts)
    final = (longs_n - shorts_n).fillna(0)
    return enforce_post_shift_strict_gmv(final, universe), alloc

# -------- Reporting --------
def report_backtest(weights, returns, universe, label, oos_idx=None):
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    if oos_idx is not None:
        weights = weights.loc[oos_idx]; returns_e = returns.loc[oos_idx]; univ_e = universe.loc[oos_idx]
    else:
        returns_e = returns; univ_e = universe
    weights = weights.reindex(columns=returns_e.columns, fill_value=0)
    univ_e = univ_e.reindex(columns=returns_e.columns, fill_value=0)
    sr, pnl = utils.backtest_portfolio(weights, returns_e, univ_e, plot_=False, print_=True)
    yr = pnl.groupby(pnl.index.year).agg(
        ann_ret=lambda s: s.mean()*252,
        ann_vol=lambda s: s.std()*np.sqrt(252),
        sharpe=lambda s: (s.mean()/s.std()*np.sqrt(252)) if s.std() > 0 else 0,
        n="count",
    )
    print("\n[Per-year]")
    print(yr.round(3).to_string())
    eq = (1+pnl).cumprod(); dd = eq/eq.cummax() - 1
    print(f"[summary] net_SR={sr}  max_DD={dd.min()*100:.1f}%  hit_days={(pnl>0).mean()*100:.1f}%")
    return {"net_sr": float(sr), "max_dd": float(dd.min()*100),
            "hit_days": float((pnl>0).mean()*100), "pnl": pnl}

# -------- Main --------
def main():
    t0 = time.time()
    ohlcv = load_data()
    universe = build_universe(ohlcv)
    returns = build_returns(ohlcv)
    universe = universe.reindex(columns=returns.columns).fillna(0).astype(int)
    print(f"[data] dates {returns.index.min().date()}..{returns.index.max().date()}  stocks={returns.shape[1]}")

    print("\n[sleeve 1] mean-reversion (3d vwap-close) ...")
    t1=time.time(); w_mr = build_mr_sleeve(ohlcv, universe); print(f"  done {time.time()-t1:.1f}s")

    print("[sleeve 2] sector-neutral 6m momentum ...")
    t1=time.time(); w_mom = build_mom_sleeve(ohlcv, returns, universe); print(f"  done {time.time()-t1:.1f}s")

    print("[sleeve 3] stat-arb kalman (from cache) ...")
    t1=time.time(); w_sa = load_sa_sleeve(returns, universe); print(f"  done {time.time()-t1:.1f}s")

    print("[sleeve 4 NEW] uniform k=5 trailing residual MR ...")
    t1=time.time(); w_res = build_residual_mr_sleeve(returns, universe, k=5); print(f"  done {time.time()-t1:.1f}s")

    # OOS window
    oos_start = returns.index.max() - pd.DateOffset(years=OOS_YEARS)
    oos_idx = returns.index[returns.index >= oos_start]
    print(f"\n[OOS] {oos_idx.min().date()}..{oos_idx.max().date()}  ({len(oos_idx)} days)")

    # ---- Individual sleeve OOS performance ----
    print("\n\n" + "#"*70)
    print("# STANDALONE SLEEVE OOS")
    print("#"*70)
    standalone = {}
    standalone["MR (3d vwap)"]    = report_backtest(w_mr,  returns, universe, "MR (3d vwap-close)",        oos_idx)
    standalone["Momentum"]        = report_backtest(w_mom, returns, universe, "Sector-Neutral Momentum",   oos_idx)
    standalone["StatArb"]         = report_backtest(w_sa,  returns, universe, "Stat-Arb Kalman",           oos_idx)
    standalone["ResMR k=5 (NEW)"] = report_backtest(w_res, returns, universe, "Residual MR k=5 (NEW)",     oos_idx)

    # ---- Baseline 3-sleeve ensemble ----
    print("\n\n" + "#"*70)
    print("# ENSEMBLE A — BASELINE (MR + Momentum + StatArb)")
    print("#"*70)
    sleeves_3 = {"MR": w_mr, "Mom": w_mom, "SA": w_sa}
    ens_3, _ = inverse_vol_blend(sleeves_3, returns, universe)
    res_3 = report_backtest(ens_3, returns, universe, "BASELINE 3-SLEEVE ENSEMBLE", oos_idx)

    # ---- New 4-sleeve ensemble ----
    print("\n\n" + "#"*70)
    print("# ENSEMBLE B — WITH NEW SLEEVE (MR + Momentum + StatArb + ResMR)")
    print("#"*70)
    sleeves_4 = {"MR": w_mr, "Mom": w_mom, "SA": w_sa, "ResMR": w_res}
    ens_4, alloc_4 = inverse_vol_blend(sleeves_4, returns, universe)
    res_4 = report_backtest(ens_4, returns, universe, "NEW 4-SLEEVE ENSEMBLE", oos_idx)

    # ---- Allocation snapshot for new sleeve ----
    alloc_resmr_oos = alloc_4["ResMR"].loc[oos_idx].dropna()
    print(f"\n[OOS allocation to ResMR]  mean={alloc_resmr_oos.mean()*100:.1f}%  "
          f"min={alloc_resmr_oos.min()*100:.1f}%  max={alloc_resmr_oos.max()*100:.1f}%")

    # ---- Pairwise PnL correlations ----
    print("\n\n" + "#"*70)
    print("# PAIRWISE OOS PnL CORRELATIONS (standalone sleeves)")
    print("#"*70)
    names = list(standalone.keys())
    pnl_df = pd.DataFrame({n: standalone[n]["pnl"] for n in names})
    print(pnl_df.corr().round(3).to_string())

    # ---- Head-to-head ----
    print("\n\n" + "#"*70)
    print("# HEAD-TO-HEAD: 3-sleeve vs 4-sleeve")
    print("#"*70)
    print(f"  Baseline 3-sleeve  : net_SR={res_3['net_sr']:.3f}  max_DD={res_3['max_dd']:.2f}%  hit_days={res_3['hit_days']:.1f}%")
    print(f"  New 4-sleeve       : net_SR={res_4['net_sr']:.3f}  max_DD={res_4['max_dd']:.2f}%  hit_days={res_4['hit_days']:.1f}%")
    print(f"  delta              : SR {res_4['net_sr']-res_3['net_sr']:+.3f}  "
          f"DD {res_4['max_dd']-res_3['max_dd']:+.2f}pp")

    pnl_3 = res_3["pnl"]; pnl_4 = res_4["pnl"]
    print(f"\n[ensemble PnL daily corr]  3-sleeve vs 4-sleeve: {pnl_3.corr(pnl_4):.3f}")
    diff = pnl_4 - pnl_3
    print(f"[difference series]  mean_ann={diff.mean()*252*100:.2f}%  "
          f"vol_ann={diff.std()*np.sqrt(252)*100:.2f}%  "
          f"info_ratio_vs_3sleeve={diff.mean()/diff.std()*np.sqrt(252):.2f}")

    # Save
    out = {
        "oos": [str(oos_idx.min().date()), str(oos_idx.max().date())],
        "standalone": {n: {k: v for k, v in d.items() if k != "pnl"} for n, d in standalone.items()},
        "baseline_3sleeve": {k: v for k, v in res_3.items() if k != "pnl"},
        "new_4sleeve": {k: v for k, v in res_4.items() if k != "pnl"},
        "delta_sr": float(res_4["net_sr"] - res_3["net_sr"]),
        "delta_dd": float(res_4["max_dd"] - res_3["max_dd"]),
        "alloc_resmr_mean": float(alloc_resmr_oos.mean()),
        "pairwise_corr": pnl_df.corr().round(3).to_dict(),
    }
    (ART / "summary.json").write_text(json.dumps(out, indent=2, default=str))
    pd.DataFrame({"baseline": pnl_3, "new4": pnl_4}).to_csv(ART / "ensemble_pnl.csv")
    print(f"\n[saved] {ART}/summary.json, ensemble_pnl.csv")
    print(f"[total runtime: {time.time()-t0:.1f}s]")

if __name__ == "__main__":
    main()
