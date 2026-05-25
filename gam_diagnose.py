"""GAM signal — full reproduction, signal quality audit, portfolio diagnosis.

  1. Build the 3 features (vol_z, rsi_rank, rel_vol) and the target
     (5d beta-residual forward log return, z-scored) — matches the notebook.
  2. Train GAM on 2010-2020, predict on 2021+ (≥5yr OOS).
  3. Diagnose signal quality with the same harness as squeeze/FFT:
       IC, decile, decay, by-year, t-stat, hit-rate.
  4. Build THREE portfolios from the same prediction and compare:
       (a) User's original conviction-hysteresis (replicate the notebook)
       (b) Continuous tilt:    weights ∝ rank(pred) - 0.5, vol-scaled, normalized
       (c) Percentile bucket:  top-decile long / bottom-decile short, equal-weighted
     This isolates whether the loss is from the signal or the portfolio mapping.
"""
from __future__ import annotations
import sys, warnings, time, json
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.append(str(Path("phase2_qrt_challenge/scripts").resolve()))
import utils  # noqa

from squeeze_breakout_pipeline import (
    load_data, build_universe, build_returns,
    daily_ic, ic_by_year, ic_decay, decile_spread,
    alpha_to_weights, enforce_post_shift_strict_gmv,
)

ART = Path("stores/gam"); ART.mkdir(exist_ok=True, parents=True)
CACHE_PRED = ART / "predictions.pkl"
CACHE_TARGET = ART / "target.pkl"

H = 5             # forward horizon (days)
BETA_WIN = 250
ZSCORE_CLIP = 3.0
TRAIN_END_YEAR = 2020   # last YEAR of IS
OOS_START_YEAR = 2021

# ---------- Feature engineering ----------
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

def build_target(ohlcv, universe, log_ret, returns):
    print("[target] building 5d beta-residual fwd log return ...")
    C = ohlcv["Adj Close"]
    fwd_log = np.log(C.shift(-H) / C)

    # Use cached SPY series
    spy_path = Path("stores/spy_adj_close.parquet")
    if spy_path.exists():
        spy = pd.read_parquet(spy_path).iloc[:, 0]
        if spy.index.tz is not None:
            spy.index = spy.index.tz_localize(None)
    else:
        import yfinance as yf
        df = yf.download('SPY', start=returns.index.min(), end=returns.index.max()+pd.Timedelta(days=10), progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        spy = df['Adj Close'].squeeze() if 'Adj Close' in df.columns else df['Close'].squeeze()
        if spy.index.tz is not None: spy.index = spy.index.tz_localize(None)
    spy = spy.reindex(returns.index).ffill()

    mkt = np.log(spy / spy.shift(1))
    mkt_var = mkt.rolling(BETA_WIN, min_periods=120).var()
    cov = log_ret.where(universe==1).rolling(BETA_WIN, min_periods=120).cov(mkt)
    raw_beta = cov.div(mkt_var, axis=0)
    beta = 0.2 + 0.8 * raw_beta   # the notebook's shrinkage

    fwd_mkt = mkt.rolling(H).sum().shift(-H)
    resid_fwd = fwd_log.sub(beta.mul(fwd_mkt, axis=0))
    resid_m = resid_fwd.where(universe == 1)
    mu = resid_m.mean(axis=1); sd = resid_m.std(axis=1)
    Y = resid_m.sub(mu, axis=0).div(sd, axis=0).clip(-ZSCORE_CLIP, ZSCORE_CLIP)
    return Y, beta

# ---------- GAM training ----------
def train_gam(vol_z, rsi_rank, rel_vol, Y):
    from pygam import LinearGAM, s, te

    print("[GAM] stacking train data ...")
    parts = {"vol": vol_z, "rsi": rsi_rank, "rel": rel_vol, "y": Y}
    df = pd.concat({k: v.stack() for k, v in parts.items()}, axis=1).dropna()
    df.index.names = ["date", "ticker"]
    train_mask = df.index.get_level_values("date").year <= TRAIN_END_YEAR
    train_df = df[train_mask]
    print(f"[GAM] train samples raw: {len(train_df)}")

    MAX = 500_000
    if len(train_df) > MAX:
        rs = np.random.default_rng(42)
        idx = rs.choice(len(train_df), MAX, replace=False)
        train_df = train_df.iloc[idx]
        print(f"[GAM] subsampled to {MAX}")

    X = train_df[["vol", "rsi", "rel"]].values
    y = train_df["y"].values
    gam = LinearGAM(
        s(0, constraints='monotonic_dec', n_splines=10)
      + te(1, 2, n_splines=(8, 8))
    )
    lam_space = np.logspace(1, 5, 10)
    print("[GAM] gridsearch lam ...")
    t0 = time.time()
    gam.gridsearch(X, y, lam=lam_space, progress=False)
    print(f"[GAM] fit done in {time.time()-t0:.1f}s   lam={gam.lam}   pseudo_r2={gam.statistics_['pseudo_r2']['explained_deviance']:.5f}")
    return gam, df

def predict_all(gam, feature_df, universe):
    """Predict over the FULL panel; restore (date, ticker) MultiIndex; unstack to wide."""
    X = feature_df[["vol", "rsi", "rel"]].values
    pred = gam.predict(X)
    s = pd.Series(pred, index=feature_df.index, name="pred")
    pred_df = s.unstack()
    pred_df = pred_df.reindex(columns=universe.columns).where(universe == 1)
    return pred_df

# ---------- Diagnostics ----------
def diagnostics(pred_oos, returns, oos_idx, label):
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    # Notebook uses 5d forward; let's report BOTH T+1 (matching ensemble convention) and T+5 (matching target)
    for h_lbl, fwd in [("T+1", returns.shift(-1)), ("T+5 (sum)", returns.shift(-1).rolling(5).sum().shift(-(5-1)))]:
        ic = daily_ic(pred_oos.reindex(oos_idx), fwd.loc[oos_idx])
        print(f"\n[IC {h_lbl}] N={len(ic)}  mean={ic.mean():.4f}  median={ic.median():.4f}  "
              f"IR_ann={ic.mean()/ic.std()*np.sqrt(252):.2f}  %>0={(ic>0).mean()*100:.1f}%  "
              f"t={ic.mean()/(ic.std()/np.sqrt(len(ic))):.2f}")
    fwd1 = returns.shift(-1).loc[oos_idx]
    print("\n[IC by year (T+1)]")
    ic = daily_ic(pred_oos.reindex(oos_idx), fwd1)
    print(ic_by_year(ic).round(4).to_string())
    print("\n[IC decay]")
    print(ic_decay(pred_oos.reindex(oos_idx), returns.loc[oos_idx]).round(4).to_string(index=False))
    print("\n[Decile spread (bps fwd T+1)]")
    dec = decile_spread(pred_oos.reindex(oos_idx), fwd1) * 1e4
    print(dec.round(2).to_string())
    ls = dec.iloc[-1] - dec.iloc[0]
    mono = dec.is_monotonic_increasing or dec.is_monotonic_decreasing
    corr_r = pd.Series(range(1,11)).corr(dec.reset_index(drop=True))
    print(f"  L-S (D10-D1): {ls:.2f} bps/d  ~  {ls*252/100:.1f}%/yr gross")
    print(f"  monotonic: {mono}   corr(decile, ret): {corr_r:.3f}")
    return {"ic_mean_T1": float(ic.mean()), "ic_IR_T1": float(ic.mean()/ic.std()*np.sqrt(252)),
            "ls_bps_T1": float(ls), "monotonic": bool(mono), "decile_corr_T1": float(corr_r)}

# ---------- Portfolio constructions ----------
def portfolio_bucket(alpha, universe, long_pct=0.90, short_pct=0.10):
    """Top decile long, bottom decile short, equal-weighted — master pipeline style."""
    rk = alpha.where(universe == 1, np.nan).rank(axis=1, pct=True)
    lm = (rk >= long_pct).astype(float)
    sm = (rk <= short_pct).astype(float)
    nL = lm.sum(axis=1).replace(0, np.nan)
    nS = sm.sum(axis=1).replace(0, np.nan)
    w = lm.div(nL, axis=0) * 0.5 + sm.div(nS, axis=0) * -0.5
    return enforce_post_shift_strict_gmv(w.fillna(0).shift(1), universe)

def portfolio_continuous(alpha, universe, vol_floor=0.005, log_ret=None):
    """Continuous rank tilt × inverse-vol weighting, dollar-neutral, capped."""
    a = alpha.where(universe == 1, np.nan)
    # cross-sectional demean -> rank centered around 0
    a_z = a.sub(a.mean(axis=1), axis=0).div(a.std(axis=1), axis=0)
    a_z = a_z.clip(-3, 3)
    if log_ret is not None:
        vol_20 = log_ret.rolling(20).std().clip(lower=vol_floor)
        a_z = a_z.div(vol_20)
    # raw -> split L/S, normalize each book to 0.5, cap 0.099
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

def portfolio_hysteresis(pred_df, universe, returns, log_ret, beta,
                         entry=2.4, exit_=0.25, vol_floor=0.005):
    """Replicate the notebook's conviction-hysteresis logic — vectorized where possible."""
    print(f"  hysteresis: entry={entry} exit={exit_}")
    dates = pred_df.dropna(how='all').index
    portfolio = pd.DataFrame(0.0, index=dates, columns=pred_df.columns)
    prev_signs = pd.Series(0.0, index=pred_df.columns)
    vol_20 = log_ret.rolling(20).std().clip(lower=vol_floor)

    for date in dates:
        alpha = pred_df.loc[date].dropna()
        if len(alpha) < 100: continue
        # beta-residualize cross-sectionally
        bt = beta.loc[date].reindex(alpha.index).dropna()
        common = alpha.index.intersection(bt.index)
        if len(common) < 100: continue
        a = alpha.loc[common].values; b = bt.loc[common].values
        m_b, m_a = b.mean(), a.mean()
        denom = ((b - m_b)**2).sum()
        if denom == 0: continue
        slope = ((b - m_b)*(a - m_a)).sum() / denom
        resid = a - (slope * b + (m_a - slope*m_b))
        sig = pd.Series(resid, index=common)
        # inverse-vol weight
        v = vol_20.loc[date].reindex(sig.index).dropna()
        sig = sig.loc[v.index] / v
        sig = sig.replace([np.inf, -np.inf], np.nan).dropna()
        z = (sig - sig.mean()) / sig.std()
        z = z.clip(-8, 8)
        # hysteresis: new entries OR previous holds passing the lax exit threshold
        prev_aligned = prev_signs.reindex(z.index).fillna(0)
        pos_mask = (z > entry) | ((z > exit_) & (prev_aligned > 0))
        neg_mask = (z < -entry) | ((z < -exit_) & (prev_aligned < 0))
        pos = z[pos_mask]
        neg = z[neg_mask].abs()
        if len(pos) < 5 or len(neg) < 5:
            prev_signs = pd.Series(0.0, index=pred_df.columns)
            continue
        wp = (pos / pos.sum()) * 0.5
        wn = -(neg / neg.sum()) * 0.5
        # cap 0.099
        wp = wp.clip(upper=0.099); wn = wn.clip(lower=-0.099)
        # re-normalize to exactly 0.5 / -0.5
        wp = wp / wp.sum() * 0.5
        wn = wn / wn.sum() * 0.5 * (-1)  # already negative; rescale
        wn = wn.abs() / wn.abs().sum() * 0.5
        weights = pd.Series(0.0, index=pred_df.columns)
        weights.loc[wp.index] = wp
        weights.loc[wn.index] = -wn
        portfolio.loc[date] = weights.values
        prev_signs = pd.Series(np.sign(weights), index=pred_df.columns)

    return enforce_post_shift_strict_gmv(portfolio.shift(1), universe)

# ---------- Backtest reporter ----------
def report(weights, returns, universe, label):
    print(f"\n--- {label} ---")
    w = weights.reindex(columns=returns.columns, fill_value=0.0)
    u = universe.reindex(columns=returns.columns, fill_value=0)
    sr, pnl = utils.backtest_portfolio(w, returns, u, plot_=False, print_=True)
    eq = (1+pnl).cumprod(); dd = eq/eq.cummax()-1
    print(f"  max_DD={dd.min()*100:.2f}%  hit_days={(pnl>0).mean()*100:.1f}%")
    yr = pnl.groupby(pnl.index.year).agg(sharpe=lambda s: (s.mean()/s.std()*np.sqrt(252)) if s.std()>0 else 0).round(2)
    print(f"  per-year SR: " + "  ".join(f"{y}={v.iloc[0]:+.2f}" for y, v in yr.iterrows()))
    n_active = (w.abs().sum(axis=1) > 0.01).sum()
    print(f"  active trading days: {n_active}/{len(w)}")
    return {"net_sr": float(sr), "max_dd": float(dd.min()*100), "active_days": int(n_active), "pnl": pnl}

# ---------- Main ----------
def main():
    t0 = time.time()
    ohlcv = load_data()
    universe = build_universe(ohlcv)
    returns = build_returns(ohlcv)
    universe = universe.reindex(columns=returns.columns).fillna(0).astype(int)

    if CACHE_PRED.exists() and CACHE_TARGET.exists():
        print(f"[cache] loading predictions from {CACHE_PRED}")
        pred_df = pd.read_pickle(CACHE_PRED)
        target_info = pd.read_pickle(CACHE_TARGET)
        Y = target_info["Y"]; beta = target_info["beta"]; log_ret = target_info["log_ret"]
    else:
        vol_z, rsi_rank, rel_vol, log_ret = build_features(ohlcv, universe)
        Y, beta = build_target(ohlcv, universe, log_ret, returns)
        gam, full_df = train_gam(vol_z, rsi_rank, rel_vol, Y)
        # predict everywhere we have features
        pred_df = predict_all(gam, full_df, universe)
        pd.to_pickle(pred_df, CACHE_PRED)
        pd.to_pickle({"Y": Y, "beta": beta, "log_ret": log_ret}, CACHE_TARGET)

    # OOS window
    oos_idx = returns.index[returns.index.year >= OOS_START_YEAR]
    print(f"\n[OOS] {oos_idx.min().date()}..{oos_idx.max().date()}  ({len(oos_idx)} days)")

    # ---- Signal diagnostics ----
    diag = diagnostics(pred_df, returns, oos_idx, "GAM SIGNAL — OOS diagnostics")

    # ---- Three portfolio constructions ----
    print("\n\n" + "#"*70); print("# PORTFOLIO CONSTRUCTIONS"); print("#"*70)
    rets_oos = returns.loc[oos_idx]
    univ_oos = universe.loc[oos_idx]
    pred_oos = pred_df.reindex(oos_idx)

    # (a) Hysteresis (the notebook's approach)
    print("\n[portfolio a] hysteresis (entry=2.4 exit=0.25) — replicating notebook")
    w_a = portfolio_hysteresis(pred_oos, univ_oos, rets_oos, log_ret.loc[oos_idx], beta.loc[oos_idx])
    res_a = report(w_a, rets_oos, univ_oos, "(a) Hysteresis entry=2.4 exit=0.25")

    # (a2) Hysteresis with looser threshold
    print("\n[portfolio a2] hysteresis (entry=1.5 exit=0.25) — looser")
    w_a2 = portfolio_hysteresis(pred_oos, univ_oos, rets_oos, log_ret.loc[oos_idx], beta.loc[oos_idx], entry=1.5)
    res_a2 = report(w_a2, rets_oos, univ_oos, "(a2) Hysteresis entry=1.5 exit=0.25")

    # (b) Continuous tilt with inverse-vol
    print("\n[portfolio b] continuous tilt (z-score * inv-vol), capped")
    w_b = portfolio_continuous(pred_oos, univ_oos, log_ret=log_ret.loc[oos_idx])
    res_b = report(w_b, rets_oos, univ_oos, "(b) Continuous tilt, inv-vol")

    # (c) Percentile bucket (master pipeline convention)
    print("\n[portfolio c] percentile bucket (top10% long, bot10% short, EW)")
    w_c = portfolio_bucket(pred_oos, univ_oos)
    res_c = report(w_c, rets_oos, univ_oos, "(c) Percentile bucket (10/10)")

    # (c2) Bucket 20/20 (wider for less concentration)
    print("\n[portfolio c2] percentile bucket (top20% long, bot20% short, EW)")
    w_c2 = portfolio_bucket(pred_oos, univ_oos, long_pct=0.80, short_pct=0.20)
    res_c2 = report(w_c2, rets_oos, univ_oos, "(c2) Percentile bucket (20/20)")

    print("\n\n" + "="*70); print("HEADLINE — same signal, four portfolios"); print("="*70)
    rows = [
        ("(a)  hysteresis e=2.4 (notebook)", res_a),
        ("(a2) hysteresis e=1.5 (looser)  ", res_a2),
        ("(b)  continuous tilt + inv-vol  ", res_b),
        ("(c)  bucket 10/10 EW            ", res_c),
        ("(c2) bucket 20/20 EW            ", res_c2),
    ]
    print(f"  {'label':40s}  {'net_SR':>7s}  {'max_DD%':>7s}  {'active_d':>8s}")
    for lbl, r in rows:
        print(f"  {lbl}  {r['net_sr']:7.3f}  {r['max_dd']:7.2f}  {r['active_days']:8d}")

    # Save
    out = {
        "oos": [str(oos_idx.min().date()), str(oos_idx.max().date())],
        "diag": diag,
        "portfolios": {lbl: {k: v for k, v in r.items() if k != "pnl"} for lbl, r in rows},
    }
    (ART / "summary.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[saved] {ART}/summary.json")
    print(f"[total runtime: {time.time()-t0:.1f}s]")

if __name__ == "__main__":
    main()
