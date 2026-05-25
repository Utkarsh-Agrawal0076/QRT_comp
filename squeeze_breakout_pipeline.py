"""Intraday Volatility Breakout (squeeze) sleeve.

Pipeline:
  1. Load OHLCV, build returns + 5M ADV universe (matches master pipeline).
  2. Build Yang-Zhang variance, log-vol z-score squeeze, CLV, signed-volume anchor.
  3. Sample split: OOS = last 5 years, IS = everything before.
  4. IS hyperparameter grid -> pick by mean Spearman IC at T+1.
  5. OOS diagnostics: IC, IR, hit rate, IC by year, decay 1..10, decile spread.
  6. OOS backtest: percentile-bucket weights, T+1 lag, water-fill to GMV=1.
"""
from __future__ import annotations
import os, sys, warnings, itertools, time, json
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.append(str(Path("phase2_qrt_challenge/scripts").resolve()))
import utils  # noqa: E402

DATA_DIR = Path("stores"); DATA_DIR.mkdir(exist_ok=True)
ART_DIR = Path("stores/squeeze"); ART_DIR.mkdir(exist_ok=True, parents=True)

OOS_YEARS = 5
HORIZON = 1           # forward-return horizon for IC (matches T+1 execution)
MIN_STOCKS = 200      # min cross-section for daily IC

# ---------- 1. Data ----------
def load_data():
    cache = ART_DIR / "ohlcv_cache.pkl"
    if cache.exists():
        print(f"[data] loading cached OHLCV from {cache}")
        return pd.read_pickle(cache)
    print("[data] loading top_5000_yf_data.pkl ...")
    t0 = time.time()
    df = pd.read_pickle("top_5000_yf_data.pkl")
    print(f"  loaded in {time.time()-t0:.1f}s, shape={df.shape}")

    def _dedup(x): return x.loc[:, ~x.columns.duplicated()]
    out = {f: _dedup(df[f]) for f in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]}
    pd.to_pickle(out, cache)
    return out

def build_universe(ohlcv):
    print("[univ] building 5M ADV universe (60d)")
    dv = (ohlcv["Close"] * ohlcv["Volume"]).fillna(0)
    adv60 = dv.rolling(60, min_periods=60).mean()
    return (adv60 >= 5_000_000).astype(int)

def build_returns(ohlcv):
    r = ohlcv["Adj Close"].pct_change(fill_method=None).fillna(0)
    return r.loc[:, ~r.columns.duplicated()]

# ---------- 2. Feature engineering ----------
def yang_zhang_var(ohlcv, k=20):
    """Yang-Zhang variance — bias-free under overnight gaps."""
    O, H, L, C = ohlcv["Open"], ohlcv["High"], ohlcv["Low"], ohlcv["Close"]
    C_prev = C.shift(1)
    # Overnight, open->close, Rogers-Satchell components (log)
    log_oc_prev = np.log(O / C_prev)            # overnight
    log_co = np.log(C / O)                      # open->close
    rs = (np.log(H / C) * np.log(H / O) + np.log(L / C) * np.log(L / O))  # Rogers-Satchell

    n = k
    overnight_var = log_oc_prev.rolling(n).var()
    oc_var = log_co.rolling(n).var()
    rs_var = rs.rolling(n).mean()
    kk = 0.34 / (1.34 + (n + 1) / (n - 1))
    yz_var = overnight_var + kk * oc_var + (1 - kk) * rs_var
    return yz_var

def squeeze_z(yz_var, k=20):
    """Log-variance z-score, lag-1 (no self-overlap)."""
    log_v = np.log(yz_var.clip(lower=1e-12))
    mu = log_v.shift(1).rolling(k, min_periods=k//2).mean()
    sd = log_v.shift(1).rolling(k, min_periods=k//2).std()
    z = (log_v.shift(1) - mu) / sd.replace(0, np.nan)
    return z  # large negative = compressed

def clv(ohlcv, m=3):
    H, L, C = ohlcv["High"], ohlcv["Low"], ohlcv["Close"]
    rng = (H - L).replace(0, np.nan)
    raw = ((C - L) - (H - C)) / rng
    return raw.rolling(m, min_periods=1).mean()

def signed_volume(ohlcv, m=5):
    """sign(C - 5d VWAP) * normalized volume."""
    H, L, C, V = ohlcv["High"], ohlcv["Low"], ohlcv["Close"], ohlcv["Volume"]
    tp = (H + L + C) / 3.0
    pv = (tp * V).rolling(m).sum()
    vv = V.rolling(m).sum().replace(0, np.nan)
    vwap = pv / vv
    sgn = np.sign(C - vwap)
    # volume z relative to its own 20d mean
    vz = (V - V.rolling(20).mean()) / V.rolling(20).std().replace(0, np.nan)
    return sgn * vz.clip(-3, 3)

def build_alpha(ohlcv, universe, k=20, m=3, tau=1.0, require_anchor_agreement=True):
    """Squeeze-weighted directional alpha.

      z       = log-vol z-score (lag-1)        : compressed when z << 0
      weight  = exp(-max(z + tau, 0))           : near 1 when z<-tau, decays to 0
      anchor  = clv * sign agrees with signed_volume? clv : 0
      alpha   = weight * anchor                 : in [-1, 1] approx
    """
    yz = yang_zhang_var(ohlcv, k=k)
    z = squeeze_z(yz, k=k)
    c = clv(ohlcv, m=m).shift(1)
    sv = signed_volume(ohlcv).shift(1)

    w = np.exp(-np.clip(z + tau, 0, None))     # squeezed names get full weight
    if require_anchor_agreement:
        agree = (np.sign(c) == np.sign(sv)) & (sv.abs() > 0) & (c.abs() > 0)
        anchor = c.where(agree, 0.0)
    else:
        anchor = c

    alpha = w * anchor
    return alpha.where(universe == 1, np.nan)

# ---------- 3. Diagnostics ----------
def daily_ic(signal, fwd, min_stocks=MIN_STOCKS):
    common = signal.dropna(how="all").index.intersection(fwd.dropna(how="all").index)
    ics, dates = [], []
    for dt in common:
        s = signal.loc[dt].dropna()
        f = fwd.loc[dt].reindex(s.index).dropna()
        if len(f) < min_stocks:
            continue
        cs = s.index.intersection(f.index)
        ics.append(s[cs].rank().corr(f[cs].rank()))
        dates.append(dt)
    return pd.Series(ics, index=pd.DatetimeIndex(dates), name="ic")

def ic_decay(signal, returns, lags=(1,2,3,5,10)):
    rows = []
    for L in lags:
        fwd = returns.shift(-L)
        ic = daily_ic(signal, fwd)
        if len(ic) == 0:
            continue
        rows.append({"lag": L, "mean_IC": ic.mean(), "IR_ann": ic.mean()/ic.std()*np.sqrt(252),
                     "pct_pos": (ic > 0).mean(), "n_days": len(ic)})
    return pd.DataFrame(rows)

def decile_spread(signal, fwd, n_bins=10):
    rks = signal.rank(axis=1, pct=True)
    out = {}
    for d in range(1, n_bins + 1):
        lo, hi = (d-1)/n_bins, d/n_bins
        mask = (rks >= lo) & (rks < hi) if d < n_bins else (rks >= lo)
        out[d] = fwd.where(mask).stack().mean()
    return pd.Series(out, name="avg_fwd_ret")

def ic_by_year(ic):
    g = ic.groupby(ic.index.year).agg(["mean", "std", "count"])
    g.columns = ["mean_IC", "std_IC", "n_days"]
    g["IR_ann"] = g["mean_IC"] / g["std_IC"] * np.sqrt(252)
    g["t_stat"] = g["mean_IC"] / (g["std_IC"] / np.sqrt(g["n_days"]))
    return g

# ---------- 4. Portfolio construction ----------
def alpha_to_weights(alpha, universe, long_pct=0.90, short_pct=0.10, gmv=1.0):
    """Top decile long, bottom decile short, equal-weighted, dollar-neutral."""
    masked = alpha.where(universe == 1, np.nan)
    rk = masked.rank(axis=1, pct=True)
    long_m = (rk >= long_pct).astype(float)
    short_m = (rk <= short_pct).astype(float)

    nL = long_m.sum(axis=1).replace(0, np.nan)
    nS = short_m.sum(axis=1).replace(0, np.nan)
    wL = long_m.div(nL, axis=0) * (gmv / 2)
    wS = short_m.div(nS, axis=0) * (-gmv / 2)
    w = (wL.fillna(0) + wS.fillna(0))
    return w

def enforce_post_shift_strict_gmv(shifted_portfolio, universe_df, max_weight=0.098):
    portfolio = shifted_portfolio * universe_df
    abs_sum = portfolio.abs().sum(axis=1)
    for date, gmv in abs_sum.items():
        if gmv <= 1e-8:
            continue
        row = (portfolio.loc[date] / gmv).copy()
        for _ in range(20):
            if row.abs().max() <= max_weight + 1e-6:
                break
            cap_mask = row.abs() > max_weight
            row[cap_mask] = np.sign(row[cap_mask]) * max_weight
            rem = 1.0 - row[cap_mask].abs().sum()
            uncap_sum = row[~cap_mask].abs().sum()
            if uncap_sum > 1e-8:
                row[~cap_mask] *= rem / uncap_sum
            else:
                break
        if abs(row.abs().sum() - 1.0) > 0.01 or row.abs().max() > 0.1001:
            portfolio.loc[date] = 0.0
        else:
            portfolio.loc[date] = row
    return portfolio

# ---------- 5. Main ----------
def main():
    ohlcv = load_data()
    universe = build_universe(ohlcv)
    returns = build_returns(ohlcv)

    universe = universe.reindex(columns=returns.columns).fillna(0).astype(int)
    print(f"[data] dates {returns.index.min().date()} .. {returns.index.max().date()}  "
          f"stocks={returns.shape[1]}")

    # ---- sample split ----
    oos_start = returns.index.max() - pd.DateOffset(years=OOS_YEARS)
    is_idx = returns.index[returns.index < oos_start]
    oos_idx = returns.index[returns.index >= oos_start]
    print(f"[split] IS: {is_idx.min().date()} .. {is_idx.max().date()}  ({len(is_idx)} days)")
    print(f"[split] OOS: {oos_idx.min().date()} .. {oos_idx.max().date()}  ({len(oos_idx)} days)")

    # ---- IS grid search ----
    grid = list(itertools.product(
        [15, 20, 30],          # k
        [3, 5],                # m
        [0.5, 1.0, 1.5],       # tau
        [True, False],         # anchor agreement
    ))
    print(f"[grid] evaluating {len(grid)} configs on IS")

    is_rets = returns.loc[is_idx]
    is_univ = universe.loc[is_idx]
    fwd_is = is_rets.shift(-HORIZON)

    rows = []
    for (k, m, tau, agr) in grid:
        a = build_alpha(ohlcv, universe, k=k, m=m, tau=tau, require_anchor_agreement=agr)
        a_is = a.reindex(is_idx)
        ic = daily_ic(a_is, fwd_is)
        if len(ic) < 100:
            continue
        rows.append({
            "k": k, "m": m, "tau": tau, "agree": agr,
            "n_days": len(ic),
            "mean_IC": ic.mean(),
            "IR_ann": ic.mean() / ic.std() * np.sqrt(252),
            "pct_pos": (ic > 0).mean(),
            "t_stat": ic.mean() / (ic.std() / np.sqrt(len(ic))),
        })
    grid_df = pd.DataFrame(rows).sort_values("mean_IC", ascending=False)
    grid_df.to_csv(ART_DIR / "is_grid.csv", index=False)
    print("\n[IS grid — top 8 by mean IC]")
    print(grid_df.head(8).round(4).to_string(index=False))

    best = grid_df.iloc[0]
    K, M, TAU, AGR = int(best.k), int(best.m), float(best.tau), bool(best.agree)
    print(f"\n[IS pick] k={K}  m={M}  tau={TAU}  agree={AGR}  "
          f"IS_IC={best.mean_IC:.4f}  IR={best.IR_ann:.2f}")

    # ---- OOS diagnostics ----
    print("\n[OOS] building alpha on full sample, slicing to OOS window")
    alpha = build_alpha(ohlcv, universe, k=K, m=M, tau=TAU, require_anchor_agreement=AGR)
    alpha_oos = alpha.reindex(oos_idx)
    oos_rets = returns.loc[oos_idx]
    oos_univ = universe.loc[oos_idx]

    fwd_oos = oos_rets.shift(-HORIZON)
    ic_oos = daily_ic(alpha_oos, fwd_oos)
    print(f"\n[OOS IC, lag=T+{HORIZON}]")
    print(f"  N days       : {len(ic_oos)}")
    print(f"  Mean IC      : {ic_oos.mean():.4f}")
    print(f"  Median IC    : {ic_oos.median():.4f}")
    print(f"  Std IC       : {ic_oos.std():.4f}")
    print(f"  IR (annual)  : {ic_oos.mean()/ic_oos.std()*np.sqrt(252):.2f}")
    print(f"  % days > 0   : {(ic_oos > 0).mean()*100:.1f}%")
    print(f"  t-stat       : {ic_oos.mean()/(ic_oos.std()/np.sqrt(len(ic_oos))):.2f}")

    print("\n[OOS IC by year]")
    print(ic_by_year(ic_oos).round(4).to_string())

    print("\n[OOS IC decay]")
    print(ic_decay(alpha_oos, oos_rets).round(4).to_string(index=False))

    print("\n[OOS decile spread, fwd T+1 return]")
    dec = decile_spread(alpha_oos, fwd_oos)
    print((dec * 1e4).round(2).rename("bps").to_string())
    ls = (dec.iloc[-1] - dec.iloc[0]) * 1e4
    print(f"  L-S spread (D10-D1): {ls:.2f} bps/day  ~  {ls*252/100:.1f}%/yr gross")

    # Hit rate per decile (% days the decile beat the cross-sectional mean)
    print("\n[OOS hit rate per decile (vs cross-sectional mean, daily)]")
    rk = alpha_oos.rank(axis=1, pct=True)
    csmean = fwd_oos.mean(axis=1)
    hits = {}
    for d in range(1, 11):
        lo, hi = (d-1)/10, d/10
        mask = (rk >= lo) & (rk < hi) if d < 10 else (rk >= lo)
        dec_ret = fwd_oos.where(mask).mean(axis=1)
        hits[d] = (dec_ret > csmean).mean()
    print(pd.Series(hits, name="hit_rate").round(3).to_string())

    # ---- OOS backtest ----
    print("\n[OOS backtest] building portfolio (top-decile long, bottom-decile short)")
    w_raw = alpha_to_weights(alpha_oos, oos_univ)
    w_shifted = w_raw.shift(1)
    w_final = enforce_post_shift_strict_gmv(w_shifted, oos_univ)

    # align cols & shape (utils.backtest_portfolio requires shape match)
    w_final = w_final.reindex(columns=oos_rets.columns, fill_value=0.0)
    oos_univ_b = oos_univ.reindex(columns=oos_rets.columns, fill_value=0)
    print(f"  shapes: weights={w_final.shape} returns={oos_rets.shape} universe={oos_univ_b.shape}")

    print("\n=== OOS BACKTEST ===")
    sr, pnl = utils.backtest_portfolio(w_final, oos_rets, oos_univ_b, plot_=False, print_=True)

    # Per-year breakdown
    yr = pnl.groupby(pnl.index.year).agg(
        mean_pnl=lambda s: s.mean()*252,
        vol_ann=lambda s: s.std()*np.sqrt(252),
        sharpe=lambda s: (s.mean()/s.std()*np.sqrt(252)) if s.std() > 0 else 0,
        n_days="count",
    )
    print("\n[OOS PnL by year (gross)]")
    print(yr.round(3).to_string())

    # Drawdown
    eq = (1 + pnl).cumprod()
    dd = eq / eq.cummax() - 1
    print(f"\n[OOS] gross Sharpe={sr}  max DD={dd.min()*100:.1f}%  hit_rate_days={(pnl>0).mean()*100:.1f}%")

    # Save artifacts
    out = {
        "is_window": [str(is_idx.min().date()), str(is_idx.max().date())],
        "oos_window": [str(oos_idx.min().date()), str(oos_idx.max().date())],
        "best_config": {"k": K, "m": M, "tau": TAU, "agree": AGR},
        "is_mean_IC": float(best.mean_IC),
        "oos_mean_IC": float(ic_oos.mean()),
        "oos_IR": float(ic_oos.mean()/ic_oos.std()*np.sqrt(252)),
        "oos_decile_spread_bps_per_day": float(ls),
        "oos_net_sharpe": float(sr),
        "oos_max_dd_pct": float(dd.min()*100),
    }
    (ART_DIR / "summary.json").write_text(json.dumps(out, indent=2))
    pnl.to_csv(ART_DIR / "oos_pnl.csv")
    ic_oos.to_csv(ART_DIR / "oos_ic.csv")
    print(f"\n[saved] {ART_DIR}/summary.json, oos_pnl.csv, oos_ic.csv, is_grid.csv")

if __name__ == "__main__":
    main()
