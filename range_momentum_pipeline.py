"""Risk-Adjusted Momentum / Volume-Anomaly / Range-Percentile sleeve — Stage 1.

Tests the proposed 3-factor strategy as a cross-sectional sleeve, comparable to
the production sleeves. STAGE 1 only computes raw factor diagnostics (no model)
so we can decide whether the factors carry signal before choosing a target/model.

Factors (computed through day t, cross-sectionally winsorized at 2/98 pct daily):
  f1  risk-adjusted momentum : (P_t/P_{t-14} - 1) / std_14(daily ret)
  f2  volume anomaly         : (V_t - SMA(V,20)) / StdDev(V,20)
  f3  range percentile       : (P_t - min(P,325)) / (max(P,325) - min(P,325))
  fC  combined               : equal-weight sum of cross-sectional z-scores

Diagnostics per factor (IS + OOS): mean IC, IR, IC decay across lags, decile
spread on T+1 fwd returns, IC by year.
"""
from __future__ import annotations
import sys, warnings, time, json
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.append(str(Path("phase2_qrt_challenge/scripts").resolve()))
import utils  # noqa: E402

ART_DIR = Path("stores/range_mom"); ART_DIR.mkdir(exist_ok=True, parents=True)
SQUEEZE_CACHE = Path("stores/squeeze/ohlcv_cache.pkl")

OOS_YEARS = 5
HORIZON = 1
MIN_STOCKS = 200
MOM_LB = 14
VOL_LB = 20
RANGE_LB = 325


# ---------- 1. Data ----------
def load_data():
    if SQUEEZE_CACHE.exists():
        print(f"[data] loading cached OHLCV from {SQUEEZE_CACHE}")
        return pd.read_pickle(SQUEEZE_CACHE)
    print("[data] loading top_5000_yf_data.pkl ...")
    t0 = time.time()
    df = pd.read_pickle("top_5000_yf_data.pkl")
    print(f"  loaded in {time.time()-t0:.1f}s, shape={df.shape}")
    def _dedup(x): return x.loc[:, ~x.columns.duplicated()]
    out = {f: _dedup(df[f]) for f in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]}
    pd.to_pickle(out, SQUEEZE_CACHE)
    return out


def build_universe(ohlcv):
    print("[univ] building 5M ADV universe (60d)")
    dv = (ohlcv["Close"] * ohlcv["Volume"]).fillna(0)
    adv60 = dv.rolling(60, min_periods=60).mean()
    return (adv60 >= 5_000_000).astype(int)


def build_returns(ohlcv):
    r = ohlcv["Adj Close"].pct_change(fill_method=None).fillna(0)
    return r.loc[:, ~r.columns.duplicated()]


# ---------- 2. Factors ----------
def winsorize_xs(df, lo=0.02, hi=0.98):
    """Cross-sectional (per-row) winsorization at lo/hi percentiles."""
    qlo = df.quantile(lo, axis=1)
    qhi = df.quantile(hi, axis=1)
    return df.clip(lower=qlo, upper=qhi, axis=0)


def f1_risk_adj_momentum(ohlcv):
    P = ohlcv["Adj Close"]
    ret = P.pct_change(fill_method=None)
    mom = P / P.shift(MOM_LB) - 1.0
    vol = ret.rolling(MOM_LB, min_periods=MOM_LB).std()
    return mom / vol.replace(0, np.nan)


def f2_volume_anomaly(ohlcv):
    V = ohlcv["Volume"]
    mu = V.rolling(VOL_LB, min_periods=VOL_LB).mean()
    sd = V.rolling(VOL_LB, min_periods=VOL_LB).std()
    return (V - mu) / sd.replace(0, np.nan)


def f3_range_percentile(ohlcv):
    P = ohlcv["Adj Close"]
    lo = P.rolling(RANGE_LB, min_periods=RANGE_LB).min()
    hi = P.rolling(RANGE_LB, min_periods=RANGE_LB).max()
    return (P - lo) / (hi - lo).replace(0, np.nan)


def zscore_xs(df):
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0)


def build_factors(ohlcv, universe):
    raw = {
        "f1_mom": f1_risk_adj_momentum(ohlcv),
        "f2_vol": f2_volume_anomaly(ohlcv),
        "f3_rng": f3_range_percentile(ohlcv),
    }
    fac = {}
    for k, v in raw.items():
        w = winsorize_xs(v.where(universe == 1, np.nan))
        fac[k] = w
    # combined = equal-weight sum of cross-sectional z-scores
    fac["fC_combo"] = sum(zscore_xs(fac[k]) for k in ["f1_mom", "f2_vol", "f3_rng"])
    return fac


# ---------- 3. Diagnostics (mirror squeeze pipeline) ----------
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


def ic_decay(signal, returns, lags=(1, 2, 3, 5, 10)):
    rows = []
    for L in lags:
        ic = daily_ic(signal, returns.shift(-L))
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


def diagnose(name, signal, rets, idx, label):
    sig = signal.reindex(idx)
    r = rets.loc[idx]
    fwd = r.shift(-HORIZON)
    ic = daily_ic(sig, fwd)
    if len(ic) == 0:
        print(f"\n[{label}] {name}: no IC days"); return None
    mean_ic = ic.mean(); ir = mean_ic/ic.std()*np.sqrt(252)
    tstat = mean_ic/(ic.std()/np.sqrt(len(ic)))
    dec = decile_spread(sig, fwd)
    ls_bps = (dec.iloc[-1] - dec.iloc[0]) * 1e4
    # monotonicity of decile pattern: corr of decile index vs avg fwd ret
    dec_corr = pd.Series(range(1, 11)).corr(pd.Series(dec.values))
    print(f"\n[{label}] {name}")
    print(f"  n_days={len(ic)}  mean_IC={mean_ic:.4f}  IR={ir:.2f}  t={tstat:.2f}  %>0={ (ic>0).mean()*100:.1f}%")
    print(f"  decile spread D10-D1 = {ls_bps:.2f} bps/d (~{ls_bps*252/100:.1f}%/yr gross)  decile_corr={dec_corr:.2f}")
    print("  decile fwd ret (bps): " + " ".join(f"{v*1e4:+.1f}" for v in dec.values))
    return {"name": name, "label": label, "n_days": len(ic), "mean_IC": float(mean_ic),
            "IR_ann": float(ir), "t_stat": float(tstat), "pct_pos": float((ic > 0).mean()),
            "decile_spread_bps": float(ls_bps), "decile_corr": float(dec_corr)}


# ---------- 4. Main ----------
def main():
    ohlcv = load_data()
    universe = build_universe(ohlcv)
    returns = build_returns(ohlcv)
    universe = universe.reindex(columns=returns.columns).fillna(0).astype(int)
    print(f"[data] dates {returns.index.min().date()} .. {returns.index.max().date()}  stocks={returns.shape[1]}")

    oos_start = returns.index.max() - pd.DateOffset(years=OOS_YEARS)
    is_idx = returns.index[returns.index < oos_start]
    oos_idx = returns.index[returns.index >= oos_start]
    print(f"[split] IS:  {is_idx.min().date()} .. {is_idx.max().date()}  ({len(is_idx)} days)")
    print(f"[split] OOS: {oos_idx.min().date()} .. {oos_idx.max().date()}  ({len(oos_idx)} days)")

    print("\n[factors] building f1/f2/f3 + combined (winsorized 2/98 daily)")
    fac = build_factors(ohlcv, universe)

    results = []
    for name in ["f1_mom", "f2_vol", "f3_rng", "fC_combo"]:
        results.append(diagnose(name, fac[name], returns, is_idx, "IS"))
        res = diagnose(name, fac[name], returns, oos_idx, "OOS")
        results.append(res)

    # OOS IC-decay + IC-by-year for the most promising single factor + combo
    print("\n[OOS IC decay]")
    for name in ["f1_mom", "f3_rng", "fC_combo"]:
        print(f"  -- {name} --")
        print(ic_decay(fac[name].reindex(oos_idx), returns.loc[oos_idx]).round(4).to_string(index=False))

    print("\n[OOS IC by year — combined factor]")
    ic_combo = daily_ic(fac["fC_combo"].reindex(oos_idx), returns.loc[oos_idx].shift(-HORIZON))
    print(ic_by_year(ic_combo).round(4).to_string())

    results = [r for r in results if r is not None]
    (ART_DIR / "stage1_factor_diag.json").write_text(json.dumps(results, indent=2))
    print(f"\n[saved] {ART_DIR}/stage1_factor_diag.json")


if __name__ == "__main__":
    main()
