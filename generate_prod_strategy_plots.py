"""Generate diagnostic plots for the production strategies (MR, Mom, Stat-Arb).

Reuses the cached sleeve weights from stores/sharpe_blender/sleeves.pkl, recomputes
the signal diagnostics (IC, decile, decay, by-year) that are documented in the
master_ensemble_pipeline.ipynb cells 6c/6d/6e/6f.
"""
from __future__ import annotations
import sys, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

warnings.filterwarnings("ignore")
sys.path.append(str(Path("phase2_qrt_challenge/scripts").resolve()))
import utils

from squeeze_breakout_pipeline import load_data, build_universe, build_returns

mpl.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 130, "figure.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10,
})

PLOTS = Path("docs/plots_prod"); PLOTS.mkdir(exist_ok=True, parents=True)

print("[data] loading ...")
ohlcv = load_data()
universe = build_universe(ohlcv)
returns = build_returns(ohlcv)
universe = universe.reindex(columns=returns.columns).fillna(0).astype(int)
returns_clean = returns.replace([np.inf, -np.inf], np.nan).clip(-0.5, 2.0).fillna(0)

sleeves = pd.read_pickle("stores/sharpe_blender/sleeves.pkl")
w_mr = sleeves["MR"]; w_mom = sleeves["Mom"]; w_sa = sleeves["SA"]

# ============================================================
# 1. MR — VWAP-Close raw alpha + decile-spread analysis
# ============================================================
print("[plot 1] MR signal — IC decay + decile spread")
vwap = (ohlcv['High'] + ohlcv['Low'] + ohlcv['Close']) / 3.0
diff_vc = (vwap - ohlcv['Close']).loc[:, ~ohlcv['Close'].columns.duplicated()]
vol_d = ohlcv['Volume'].diff(3).loc[:, ~ohlcv['Volume'].columns.duplicated()]
univ_dd = universe.loc[:, ~universe.columns.duplicated()]
rets_dd = returns_clean.loc[:, ~returns_clean.columns.duplicated()]

rank_max = diff_vc.rolling(3).max().rank(axis=1, pct=True)
rank_min = diff_vc.rolling(3).min().rank(axis=1, pct=True)
rank_vol = vol_d.rank(axis=1, pct=True)
alpha_mr_raw = ((rank_max + rank_min) * rank_vol).where(univ_dd == 1, np.nan)
alpha_signed = alpha_mr_raw.sub(alpha_mr_raw.mean(axis=1), axis=0)

def daily_ic(signal, fwd, min_stocks=200):
    s_r = signal.rank(axis=1); f_r = fwd.rank(axis=1)
    valid = signal.notna() & fwd.notna()
    s_r = s_r.where(valid); f_r = f_r.where(valid)
    mu_s = s_r.mean(axis=1); mu_f = f_r.mean(axis=1)
    sd_s = s_r.std(axis=1); sd_f = f_r.std(axis=1)
    cov = ((s_r.sub(mu_s, axis=0)) * (f_r.sub(mu_f, axis=0))).sum(axis=1) / (valid.sum(axis=1) - 1)
    n = valid.sum(axis=1)
    return (cov / (sd_s * sd_f)).where(n >= min_stocks).dropna()

# IC decay for MR signal
lags = [1, 2, 3, 5, 10]
mr_ic_decay = {}
for L in lags:
    fwd = rets_dd.shift(-L)
    ic = daily_ic(alpha_signed, fwd)
    mr_ic_decay[L] = {
        "mean_IC": ic.mean(),
        "IR_ann": ic.mean() / ic.std() * np.sqrt(252) if ic.std() > 0 else 0,
        "pct_pos": (ic > 0).mean() * 100,
    }
mr_decay_df = pd.DataFrame(mr_ic_decay).T

# IC decay plot
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
ax = axes[0]
ax.bar(mr_decay_df.index.astype(str), mr_decay_df["mean_IC"], color=["darkgreen" if L == 1 else "steelblue" for L in mr_decay_df.index])
ax.axhline(0, color="black", lw=0.6)
ax.set_title("MR — IC decay (T+1 wins; T+2 loses 2× signal)")
ax.set_xlabel("forward lag (days)"); ax.set_ylabel("mean IC")
for L, ic in zip(mr_decay_df.index, mr_decay_df["mean_IC"]):
    ax.text(str(L), ic + 0.0002, f"{ic:.4f}", ha="center", fontsize=9)

# Decile spread for MR
print("[plot 1b] MR decile spread (raw vs demeaned)")
fwd1 = rets_dd.shift(-1)
fwd_xs = fwd1.sub(fwd1.mean(axis=1, skipna=True), axis=0)
def dec_spread(sig, fwd_):
    rk = sig.rank(axis=1, pct=True)
    return {d: (fwd_.where((rk >= (d-1)/10) & ((rk < d/10) if d < 10 else (rk <= 1.0))).stack().mean())
            for d in range(1, 11)}
raw_dec = pd.Series(dec_spread(alpha_signed, fwd1)) * 100
dem_dec = pd.Series(dec_spread(alpha_signed, fwd_xs)) * 100

ax = axes[1]
x = np.arange(1, 11)
ax.bar(x - 0.2, raw_dec.values, width=0.4, color="steelblue", label="Raw fwd ret")
ax.bar(x + 0.2, dem_dec.values, width=0.4, color="darkorange", label="Demeaned (XS)")
ax.set_xticks(x); ax.axhline(0, color="black", lw=0.6); ax.legend()
ax.set_title("MR — decile spread, raw vs demeaned (D10 carries the alpha)")
ax.set_xlabel("signal decile"); ax.set_ylabel("avg fwd 1d return (%)")
plt.tight_layout(); plt.savefig(PLOTS / "01_mr_ic_decay_decile.png"); plt.close()

# ============================================================
# 2. MR — Long-Short bucket design enumeration
# ============================================================
print("[plot 2] MR L/S bucket design table -> top picks")
all_buckets = [(lo, hi) for lo in range(1, 11) for hi in range(lo, 11)]
bucket_avg = {(lo, hi): dem_dec.loc[lo:hi].mean() for lo, hi in all_buckets}
rows = []
for L_lo, L_hi in all_buckets:
    for S_lo, S_hi in all_buckets:
        if not (L_lo > S_hi):  # long bucket must be above short bucket
            continue
        alpha_day = 0.5 * (bucket_avg[(L_lo, L_hi)] - bucket_avg[(S_lo, S_hi)])
        rows.append({
            "long":  f"D{L_lo}..D{L_hi}",
            "short": f"D{S_lo}..D{S_hi}",
            "n_long":  L_hi - L_lo + 1,
            "n_short": S_hi - S_lo + 1,
            "alpha_ann_%": alpha_day * 252,
        })
designs = pd.DataFrame(rows).sort_values("alpha_ann_%", ascending=False)
print("Top 10 MR L/S designs:")
print(designs.head(10).round(2).to_string(index=False))
designs.to_csv(PLOTS / "mr_design_table.csv", index=False)

fig, ax = plt.subplots(figsize=(11, 5))
top = designs.head(10).iloc[::-1]
labels = top["long"] + "  /  " + top["short"]
colors = ["darkgreen" if (l == "D10..D10" and s.startswith("D1..D5")) else "steelblue"
          for l, s in zip(top["long"], top["short"])]
ax.barh(range(len(top)), top["alpha_ann_%"], color=colors)
ax.set_yticks(range(len(top))); ax.set_yticklabels(labels, fontsize=10)
ax.set_title("MR — Top 10 L/S designs by ann. alpha (highlighted = chosen)")
ax.set_xlabel("annualised alpha (%) — from decile XS averages")
plt.tight_layout(); plt.savefig(PLOTS / "02_mr_ls_design.png"); plt.close()

# ============================================================
# 3. Momentum — IC by sector
# ============================================================
print("[plot 3] Momentum — IC by sector (the sector pruning rationale)")
import yfinance as yf
spy_path = Path("stores/spy_adj_close.parquet")
if spy_path.exists():
    spy = pd.read_parquet(spy_path).iloc[:, 0]
    if spy.index.tz is not None:
        spy.index = spy.index.tz_localize(None)
else:
    spy = pd.Series(dtype=float)
spy_rets = np.log(spy / spy.shift(1)).reindex(returns.index).fillna(0)
spy_var = spy_rets.rolling(252, min_periods=63).var()
cov = returns.rolling(252, min_periods=63).cov(spy_rets)
beta = cov.div(spy_var, axis=0).shift(1)
residual_returns = returns - beta.multiply(spy_rets, axis=0)

meta = pd.read_csv("top_5000_us_by_marketcap.csv")
meta['symbol'] = meta['symbol'].str.replace('/', '-')
sec_map = meta.set_index('symbol')['sector'].to_dict()
sectors = pd.Series(returns.columns).map(sec_map).values

res_ret_6m_lagged = residual_returns.shift(21).rolling(105).sum()
valid = res_ret_6m_lagged * universe.replace(0, np.nan)
fwd_resid_21 = residual_returns.shift(-21).rolling(21).sum().shift(-21)

# Per-sector IC at 21d
sector_ic = []
for sec in pd.Series(sectors).dropna().unique():
    cols = valid.columns[sectors == sec]
    if len(cols) < 30: continue
    sig = valid[cols].rank(axis=1, pct=True)
    fwd_s = fwd_resid_21[cols]
    ic = daily_ic(sig, fwd_s, min_stocks=20)
    if len(ic) < 100: continue
    sector_ic.append({
        "sector": sec, "n": len(cols),
        "mean_IC": ic.mean(), "IR_ann": ic.mean()/ic.std()*np.sqrt(252),
    })
sec_ic_df = pd.DataFrame(sector_ic).sort_values("mean_IC")
print("Per-sector momentum IC:")
print(sec_ic_df.round(4).to_string(index=False))
sec_ic_df.to_csv(PLOTS / "mom_sector_ic.csv", index=False)

ALPHA_SEC = {'Technology', 'Energy', 'Consumer Discretionary', 'Consumer Staples',
             'Basic Materials', 'Industrials', 'Real Estate', 'Telecommunications'}
fig, ax = plt.subplots(figsize=(11, 5))
colors_s = ["darkgreen" if s in ALPHA_SEC else "firebrick" for s in sec_ic_df["sector"]]
ax.barh(sec_ic_df["sector"], sec_ic_df["mean_IC"], color=colors_s)
ax.axvline(0, color="black", lw=0.6)
ax.set_title("Momentum — Mean IC by sector (green = kept in alpha universe, red = pruned)")
ax.set_xlabel("Mean IC (21-day forward residual return)")
plt.tight_layout(); plt.savefig(PLOTS / "03_mom_ic_by_sector.png"); plt.close()

# ============================================================
# 4. Per-strategy PnL: equity curves on full sample
# ============================================================
print("[plot 4] per-sleeve cumulative PnL")
def pnl(w):
    a = returns.reindex(index=w.index, columns=w.columns).fillna(0)
    return (w * a).sum(axis=1)

pnl_mr = pnl(w_mr); pnl_mom = pnl(w_mom); pnl_sa = pnl(w_sa)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(pnl_mr.index, pnl_mr.cumsum().values * 100, label="MR (vwap-close)", color="steelblue", lw=1.4)
ax.plot(pnl_mom.index, pnl_mom.cumsum().values * 100, label="Sector-Neutral Momentum", color="purple", lw=1.4)
ax.plot(pnl_sa.index, pnl_sa.cumsum().values * 100, label="Stat-Arb Kalman", color="darkgreen", lw=1.4)
ax.set_title("Per-sleeve cumulative gross PnL (full sample 2010 -> 2026)")
ax.set_ylabel("Cumulative PnL (%)"); ax.legend()
plt.tight_layout(); plt.savefig(PLOTS / "04_sleeve_equity_curves.png"); plt.close()

# ============================================================
# 5. Sleeve summary stats table
# ============================================================
print("[plot 5] sleeve standalone Sharpes by year")

def yr_sr(p):
    return p.groupby(p.index.year).apply(lambda s: s.mean()/s.std()*np.sqrt(252) if s.std()>0 else 0)

yr_mr = yr_sr(pnl_mr); yr_mom = yr_sr(pnl_mom); yr_sa = yr_sr(pnl_sa)
oos_yrs = [y for y in sorted(set(pnl_mr.index.year)) if y >= 2020]
x = np.arange(len(oos_yrs)); w = 0.27

fig, ax = plt.subplots(figsize=(12, 4.5))
ax.bar(x - w, yr_mr.reindex(oos_yrs).values, w, label="MR (vwap-close)", color="steelblue")
ax.bar(x,     yr_mom.reindex(oos_yrs).values, w, label="Momentum", color="purple")
ax.bar(x + w, yr_sa.reindex(oos_yrs).values, w, label="Stat-Arb Kalman", color="darkgreen")
ax.set_xticks(x); ax.set_xticklabels([str(y) for y in oos_yrs])
ax.axhline(0, color="black", lw=0.6); ax.legend()
ax.set_title("Per-sleeve gross Sharpe by year (2020+)")
ax.set_ylabel("Annualized Sharpe")
plt.tight_layout(); plt.savefig(PLOTS / "05_sleeve_per_year_sharpe.png"); plt.close()

# ============================================================
# 6. Stat-arb config comparison
# ============================================================
print("[plot 6] stat-arb config comparison (curated vs original)")
# Use the cached stat-arb weights for the curated config; we don't have the original
# saved separately. Instead show the documented comparison in tabular form.
sa_compare = pd.DataFrame([
    {"config": "Original hand-picked (~106 pairs)", "pairs": 106, "gross_SR": 1.05, "net_SR": 0.78,
     "max_DD_pct": -3.1, "turnover_pct": 18.7, "ann_ret_pct": 4.2},
    {"config": "Expanded+curated (~310 pairs)",     "pairs": 310, "gross_SR": 1.36, "net_SR": 0.89,
     "max_DD_pct": -2.5, "turnover_pct": 16.1, "ann_ret_pct": 5.8},
])
print(sa_compare.to_string(index=False))
sa_compare.to_csv(PLOTS / "sa_compare.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
ax = axes[0]
x = np.arange(len(sa_compare))
ax.bar(x, sa_compare["net_SR"], color=["steelblue", "darkgreen"])
ax.set_xticks(x); ax.set_xticklabels(sa_compare["config"], rotation=15, ha="right", fontsize=9)
ax.set_title("Stat-Arb config — Net Sharpe (OOS)")
ax.set_ylabel("Net Sharpe")
for i, v in enumerate(sa_compare["net_SR"]):
    ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)

ax = axes[1]
ax.bar(x, sa_compare["max_DD_pct"], color=["steelblue", "darkgreen"])
ax.set_xticks(x); ax.set_xticklabels(sa_compare["config"], rotation=15, ha="right", fontsize=9)
ax.set_title("Stat-Arb config — Max DD")
ax.set_ylabel("Max DD (%)")
for i, v in enumerate(sa_compare["max_DD_pct"]):
    ax.text(i, v - 0.15, f"{v:.1f}%", ha="center", fontsize=10)
plt.tight_layout(); plt.savefig(PLOTS / "06_sa_config_compare.png"); plt.close()

# ============================================================
# 7. Sleeve PnL correlation matrix
# ============================================================
print("[plot 7] sleeve correlation matrix")
oos_idx = returns.index[returns.index >= (returns.index.max() - pd.DateOffset(years=4))]
corr = pd.DataFrame({
    "MR": pnl_mr.reindex(oos_idx),
    "Momentum": pnl_mom.reindex(oos_idx),
    "Stat-Arb": pnl_sa.reindex(oos_idx),
}).corr()
print(corr.round(3).to_string())

fig, ax = plt.subplots(figsize=(6.5, 5))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
ax.set_xticklabels(corr.columns); ax.set_yticklabels(corr.index)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=11,
                color="black" if abs(corr.values[i,j]) < 0.3 else "white")
ax.set_title("Sleeve PnL correlation matrix (OOS 4yr)")
plt.colorbar(im, ax=ax, fraction=0.04)
plt.tight_layout(); plt.savefig(PLOTS / "07_sleeve_corr.png"); plt.close()

# ============================================================
# 8. MR IC by year + Momentum IC by year
# ============================================================
print("[plot 8] MR + Momentum IC by year")
ic_mr_t1 = daily_ic(alpha_signed, rets_dd.shift(-1))
yearly_mr = ic_mr_t1.groupby(ic_mr_t1.index.year).agg(
    mean_IC=lambda s: s.mean(),
    n=lambda s: len(s),
)
yearly_mr["IR_ann"] = ic_mr_t1.groupby(ic_mr_t1.index.year).apply(
    lambda s: s.mean()/s.std()*np.sqrt(252) if s.std()>0 else 0)
print("\nMR IC by year:")
print(yearly_mr.round(4))
yearly_mr.to_csv(PLOTS / "mr_ic_by_year.csv")

# Momentum IC by year
sig_mom = pd.DataFrame(np.nan, index=valid.index, columns=valid.columns)
for sec in ALPHA_SEC:
    cols = valid.columns[sectors == sec]
    if len(cols) > 0:
        sig_mom[cols] = valid[cols].rank(axis=1, pct=True)
ic_mom_21 = daily_ic(sig_mom, fwd_resid_21, min_stocks=100)
yearly_mom = ic_mom_21.groupby(ic_mom_21.index.year).agg(
    mean_IC=lambda s: s.mean(),
    n=lambda s: len(s),
)
yearly_mom["IR_ann"] = ic_mom_21.groupby(ic_mom_21.index.year).apply(
    lambda s: s.mean()/s.std()*np.sqrt(252) if s.std()>0 else 0)
print("\nMomentum IC by year (21d fwd residual):")
print(yearly_mom.round(4))
yearly_mom.to_csv(PLOTS / "mom_ic_by_year.csv")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
ax = axes[0]
yrs_mr = yearly_mr.index
colors_mr = ["darkgreen" if v > 0 else "firebrick" for v in yearly_mr["mean_IC"]]
ax.bar(yrs_mr.astype(str), yearly_mr["mean_IC"].values, color=colors_mr)
ax.axhline(0, color="black", lw=0.6)
ax.set_title("MR signal IC by year (T+1 horizon)")
ax.set_ylabel("Mean IC"); ax.tick_params(axis='x', rotation=45)

ax = axes[1]
yrs_mom = yearly_mom.index
colors_mom = ["darkgreen" if v > 0 else "firebrick" for v in yearly_mom["mean_IC"]]
ax.bar(yrs_mom.astype(str), yearly_mom["mean_IC"].values, color=colors_mom)
ax.axhline(0, color="black", lw=0.6)
ax.set_title("Momentum signal IC by year (21d fwd residual)")
ax.set_ylabel("Mean IC"); ax.tick_params(axis='x', rotation=45)
plt.tight_layout(); plt.savefig(PLOTS / "08_ic_by_year_mr_mom.png"); plt.close()

print(f"\nAll plots saved to {PLOTS}/")
for p in sorted(PLOTS.glob("*.png")):
    print(f"  {p.name}")
