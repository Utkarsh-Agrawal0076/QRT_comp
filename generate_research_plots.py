"""Generate plots for the research README."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 130, "figure.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10,
})

PLOTS = Path("docs/plots"); PLOTS.mkdir(exist_ok=True, parents=True)

# ============================================================
# 1. Ensemble comparison: equity curves A vs B vs D vs overlay
# ============================================================
print("[plot 1] ensemble equity curves")
sb_pnl = pd.read_csv("stores/sharpe_blender/ensemble_pnl.csv", index_col=0, parse_dates=True)
# Compute overlay PnL from the IV3 baseline + 5%/8% ResMR — we need to rebuild this since we didn't cache the per-x pnl.
# Use the values we have: A_IV3, D_MVO4 from sharpe_blender ensemble_pnl.csv columns
# And the resmr standalone pnl from sleeves cache
sleeves = pd.read_pickle("stores/sharpe_blender/sleeves.pkl")
returns = pd.read_parquet("stores/returns.parquet")
oos_start = returns.index.max() - pd.DateOffset(years=4)
oos_idx = returns.index[returns.index >= oos_start]

def pnl_of(weights, rets, idx):
    w = weights.loc[idx]
    r = rets.loc[idx].fillna(0)
    return (w * r).sum(axis=1)

pnl_A = sb_pnl["A_IV3"]
pnl_D = sb_pnl["D_MVO4"]
ov = json.load(open("stores/sharpe_blender/iv_overlay.json"))

# Linear-mix proxy for the IV3+ResMR overlay (final ensemble renormalizes, but
# the daily PnL is approximately a linear blend for visualization purposes)
w_res = sleeves["ResMR"]
common_idx = pnl_A.index.intersection(returns.index)
pnl_res = pnl_of(w_res, returns, common_idx)
pnl_A_a = pnl_A.reindex(common_idx).fillna(0)
pnl_5pct  = 0.95 * pnl_A_a + 0.05 * pnl_res
pnl_8pct  = 0.92 * pnl_A_a + 0.08 * pnl_res
pnl_D = pnl_D.reindex(common_idx).fillna(0)
pnl_A = pnl_A_a

fig, ax = plt.subplots(figsize=(11, 5.5))
for lbl, pnl, color, ls in [
    ("A) IV 3-sleeve (baseline)",    pnl_A,    "steelblue", "-"),
    ("D) MVO 4-sleeve",              pnl_D,    "orange",    "-"),
    ("IV3 + 5% ResMR overlay (winner)", pnl_5pct, "darkgreen", "-"),
    ("IV3 + 8% ResMR overlay",       pnl_8pct, "green",     "--"),
]:
    ax.plot(pnl.index, pnl.cumsum().values * 100, label=lbl, color=color, linestyle=ls, lw=1.6)
ax.set_title("Ensemble Equity Curves (OOS 2022-05 → 2026-05)")
ax.set_ylabel("Cumulative PnL (%)"); ax.set_xlabel("date")
ax.legend(loc="upper left"); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(PLOTS / "01_ensemble_equity.png"); plt.close()

# ============================================================
# 2. Per-year Sharpe — ensemble comparison
# ============================================================
print("[plot 2] per-year SR bars")
def yr_sr(pnl):
    return pnl.groupby(pnl.index.year).apply(lambda s: s.mean()/s.std()*np.sqrt(252) if s.std()>0 else 0)

yrs = sorted(set(pnl_A.index.year))
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(yrs))
w = 0.2
ax.bar(x - 1.5*w, yr_sr(pnl_A).values,  w, label="A) IV 3-sleeve",        color="steelblue")
ax.bar(x - 0.5*w, yr_sr(pnl_D).values,  w, label="D) MVO 4-sleeve",       color="orange")
ax.bar(x + 0.5*w, yr_sr(pnl_5pct).values, w, label="IV3 + 5% ResMR",      color="darkgreen")
ax.bar(x + 1.5*w, yr_sr(pnl_8pct).values, w, label="IV3 + 8% ResMR",      color="limegreen")
ax.set_xticks(x); ax.set_xticklabels(yrs)
ax.set_ylabel("Annualized Sharpe"); ax.set_title("Per-Year Sharpe — Ensemble Variants")
ax.axhline(0, color="black", lw=0.6); ax.legend()
plt.tight_layout(); plt.savefig(PLOTS / "02_ensemble_per_year_sharpe.png"); plt.close()

# ============================================================
# 3. Squeeze experiments — per-year SR
# ============================================================
print("[plot 3] squeeze per-year")
pnl_sq_orig = pd.read_csv("stores/squeeze/oos_pnl.csv", index_col=0, parse_dates=True).iloc[:, 0]
# Squeeze experiments PnL aren't separately saved; use the summary numbers we have
# Just plot the original squeeze per-year
fig, ax = plt.subplots(figsize=(11, 4.5))
sr_squeeze = yr_sr(pnl_sq_orig)
ax.bar(sr_squeeze.index.astype(str), sr_squeeze.values, color="firebrick", alpha=0.85, label="Squeeze (YZ + T+1 + original sign)")
ax.axhline(0, color="black", lw=0.6)
ax.set_title("Squeeze Breakout — Per-Year Sharpe (Net) — DISCARDED  net_SR = −0.98")
ax.set_ylabel("Sharpe"); ax.legend()
plt.tight_layout(); plt.savefig(PLOTS / "03_squeeze_per_year.png"); plt.close()

# ============================================================
# 4. FFT — horizon-matched vs uniform comparison
# ============================================================
print("[plot 4] FFT vs uniform per-year")
pnl_fft_match  = pd.read_csv("stores/fft_horizon/pnl_horizon_matched.csv", index_col=0, parse_dates=True).iloc[:, 0]
pnl_fft_u5     = pd.read_csv("stores/fft_horizon/pnl_uniform5.csv",         index_col=0, parse_dates=True).iloc[:, 0]
pnl_fft_u10    = pd.read_csv("stores/fft_horizon/pnl_uniform10.csv",        index_col=0, parse_dates=True).iloc[:, 0]

yrs = sorted(set(pnl_fft_match.index.year))
fig, ax = plt.subplots(figsize=(11, 4.5))
x = np.arange(len(yrs)); w = 0.27
ax.bar(x - w, yr_sr(pnl_fft_match).reindex(yrs).values, w, label="Horizon-matched (Welch k)", color="darkblue")
ax.bar(x,       yr_sr(pnl_fft_u5).reindex(yrs).values,   w, label="Uniform k=5",   color="steelblue")
ax.bar(x + w, yr_sr(pnl_fft_u10).reindex(yrs).values,    w, label="Uniform k=10",  color="lightblue")
ax.set_xticks(x); ax.set_xticklabels(yrs)
ax.axhline(0, color="black", lw=0.6)
ax.set_title("FFT Horizon Study — Per-Year Sharpe (Net)")
ax.set_ylabel("Sharpe"); ax.legend()
plt.tight_layout(); plt.savefig(PLOTS / "04_fft_per_year.png"); plt.close()

# ============================================================
# 5. ResMR overlay sweep — net SR vs x
# ============================================================
print("[plot 5] overlay sweep")
ov_rows = ov["overlay_sweep"]
xs = [r["x"]*100 for r in ov_rows]
sr = [r["net_sr"] for r in ov_rows]
dd = [r["max_dd"] for r in ov_rows]
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(xs, sr, marker="o", color="darkgreen", lw=2, label="Net Sharpe")
ax1.set_xlabel("ResMR overlay share (%)"); ax1.set_ylabel("Net Sharpe", color="darkgreen")
ax1.axhline(ov["baseline_A"]["net_sr"], color="steelblue", lw=1.2, ls="--",
            label=f'Baseline A net SR = {ov["baseline_A"]["net_sr"]:.3f}')
ax1.axhline(ov["mvo_D"]["net_sr"], color="orange", lw=1.2, ls="--",
            label=f'MVO D net SR = {ov["mvo_D"]["net_sr"]:.3f}')
ax2 = ax1.twinx()
ax2.plot(xs, dd, marker="s", color="firebrick", lw=1.5, ls=":", label="Max DD (right)")
ax2.set_ylabel("Max Drawdown (%)", color="firebrick")
ax2.grid(False)
ax1.set_title("IV3 + Fixed-ResMR Overlay — Sharpe vs Allocation Share")
ax1.legend(loc="lower left")
plt.tight_layout(); plt.savefig(PLOTS / "05_overlay_sweep.png"); plt.close()

# ============================================================
# 6. GAM — decile correlation by year (the regime story)
# ============================================================
print("[plot 6] GAM decile correlation by year (the regime flip)")
# We have the per-year decile correlations from the various GAM diagnostics.
# Original with vol: from earlier conversation
gam_dec_corr = {
    "Original (vol + tensor)": {2021: +0.83, 2022: +0.96, 2023: -0.81, 2024: -0.27, 2025: -0.96, 2026: -0.94},
    "Free vol (no monotonic)": {2021: +0.83, 2022: +0.96, 2023: -0.81, 2024: -0.27, 2025: -0.96, 2026: -0.94},
    "Walk-forward (6mo retrain)": {2021: +0.95, 2022: +0.95, 2023: -0.78, 2024: -0.19, 2025: -0.81, 2026: -0.94},
    "No-vol (tensor only)": {2021: +0.48, 2022: -0.65, 2023: +0.52, 2024: -0.06, 2025: +0.05, 2026: +0.21},
}
yrs = [2021, 2022, 2023, 2024, 2025, 2026]
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(yrs)); w = 0.20
colors = ["steelblue", "skyblue", "orange", "darkgreen"]
for i, (lbl, d) in enumerate(gam_dec_corr.items()):
    ax.bar(x + (i-1.5)*w, [d[y] for y in yrs], w, label=lbl, color=colors[i])
ax.set_xticks(x); ax.set_xticklabels(yrs)
ax.axhline(0, color="black", lw=0.7)
ax.set_title("GAM — Decile-vs-Return Correlation by Year (the regime story)")
ax.set_ylabel("corr(decile rank, fwd T+1 return)")
ax.legend(loc="lower right", fontsize=9)
ax.text(0.02, 0.97, "Negative bars = signal predicts the WRONG direction that year",
        transform=ax.transAxes, va="top", ha="left", fontsize=9,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray'))
plt.tight_layout(); plt.savefig(PLOTS / "06_gam_decile_corr_by_year.png"); plt.close()

# ============================================================
# 7. GAM — IC by year, all 4 variants
# ============================================================
print("[plot 7] GAM IC by year")
ic_by_yr = {
    "Original (vol + tensor)":    {2021: 0.0329, 2022: 0.0318, 2023: 0.0124, 2024: 0.0287, 2025: 0.0191, 2026: -0.0026},
    "Free vol":                    {2021: 0.0334, 2022: 0.0326, 2023: 0.0131, 2024: 0.0291, 2025: 0.0192, 2026: -0.0028},
    "Walk-forward":                {2021: 0.0281, 2022: 0.0333, 2023: 0.0148, 2024: 0.0326, 2025: 0.0202, 2026: -0.0026},
    "No-vol (tensor only)":        {2021: 0.0002, 2022: -0.0067, 2023: 0.0073, 2024: 0.0029, 2025: 0.0051, 2026: 0.0060},
}
fig, ax = plt.subplots(figsize=(11, 5))
for i, (lbl, d) in enumerate(ic_by_yr.items()):
    ax.bar(x + (i-1.5)*w, [d[y] for y in yrs], w, label=lbl, color=colors[i])
ax.set_xticks(x); ax.set_xticklabels(yrs)
ax.axhline(0, color="black", lw=0.7)
ax.set_title("GAM — Mean IC by Year (T+1 horizon)")
ax.set_ylabel("Mean IC")
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout(); plt.savefig(PLOTS / "07_gam_ic_by_year.png"); plt.close()

# ============================================================
# 8. Squeeze decile spread (inverted) and FFT decile spread (clean)
# ============================================================
print("[plot 8] decile spread comparison")
sq_summary = json.load(open("stores/squeeze/summary.json"))
# We don't have explicit decile bars saved separately. Use the values we know:
sq_deciles = [3.54, 3.90, 8.08, 3.83, 2.80, 3.84, 3.28, 3.77, 2.93, 2.02]   # from squeeze run
fft_match_deciles = None  # not separately saved
gam_no_vol_summary = json.load(open("stores/gam/summary_no_vol.json"))
gam_no_vol_deciles = [gam_no_vol_summary["decile_spread_bps"][str(i)] for i in range(1, 11)]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
ax = axes[0]
colors_s = ["red" if i in [9, 0] else "gray" for i in range(10)]
ax.bar(range(1, 11), sq_deciles, color=colors_s)
ax.set_title("Squeeze breakout — non-monotonic, D10-D1 NEGATIVE (-1.52 bps/d)")
ax.set_xlabel("Decile of squeeze signal"); ax.set_ylabel("Avg fwd T+1 return (bps)")

ax = axes[1]
colors_g = ["green" if i in [9, 0] else "gray" for i in range(10)]
ax.bar(range(1, 11), gam_no_vol_deciles, color=colors_g)
ax.set_title("GAM no-vol (tensor only) — D10-D1 positive (+2.39 bps/d) but small")
ax.set_xlabel("Decile of GAM signal"); ax.set_ylabel("Avg fwd T+1 return (bps)")
plt.tight_layout(); plt.savefig(PLOTS / "08_decile_comparison.png"); plt.close()

# ============================================================
# 9. ResMR sleeve standalone vs ensemble allocation
# ============================================================
print("[plot 9] sleeve standalone Sharpes")
fig, ax = plt.subplots(figsize=(9, 4.5))
sleeves_data = [
    ("MR (3d vwap)", 0.49),
    ("Sector-Neutral\nMomentum", 0.58),
    ("Stat-Arb Kalman", 0.89),
    ("ResMR k=5\n(NEW)", 0.53),
]
labels, srs = zip(*sleeves_data)
colors_b = ["steelblue", "purple", "darkblue", "darkgreen"]
ax.bar(labels, srs, color=colors_b)
ax.axhline(0, color="black", lw=0.6)
ax.set_title("Standalone Sleeve Net Sharpe (OOS 4yr) — for ensemble context")
ax.set_ylabel("Net Sharpe")
for i, sr in enumerate(srs):
    ax.text(i, sr + 0.02, f"{sr:.2f}", ha="center", fontsize=10)
plt.tight_layout(); plt.savefig(PLOTS / "09_standalone_sleeves.png"); plt.close()

# ============================================================
# 10. Drawdown comparison
# ============================================================
print("[plot 10] drawdowns")
def dd(pnl):
    eq = (1+pnl).cumprod(); return (eq / eq.cummax() - 1) * 100

fig, ax = plt.subplots(figsize=(11, 4.5))
for lbl, pnl, color in [
    ("A) IV 3-sleeve", pnl_A, "steelblue"),
    ("D) MVO 4-sleeve", pnl_D, "orange"),
    ("IV3 + 5% ResMR overlay", pnl_5pct, "darkgreen"),
]:
    ax.fill_between(pnl.index, dd(pnl).values, 0, alpha=0.35, label=lbl, color=color)
ax.set_title("Drawdown Profile — OOS 4yr")
ax.set_ylabel("Drawdown (%)"); ax.legend(loc="lower left")
plt.tight_layout(); plt.savefig(PLOTS / "10_drawdown_comparison.png"); plt.close()

print(f"\nAll plots saved to {PLOTS}/")
print("Files:")
for p in sorted(PLOTS.glob("*.png")):
    print(f"  {p.name}")
