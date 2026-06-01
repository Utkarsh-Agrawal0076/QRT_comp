"""
Build a consolidated report for the SELECTED ("worth trying") alphas and their
correlation structure.

For every alpha in alpha101_results/shortlist.csv it (re)computes, on the
in-sample window at the live T+1 lag over the 5M-ADV universe:
  * the daily rank-IC series and its summary stats
  * a direction-adjusted dollar-neutral decile long/short PnL series
    (alphas with negative IC are flipped so the traded strategy is the
    profitable orientation)

Outputs (in alpha101_results/):
  * SELECTED_REPORT.md      - full write-up: metrics table, correlation
                              clusters, a de-correlated subset, per-year IC appendix
  * corr_matrix.csv         - L/S PnL correlation matrix of the selected alphas
  * corr_heatmap.png        - heatmap of that matrix

Usage:
  python alpha101_report.py            # in-sample 2010-2020 (matches the screen)
  python alpha101_report.py --start 2021-01-01   # OOS view
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import alpha101 as a
from alpha101_pipeline import compute_ic, decile_backtest, OUT_DIR

CORR_REDUNDANT = 0.70   # |corr| above which two alphas are treated as duplicates


def sharpe(pnl):
    return float(pnl.mean() / pnl.std() * np.sqrt(252)) if pnl.std() > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shortlist", default=os.path.join(OUT_DIR, "shortlist.csv"))
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default="2020-12-31")
    ap.add_argument("--lag", type=int, default=1)
    ap.add_argument("--use-cache", action="store_true",
                    help="reuse cached PnL/perf instead of recomputing signals")
    args = ap.parse_args()

    sl = pd.read_csv(args.shortlist)
    sl["n"] = sl["alpha"].str.replace("alpha", "").astype(int)
    nums = sl["n"].tolist()
    dir_sign = dict(zip(sl["alpha"], np.where(sl["direction"] == "long", 1.0, -1.0)))
    print(f"Reporting on {len(nums)} selected alphas "
          f"({args.start or '2010'}..{args.end}, T+{args.lag})")

    pnl_cache = os.path.join(OUT_DIR, "selected_pnl.parquet")
    perf_cache = os.path.join(OUT_DIR, "selected_perf.csv")

    if args.use_cache and os.path.exists(pnl_cache) and os.path.exists(perf_cache):
        print("Using cached PnL / perf.")
        pnl_df = pd.read_parquet(pnl_cache)
        perf = pd.read_csv(perf_cache).set_index("alpha")
    else:
        data, returns, universe = a.load_panel(start=args.start, end=args.end)
        pnl_dict, rows = {}, []
        for n in nums:
            name = f"alpha{n:03d}"
            sig = a.get_alpha(n)(data)
            ic = compute_ic(sig, returns, universe, lag=args.lag)
            pnl, _ = decile_backtest(sig, returns, universe, lag=args.lag)
            s = dir_sign[name]
            pnl = pnl * s                      # trade in the profitable direction
            pnl_dict[name] = pnl
            ic_adj = ic * s
            rows.append({
                "alpha": name,
                "mean_ic_dir": round(float(ic_adj.mean()), 4),
                "hit": round(float((ic_adj > 0).mean()), 3),
                "ic_ir": round(float(ic_adj.mean() / ic_adj.std()), 3) if ic_adj.std() else None,
                "ls_sharpe": round(sharpe(pnl), 2),
                "ann_ret_%": round(float(pnl.mean() * 252 * 100), 2),
            })
            print(f"  {name}: IC {ic_adj.mean():+.4f}  L/S Sharpe {sharpe(pnl):+.2f}")
        # cache the expensive part immediately
        pnl_df = pd.DataFrame(pnl_dict).dropna(how="all")
        pnl_df.to_parquet(pnl_cache)
        perf = pd.DataFrame(rows).set_index("alpha")
        perf.to_csv(perf_cache)

    # join screen metrics with recomputed direction-adjusted perf (drop overlaps)
    merged = sl.set_index("alpha").drop(columns=["ic_ir", "ls_sharpe"]).join(perf)

    # ---- correlation matrix of direction-adjusted L/S PnL ----
    corr = pnl_df.corr()
    corr.to_csv(os.path.join(OUT_DIR, "corr_matrix.csv"))

    # heatmap
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.index, fontsize=6)
    ax.set_title(f"Selected alphas — L/S PnL correlation ({args.start or '2010'}..{args.end}, T+{args.lag})")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    heatmap_path = os.path.join(OUT_DIR, "corr_heatmap.png")
    fig.savefig(heatmap_path, dpi=130)
    plt.close(fig)

    # highly-correlated pairs
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c = corr.iloc[i, j]
            if abs(c) >= CORR_REDUNDANT:
                pairs.append((cols[i], cols[j], round(float(c), 2)))
    pairs.sort(key=lambda x: -abs(x[2]))

    # greedy de-correlated subset: keep highest |t|, drop anything correlated > thresh
    order = merged.sort_values("abs_t", ascending=False).index.tolist()
    keep = []
    for name in order:
        if all(abs(corr.loc[name, k]) < CORR_REDUNDANT for k in keep):
            keep.append(name)
    avg_pair_corr = float(corr.where(~np.eye(len(corr), dtype=bool)).abs().stack().mean())

    write_report(merged, corr, pairs, keep, avg_pair_corr, heatmap_path, args)
    print(f"\nSaved -> {os.path.join(OUT_DIR, 'SELECTED_REPORT.md')}")
    print(f"        {os.path.join(OUT_DIR, 'corr_matrix.csv')}")
    print(f"        {heatmap_path}")
    print(f"De-correlated subset ({len(keep)}): {keep}")


def write_report(merged, corr, pairs, keep, avg_pair_corr, heatmap_path, args):
    yearly = {}
    yd_path = os.path.join(OUT_DIR, "yearly_detail.json")
    if os.path.exists(yd_path):
        yearly = json.load(open(yd_path))

    lines = []
    L = lines.append
    L(f"# Selected Alphas — Test Report & Correlation Structure\n")
    L(f"*Window:* {args.start or '2010-01-01'} to {args.end} (in-sample screen) "
      f"| *Lag:* T+{args.lag} | *Universe:* $5M / 60-day ADV (`universe_5m`)\n")
    L("**Selection:** the literal mean-IC>0.02 rule selected nothing at T+1 on this "
      "universe, so 'selected' here = the *worth-trying* shortlist "
      "(|t|>=3 and |mean IC|>=0.004, traded in the IC-implied direction). "
      "IC = daily cross-sectional Spearman (verified vs scipy). L/S = dollar-neutral "
      "top-vs-bottom decile, gross.\n")

    # --- main metrics table ---
    L("## 1. Test results (direction-adjusted)\n")
    L("| alpha | tier | dir | mean IC | t | IC-IR | L/S Sharpe | ann ret % | hit | yr spread | flags |")
    L("|---|---|---|---|---|---|---|---|---|---|---|")
    m = merged.sort_values(["tier", "abs_t"], ascending=[True, False])
    for name, r in m.iterrows():
        flags = []
        if r.get("regime_candidate"): flags.append("regime")
        if r.get("indneutral_approx"): flags.append("indneut~")
        if r.get("uses_cap_proxy"): flags.append("cap~")
        L(f"| {name} | {r['tier']} | {'L' if r['direction']=='long' else 'S(flip)'} "
          f"| {r['mean_ic_dir']:+.4f} | {r['abs_t']:.1f} | {r['ic_ir']} "
          f"| {r['ls_sharpe']} | {r.get('ann_ret_%','')} | {r['hit_rate']:.0%} "
          f"| {r['year_spread']:.4f} | {','.join(flags)} |")
    L("")

    # --- correlation ---
    L("## 2. Correlation structure (L/S PnL)\n")
    L(f"Average absolute pairwise correlation across the {len(corr)} selected "
      f"alphas: **{avg_pair_corr:.2f}**. Heatmap: `{os.path.basename(heatmap_path)}`; "
      f"full matrix: `corr_matrix.csv`.\n")
    L(f"### Redundant pairs (|corr| >= {CORR_REDUNDANT})\n")
    if pairs:
        L("| alpha A | alpha B | corr |")
        L("|---|---|---|")
        for x, y, c in pairs:
            L(f"| {x} | {y} | {c:+.2f} |")
    else:
        L("_None — all selected alphas are below the redundancy threshold._")
    L("")

    # --- de-correlated subset ---
    L("## 3. Recommended de-correlated subset\n")
    L(f"Greedy pick (highest |t| first, drop anything with |corr| >= {CORR_REDUNDANT} "
      f"to an already-kept alpha) — **{len(keep)} alphas** that capture the distinct "
      f"bets without piling into one family:\n")
    sub = merged.loc[keep].sort_values("abs_t", ascending=False)
    L("| alpha | dir | mean IC | t | L/S Sharpe |")
    L("|---|---|---|---|---|")
    for name, r in sub.iterrows():
        L(f"| {name} | {'L' if r['direction']=='long' else 'S(flip)'} "
          f"| {r['mean_ic_dir']:+.4f} | {r['abs_t']:.1f} | {r['ls_sharpe']} |")
    L(f"\n`{keep}`\n")

    # --- per-year IC appendix ---
    L("## 4. Appendix — year-by-year mean IC (raw direction)\n")
    if yearly:
        years = sorted({yy["year"] for nm in merged.index if nm in yearly for yy in yearly[nm]})
        L("| alpha | " + " | ".join(str(y) for y in years) + " |")
        L("|---" * (len(years) + 1) + "|")
        for name in m.index:
            if name not in yearly:
                continue
            ymap = {yy["year"]: yy["mean_ic"] for yy in yearly[name]}
            cells = [f"{ymap.get(y, ''):+.3f}" if y in ymap else "" for y in years]
            L(f"| {name} | " + " | ".join(cells) + " |")
    L("")

    with open(os.path.join(OUT_DIR, "SELECTED_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
