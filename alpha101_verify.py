"""
Verification package for the 12 net-tradeable alphas (cell-14 methodology):
  A. exact bucket design per alpha + gross/net Sharpe/turnover (selected_designs.csv)
  B. decile bar diagrams with the CHOSEN long (green) / short (red) buckets
     highlighted so the design can be eyeballed       (selected_decile_charts.png)
  C. IC of every selected alpha (and the dropped ones) (selected_ic.csv)
  D. alphas selected earlier (13-core) but dropped now, with the reason (dropped_alphas.csv)
  E. correlation matrix of the selected alphas' best-design L/S PnL
     (selected_corr_matrix.csv + selected_corr_heatmap.png)

All on the in-sample window 2010-2020, T+1, 5M-ADV universe, with the user's
processing: clip returns [-0.5, 2.0], 3-day smooth, cross-sectional demean.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import alpha101 as a
from alpha101_bucket_enum import clean_returns, bucket_masks, LAG, EXEC_BPS, FIN_ANNUAL

EARLIER_CORE = ["alpha044", "alpha024", "alpha055", "alpha078", "alpha022", "alpha084",
                "alpha023", "alpha005", "alpha040", "alpha083", "alpha030", "alpha073", "alpha059"]


def parse(s):
    lo, hi = s.replace("D", "").split("-")
    return int(lo), int(hi)


def design_pnl_series(rp, returns, long_b, short_b):
    lo_l, hi_l = (long_b[0] - 1) / 10, long_b[1] / 10
    lo_s, hi_s = (short_b[0] - 1) / 10, short_b[1] / 10
    longs = ((rp >= lo_l) & (rp <= hi_l + 1e-9)).astype(float)
    shorts = ((rp >= lo_s) & (rp < hi_s)).astype(float)
    lw = longs.div(longs.sum(axis=1), axis=0) * 0.5
    sw = shorts.div(shorts.sum(axis=1), axis=0) * 0.5
    w = (lw - sw).shift(LAG).fillna(0.0)
    gross = (w * returns.fillna(0.0)).sum(axis=1)
    traded = w.diff().abs().sum(axis=1).fillna(0.0)
    book = w.abs().sum(axis=1)
    net = gross - traded * EXEC_BPS - book * (FIN_ANNUAL / 252)
    net = net.loc[gross.ne(0).cumsum() > 0]
    return net


def decile_demeaned(rp, fwd_xs):
    out = []
    for d in range(1, 11):
        lo, hi = (d - 1) / 10, d / 10
        m = (rp >= lo) & (rp < hi) if d < 10 else (rp >= lo)
        out.append(np.nanmean(fwd_xs.where(m).values) * 1e4)  # bps/day
    return np.array(out)


def daily_ic(sig_smooth, fwd, uni):
    al = sig_smooth.shift(LAG).where(uni)
    sr = al.rank(axis=1); rr = fwd.where(uni).rank(axis=1)
    mask = sr.notna() & rr.notna()
    sr, rr = sr.where(mask), rr.where(mask)
    n = mask.sum(axis=1)
    da = sr.sub(sr.mean(axis=1), axis=0); db = rr.sub(rr.mean(axis=1), axis=0)
    ic = (da * db).sum(axis=1) / np.sqrt((da**2).sum(axis=1) * (db**2).sum(axis=1))
    return ic.where(n >= 50).dropna()


def main():
    enum = pd.read_csv("alpha101_results/bucket_enum.csv").set_index("alpha")
    selected = list(enum[enum["net_Sh"] >= 0.5].index)
    union = sorted(set(selected) | set(EARLIER_CORE),
                   key=lambda x: int(x.replace("alpha", "")))

    data, returns, universe = a.load_panel(start="2010-01-01", end="2020-12-31", verbose=True)
    returns = clean_returns(returns)
    uni = universe.astype(bool)
    fwd = returns.shift(-1)
    fwd_xs = fwd.sub(fwd.mean(axis=1), axis=0)

    profiles, ic_rows, pnl_sel = {}, [], {}
    for name in union:
        n = int(name.replace("alpha", ""))
        sig = a.get_alpha(n)(data)
        smooth = sig.replace([np.inf, -np.inf], np.nan).rolling(3, min_periods=1).mean()
        rp = bucket_masks(sig, universe)
        profiles[name] = decile_demeaned(rp, fwd_xs)
        ic = daily_ic(smooth, fwd, uni)
        ic_rows.append({"alpha": name, "mean_ic": round(ic.mean(), 4),
                        "hit_rate": round((ic > 0).mean(), 3),
                        "ic_ir": round(ic.mean() / ic.std(), 3) if ic.std() else None,
                        "t_stat": round(ic.mean() / ic.std() * np.sqrt(len(ic)), 1) if ic.std() else None,
                        "selected": name in selected})
        if name in selected:
            lb, sb = parse(enum.loc[name, "best_long"]), parse(enum.loc[name, "best_short"])
            pnl_sel[name] = design_pnl_series(rp, returns, lb, sb)
        print(f"  {name}: IC {ic.mean():+.4f}")

    # A. designs table
    designs = enum.loc[selected, ["best_long", "best_short", "gross_Sh", "net_Sh", "turn_%", "annret_%"]].copy()
    ic_df = pd.DataFrame(ic_rows).set_index("alpha")
    designs = designs.join(ic_df[["mean_ic", "hit_rate", "ic_ir", "t_stat"]])
    designs = designs.sort_values("net_Sh", ascending=False)
    designs.to_csv("alpha101_results/selected_designs.csv")
    print("\n===== A. SELECTED ALPHAS: exact design + IC =====")
    print(designs.to_string())

    # B. decile diagrams with chosen buckets highlighted
    nsel = len(selected)
    ncol = 3; nrow = int(np.ceil(nsel / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 3.2 * nrow))
    axes = np.array(axes).reshape(-1)
    for ax, name in zip(axes, designs.index):
        lb, sb = parse(enum.loc[name, "best_long"]), parse(enum.loc[name, "best_short"])
        prof = profiles[name]
        colors = []
        for d in range(1, 11):
            if lb[0] <= d <= lb[1]:
                colors.append("#2ca02c")   # long
            elif sb[0] <= d <= sb[1]:
                colors.append("#d62728")   # short
            else:
                colors.append("#cccccc")
        ax.bar(range(1, 11), prof, color=colors, edgecolor="k", linewidth=0.4)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"{name}  L:{enum.loc[name,'best_long']} S:{enum.loc[name,'best_short']} "
                     f"(net {enum.loc[name,'net_Sh']})", fontsize=9)
        ax.set_xticks(range(1, 11)); ax.tick_params(labelsize=7)
    for ax in axes[nsel:]:
        ax.axis("off")
    fig.suptitle("Selected alphas — demeaned fwd return by decile (green=long bucket, red=short bucket)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig("alpha101_results/selected_decile_charts.png", dpi=130)
    plt.close(fig)

    # C. IC table
    ic_df.to_csv("alpha101_results/selected_ic.csv")
    print("\n===== C. IC (all selected + dropped) =====")
    print(ic_df.sort_values("mean_ic", ascending=False).to_string())

    # D. dropped (earlier core, not selected now)
    dropped = [x for x in EARLIER_CORE if x not in selected]
    drows = []
    for name in dropped:
        net = enum.loc[name, "net_Sh"] if name in enum.index else np.nan
        gross = enum.loc[name, "gross_Sh"] if name in enum.index else np.nan
        turn = enum.loc[name, "turn_%"] if name in enum.index else np.nan
        if name == "alpha059":
            reason = "degenerate (single-year/indneutralize artifact)"
        elif pd.notna(net) and net < 0.5 and pd.notna(gross) and gross >= 1.0:
            reason = f"gross OK ({gross}) but costs/turnover kill net ({net}, turn {turn:.0f}%)"
        else:
            reason = f"net Sharpe {net} below 0.5"
        drows.append({"alpha": name, "gross_Sh": gross, "net_Sh": net, "turn_%": turn,
                      "mean_ic": ic_df.loc[name, "mean_ic"], "reason": reason})
    dropped_df = pd.DataFrame(drows)
    dropped_df.to_csv("alpha101_results/dropped_alphas.csv", index=False)
    print("\n===== D. DROPPED (earlier core -> not selected now) =====")
    print(dropped_df.to_string(index=False))

    # E. correlation matrix of selected best-design L/S PnL
    pnl_df = pd.DataFrame(pnl_sel).dropna(how="all")
    corr = pnl_df.corr()
    corr.to_csv("alpha101_results/selected_corr_matrix.csv")
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.index, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=6)
    ax.set_title("Selected alphas — best-design L/S PnL correlation")
    fig.colorbar(im, fraction=0.046, pad=0.04); fig.tight_layout()
    fig.savefig("alpha101_results/selected_corr_heatmap.png", dpi=130)
    plt.close(fig)
    avg = corr.where(~np.eye(len(corr), dtype=bool)).abs().stack().mean()
    print("\n===== E. CORRELATION (selected best-design L/S PnL) =====")
    print(corr.round(2).to_string())
    print(f"\navg |pairwise corr|: {avg:.2f}")
    print("\nSaved: selected_designs.csv, selected_decile_charts.png, selected_ic.csv, "
          "dropped_alphas.csv, selected_corr_matrix.csv, selected_corr_heatmap.png")


if __name__ == "__main__":
    main()
