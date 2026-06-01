"""
Per-alpha decile bar charts of forward MARKET-EXCESS return (drift removed),
in the style of the user's residual-decile plot. Each bucket is coloured by the
statistical significance of its daily excess return:

    green  = significantly POSITIVE  (t > +2)   -> LONG bucket
    red    = significantly NEGATIVE  (t < -2)    -> SHORT bucket
    grey   = insignificant                       -> HYSTERESIS / neutral band

This identifies, per alpha, exactly which deciles to long and which to short
(capturing the whole edge on the table) and leaves a neutral buffer to cut
turnover.

In-sample 2010-2020, 5M-ADV universe. Horizon configurable (default T+1).
"""

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import alpha101 as a

NDEC = 10
TSIG = 2.0


def bucket_stats(sig, fwd_excess, uni, lag, horizon):
    s = sig.replace([np.inf, -np.inf], np.nan).shift(lag).where(uni)
    r = s.rank(axis=1, pct=True)
    b = (r * NDEC).clip(upper=NDEC - 1e-9).where(r.notna())
    means, tstats = [], []
    for d in range(NDEC):
        daily = fwd_excess.where((b >= d) & (b < d + 1)).mean(axis=1).dropna()
        # horizon-day non-overlapping-ish scaling: report per-`horizon`-day return
        m = daily.mean() * horizon * 1e4  # bps over the horizon
        t = daily.mean() / daily.std() * np.sqrt(len(daily)) if daily.std() else 0.0
        means.append(m); tstats.append(t)
    return np.array(means), np.array(tstats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=1, help="forward horizon in days (display scaling)")
    ap.add_argument("--lag", type=int, default=1)
    ap.add_argument("--set", default="core", choices=["core", "shortlist"])
    args = ap.parse_args()

    if args.set == "core":
        names = json.load(open("alpha101_results/genuine_core.json"))["decorrelated_core"]
        names = [n for n in names if n != "alpha059"]
    else:
        names = list(pd.read_csv("alpha101_results/shortlist.csv")["alpha"])
    nums = [int(n.replace("alpha", "")) for n in names]

    data, returns, universe = a.load_panel(start="2010-01-01", end="2020-12-31", verbose=True)
    uni = universe.astype(bool)
    fwd = returns.where(uni)
    fwd_excess = fwd.sub(fwd.mean(axis=1), axis=0)   # market-excess (residual drift removed)

    ncol = 3
    nrow = int(np.ceil(len(nums) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 3.2 * nrow))
    axes = np.array(axes).reshape(-1)

    classification = {}
    for ax, n in zip(axes, nums):
        name = f"alpha{n:03d}"
        sig = a.get_alpha(n)(data)
        means, tstats = bucket_stats(sig, fwd_excess, uni, args.lag, args.horizon)
        colors = ["green" if t > TSIG else "red" if t < -TSIG else "lightgrey" for t in tstats]
        ax.bar(range(1, NDEC + 1), means, color=colors, edgecolor="k", linewidth=0.4)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"{name}   (D10-D1 = {means[-1]-means[0]:.1f} bps)", fontsize=10)
        ax.set_xticks(range(1, NDEC + 1))
        ax.tick_params(labelsize=7)
        longs = [d + 1 for d in range(NDEC) if tstats[d] > TSIG]
        shorts = [d + 1 for d in range(NDEC) if tstats[d] < -TSIG]
        hyst = [d + 1 for d in range(NDEC) if abs(tstats[d]) <= TSIG]
        classification[name] = {"long": longs, "short": shorts, "hysteresis": hyst,
                                "means_bps": [round(x, 2) for x in means],
                                "tstats": [round(x, 1) for x in tstats]}
        print(f"{name}: SHORT D{shorts}  | hysteresis D{hyst} | LONG D{longs}")

    for ax in axes[len(nums):]:
        ax.axis("off")
    fig.suptitle(f"Forward market-excess return by signal decile "
                 f"(T+{args.lag}, {args.horizon}d horizon)  — red=short, grey=hysteresis, green=long",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = f"alpha101_results/decile_charts_{args.set}.png"
    fig.savefig(out, dpi=130)
    json.dump(classification, open(f"alpha101_results/decile_buckets_{args.set}.json", "w"), indent=2)
    print(f"\nSaved {out} and decile_buckets_{args.set}.json")


if __name__ == "__main__":
    main()
