"""
Filter the screened 101 alphas down to a practical "worth trying" shortlist.

The literal mean-IC>0.02 bar is too high for single formulaic alphas at T+1 on
this universe (ICs run ~0.003-0.013). Instead we keep alphas with a
*statistically reliable* edge of meaningful size, in EITHER direction (a strong
negative-IC alpha is traded by flipping the sign).

Default "worth trying" gate (all on the 2010-2020 in-sample screen):
    |t-stat|  >= 3.0     # IC reliably different from zero over the sample
    |mean IC| >= 0.004   # at least as strong as the textbook 1-day reversal
Tiers:
    A : |t|>=4  and |mean IC|>=0.006     (lead candidates)
    B : |t|>=3  and |mean IC|>=0.004     (worth trying)
Regime-only alphas (fail the gate but flagged regime_candidate) are listed
separately as conditional/timing candidates.

Usage:
    python alpha101_shortlist.py
    python alpha101_shortlist.py --tmin 3 --icmin 0.004
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

OUT_DIR = "alpha101_results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmin", type=float, default=3.0, help="min |t-stat|")
    ap.add_argument("--icmin", type=float, default=0.004, help="min |mean IC|")
    ap.add_argument("--metrics", default=os.path.join(OUT_DIR, "all_metrics.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.metrics)
    df["abs_ic"] = df["mean_ic"].abs()
    df["abs_t"] = df["t_stat"].abs()
    df["abs_sharpe"] = df["ls_sharpe"].abs()
    df["direction"] = np.where(df["mean_ic"] >= 0, "long", "short(flip)")

    gate = (df["abs_t"] >= args.tmin) & (df["abs_ic"] >= args.icmin)
    short = df[gate].copy()

    def tier(r):
        if r["abs_t"] >= 4 and r["abs_ic"] >= 0.006:
            return "A"
        if r["abs_t"] >= 3 and r["abs_ic"] >= 0.004:
            return "B"
        return "C"
    short["tier"] = short.apply(tier, axis=1)
    short = short.sort_values(["tier", "abs_t"], ascending=[True, False])

    cols = ["alpha", "tier", "direction", "mean_ic", "abs_t", "ic_ir",
            "ls_sharpe", "hit_rate", "year_spread", "regime_candidate",
            "indneutral_approx", "uses_cap_proxy"]
    print("=" * 80)
    print(f"WORTH-TRYING SHORTLIST  (gate: |t|>={args.tmin}, |IC|>={args.icmin}; "
          f"in-sample 2010-2020, T+1)")
    print("=" * 80)
    if len(short):
        show = short[cols].copy()
        show["mean_ic"] = show["mean_ic"].map(lambda v: f"{v:+.4f}")
        show["abs_t"] = show["abs_t"].map(lambda v: f"{v:.1f}")
        print(show.to_string(index=False))
    else:
        print("  (nothing clears the gate — loosen --tmin/--icmin)")

    # regime-only (fail gate but flagged regime candidate)
    reg = df[(~gate) & (df["regime_candidate"])].sort_values("year_spread", ascending=False)
    print("\n" + "-" * 80)
    print(f"REGIME / TIMING-ONLY candidates ({len(reg)}): strong some years, weak others")
    print("-" * 80)
    if len(reg):
        rcols = ["alpha", "mean_ic", "best_year", "best_year_ic",
                 "worst_year", "worst_year_ic", "year_spread"]
        print(reg[rcols].to_string(index=False))

    # persist
    out_csv = os.path.join(OUT_DIR, "shortlist.csv")
    short[cols].to_csv(out_csv, index=False)
    payload = {
        "gate": {"abs_t_min": args.tmin, "abs_ic_min": args.icmin,
                 "sample": "2010-2020 in-sample", "lag": "T+1",
                 "universe": "5M 60-day ADV"},
        "shortlist": short[cols].to_dict("records"),
        "regime_only": reg[["alpha", "mean_ic", "year_spread"]].to_dict("records"),
    }
    with open(os.path.join(OUT_DIR, "shortlist.json"), "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print(f"Tier A: {list(short[short.tier=='A']['alpha'])}")
    print(f"Tier B: {list(short[short.tier=='B']['alpha'])}")
    print(f"Total worth-trying: {len(short)} | regime-only: {len(reg)}")
    print(f"Saved -> {out_csv} and shortlist.json")


if __name__ == "__main__":
    main()
