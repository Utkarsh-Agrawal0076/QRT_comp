"""
FINAL configuration (locked): equal-vol across three styles.

  Reversal super-sleeve = inverse-vol blend of {MR(=a11), ResMR, a024, a041, a100}
  FINAL book            = inverse-vol (equal-vol) blend of {Reversal, Mom, SA}

Notebook conventions: Adj-Close returns, normalize_and_cap books, backtest cost
model (2bps + 0.5% financing). Produces the full per-year tearsheet (gross & net)
vs the live 3-sleeve book, and saves the deployable weight matrices.
"""

import numpy as np
import pandas as pd

import alpha101 as a
from alpha101_bucket_enum import bucket_masks, LAG
from alpha101_ensemble import DESIGNS, hysteresis_state
from alpha101_consistent_ensemble import dd, nc_signed, bt, sh, invvol_blend

OOS = "2021-01-01"
NEW = [24, 41, 100]


def heads(g, n, lbl):
    def w(p, m=None):
        pp = p if m is None else p[m]
        return sh(pp)
    ism, oosm = g.index < OOS, g.index >= OOS
    print(f"  {lbl:12s} | gross: full {sh(g):.2f} IS {w(g,ism):.2f} OOS {w(g,oosm):.2f} 1y {sh(g.tail(252)):.2f}"
          f"  | net: full {sh(n):.2f} IS {w(n,ism):.2f} OOS {w(n,oosm):.2f} 1y {sh(n.tail(252)):.2f}")


def main():
    h = pd.read_pickle("top_5000_yf_data.pkl")
    rets = dd(h["Adj Close"]).pct_change(fill_method=None).fillna(0)
    advv = dd(h["Close"]).mul(dd(h["Volume"])).fillna(0)
    uni = dd((advv.rolling(60, min_periods=60).mean() >= 5e6).astype(int)).reindex(columns=rets.columns, fill_value=0)
    data, _, universe = a.load_panel(start="2010-01-01", end=None, verbose=False)

    new_w = {}
    for n in NEW:
        sig = a.get_alpha(n)(data)
        st = hysteresis_state(bucket_masks(sig, universe), **DESIGNS[n]).astype(float)
        new_w[n] = nc_signed(nc_signed(st).shift(LAG))
        print(f"  built alpha{n:03d}")
    sl = pd.read_pickle("stores/sharpe_blender/sleeves.pkl")
    MR, Mom, SA, ResMR = sl["MR"], sl["Mom"], sl["SA"], sl["ResMR"]

    # Level 1: reversal super-sleeve
    rev = invvol_blend({"MR": MR, "ResMR": ResMR, "a024": new_w[24],
                        "a041": new_w[41], "a100": new_w[100]}, rets)
    # Level 2: equal-vol across the three styles
    final = invvol_blend({"Reversal": rev, "Mom": Mom, "SA": SA}, rets)
    live3 = invvol_blend({"MR": MR, "Mom": Mom, "SA": SA}, rets)

    gf, nf = bt(final, rets, uni)
    gl, nl = bt(live3, rets, uni)

    print("\n================= HEADLINE (gross & net Sharpe) =================")
    heads(gl, nl, "LIVE-3")
    heads(gf, nf, "FINAL eqvol")

    print("\n================= PER-YEAR TEARSHEET =================")
    print(f"  {'year':>5s} | {'LIVE net%':>9s} {'LIVE Sh':>7s} | {'FINAL net%':>10s} {'FINAL Sh':>8s} "
          f"| {'FINAL gross%':>12s}")
    for y in range(2010, gf.index[-1].year + 1):
        nly = nl[nl.index.year == y]; nfy = nf[nf.index.year == y]; gfy = gf[gf.index.year == y]
        if len(nfy) < 20:
            continue
        def s(p): return p.mean() / p.std() * np.sqrt(252) if p.std() > 0 else np.nan
        star = " *OOS" if y >= 2021 else ""
        print(f"  {y:>5d} | {nly.sum()*100:8.1f}% {s(nly):7.2f} | {nfy.sum()*100:9.1f}% {s(nfy):8.2f} "
              f"| {gfy.sum()*100:11.1f}%{star}")

    # save deployable weights
    final.to_parquet("alpha101_results/FINAL_ensemble_weights.parquet")
    rev.to_parquet("alpha101_results/FINAL_reversal_supersleeve.parquet")
    for n in NEW:
        new_w[n].to_parquet(f"alpha101_results/FINAL_alpha{n:03d}_weights.parquet")
    print("\nSaved FINAL_ensemble_weights.parquet, FINAL_reversal_supersleeve.parquet, "
          "FINAL_alpha0{24,41,100}_weights.parquet")


if __name__ == "__main__":
    main()
