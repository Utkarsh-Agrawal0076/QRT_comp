"""
Performance of the AS-WIRED 4-leg reversal super-sleeve (Alpha#11 + a024 + a041
+ a100, NO ResMR) vs the 5-leg version and the live 3-sleeve book.
Notebook conventions (Adj-Close, normalize_and_cap, backtest cost model).
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
    ism, oosm = g.index < OOS, g.index >= OOS
    print(f"  {lbl:18s} | gross: full {sh(g):.2f} OOS {sh(g[oosm]):.2f} 1y {sh(g.tail(252)):.2f}"
          f"  | net: full {sh(n):.2f} IS {sh(n[ism]):.2f} OOS {sh(n[oosm]):.2f} 1y {sh(n.tail(252)):.2f}")


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

    rev4 = invvol_blend({"MR": MR, "a024": new_w[24], "a041": new_w[41], "a100": new_w[100]}, rets)
    rev5 = invvol_blend({"MR": MR, "ResMR": ResMR, "a024": new_w[24], "a041": new_w[41], "a100": new_w[100]}, rets)

    live3 = invvol_blend({"MR": MR, "Mom": Mom, "SA": SA}, rets)
    final4 = invvol_blend({"Rev": rev4, "Mom": Mom, "SA": SA}, rets)
    final5 = invvol_blend({"Rev": rev5, "Mom": Mom, "SA": SA}, rets)

    print("\n  legend: full / IS<=2020 / OOS>=2021 / 1y=trailing252\n")
    heads(*bt(live3, rets, uni), "LIVE-3")
    heads(*bt(final4, rets, uni), "FINAL 4-leg (WIRED)")
    heads(*bt(final5, rets, uni), "FINAL 5-leg (+ResMR)")

    # per-year net for the wired 4-leg vs live
    gl, nl = bt(live3, rets, uni)
    g4, n4 = bt(final4, rets, uni)
    print("\n  per-year net % (Sharpe):  LIVE-3   vs   FINAL 4-leg")
    for y in range(2010, n4.index[-1].year + 1):
        a_ = nl[nl.index.year == y]; b_ = n4[n4.index.year == y]
        if len(b_) < 20:
            continue
        def s(p): return p.mean()/p.std()*np.sqrt(252) if p.std() > 0 else np.nan
        star = " *OOS" if y >= 2021 else ""
        print(f"    {y}:  {a_.sum()*100:+5.1f}% ({s(a_):+.2f})   {b_.sum()*100:+5.1f}% ({s(b_):+.2f}){star}")


if __name__ == "__main__":
    main()
