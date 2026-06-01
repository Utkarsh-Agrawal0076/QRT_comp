"""
Show Momentum's CAPITAL weight vs RISK contribution in the HIER allocation, and
sweep the Momentum floor / risk-parity to see the trade-off (OOS Sharpe vs 2024).
Notebook conventions (Adj Close, normalize_and_cap, backtest cost model).
"""

import numpy as np
import pandas as pd

import alpha101 as a
from alpha101_bucket_enum import bucket_masks, LAG
from alpha101_ensemble import DESIGNS, hysteresis_state
from alpha101_consistent_ensemble import dd, nc_signed, bt, sh, invvol_blend

OOS = "2021-01-01"
REV_CAP = 0.375
NEW = [24, 41, 100]


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

    rev = invvol_blend({"MR": MR, "ResMR": ResMR, "a024": new_w[24],
                        "a041": new_w[41], "a100": new_w[100]}, rets)
    styles = {"Reversal": rev, "Mom": Mom, "SA": SA}
    spdf = pd.DataFrame({k: bt(v, rets, uni)[1] for k, v in styles.items()})
    vol = (spdf.rolling(60, min_periods=20).std() * np.sqrt(252)).clip(lower=0.05)
    esh = (spdf.expanding(min_periods=120).mean() / spdf.expanding(min_periods=120).std()
           * np.sqrt(252)).clip(lower=0.1).shift(1).fillna(1.0)

    def alloc(method, floor=0.15):
        if method == "risk_parity":
            rw = pd.DataFrame({k: 1/3 for k in styles}, index=esh.index)
        else:  # sharpe-tilt + reversal cap + mom floor
            base = esh.div(esh.sum(axis=1), axis=0)
            mom_w = base["Mom"].clip(lower=floor)
            rem = 1 - mom_w
            den = (esh["Reversal"] + esh["SA"]).replace(0, np.nan)
            rev_w = (rem * esh["Reversal"] / den).clip(upper=REV_CAP)
            rw = pd.DataFrame({"Reversal": rev_w, "Mom": mom_w, "SA": rem - rev_w})
        return rw.div(vol)

    def evaluate(ca):
        contribs = {k: dd(styles[k]).mul(ca[k], axis=0) for k in styles}
        raw = None
        for k in styles:
            raw = contribs[k] if raw is None else raw.add(contribs[k], fill_value=0)
        master = nc_signed(raw)
        g, n = bt(master, rets, uni)
        # capital share per style (fraction of gross notional, pre-normalisation)
        tot = sum(contribs[k].abs().sum(axis=1) for k in styles).replace(0, np.nan)
        capw = {k: (contribs[k].abs().sum(axis=1) / tot).mean() for k in styles}
        # risk contribution (net)
        cn = {k: bt(contribs[k], rets, uni)[1] for k in styles}
        df = pd.DataFrame(cn); port = df.sum(axis=1)
        rc = {k: np.cov(df[k][df.index >= OOS], port[df.index >= OOS])[0, 1] / port[df.index >= OOS].var() for k in styles}
        s = sum(rc.values())
        r24 = n[n.index.year == 2024]
        return n, capw, {k: rc[k]/s for k in styles}, r24.sum()*100

    print("\n  method            | n.full n.OOS g.1y | 2024 | Mom CAPw / RISKc | Rev CAPw | SA CAPw")
    configs = [("sharpe+floor0.15", lambda: alloc("st", 0.15)),
               ("sharpe+floor0.25", lambda: alloc("st", 0.25)),
               ("sharpe+floor0.35", lambda: alloc("st", 0.35)),
               ("risk_parity",      lambda: alloc("risk_parity"))]
    for name, fn in configs:
        ca = fn()
        npnl, capw, rc, r24 = evaluate(ca)
        # gross 1y from same master
        cont = {k: dd(styles[k]).mul(ca[k], axis=0) for k in styles}
        raw = None
        for k in styles:
            raw = cont[k] if raw is None else raw.add(cont[k], fill_value=0)
        gg, _ = bt(nc_signed(raw), rets, uni)
        print(f"  {name:17s} | {sh(npnl):6.2f} {sh(npnl[npnl.index>=OOS]):5.2f} {sh(gg.tail(252)):5.2f} | "
              f"{r24:+4.1f}% | {capw['Mom']*100:3.0f}% / {rc['Mom']*100:3.0f}% | "
              f"{capw['Reversal']*100:3.0f}% | {capw['SA']*100:3.0f}%")


if __name__ == "__main__":
    main()
