"""
New-alpha hierarchical allocation, in the master_ensemble notebook's conventions:
  - returns = Adj Close pct_change
  - universe_5m = adv60 >= 5M
  - books built with normalize_and_cap (split long/short to 0.5, cap 0.099)
  - gross & net per backtest_portfolio cost model (2bps + 0.5% financing)

Compares:
  LIVE-3     : inv-vol {MR(=a11), Mom, SA}                       (the 2.63 gross-1y book)
  LIVE+a024  : inv-vol {MR, Mom, SA, a024}                       (minimal addition)
  HIER       : Reversal super-sleeve {MR,ResMR,a024,a041,a100}
               + {Mom, SA}, Sharpe-tilt, reversal cap 0.375, Mom floor 0.15
Reports gross/net Sharpe full, trailing-1y, IS(<=2020), OOS(2021+) + risk contribution.
"""

import numpy as np
import pandas as pd

import alpha101 as a
from alpha101_bucket_enum import bucket_masks, LAG
from alpha101_ensemble import DESIGNS, hysteresis_state

OOS = "2021-01-01"
REV_CAP, MOM_FLOOR = 0.375, 0.15
NEW = [24, 41, 100]


def dd(df): return df.loc[:, ~df.columns.duplicated()]


def nc(w, target=0.50, cap=0.099):
    rs = w.sum(axis=1) + 1e-10
    w = (w.div(rs, axis=0) * target).clip(upper=cap)
    rs2 = w.sum(axis=1) + 1e-10
    return w.div(rs2, axis=0) * target


def nc_signed(w):
    return (nc(w.where(w > 0, 0)) - nc(w.where(w < 0, 0).abs())).fillna(0)


def bt(w, rets, uni):
    w = dd(w).reindex(index=rets.index, columns=rets.columns).fillna(0) * uni
    gross = (w * rets.fillna(0)).sum(axis=1)
    traded = w.diff().abs().sum(axis=1).fillna(0)
    book = w.abs().sum(axis=1)
    net = gross - traded * 2e-4 - book * (0.005 / 252)
    return gross, net


def sh(p):
    p = p[p.ne(0).cumsum() > 0]
    return p.mean() / p.std() * np.sqrt(252) if p.std() > 0 else np.nan


def invvol_blend(wlist, rets):
    iv = {}
    for k, w in wlist.items():
        pnl = (dd(w).reindex(index=rets.index, columns=rets.columns).fillna(0) * rets.fillna(0)).sum(axis=1)
        iv[k] = 1 / (pnl.rolling(60, min_periods=20).std() * np.sqrt(252)).clip(lower=0.05)
    s = sum(iv.values())
    out = None
    for k, w in wlist.items():
        c = dd(w).mul((iv[k] / s).shift(1).fillna(1 / len(wlist)), axis=0)
        out = c if out is None else out.add(c, fill_value=0)
    return nc_signed(out)


def report(name, w, rets, uni):
    g, n = bt(w, rets, uni)
    print(f"  {name:14s} | g.full {sh(g):5.2f} n.full {sh(n):5.2f} | "
          f"g.1y {sh(g.tail(252)):5.2f} n.1y {sh(n.tail(252)):5.2f} | "
          f"n.IS {sh(n[n.index < OOS]):5.2f} n.OOS {sh(n[n.index >= OOS]):5.2f}")
    return g, n


def main():
    h = pd.read_pickle("top_5000_yf_data.pkl")
    rets = dd(h["Adj Close"]).pct_change(fill_method=None).fillna(0)
    advv = dd(h["Close"]).mul(dd(h["Volume"])).fillna(0)
    uni = dd((advv.rolling(60, min_periods=60).mean() >= 5_000_000).astype(int)).reindex(columns=rets.columns, fill_value=0)

    data, _, universe = a.load_panel(start="2010-01-01", end=None, verbose=False)

    # new alpha books (hysteresis -> dollar-neutral, T+1, capped)
    new_w = {}
    for nidx in NEW:
        sig = a.get_alpha(nidx)(data)
        rp = bucket_masks(sig, universe)
        st = hysteresis_state(rp, **DESIGNS[nidx]).astype(float)
        new_w[nidx] = nc_signed(nc_signed(st).shift(LAG))
        print(f"  built alpha{nidx:03d}")

    sl = pd.read_pickle("stores/sharpe_blender/sleeves.pkl")
    MR, Mom, SA, ResMR = sl["MR"], sl["Mom"], sl["SA"], sl["ResMR"]

    print("\n  legend: g=gross n=net | full / 1y=trailing252 / IS<=2020 / OOS>=2021\n")

    # LIVE-3
    live3 = invvol_blend({"MR": MR, "Mom": Mom, "SA": SA}, rets)
    report("LIVE-3", live3, rets, uni)

    # LIVE + a024
    live_a024 = invvol_blend({"MR": MR, "Mom": Mom, "SA": SA, "a024": new_w[24]}, rets)
    report("LIVE+a024", live_a024, rets, uni)

    # HIER
    rev = invvol_blend({"MR": MR, "ResMR": ResMR, "a024": new_w[24],
                        "a041": new_w[41], "a100": new_w[100]}, rets)
    styles = {"Reversal": rev, "Mom": Mom, "SA": SA}
    spdf = pd.DataFrame({k: bt(v, rets, uni)[1] for k, v in styles.items()})
    vol = (spdf.rolling(60, min_periods=20).std() * np.sqrt(252)).clip(lower=0.05)
    esh = (spdf.expanding(min_periods=120).mean() / spdf.expanding(min_periods=120).std()
           * np.sqrt(252)).clip(lower=0.1).shift(1).fillna(1.0)
    base = esh.div(esh.sum(axis=1), axis=0)
    mom_w = base["Mom"].clip(lower=MOM_FLOOR)
    rem = 1 - mom_w
    den = (esh["Reversal"] + esh["SA"]).replace(0, np.nan)
    rev_w = (rem * esh["Reversal"] / den).clip(upper=REV_CAP)
    sa_w = rem - rev_w
    rw = pd.DataFrame({"Reversal": rev_w, "Mom": mom_w, "SA": sa_w})
    ca = rw.div(vol)
    master = None
    for k in styles:
        c = dd(styles[k]).mul(ca[k], axis=0)
        master = c if master is None else master.add(c, fill_value=0)
    master = nc_signed(master)
    report("HIER", master, rets, uni)

    # risk contribution (net) per style
    print("\n  risk contribution per style (net pnl):")
    contrib = {k: bt(dd(styles[k]).mul(ca[k], axis=0), rets, uni)[1] for k in styles}
    for lab, msk in [("full", slice(None)), ("OOS", master.index >= OOS)]:
        df = pd.DataFrame({k: v.loc[msk] if lab != "full" else v for k, v in contrib.items()})
        port = df.sum(axis=1)
        rc = {k: np.cov(df[k], port)[0, 1] / port.var() for k in df}
        s = sum(rc.values())
        print(f"    {lab:4s}: " + "  ".join(f"{k}={rc[k]/s*100:3.0f}%" for k in df))


if __name__ == "__main__":
    main()
