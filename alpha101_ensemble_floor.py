"""
Hierarchical allocator WITH a Momentum floor vs WITHOUT, to confirm the floor
protects the 2024/2026 momentum payoffs (the regime hedge for reversal).

Level 1: inv-vol reversal super-sleeve (MR+ResMR+a024+a041+a100).
Level 2: Sharpe-tilt risk budget across {Reversal, Mom, SA} with
         reversal cap 0.375 and an optional Momentum floor.
"""

import numpy as np
import pandas as pd

import alpha101 as a
from alpha101_bucket_enum import clean_returns, bucket_masks, LAG
from alpha101_ensemble import DESIGNS, hysteresis_state, normalize_dn
from alpha101_portfolio import net_pnl, inv_vol_blend

OOS = "2021-01-01"
REV_CAP = 0.375
NEW = [24, 41, 100]


def build_master(styles, returns, mom_floor):
    spdf = pd.DataFrame({k: net_pnl(v, returns) for k, v in styles.items()})
    vol = (spdf.rolling(60, min_periods=20).std() * np.sqrt(252)).clip(lower=0.05)
    sh = (spdf.expanding(min_periods=120).mean() / spdf.expanding(min_periods=120).std()
          * np.sqrt(252)).clip(lower=0.1).shift(1).fillna(1.0)
    base = sh.div(sh.sum(axis=1), axis=0)
    mom_w = base["Mom"].clip(lower=mom_floor)
    rem = 1 - mom_w
    denom = (sh["Reversal"] + sh["SA"]).replace(0, np.nan)
    rev_w = (rem * sh["Reversal"] / denom).clip(upper=REV_CAP)
    sa_w = rem - rev_w
    rw = pd.DataFrame({"Reversal": rev_w, "Mom": mom_w, "SA": sa_w})
    cap_alloc = rw.div(vol)
    master = None
    for k in styles:
        c = styles[k].mul(cap_alloc[k], axis=0)
        master = c if master is None else master.add(c, fill_value=0)
    return normalize_dn(master), cap_alloc, {k: styles[k].mul(cap_alloc[k], axis=0) for k in styles}


def stats(net, lab):
    net = net[net.ne(0).cumsum() > 0]
    def sh(x): return x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else np.nan
    ism, oosm = net.index < OOS, net.index >= OOS
    dd_o = (net[oosm].cumsum() - net[oosm].cumsum().cummax()).min()
    r24 = net[net.index.year == 2024]
    print(f"  {lab:18s} IS {sh(net[ism]):5.2f} | OOS {sh(net[oosm]):5.2f} "
          f"| OOS ret {net[oosm].mean()*252*100:4.1f}% | OOS maxDD {dd_o*100:5.1f}% "
          f"| 2024 {r24.sum()*100:+5.1f}% ({sh(r24):+.2f})")
    return net


def main():
    data, returns, universe = a.load_panel(start="2010-01-01", end=None, verbose=True)
    returns = clean_returns(returns)
    uni = universe.astype(bool)

    wmats = {}
    for n in NEW:
        sig = a.get_alpha(n)(data)
        rp = bucket_masks(sig, universe)
        st = hysteresis_state(rp, **DESIGNS[n])
        wmats[n] = normalize_dn(normalize_dn(st.astype(float)).shift(LAG).where(uni, 0.0))
        print(f"  built alpha{n:03d}")
    sl = pd.read_pickle("stores/sharpe_blender/sleeves.pkl")
    for k in ["MR", "Mom", "SA", "ResMR"]:
        wmats[k] = sl[k].reindex(index=returns.index, columns=returns.columns).fillna(0)

    rev = inv_vol_blend(wmats, returns, ["MR", "ResMR", 24, 41, 100])
    styles = {"Reversal": rev, "Mom": wmats["Mom"], "SA": wmats["SA"]}

    print("\n========= HIERARCHICAL: no-floor vs Momentum floor =========")
    m0, a0, c0 = build_master(styles, returns, mom_floor=0.0)
    net0 = stats(net_pnl(m0, returns), "no floor")
    m1, a1, c1 = build_master(styles, returns, mom_floor=0.15)
    net1 = stats(net_pnl(m1, returns), "Mom floor 15%")

    print("\n  --- OOS yearly net return % (Sharpe) : Mom-floor ensemble ---")
    for y in range(2021, net1.index[-1].year + 1):
        p = net1[net1.index.year == y]
        if len(p) < 20:
            continue
        sh = p.mean() / p.std() * np.sqrt(252) if p.std() > 0 else np.nan
        print(f"    {y}: {p.sum()*100:+5.1f}% ({sh:+.2f})")

    def rc(contribs, msk, lab):
        df = pd.DataFrame({k: (v * returns.fillna(0)).sum(axis=1)[msk] for k, v in contribs.items()})
        port = df.sum(axis=1)
        r = {k: np.cov(df[k], port)[0, 1] / port.var() for k in df}
        s = sum(r.values())
        print(f"    {lab}: " + "  ".join(f"{k}={r[k]/s*100:3.0f}%" for k in df))

    print("\n  --- OOS realised risk contribution ---")
    rc(c0, m0.index >= OOS, "no floor   ")
    rc(c1, m1.index >= OOS, "Mom floor  ")
    print(f"\n  --- avg Momentum capital alloc (OOS) ---")
    print(f"    no floor: {a0['Mom'][a0.index>=OOS].mean():.3f} | "
          f"floor: {a1['Mom'][a1.index>=OOS].mean():.3f}")

    net1.to_frame("net").to_csv("alpha101_results/ensemble_floor_net_pnl.csv")
    print("\nSaved ensemble_floor_net_pnl.csv")


if __name__ == "__main__":
    main()
