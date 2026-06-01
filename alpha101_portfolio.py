"""
Hierarchical (two-level) portfolio allocation vs naive flat inverse-vol.

Sleeves:
  Reversal cluster : MR(=Alpha#11), ResMR, a024, a041, a100   (the 3 new = hysteresis designs)
  Momentum         : Mom
  Stat-Arb         : SA

Level 1: equal-risk (inverse-vol) blend WITHIN the reversal cluster -> 1 super-sleeve
         (keeps all sleeves; trades net; higher net Sharpe than alpha11 alone)
Level 2: across {Reversal, Mom, StatArb} a SHARPE-TILTED risk budget with a REVERSAL CAP
         (StatArb stays dominant by Sharpe; reversal can't exceed the cap)

Compared against:
  BASE-prod : flat inverse-vol over {MR, Mom, SA, ResMR}      (current production, no new alphas)
  BASE-naive: flat inverse-vol over all 7 sleeves             (the pathological 'reversal swamp')

Reports IS 2010-2020 / OOS 2021+ net Sharpe and realised risk contribution per style.
"""

import numpy as np
import pandas as pd

import alpha101 as a
from alpha101_bucket_enum import clean_returns, bucket_masks, LAG, EXEC_BPS, FIN_ANNUAL
from alpha101_ensemble import DESIGNS, hysteresis_state, normalize_dn

OOS_START = "2021-01-01"
REV_CAP = 0.375            # max RISK fraction to the reversal style
NEW = [24, 41, 100]


def net_pnl(w, returns):
    gross = (w * returns.fillna(0)).sum(axis=1)
    traded = w.diff().abs().sum(axis=1).fillna(0)
    book = w.abs().sum(axis=1)
    return gross - traded * EXEC_BPS - book * (FIN_ANNUAL / 252)


def inv_vol_blend(wmats, returns, names):
    """Equal-risk blend of weight matrices -> combined dollar-neutral weights."""
    pdf = pd.DataFrame({n: (wmats[n] * returns.fillna(0)).sum(axis=1) for n in names})
    vol = (pdf.rolling(60, min_periods=20).std() * np.sqrt(252)).clip(lower=0.05)
    alloc = (1 / vol).div((1 / vol).sum(axis=1), axis=0).shift(1).fillna(1.0 / len(names))
    out = None
    for n in names:
        c = wmats[n].mul(alloc[n], axis=0)
        out = c if out is None else out.add(c, fill_value=0)
    return normalize_dn(out)


def report(name, w, returns):
    net = net_pnl(w, returns)
    net = net[net.ne(0).cumsum() > 0]
    def sh(x): return x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else np.nan
    ism, oosm = net.index < OOS_START, net.index >= OOS_START
    dd = (net.cumsum() - net.cumsum().cummax()).min()
    print(f"  {name:28s} IS {sh(net[ism]):5.2f} | OOS {sh(net[oosm]):5.2f} | "
          f"full {sh(net):5.2f} | maxDD {dd*100:5.1f}%")
    return net


def risk_contrib(style_pnls, label):
    """Fractional risk contribution of each style to the combined book."""
    df = pd.DataFrame(style_pnls)
    port = df.sum(axis=1)
    rc = {k: np.cov(df[k], port)[0, 1] / port.var() for k in df.columns}
    s = sum(rc.values())
    print(f"    risk contribution [{label}]: " +
          "  ".join(f"{k}={rc[k]/s*100:4.0f}%" for k in df.columns))


def main():
    data, returns, universe = a.load_panel(start="2010-01-01", end=None, verbose=True)
    returns = clean_returns(returns)
    uni = universe.astype(bool)

    # new alpha sleeves (hysteresis)
    wmats = {}
    for n in NEW:
        sig = a.get_alpha(n)(data)
        rp = bucket_masks(sig, universe)
        st = hysteresis_state(rp, **DESIGNS[n])
        wmats[n] = normalize_dn(normalize_dn(st.astype(float)).shift(LAG).where(uni, 0.0))
        print(f"  built alpha{n:03d}")

    # live sleeves
    sl = pd.read_pickle("stores/sharpe_blender/sleeves.pkl")
    for key in ["MR", "Mom", "SA", "ResMR"]:
        wmats[key] = sl[key].reindex(index=returns.index, columns=returns.columns).fillna(0)

    print("\n========== ALLOCATION COMPARISON (net Sharpe) ==========")

    # --- baselines ---
    base_prod = inv_vol_blend(wmats, returns, ["MR", "Mom", "SA", "ResMR"])
    report("BASE-prod {MR,Mom,SA,Res}", base_prod, returns)

    naive_all = inv_vol_blend(wmats, returns, ["MR", "ResMR", 24, 41, 100, "Mom", "SA"])
    report("BASE-naive flat 7-sleeve", naive_all, returns)

    # --- hierarchical ---
    rev_names = ["MR", "ResMR", 24, 41, 100]
    rev_super = inv_vol_blend(wmats, returns, rev_names)          # Level 1
    styles = {"Reversal": rev_super, "Mom": wmats["Mom"], "SA": wmats["SA"]}

    # Level 2: Sharpe-tilted risk budget with reversal cap (causal: expanding Sharpe, shift 1)
    spnl = {k: net_pnl(v, returns) for k, v in styles.items()}
    spdf = pd.DataFrame(spnl)
    vol = (spdf.rolling(60, min_periods=20).std() * np.sqrt(252)).clip(lower=0.05)
    exp_sh = (spdf.expanding(min_periods=120).mean() / spdf.expanding(min_periods=120).std()
              * np.sqrt(252)).clip(lower=0.1).shift(1).fillna(1.0)
    rw = exp_sh.div(exp_sh.sum(axis=1), axis=0)                   # Sharpe-proportional risk weights
    # apply reversal cap, redistribute to Mom/SA by their Sharpe
    over = (rw["Reversal"] - REV_CAP).clip(lower=0)
    rw["Reversal"] = rw["Reversal"].clip(upper=REV_CAP)
    ms = exp_sh[["Mom", "SA"]].div(exp_sh[["Mom", "SA"]].sum(axis=1), axis=0)
    rw["Mom"] = rw["Mom"] + over * ms["Mom"]
    rw["SA"] = rw["SA"] + over * ms["SA"]
    cap_alloc = rw.div(vol)                                       # risk-weight -> capital, inverse vol
    master = None
    for k in styles:
        c = styles[k].mul(cap_alloc[k], axis=0)
        master = c if master is None else master.add(c, fill_value=0)
    master = normalize_dn(master)
    hier_net = report("HIER Sharpe-tilt+cap", master, returns)

    print("\n========== RISK CONTRIBUTION PER STYLE ==========")
    # decompose each method into the 3 styles' net pnl (using each method's implied style weights)
    def style_pnls_for(weight_combo_alloc):
        return {k: (cap_alloc[k] if weight_combo_alloc else 1) for k in styles}

    rc_styles_hier = {k: styles[k].mul(cap_alloc[k], axis=0) for k in styles}
    rc_styles_hier = {k: (v * returns.fillna(0)).sum(axis=1) for k, v in rc_styles_hier.items()}
    for lab, msk in [("IS 2010-2020", master.index < OOS_START), ("OOS 2021+", master.index >= OOS_START)]:
        risk_contrib({k: v[msk] for k, v in rc_styles_hier.items()}, lab)

    # naive flat: each individual sleeve equal-risk -> show reversal style share
    print("\n  (naive flat 7-sleeve, for contrast — reversal = MR+ResMR+a024+a041+a100)")
    rev_pnl_naive = sum((wmats[n] * returns.fillna(0)).sum(axis=1) for n in ["MR", "ResMR", 24, 41, 100])
    flat = {"Reversal": rev_pnl_naive,
            "Mom": (wmats["Mom"] * returns.fillna(0)).sum(axis=1),
            "SA": (wmats["SA"] * returns.fillna(0)).sum(axis=1)}
    # equal-risk weights from inverse vol of the 7 sleeves -> approximate by equal weight here
    for lab, msk in [("IS 2010-2020", flat["Mom"].index < OOS_START), ("OOS 2021+", flat["Mom"].index >= OOS_START)]:
        risk_contrib({k: v[msk] for k, v in flat.items()}, lab)

    pd.DataFrame({"hier_net": hier_net}).to_csv("alpha101_results/portfolio_hier_net_pnl.csv")
    print("\nSaved portfolio_hier_net_pnl.csv")


if __name__ == "__main__":
    main()
