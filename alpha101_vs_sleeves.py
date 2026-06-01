"""
Correlation of the 4 new alpha sleeves (024/041/100/043, hysteresis designs) and
their inverse-vol ensemble against the live master_ensemble sleeves:
MR (=Alpha#11), Momentum, Stat-Arb, ResMR. Daily gross-PnL correlation,
full-sample and OOS (2021+).
"""

import numpy as np
import pandas as pd

import alpha101 as a
from alpha101_bucket_enum import clean_returns, bucket_masks, LAG
from alpha101_ensemble import DESIGNS, hysteresis_state, normalize_dn

OOS_START = "2021-01-01"
SLEEVES = [24, 41, 100, 43]


def main():
    data, returns, universe = a.load_panel(start="2010-01-01", end=None, verbose=True)
    returns = clean_returns(returns)
    uni = universe.astype(bool)
    rfill = returns.fillna(0)

    # --- new alpha sleeves ---
    pnl = {}
    wmats = {}
    for n in SLEEVES:
        sig = a.get_alpha(n)(data)
        rp = bucket_masks(sig, universe)
        st = hysteresis_state(rp, **DESIGNS[n])
        w = normalize_dn(normalize_dn(st.astype(float)).shift(LAG).where(uni, 0.0))
        wmats[n] = w
        pnl[f"a{n:03d}"] = (w * rfill).sum(axis=1)
        print(f"  built alpha{n:03d}")

    # inverse-vol ensemble of the 4
    pdf = pd.DataFrame({n: (wmats[n] * rfill).sum(axis=1) for n in SLEEVES})
    vol = (pdf.rolling(60, min_periods=20).std() * np.sqrt(252)).clip(lower=0.05)
    alloc = (1 / vol).div((1 / vol).sum(axis=1), axis=0).shift(1).fillna(0.25)
    master = None
    for n in SLEEVES:
        c = wmats[n].mul(alloc[n], axis=0)
        master = c if master is None else master.add(c, fill_value=0)
    master = normalize_dn(master)
    pnl["ENS4"] = (master * rfill).sum(axis=1)

    # --- live sleeves ---
    sl = pd.read_pickle("stores/sharpe_blender/sleeves.pkl")
    for name, key in [("MR(=a11)", "MR"), ("Mom", "Mom"), ("StatArb", "SA"), ("ResMR", "ResMR")]:
        w = sl[key].reindex(index=returns.index, columns=returns.columns).fillna(0)
        pnl[name] = (w * rfill).sum(axis=1)

    P = pd.DataFrame(pnl)
    order = ["a024", "a041", "a100", "a043", "ENS4", "MR(=a11)", "Mom", "StatArb", "ResMR"]
    P = P[order]

    def show(df, label):
        c = df.corr()
        print(f"\n===== {label} daily PnL correlation =====")
        print(c.round(2).to_string())
        # focus block: new vs live
        new = ["a024", "a041", "a100", "a043", "ENS4"]
        live = ["MR(=a11)", "Mom", "StatArb", "ResMR"]
        print(f"\n  {label}: new-sleeve vs live-sleeve correlations")
        print(c.loc[new, live].round(2).to_string())

    full = P.loc[P["a024"].ne(0).cumsum() > 0]
    show(full, "FULL 2010-2026")
    show(full[full.index >= OOS_START], "OOS 2021+")

    full.corr().to_csv("alpha101_results/vs_sleeves_corr_full.csv")
    print("\nSaved vs_sleeves_corr_full.csv")


if __name__ == "__main__":
    main()
