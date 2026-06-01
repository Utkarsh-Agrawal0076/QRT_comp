"""
4-sleeve hysteresis inverse-vol ensemble (IC-significant survivors only:
alpha024, alpha041, alpha100, alpha043). Same frozen designs / blend logic;
reports IS 2010-2020 and OOS 2021+.
"""

import numpy as np
import pandas as pd

import alpha101 as a
from alpha101_bucket_enum import clean_returns, bucket_masks, LAG, EXEC_BPS, FIN_ANNUAL
from alpha101_ensemble import DESIGNS, hysteresis_state, normalize_dn

OOS_START = "2021-01-01"
SLEEVES = [24, 41, 100]


def period(gross, net, traded, book, label):
    def sh(x): return x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else np.nan
    turn = traded.mean() / book.mean() * 100 if book.mean() else np.nan
    dd = (net.cumsum() - net.cumsum().cummax()).min()
    print(f"  {label:20s}: Net Sh {sh(net):.2f} | Gross Sh {sh(gross):.2f} | "
          f"turn {turn:.0f}% | net ret {net.mean()*252*100:.1f}% | "
          f"vol {net.std()*np.sqrt(252)*100:.1f}% | maxDD {dd*100:.1f}%")
    return net


def main():
    data, returns, universe = a.load_panel(start="2010-01-01", end=None, verbose=True)
    returns = clean_returns(returns)
    uni = universe.astype(bool)

    sleeves = {}
    for n in SLEEVES:
        sig = a.get_alpha(n)(data)
        rp = bucket_masks(sig, universe)
        state = hysteresis_state(rp, **DESIGNS[n])
        w = normalize_dn(normalize_dn(state.astype(float)).shift(LAG).where(uni, 0.0))
        sleeves[n] = w
        print(f"  built alpha{n:03d}")

    pnl_df = pd.DataFrame({n: (sleeves[n] * returns.fillna(0)).sum(axis=1) for n in SLEEVES})
    vol = (pnl_df.rolling(60, min_periods=20).std() * np.sqrt(252)).clip(lower=0.05)
    inv = 1.0 / vol
    alloc = inv.div(inv.sum(axis=1), axis=0).shift(1).fillna(1.0 / len(SLEEVES))

    master = None
    for n in SLEEVES:
        c = sleeves[n].mul(alloc[n], axis=0)
        master = c if master is None else master.add(c, fill_value=0)
    master = normalize_dn(master)

    gross = (master * returns.fillna(0)).sum(axis=1)
    traded = master.diff().abs().sum(axis=1).fillna(0)
    book = master.abs().sum(axis=1)
    net = gross - traded * EXEC_BPS - book * (FIN_ANNUAL / 252)
    v = gross.ne(0).cumsum() > 0
    gross, net, traded, book = gross[v], net[v], traded[v], book[v]

    print("\n" + "=" * 70)
    print(f"3-SLEEVE ENSEMBLE (alpha024/041/100) — IS vs OOS")
    print("=" * 70)
    ism = net.index < OOS_START
    oosm = net.index >= OOS_START
    period(gross[ism], net[ism], traded[ism], book[ism], "IN-SAMPLE 2010-2020")
    period(gross[oosm], net[oosm], traded[oosm], book[oosm], "OUT-OF-SAMPLE 2021+")

    print("\n  --- Year-by-year net Sharpe ---")
    for y in range(2010, net.index[-1].year + 1):
        ny = net[net.index.year == y]
        if len(ny) < 20:
            continue
        sh = ny.mean() / ny.std() * np.sqrt(252) if ny.std() > 0 else np.nan
        print(f"    {y}: {sh:5.2f}{'  *OOS' if y >= 2021 else ''}")

    net.to_frame("net_pnl").to_csv("alpha101_results/ensemble_3sleeve_net_pnl.csv")
    print("\nSaved ensemble_3sleeve_net_pnl.csv")


if __name__ == "__main__":
    main()
