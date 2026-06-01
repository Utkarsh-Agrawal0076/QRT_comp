"""
Out-of-sample test of the frozen 6-alpha hysteresis inverse-vol ensemble.

Designs (which alphas, buckets, buffers) were fixed on 2010-2020. Here we run the
SAME logic continuously over the full history and report IS (2010-2020) vs
OOS (2021+) with no refitting. The inverse-vol allocation is causal (trailing
60d vol, shifted 1), so letting it run across the boundary introduces no leakage.
"""

import numpy as np
import pandas as pd

import alpha101 as a
from alpha101_bucket_enum import clean_returns, bucket_masks, LAG, EXEC_BPS, FIN_ANNUAL
from alpha101_ensemble import DESIGNS, hysteresis_state, normalize_dn

OOS_START = "2021-01-01"


def period_stats(gross, net, traded, book, label):
    def sh(x): return x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else np.nan
    turn = traded.mean() / book.mean() * 100 if book.mean() else np.nan
    dd = (net.cumsum() - net.cumsum().cummax()).min()
    print(f"\n  {label}: {net.index[0].date()} -> {net.index[-1].date()}  ({len(net)} days)")
    print(f"    Gross Sharpe {sh(gross):.2f} | Net Sharpe {sh(net):.2f} | "
          f"turnover {turn:.0f}% | net ann ret {net.mean()*252*100:.1f}% | "
          f"vol {net.std()*np.sqrt(252)*100:.1f}% | maxDD {dd*100:.1f}%")


def main():
    data, returns, universe = a.load_panel(start="2010-01-01", end=None, verbose=True)
    returns = clean_returns(returns)
    uni = universe.astype(bool)

    sleeves = {}
    for n, d in DESIGNS.items():
        sig = a.get_alpha(n)(data)
        rp = bucket_masks(sig, universe)
        state = hysteresis_state(rp, **d)
        w = normalize_dn(state.astype(float)).shift(LAG)
        w = w.where(uni, 0.0)
        sleeves[n] = normalize_dn(w)
        print(f"  built alpha{n:03d}")

    pnl_df = pd.DataFrame({n: (sleeves[n] * returns.fillna(0)).sum(axis=1) for n in DESIGNS})
    vol = (pnl_df.rolling(60, min_periods=20).std() * np.sqrt(252)).clip(lower=0.05)
    inv = 1.0 / vol
    alloc = inv.div(inv.sum(axis=1), axis=0).shift(1).fillna(1.0 / len(DESIGNS))

    master = None
    for n in DESIGNS:
        contrib = sleeves[n].mul(alloc[n], axis=0)
        master = contrib if master is None else master.add(contrib, fill_value=0)
    master = normalize_dn(master)

    gross = (master * returns.fillna(0)).sum(axis=1)
    traded = master.diff().abs().sum(axis=1).fillna(0)
    book = master.abs().sum(axis=1)
    net = gross - traded * EXEC_BPS - book * (FIN_ANNUAL / 252)
    valid = gross.ne(0).cumsum() > 0
    gross, net, traded, book = gross[valid], net[valid], traded[valid], book[valid]

    print("\n" + "=" * 66)
    print("FROZEN 6-ALPHA HYSTERESIS INVERSE-VOL ENSEMBLE — IS vs OOS")
    print("=" * 66)
    is_m = net.index < OOS_START
    oos_m = net.index >= OOS_START
    period_stats(gross[is_m], net[is_m], traded[is_m], book[is_m], "IN-SAMPLE 2010-2020")
    period_stats(gross[oos_m], net[oos_m], traded[oos_m], book[oos_m], "OUT-OF-SAMPLE 2021+")

    print("\n  --- Year-by-year net Sharpe (OOS years bold-starred) ---")
    print(f"  {'year':>5s} {'netSharpe':>10s} {'annret%':>8s}")
    for y in range(2010, net.index[-1].year + 1):
        ny = net[net.index.year == y]
        if len(ny) < 20:
            continue
        sh = ny.mean() / ny.std() * np.sqrt(252) if ny.std() > 0 else np.nan
        star = " *OOS" if y >= 2021 else ""
        print(f"  {y:>5d} {sh:10.2f} {ny.mean()*252*100:8.1f}{star}")

    net.to_frame("net_pnl").to_csv("alpha101_results/ensemble_oos_net_pnl.csv")
    print("\nSaved ensemble_oos_net_pnl.csv")


if __name__ == "__main__":
    main()
