"""
Per-sleeve IS vs OOS decomposition of the 6 hysteresis alphas (standalone,
dollar-neutral, T+1). Shows which legs generalised and which decayed.
"""

import numpy as np
import pandas as pd

import alpha101 as a
from alpha101_bucket_enum import clean_returns, bucket_masks, LAG, EXEC_BPS, FIN_ANNUAL
from alpha101_ensemble import DESIGNS, hysteresis_state, normalize_dn

OOS_START = "2021-01-01"


def stats(net, traded, book):
    sh = net.mean() / net.std() * np.sqrt(252) if net.std() > 0 else np.nan
    turn = traded.mean() / book.mean() * 100 if book.mean() else np.nan
    dd = (net.cumsum() - net.cumsum().cummax()).min()
    return sh, turn, net.mean() * 252 * 100, dd * 100


def main():
    data, returns, universe = a.load_panel(start="2010-01-01", end=None, verbose=True)
    returns = clean_returns(returns)
    uni = universe.astype(bool)

    print(f"\n{'alpha':8s} | {'IS netSh':>8s} {'IS turn':>7s} {'IS ret%':>7s} | "
          f"{'OOS netSh':>9s} {'OOS turn':>8s} {'OOS ret%':>8s} {'OOS DD%':>7s} | {'retain':>6s}")
    print("-" * 92)
    for n, d in DESIGNS.items():
        sig = a.get_alpha(n)(data)
        rp = bucket_masks(sig, universe)
        state = hysteresis_state(rp, **d)
        w = normalize_dn(normalize_dn(state.astype(float)).shift(LAG).where(uni, 0.0))
        gross = (w * returns.fillna(0)).sum(axis=1)
        traded = w.diff().abs().sum(axis=1).fillna(0)
        book = w.abs().sum(axis=1)
        net = gross - traded * EXEC_BPS - book * (FIN_ANNUAL / 252)
        valid = gross.ne(0).cumsum() > 0
        net, traded, book = net[valid], traded[valid], book[valid]
        ism = net.index < OOS_START
        oosm = net.index >= OOS_START
        is_sh, is_t, is_r, _ = stats(net[ism], traded[ism], book[ism])
        oos_sh, oos_t, oos_r, oos_dd = stats(net[oosm], traded[oosm], book[oosm])
        retain = oos_sh / is_sh if is_sh and is_sh > 0 else np.nan
        print(f"alpha{n:03d} | {is_sh:8.2f} {is_t:6.0f}% {is_r:7.1f} | "
              f"{oos_sh:9.2f} {oos_t:7.0f}% {oos_r:8.1f} {oos_dd:7.1f} | {retain:6.0%}")


if __name__ == "__main__":
    main()
