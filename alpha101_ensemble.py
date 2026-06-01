"""
Inverse-volatility ensemble of the 6 orthogonal alphas, each traded with an
ASYMMETRIC HYSTERESIS design (enter on core band, hold through buffer band, exit
beyond it -> lower turnover), per the user's per-alpha spec.

Blend follows master_ensemble cell 5: 60d rolling-vol inverse-vol allocation
(shifted 1 day, causal), combine the WEIGHT matrices so trades net internally,
then re-normalize each book to dollar-neutral GMV=1.

Test window 2010-2020, T+1, 5M-ADV universe. Costs: 2bps/trade + 0.5% ann financing.
"""

import numpy as np
import pandas as pd

import alpha101 as a
from alpha101_bucket_enum import clean_returns, bucket_masks, LAG, EXEC_BPS, FIN_ANNUAL

# (long_enter, long_exit, short_enter, short_exit) in pct-rank units
DESIGNS = {
    24:  dict(le=0.70, lx=0.60, se=0.10, sx=0.20),
    41:  dict(le=0.70, lx=0.60, se=0.10, sx=0.20),
    100: dict(le=0.90, lx=0.80, se=0.50, sx=0.50),
    43:  dict(le=0.90, lx=0.90, se=0.35, sx=0.40),
    19:  dict(le=0.80, lx=0.70, se=0.40, sx=0.50),
    73:  dict(le=0.80, lx=0.70, se=0.10, sx=0.20),
}


def hysteresis_state(rp, le, lx, se, sx):
    """Stateful per-stock long(+1)/short(-1)/flat(0) with entry/exit bands."""
    R = rp.values
    T, N = R.shape
    state = np.zeros(N)
    out = np.zeros((T, N))
    for t in range(T):
        r = R[t]
        valid = ~np.isnan(r)
        new = np.zeros(N)
        # maintain through buffer band
        new[(state == 1) & valid & (r >= lx)] = 1
        new[(state == -1) & valid & (r < sx)] = -1
        # fresh entries override
        new[valid & (r >= le)] = 1
        new[valid & (r < se)] = -1
        state = new
        out[t] = new
    return pd.DataFrame(out, index=rp.index, columns=rp.columns)


def normalize_dn(w):
    """Dollar-neutral, GMV=1: long book -> +0.5, short book -> -0.5."""
    longs = w.clip(lower=0); shorts = w.clip(upper=0)
    ls = longs.sum(axis=1).replace(0, np.nan)
    ss = shorts.abs().sum(axis=1).replace(0, np.nan)
    return (longs.div(ls, axis=0) * 0.5).fillna(0) + (shorts.div(ss, axis=0) * 0.5).fillna(0)


def pnl_stats(w, returns):
    gross = (w * returns.fillna(0.0)).sum(axis=1)
    traded = w.diff().abs().sum(axis=1).fillna(0.0)
    book = w.abs().sum(axis=1)
    net = gross - traded * EXEC_BPS - book * (FIN_ANNUAL / 252)
    idx = gross.ne(0).cumsum() > 0
    gross, net, traded, book = gross[idx], net[idx], traded[idx], book[idx]
    def sh(x): return x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else np.nan
    turn = traded.mean() / book.mean() * 100 if book.mean() else np.nan
    return gross, net, sh(gross), sh(net), turn


def main():
    data, returns, universe = a.load_panel(start="2010-01-01", end="2020-12-31", verbose=True)
    returns = clean_returns(returns)

    sleeves, pnls_net = {}, {}
    print("\n--- Standalone sleeves (hysteresis design) ---")
    print(f"{'alpha':8s} {'gross Sh':>8s} {'net Sh':>7s} {'turn%':>6s} {'annret%':>8s}")
    sum_turn = 0
    for n, d in DESIGNS.items():
        sig = a.get_alpha(n)(data)
        rp = bucket_masks(sig, universe)
        state = hysteresis_state(rp, **d)
        w = normalize_dn(state.astype(float)).shift(LAG)
        w = w.where(universe.astype(bool), 0.0)
        w = normalize_dn(w)
        g, net, shg, shn, turn = pnl_stats(w, returns)
        sleeves[n] = w
        pnls_net[n] = net
        sum_turn += turn
        print(f"alpha{n:03d} {shg:8.2f} {shn:7.2f} {turn:6.0f} {g.mean()*252*100:8.1f}")

    # --- inverse-vol allocation (causal: shift 1) ---
    pnl_df = pd.DataFrame({n: (sleeves[n] * returns.fillna(0)).sum(axis=1) for n in DESIGNS})
    vol = pnl_df.rolling(60, min_periods=20).std() * np.sqrt(252)
    vol = vol.clip(lower=0.05)
    inv = 1.0 / vol
    alloc = inv.div(inv.sum(axis=1), axis=0).shift(1).fillna(1.0 / len(DESIGNS))

    # --- combine weight matrices (internal netting) ---
    master = None
    for n in DESIGNS:
        contrib = sleeves[n].mul(alloc[n], axis=0)
        master = contrib if master is None else master.add(contrib, fill_value=0)
    master = normalize_dn(master)

    g, net, shg, shn, turn = pnl_stats(master, returns)

    # equal-weight blend for comparison
    eq = None
    for n in DESIGNS:
        eq = sleeves[n] if eq is None else eq.add(sleeves[n], fill_value=0)
    eq = normalize_dn(eq)
    _, eqnet, eqshg, eqshn, eqturn = pnl_stats(eq, returns)

    print("\n" + "=" * 60)
    print("ENSEMBLE (inverse-vol, internal netting)")
    print("=" * 60)
    print(f"  Gross Sharpe : {shg:.2f}")
    print(f"  Net   Sharpe : {shn:.2f}")
    print(f"  Turnover     : {turn:.0f}%   (sum of standalone sleeves: {sum_turn:.0f}% "
          f"-> netting saves {(1-turn/sum_turn)*100:.0f}%)")
    print(f"  Net ann ret  : {net.mean()*252*100:.1f}%   vol {net.std()*np.sqrt(252)*100:.1f}%")
    dd = (net.cumsum() - net.cumsum().cummax()).min()
    print(f"  Max drawdown : {dd*100:.1f}% (of GMV-days)")
    print(f"  [equal-weight blend: net Sharpe {eqshn:.2f}, turn {eqturn:.0f}%]")

    print("\n--- Year-by-year (inverse-vol ensemble, net) ---")
    print(f"  {'year':>5s} {'net Sharpe':>11s} {'ann ret %':>10s}")
    for y in range(2010, 2021):
        ny = net[net.index.year == y]
        sh = ny.mean() / ny.std() * np.sqrt(252) if ny.std() > 0 else np.nan
        print(f"  {y:>5d} {sh:11.2f} {ny.mean()*252*100:10.1f}")

    master.to_parquet("alpha101_results/ensemble_weights.parquet")
    net.to_frame("net_pnl").to_csv("alpha101_results/ensemble_net_pnl.csv")
    print("\nSaved ensemble_weights.parquet, ensemble_net_pnl.csv")


if __name__ == "__main__":
    main()
