"""
Robustness tests on the 13-alpha genuine core:
  1. Year-by-year L/S Sharpe + concentration  (is it 1-2 lucky years?)
  2. Full per-decile forward-return profile (10 buckets)
  3. Width sweep: long/short top/bottom {10,20,30,40,50}% -> gross & NET Sharpe,
     ann return, turnover. Tells us whether widening the legs lowers risk and
     captures more of the spread after costs.

In-sample 2010-2020, T+1, 5M-ADV universe. Costs (competition model):
2 bps per unit traded + 0.5% annual financing on gross book.
"""

import json
import numpy as np
import pandas as pd

import alpha101 as a

LAG = 1
EXEC_BPS = 2e-4
FIN_ANNUAL = 0.005
WIDTHS = [0.1, 0.2, 0.3, 0.4, 0.5]


def ls_weights(sig, universe, q):
    s = sig.replace([np.inf, -np.inf], np.nan).where(universe.astype(bool))
    r = s.rank(axis=1, pct=True)
    longs = (r >= 1 - q).astype(float)
    shorts = (r <= q).astype(float)
    lw = longs.div(longs.sum(axis=1), axis=0) * 0.5
    sw = shorts.div(shorts.sum(axis=1), axis=0) * 0.5
    return (lw - sw).shift(LAG).fillna(0.0)


def pnl_stats(w, returns):
    gross = (w * returns.fillna(0.0)).sum(axis=1)
    traded = w.diff().abs().sum(axis=1).fillna(0.0)
    book = w.abs().sum(axis=1)
    net = gross - traded * EXEC_BPS - book * (FIN_ANNUAL / 252)
    gross = gross.loc[gross.ne(0).cumsum() > 0]
    net = net.reindex(gross.index)
    turn = (traded.mean() / book.mean() * 100) if book.mean() else np.nan
    def sh(x): return x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else np.nan
    return gross, net, sh(gross), sh(net), turn


def decile_profile(sig, returns, universe):
    s = sig.replace([np.inf, -np.inf], np.nan)
    al = s.shift(LAG).where(universe.astype(bool))
    r = al.rank(axis=1, pct=True)
    b = (r * 10).clip(upper=9.999).where(r.notna())
    fwd = returns.where(universe.astype(bool))
    return [fwd.where((b >= d) & (b < d + 1)).mean(axis=1).mean() * 1e4 for d in range(10)]


def main():
    core = json.load(open("alpha101_results/genuine_core.json"))["decorrelated_core"]
    data, returns, universe = a.load_panel(start="2010-01-01", end="2020-12-31", verbose=True)
    nums = [int(x.replace("alpha", "")) for x in core]

    yearly_rows, decile_rows, width_rows = [], [], []
    combo_pnl = None

    for n in nums:
        name = f"alpha{n:03d}"
        sig = a.get_alpha(n)(data)

        # --- decile L/S (q=0.1) for yearly + concentration ---
        w = ls_weights(sig, universe, 0.1)
        gross, net, shg, shn, turn = pnl_stats(w, returns)
        combo_pnl = gross if combo_pnl is None else combo_pnl.add(gross, fill_value=0)

        yr = {"alpha": name}
        cum_by_year = {}
        for y in range(2010, 2021):
            g = gross[gross.index.year == y]
            yr[y] = round(g.mean() / g.std() * np.sqrt(252), 2) if len(g) and g.std() > 0 else None
            cum_by_year[y] = g.sum()
        pos = sum(1 for y in range(2010, 2021) if yr[y] is not None and yr[y] > 0)
        total = sum(v for v in cum_by_year.values())
        best2 = sum(sorted(cum_by_year.values(), reverse=True)[:2])
        yr["pos_yrs"] = f"{pos}/11"
        yr["best2_share"] = round(best2 / total * 100, 0) if total > 0 else None
        yearly_rows.append(yr)

        # --- decile profile ---
        prof = decile_profile(sig, returns, universe)
        decile_rows.append({"alpha": name, **{f"D{i+1}": round(prof[i], 1) for i in range(10)}})

        # --- width sweep ---
        wr = {"alpha": name}
        for q in WIDTHS:
            ww = ls_weights(sig, universe, q)
            _, _, shg_q, shn_q, turn_q = pnl_stats(ww, returns)
            wr[f"g{int(q*100)}"] = round(shg_q, 2)
            wr[f"n{int(q*100)}"] = round(shn_q, 2)
            wr[f"t{int(q*100)}"] = round(turn_q, 0)
        width_rows.append(wr)
        print(f"  {name}: decile gross Sh {shg:.2f} / net {shn:.2f} / turn {turn:.0f}%")

    yearly = pd.DataFrame(yearly_rows).set_index("alpha")
    decile = pd.DataFrame(decile_rows).set_index("alpha")
    width = pd.DataFrame(width_rows).set_index("alpha")

    print("\n================ 1. YEAR-BY-YEAR decile L/S Sharpe ================")
    print(yearly.to_string())
    print("\n================ 2. PER-DECILE forward return (bps/day) ===========")
    print(decile.to_string())
    print("\n================ 3. WIDTH SWEEP (g=gross Sh, n=net Sh, t=turn%) ====")
    print("  columns g10/n10/t10 = top&bottom 10% ... g50/n50/t50 = top&bottom 50%")
    print(width.to_string())

    # equal-weight combo yearly (decile L/S, gross)
    print("\n================ BONUS: equal-weight combo of 13 (decile, gross) ===")
    cy = {y: combo_pnl[combo_pnl.index.year == y] for y in range(2010, 2021)}
    print("  year   Sharpe")
    for y in range(2010, 2021):
        g = cy[y]
        sh = g.mean() / g.std() * np.sqrt(252) if len(g) and g.std() > 0 else float("nan")
        print(f"  {y}   {sh:6.2f}")
    full = combo_pnl
    print(f"  FULL combo gross Sharpe: {full.mean()/full.std()*np.sqrt(252):.2f}")

    yearly.to_csv("alpha101_results/core_yearly_sharpe.csv")
    decile.to_csv("alpha101_results/core_decile_profile.csv")
    width.to_csv("alpha101_results/core_width_sweep.csv")
    print("\nSaved core_yearly_sharpe.csv, core_decile_profile.csv, core_width_sweep.csv")


if __name__ == "__main__":
    main()
