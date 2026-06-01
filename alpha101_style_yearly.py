"""
Standalone yearly net return + Sharpe of the three styles:
  Reversal super-sleeve = inv-vol blend of MR(a11)+ResMR+a024+a041+a100
  Momentum (Mom)
  Stat-Arb (SA)  -- NOTE: pairs selected on recent ~18m data => its historical
                  backtest is look-ahead-contaminated; shown for reference only.
Reversal & Momentum are rolling all-period signals and ARE comparable year by year.
"""

import numpy as np
import pandas as pd

import alpha101 as a
from alpha101_bucket_enum import clean_returns, bucket_masks, LAG, EXEC_BPS, FIN_ANNUAL
from alpha101_ensemble import DESIGNS, hysteresis_state, normalize_dn
from alpha101_portfolio import net_pnl, inv_vol_blend

NEW = [24, 41, 100]


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
    for key in ["MR", "Mom", "SA", "ResMR"]:
        wmats[key] = sl[key].reindex(index=returns.index, columns=returns.columns).fillna(0)

    rev_super = inv_vol_blend(wmats, returns, ["MR", "ResMR", 24, 41, 100])
    styles = {"Reversal": rev_super, "Momentum": wmats["Mom"], "StatArb*": wmats["SA"]}
    pnl = {k: net_pnl(v, returns) for k, v in styles.items()}
    for k in pnl:
        pnl[k] = pnl[k][pnl[k].ne(0).cumsum() > 0]

    years = range(2010, max(p.index[-1].year for p in pnl.values()) + 1)
    print("\n================ STANDALONE YEARLY NET RETURN % (Sharpe) ================")
    print(f"  {'year':>6s} | " + " | ".join(f"{k:>16s}" for k in styles))
    for y in years:
        cells = []
        for k in styles:
            p = pnl[k][pnl[k].index.year == y]
            if len(p) < 20:
                cells.append(f"{'-':>16s}"); continue
            ret = p.sum() * 100
            sh = p.mean() / p.std() * np.sqrt(252) if p.std() > 0 else np.nan
            cells.append(f"{ret:7.1f}% ({sh:4.2f})")
        star = " *OOS" if y >= 2021 else ""
        print(f"  {y:>6d} | " + " | ".join(cells) + star)

    print("\n  ---- OOS 2021+ aggregate ----")
    for k in styles:
        p = pnl[k][pnl[k].index >= "2021-01-01"]
        sh = p.mean() / p.std() * np.sqrt(252)
        print(f"    {k:16s}: net ann ret {p.mean()*252*100:5.1f}% | Sharpe {sh:4.2f} | "
              f"cum {p.sum()*100:5.1f}%")
    print("\n  NOTE: StatArb* pairs are selected on recent ~18m data -> its pre-current"
          "\n        backtest has look-ahead; compare Reversal vs Momentum directly, treat"
          "\n        StatArb* years as indicative only.")


if __name__ == "__main__":
    main()
