"""Stage 5 — decile demeaned returns + L/S bucket-design enumeration.

Stage 3 measured long/short PnL on RAW returns (drift-contaminated), which made
the short book look like the loser. A dollar-neutral book only earns the DEMEANED
(relative) return. This recomputes per-decile demeaned returns against the
tradeable-set benchmark, then enumerates (long-bucket, short-bucket) designs to
find the weighting schema that maximizes market-neutral alpha -- exactly the tool
used to design the MR sleeve (PRODUCTION_STRATEGIES Diagnostic 5).
"""
from __future__ import annotations
import sys, warnings, json, itertools
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.append(str(Path("phase2_qrt_challenge/scripts").resolve()))
import utils  # noqa: E402
import range_momentum_pipeline as s1
import range_momentum_stage2 as s2
import range_momentum_stage3 as s3
import range_momentum_stage4 as s4

ART = Path("stores/range_mom")
SQRT252 = np.sqrt(252)


def main():
    ohlcv = s1.load_data()
    universe = s1.build_universe(ohlcv)
    returns = s1.build_returns(ohlcv)
    universe = universe.reindex(columns=returns.columns).fillna(0).astype(int)
    oos_start = returns.index.max() - pd.DateOffset(years=s2.OOS_YEARS)
    is_idx = returns.index[returns.index < oos_start]
    oos_idx = returns.index[returns.index >= oos_start]
    score = s3.build_score(ohlcv, universe, returns, is_idx, oos_idx)
    rets = returns.loc[oos_idx]
    univ = universe.loc[oos_idx]

    fwd1 = rets.shift(-1).replace([np.inf, -np.inf], np.nan)
    rk = score.where(univ == 1, np.nan).rank(axis=1, pct=True)
    tradeable = (univ == 1) & rk.notna()
    # benchmark = equal-weight mean over the SAME tradeable set we rank within
    bench = fwd1.where(tradeable).mean(axis=1)
    dem = fwd1.sub(bench, axis=0)

    # per-decile demeaned return (bps) and its standalone Sharpe as a one-sided book
    print("[decile demeaned next-day return — the market-neutral building block]")
    masks, recs = {}, []
    for d in range(1, 11):
        lo, hi = (d-1)/10, d/10
        m = ((rk >= lo) & (rk < hi)) if d < 10 else (rk >= lo)
        masks[d] = m & tradeable
        daily = dem.where(masks[d]).mean(axis=1)
        recs.append({"decile": d,
                     "demean_bps": daily.mean()*1e4,
                     "ann_%": daily.mean()*252*100,
                     "SR": daily.mean()/daily.std()*SQRT252 if daily.std() else 0,
                     "n_avg": int(masks[d].sum(axis=1).mean())})
    tab = pd.DataFrame(recs)
    print(tab.round(2).to_string(index=False))
    print("  -> long a bucket = +demean_bps ; short a bucket = -demean_bps")
    print(f"  best long  bucket: D{int(tab.loc[tab.demean_bps.idxmax(),'decile'])} "
          f"({tab.demean_bps.max():+.2f} bps)")
    print(f"  best short bucket: D{int(tab.loc[tab.demean_bps.idxmin(),'decile'])} "
          f"({tab.demean_bps.min():+.2f} bps)  -> shorting it earns {-tab.demean_bps.min():+.2f} bps")

    # daily demeaned series per decile (for combo enumeration)
    dser = {d: dem.where(masks[d]).mean(axis=1).fillna(0) for d in range(1, 11)}

    def combo_stats(long_ds, short_ds):
        l = sum(dser[d] for d in long_ds) / len(long_ds)
        s = sum(dser[d] for d in short_ds) / len(short_ds)
        pnl = 0.5 * l - 0.5 * s              # 0.5 GMV each side, dollar-neutral
        return pnl.mean()*252*100, (pnl.mean()/pnl.std()*SQRT252 if pnl.std() else 0)

    # enumerate contiguous long/short bucket ranges
    print("\n[bucket-design enumeration — top 12 by gross market-neutral Sharpe]")
    rows = []
    deciles = list(range(1, 11))
    ranges = [tuple(deciles[i:j]) for i in range(10) for j in range(i+1, 11)]
    for L, S in itertools.product(ranges, ranges):
        if set(L) & set(S):
            continue
        ann, sr = combo_stats(L, S)
        rows.append({"long": f"D{L[0]}-D{L[-1]}", "short": f"D{S[0]}-D{S[-1]}",
                     "ann_gross_%": ann, "gross_SR": sr})
    comb = pd.DataFrame(rows).sort_values("gross_SR", ascending=False)
    print(comb.head(12).round(2).to_string(index=False))
    print("\n[current design D10 long / D1 short]")
    print(comb[(comb.long == "D10-D10") & (comb.short == "D1-D1")].round(2).to_string(index=False))

    (ART / "stage5_deciles.json").write_text(
        json.dumps({"deciles": tab.to_dict("records"),
                    "top_designs": comb.head(12).to_dict("records")}, indent=2, default=float))
    print(f"\n[saved] {ART}/stage5_deciles.json")
    return score, univ, rets, masks, rk, tradeable


if __name__ == "__main__":
    main()
