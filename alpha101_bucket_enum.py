"""
Find tradeable alphas using the user's master_ensemble cell-14 methodology
(instead of a fixed D10/D1 decile L/S, which is wrong for non-monotone signals).

Per alpha, faithfully matching cell 2 / cell 14:
  * clean returns: drop +/-inf, CLIP to [-0.5, +2.0] (kills outlier-driven buckets)
  * 3-day rolling-mean smoothing of the raw alpha (cumulative IC peaks ~h=3)
  * cross-sectionally demean the signal
  * evaluate forward (T+1) MARKET-DEMEANED returns per decile
  * enumerate asymmetric tail L/S designs (long top {1,2,3} deciles vs short bottom
    {1..5} deciles) and pick the best by realized NET Sharpe
  * compare to the naive D10/D1 design (what my screen used)

This shows which alphas are tradeable with the proper construction -- e.g. Alpha#11
(the production MR sleeve) revives once the short book is widened to D1-D5.
"""

import json
import numpy as np
import pandas as pd

import alpha101 as a

LAG = 1
SMOOTH = 3
EXEC_BPS = 2e-4
FIN_ANNUAL = 0.005
LONG_OPTS = [(10, 10), (9, 10), (8, 10)]          # top 1 / 2 / 3 deciles
SHORT_OPTS = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)]  # bottom 1..5 deciles


def clean_returns(returns):
    return returns.replace([np.inf, -np.inf], np.nan).clip(lower=-0.5, upper=2.0)


def bucket_masks(sig, universe):
    s = sig.replace([np.inf, -np.inf], np.nan).rolling(SMOOTH, min_periods=1).mean()
    s = s.where(universe.astype(bool))
    s = s.sub(s.mean(axis=1), axis=0)            # demean (sign convention)
    r = s.rank(axis=1, pct=True)
    return r


def design_pnl(rank_pct, returns, long_b, short_b):
    lo_l, hi_l = (long_b[0] - 1) / 10, long_b[1] / 10
    lo_s, hi_s = (short_b[0] - 1) / 10, short_b[1] / 10
    longs = ((rank_pct >= lo_l) & (rank_pct <= hi_l + 1e-9)).astype(float)
    shorts = ((rank_pct >= lo_s) & (rank_pct < hi_s)).astype(float)
    lw = longs.div(longs.sum(axis=1), axis=0) * 0.5
    sw = shorts.div(shorts.sum(axis=1), axis=0) * 0.5
    w = (lw - sw).shift(LAG).fillna(0.0)
    gross = (w * returns.fillna(0.0)).sum(axis=1)
    traded = w.diff().abs().sum(axis=1).fillna(0.0)
    book = w.abs().sum(axis=1)
    net = gross - traded * EXEC_BPS - book * (FIN_ANNUAL / 252)
    gross = gross.loc[gross.ne(0).cumsum() > 0]
    net = net.reindex(gross.index)
    def sh(x): return x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else np.nan
    turn = traded.mean() / book.mean() * 100 if book.mean() else np.nan
    return sh(gross), sh(net), turn, gross.mean() * 252 * 100


def main():
    names = list(pd.read_csv("alpha101_results/shortlist.csv")["alpha"])
    if "alpha011" not in names:
        names.append("alpha011")
    nums = sorted(int(n.replace("alpha", "")) for n in names)

    data, returns, universe = a.load_panel(start="2010-01-01", end="2020-12-31", verbose=True)
    returns = clean_returns(returns)

    rows = []
    for n in nums:
        name = f"alpha{n:03d}"
        sig = a.get_alpha(n)(data)
        rp = bucket_masks(sig, universe)

        best = None
        for lb in LONG_OPTS:
            for sb in SHORT_OPTS:
                g, net, turn, annret = design_pnl(rp, returns, lb, sb)
                if best is None or (net is not None and net > best["net"]):
                    best = {"long": f"D{lb[0]}-D{lb[1]}", "short": f"D{sb[0]}-D{sb[1]}",
                            "gross": g, "net": net, "turn": turn, "annret": annret}
        # naive D10/D1 for comparison
        g0, net0, turn0, _ = design_pnl(rp, returns, (10, 10), (1, 1))
        rows.append({
            "alpha": name,
            "best_long": best["long"], "best_short": best["short"],
            "gross_Sh": round(best["gross"], 2), "net_Sh": round(best["net"], 2),
            "turn_%": round(best["turn"], 0), "annret_%": round(best["annret"], 1),
            "naive_D10D1_net": round(net0, 2),
        })
        print(f"  {name}: best {best['long']}/{best['short']}  net {best['net']:+.2f} "
              f"(naive D10/D1 net {net0:+.2f})")

    df = pd.DataFrame(rows).sort_values("net_Sh", ascending=False)
    df.to_csv("alpha101_results/bucket_enum.csv", index=False)
    print("\n" + "=" * 78)
    print("TRADEABLE ALPHAS by best-design NET Sharpe (cell-14 methodology, T+1)")
    print("=" * 78)
    print(df.to_string(index=False))
    tradeable = df[df["net_Sh"] >= 0.5]
    print(f"\nNet-tradeable (net Sharpe >= 0.5): {len(tradeable)} alphas")
    print(list(tradeable["alpha"]))
    revived = df[(df["net_Sh"] >= 0.5) & (df["naive_D10D1_net"] < 0.3)]
    print(f"\nREVIVED by proper bucket design (good now, dead under naive D10/D1): "
          f"{list(revived['alpha'])}")
    print("\nSaved alpha101_results/bucket_enum.csv")


if __name__ == "__main__":
    main()
