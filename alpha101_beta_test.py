"""
Hypothesis test (senior-quant): the non-monotone decile profiles of some
high-IC alphas are a MARKET-BETA artifact, not a signal failure.

Mechanism: the naive decile L/S is dollar-neutral but NOT beta-neutral. If the
extreme deciles carry different betas, then over a bull sample the high-beta
decile earns raw return from market exposure, distorting the profile. Removing
each stock's beta*market component (idiosyncratic residual) should restore
monotonicity.

For each alpha we report, on the in-sample window at T+1:
  * raw vs beta-residual decile forward-return profile (bps/day) + monotonicity
  * average market beta per decile  (smoking gun if D1 vs D10 differ)
  * the L/S net beta (D10 minus D1 beta) and the residual D10-D1 Sharpe

Market factor = equal-weight mean of universe returns (the common factor of the
book). Beta = trailing 252-day regression slope, lagged 1 day (no look-ahead).
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import alpha101 as a
from alpha101_pipeline import compute_ic

LAG = 1
BETA_WIN = 252


def build_residual_returns(returns, rm):
    """Residualise each stock against market return series rm (trailing beta)."""
    r = returns
    mean_i = r.rolling(BETA_WIN, min_periods=BETA_WIN).mean()
    mean_m = rm.rolling(BETA_WIN, min_periods=BETA_WIN).mean()
    cov = (r.mul(rm, axis=0)).rolling(BETA_WIN, min_periods=BETA_WIN).mean().sub(mean_i.mul(mean_m, axis=0))
    var_m = rm.rolling(BETA_WIN, min_periods=BETA_WIN).var()
    beta = cov.div(var_m, axis=0)
    beta_lag = beta.shift(1)
    resid = returns.sub(beta_lag.mul(rm, axis=0))
    return resid, beta_lag


def profile(sig, fwd, uni, lag=LAG, beta=None):
    al = sig.shift(lag).where(uni)
    r = al.rank(axis=1, pct=True)
    b = (r * 10).clip(upper=9.999).where(r.notna())
    prof, betas = [], []
    f = fwd.where(uni)
    for d in range(10):
        mask = (b >= d) & (b < d + 1)
        prof.append(f.where(mask).mean(axis=1).mean() * 1e4)
        if beta is not None:
            betas.append(beta.where(mask).mean(axis=1).mean())
    top = (b >= 9); bot = (b < 1)
    spread = f.where(top).mean(axis=1) - f.where(bot).mean(axis=1)
    sh = spread.mean() / spread.std() * np.sqrt(252)
    mono, _ = spearmanr(range(10), prof)
    return np.array(prof), sh, mono, (np.array(betas) if beta is not None else None)


def main():
    data, returns, universe = a.load_panel(start="2010-01-01", end="2020-12-31", verbose=True)
    uni = universe.astype(bool)
    r_in_uni = returns.where(uni)

    # two market factors: EW-universe (endogenous) and SPY (exogenous index)
    mkt_ew = r_in_uni.mean(axis=1)
    spy = pd.read_parquet("stores/spy_adj_close.parquet")["SPY"].pct_change()
    mkt_spy = spy.reindex(returns.index)
    print(f"EW-universe ann ret {mkt_ew.mean()*252*100:.1f}% | "
          f"SPY ann ret {mkt_spy.mean()*252*100:.1f}% (both strongly positive => bull sample)\n")

    print("Building beta-residual returns vs EW and vs SPY...")
    resid_ew, beta_ew = build_residual_returns(returns, mkt_ew)
    resid_spy, beta_spy = build_residual_returns(returns, mkt_spy)

    alphas = [44, 16, 24, 15, 50, 11, 9, 35]
    for n in alphas:
        sig = a.get_alpha(n)(data).replace([np.inf, -np.inf], np.nan)
        ic = compute_ic(sig, returns, universe, lag=LAG).mean()
        praw, shraw, mraw, dbeta_spy = profile(sig, returns, uni, beta=beta_spy)
        pew, shew, mew, _ = profile(sig, resid_ew, uni)
        pspy, shspy, mspy, _ = profile(sig, resid_spy, uni)
        print(f"=== alpha{n:03d}  IC {ic:+.4f} ===")
        print("  decile:           " + " ".join(f"D{i+1:>2}" for i in range(10)))
        print("  raw ret(bps):     " + " ".join(f"{x:5.1f}" for x in praw))
        print("  resid-EW(bps):    " + " ".join(f"{x:5.1f}" for x in pew))
        print("  resid-SPY(bps):   " + " ".join(f"{x:5.1f}" for x in pspy))
        print("  avg SPY-beta:     " + " ".join(f"{x:5.2f}" for x in dbeta_spy))
        print(f"  RAW      : D10-D1 Sharpe {shraw:+.2f} | mono {mraw:+.2f} | "
              f"net L/S SPY-beta {dbeta_spy[9]-dbeta_spy[0]:+.2f}")
        print(f"  resid-EW : D10-D1 Sharpe {shew:+.2f} | mono {mew:+.2f}")
        print(f"  resid-SPY: D10-D1 Sharpe {shspy:+.2f} | mono {mspy:+.2f}")
        best = max(mew, mspy)
        verdict = ("BETA ARTIFACT (SPY)" if (mspy - mraw) > 0.2 else
                   "partly beta" if (best - mraw) > 0.08 else "NOT beta - real tail effect")
        print(f"  --> mono raw {mraw:+.2f} -> EW {mew:+.2f} / SPY {mspy:+.2f}   [{verdict}]\n")


if __name__ == "__main__":
    main()
