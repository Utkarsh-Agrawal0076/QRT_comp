"""Stage 4 — does any MARKET-NEUTRAL alpha survive?

Stage 3 showed the long-only profit is ~all market beta. This strips the drift
three ways and asks whether the f1/f2/f3 classifier score has residual cross-
sectional alpha at the extremes where an L/S book trades:

  1. Per-decile next-day return: raw vs demeaned (XS) vs beta-residual (vs SPX)
  2. L/S decile-spread alpha (D10-D1) on each return definition  -> gross %/yr
  3. Sector-neutral ranking variant: net SR through the harness
  4. Turnover-reduced variant (smooth score + bucket hysteresis): can the thin
     residual spread be captured net of 2bps?
"""
from __future__ import annotations
import sys, warnings, json
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.append(str(Path("phase2_qrt_challenge/scripts").resolve()))
import utils  # noqa: E402
import range_momentum_pipeline as s1
import range_momentum_stage2 as s2
import range_momentum_stage3 as s3

ART = Path("stores/range_mom")
SQRT252 = np.sqrt(252)


def beta_residual_returns(rets):
    """ret - beta*spy_ret, beta = 252d rolling cov/var, shrunk 0.2+0.8*beta."""
    spy = pd.read_parquet("stores/spy_adj_close.parquet")
    spy = spy.iloc[:, 0] if isinstance(spy, pd.DataFrame) else spy
    spy_ret = spy.reindex(rets.index).pct_change(fill_method=None).fillna(0)
    cov = rets.rolling(252, min_periods=120).cov(spy_ret)
    var = spy_ret.rolling(252, min_periods=120).var()
    beta = (0.2 + 0.8 * cov.div(var, axis=0)).clip(-1, 3)
    return rets.sub(beta.mul(spy_ret, axis=0))


def ls_weights(score, universe, lp=0.90, sp=0.10, gmv=1.0):
    rk = score.where(universe == 1, np.nan).rank(axis=1, pct=True)
    L = (rk >= lp).astype(float); S = (rk <= sp).astype(float)
    nL = L.sum(axis=1).replace(0, np.nan); nS = S.sum(axis=1).replace(0, np.nan)
    return L.div(nL, axis=0).fillna(0) * (gmv/2) - S.div(nS, axis=0).fillna(0) * (gmv/2)


def sector_neutral_rank(score, sector_of):
    """Percentile rank within each sector, recombined into one matrix."""
    out = pd.DataFrame(np.nan, index=score.index, columns=score.columns)
    for sec, cols in sector_of.groupby(sector_of).groups.items():
        cc = [c for c in cols if c in score.columns]
        if len(cc) < 20:
            continue
        out[cc] = score[cc].rank(axis=1, pct=True)
    return out


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

    # return definitions, next-day
    raw1 = rets.shift(-1).replace([np.inf, -np.inf], np.nan)
    dem1 = raw1.sub(raw1.mean(axis=1), axis=0)
    res_full = beta_residual_returns(returns).loc[oos_idx]
    res1 = res_full.shift(-1).replace([np.inf, -np.inf], np.nan)

    print("\n[per-decile next-day ret (bps): raw vs XS-demeaned vs beta-residual]")
    rk = score.where(univ == 1, np.nan).rank(axis=1, pct=True)
    rows = []
    for d in range(1, 11):
        lo, hi = (d-1)/10, d/10
        m = ((rk >= lo) & (rk < hi)) if d < 10 else (rk >= lo)
        rows.append({"decile": d,
                     "raw": raw1.where(m).stack().mean()*1e4,
                     "demean": dem1.where(m).stack().mean()*1e4,
                     "beta_resid": res1.where(m).stack().mean()*1e4})
    dec = pd.DataFrame(rows)
    print(dec.round(2).to_string(index=False))
    for col in ["raw", "demean", "beta_resid"]:
        ls = dec[col].iloc[-1] - dec[col].iloc[0]
        mono = pd.Series(range(1, 11)).corr(dec[col])
        print(f"  {col:11s}: D10-D1 = {ls:+.2f} bps/d (~{ls*252/100:+.1f}%/yr)  monotonicity={mono:+.2f}")

    # ---- baseline dollar-neutral L/S net SR ----
    print("\n[baseline L/S (D10/D1)]")
    base = ls_weights(score, univ)
    base_f = _wf(base, univ, rets)
    sr, _ = utils.backtest_portfolio(base_f, rets, _ub(univ, rets), False, True)

    # ---- sector-neutral ranking ----
    smap = pd.read_csv("top_5000_us_by_marketcap.csv").dropna(subset=["sector"])
    sector_of = smap.set_index("symbol")["sector"]
    sector_of = sector_of[~sector_of.index.duplicated()].reindex(score.columns).dropna()
    sn = sector_neutral_rank(score, sector_of)
    print(f"\n[sector-neutral ranking, {sector_of.nunique()} sectors, "
          f"{sector_of.notna().sum()} mapped stocks]")
    w_sn = ls_weights(sn, univ)
    sr_sn, _ = utils.backtest_portfolio(_wf(w_sn, univ, rets), rets, _ub(univ, rets), False, True)

    # ---- turnover reduction: smooth score + bucket hysteresis ----
    print("\n[turnover reduction sweeps]")
    for k in [1, 3, 5, 10]:
        sm = score.rolling(k, min_periods=1).mean()
        w_sm = ls_weights(sm, univ)
        srk, _ = utils.backtest_portfolio(_wf(w_sm, univ, rets), rets, _ub(univ, rets), False, False)
        # quick turnover read
        wt = _wf(w_sm, univ, rets)
        turn = (wt.diff().abs().sum(1).mean() / wt.abs().sum(1).mean()) * 100
        print(f"  smooth k={k:2d}: net_SR={srk:+.3f}  turnover={turn:.0f}%")

    print("\n[hysteresis on smoothed-5 score (enter 0.90/0.10, exit 0.80/0.20)]")
    w_hy = _hysteresis_ls(score.rolling(5, min_periods=1).mean(), univ)
    sr_hy, _ = utils.backtest_portfolio(_wf(w_hy, univ, rets), rets, _ub(univ, rets), False, True)

    out = {"decile_raw_spread": float(dec["raw"].iloc[-1]-dec["raw"].iloc[0]),
           "decile_demean_spread": float(dec["demean"].iloc[-1]-dec["demean"].iloc[0]),
           "decile_resid_spread": float(dec["beta_resid"].iloc[-1]-dec["beta_resid"].iloc[0]),
           "baseline_net_SR": float(sr), "sector_neutral_net_SR": float(sr_sn),
           "hysteresis_net_SR": float(sr_hy)}
    (ART / "stage4_neutral.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[saved] {ART}/stage4_neutral.json")


def _wf(w, univ, rets):
    return _waterfill(w.shift(1), univ).reindex(columns=rets.columns, fill_value=0.0)


def _ub(univ, rets):
    return univ.reindex(columns=rets.columns, fill_value=0)


def _waterfill(shifted, universe_df, max_weight=0.098):
    p = (shifted * universe_df)
    asum = p.abs().sum(axis=1)
    for date, gmv in asum.items():
        if gmv <= 1e-8:
            continue
        row = (p.loc[date] / gmv).copy()
        for _ in range(20):
            if row.abs().max() <= max_weight + 1e-6:
                break
            cap = row.abs() > max_weight
            row[cap] = np.sign(row[cap]) * max_weight
            rem = 1.0 - row[cap].abs().sum()
            us = row[~cap].abs().sum()
            if us > 1e-8:
                row[~cap] *= rem / us
            else:
                break
        p.loc[date] = 0.0 if (abs(row.abs().sum()-1) > 0.01 or row.abs().max() > 0.1001) else row
    return p


def _hysteresis_ls(score, universe, enterL=0.90, exitL=0.80, enterS=0.10, exitS=0.20, gmv=1.0):
    rk = score.where(universe == 1, np.nan).rank(axis=1, pct=True)
    inL = pd.DataFrame(False, index=rk.index, columns=rk.columns)
    inS = pd.DataFrame(False, index=rk.index, columns=rk.columns)
    prevL = pd.Series(False, index=rk.columns); prevS = pd.Series(False, index=rk.columns)
    for dt in rk.index:
        r = rk.loc[dt]
        curL = (prevL & (r >= exitL)) | (r >= enterL)
        curS = (prevS & (r <= exitS)) | (r <= enterS)
        curL = curL & r.notna(); curS = curS & r.notna()
        inL.loc[dt] = curL.values; inS.loc[dt] = curS.values
        prevL, prevS = curL.fillna(False), curS.fillna(False)
    L = inL.astype(float); S = inS.astype(float)
    nL = L.sum(1).replace(0, np.nan); nS = S.sum(1).replace(0, np.nan)
    return L.div(nL, axis=0).fillna(0)*(gmv/2) - S.div(nS, axis=0).fillna(0)*(gmv/2)


if __name__ == "__main__":
    main()
