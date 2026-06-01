"""
Consistency check vs master_ensemble_pipeline.ipynb.

Reproduces the notebook's conventions exactly:
  - returns from ADJ CLOSE (not Close / returns.parquet)
  - universe_5m = adv60 >= 5M
  - 3-sleeve inv-vol ensemble (MR=a11, Mom, SA) per cell 5
  - utils.backtest_portfolio cost model (2bps + 0.5% financing)
Reports GROSS & NET Sharpe, FULL and trailing-252d (last 1 year), under both
Adj-Close and Close returns, so we can see where my earlier numbers diverged.
"""

import numpy as np
import pandas as pd

DATA = "top_5000_yf_data.pkl"


def dd(df):
    return df.loc[:, ~df.columns.duplicated()]


def normalize_and_cap(w, target=0.50, cap=0.099):
    rs = w.sum(axis=1) + 1e-10
    w = w.div(rs, axis=0) * target
    w = w.clip(upper=cap)
    rs2 = w.sum(axis=1) + 1e-10
    return w.div(rs2, axis=0) * target


def bt(w, rets, uni):
    w = dd(w).reindex(index=rets.index, columns=rets.columns).fillna(0)
    w = w * uni
    gross = (w * rets.fillna(0)).sum(axis=1)
    traded = w.diff().abs().sum(axis=1).fillna(0)
    book = w.abs().sum(axis=1)
    net = gross - traded * 2e-4 - book * (0.005 / 252)
    return gross, net


def sh(p):
    p = p[p.ne(0).cumsum() > 0]
    return p.mean() / p.std() * np.sqrt(252) if p.std() > 0 else np.nan


def line(name, w, rets, uni):
    g, n = bt(w, rets, uni)
    print(f"  {name:16s} | gross full {sh(g):5.2f}  net full {sh(n):5.2f} "
          f"| gross 1y {sh(g.tail(252)):5.2f}  net 1y {sh(n.tail(252)):5.2f} "
          f"| 1y gross ret {g.tail(252).sum()*100:+5.1f}%")


def main():
    h = pd.read_pickle(DATA)
    ret_adj = dd(h["Adj Close"]).pct_change(fill_method=None).fillna(0)
    ret_cls = dd(h["Close"]).pct_change(fill_method=None).fillna(0)
    advv = dd(h["Close"]).mul(dd(h["Volume"])).fillna(0)
    uni = (advv.rolling(60, min_periods=60).mean() >= 5_000_000).astype(int)
    uni = dd(uni)
    cols = ret_adj.columns
    uni = uni.reindex(columns=cols, fill_value=0)
    ret_cls = ret_cls.reindex(columns=cols, fill_value=0)

    sl = pd.read_pickle("stores/sharpe_blender/sleeves.pkl")
    MR, Mom, SA = sl["MR"], sl["Mom"], sl["SA"]

    print(f"data through {ret_adj.index[-1].date()}; 1y window = "
          f"{ret_adj.index[-252].date()} -> {ret_adj.index[-1].date()}\n")

    for lbl, rets in [("ADJ CLOSE (notebook)", ret_adj), ("CLOSE (my earlier runs)", ret_cls)]:
        print(f"===== returns = {lbl} =====")
        line("MR (=a11)", MR, rets, uni)
        line("Momentum", Mom, rets, uni)
        line("StatArb", SA, rets, uni)

        # reproduce cell-5 3-sleeve inv-vol ensemble (vol uses these returns)
        def rvol(w):
            ar = rets.reindex(index=w.index, columns=w.columns).fillna(0)
            pnl = (dd(w) * dd(ar)).sum(axis=1)
            return (pnl.rolling(60, min_periods=20).std() * np.sqrt(252)).clip(lower=0.05)
        iv = {k: 1 / rvol(v) for k, v in [("MR", MR), ("Mom", Mom), ("SA", SA)]}
        s = iv["MR"] + iv["Mom"] + iv["SA"]
        cM, cO, cS = (iv["MR"] / s).shift(1).fillna(1/3), (iv["Mom"] / s).shift(1).fillna(1/3), (iv["SA"] / s).shift(1).fillna(1/3)
        ew = dd(MR).mul(cM, axis=0).add(dd(Mom).mul(cO, axis=0), fill_value=0).add(dd(SA).mul(cS, axis=0), fill_value=0)
        longs = normalize_and_cap(ew.where(ew > 0, 0))
        shorts = normalize_and_cap(ew.where(ew < 0, 0).abs())
        ens = (longs - shorts).fillna(0)
        line("ENSEMBLE 3-sleeve", ens, rets, uni)
        print()


if __name__ == "__main__":
    main()
