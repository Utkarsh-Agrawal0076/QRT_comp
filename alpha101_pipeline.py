"""
Backtesting / screening pipeline for the 101 Formulaic Alphas.

For each alpha it computes, under the competition's T+2 execution lag:

  * Rank IC   : daily cross-sectional Spearman corr(signal_{t-lag}, return_t)
  * Hit rate  : fraction of days with IC > 0
  * IC IR     : mean(IC) / std(IC)   (and a t-stat)
  * Year-by-year table of mean IC / hit rate
  * A decile long/short backtest (top vs bottom decile, dollar-neutral) with
    an overall and per-year gross Sharpe.

Selection rule (as requested):  mean IC > 0.02  AND  hit rate > 50%
  -> alpha is appended to alpha101_results/selected_alphas.json

Regime detection:  an alpha that is strong in some years but weak/negative in
others (high spread of yearly IC, sign flips) is flagged to
alpha101_results/regime_alphas.json as a candidate regime-dependent signal.

Usage
-----
  python alpha101_pipeline.py --alphas 1-10          # first 10 (then 11-101)
  python alpha101_pipeline.py --alphas 1-101
  python alpha101_pipeline.py --alphas 5,9,25 --start 2014-01-01
"""

import os
import json
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

import alpha101 as a

warnings.filterwarnings("ignore")

OUT_DIR = "alpha101_results"

# selection thresholds
IC_THRESHOLD = 0.02
HIT_THRESHOLD = 0.50
MIN_NAMES = 50          # min cross-sectional names for a day's IC to count

# regime thresholds — calibrated to the observed T+2 IC scale on this universe
# (full-sample mean IC ~0.002-0.005; yearly swings ~0.01-0.02). A regime signal
# is one with a strong year (|IC| above STRONG) and a large best-vs-worst spread.
REGIME_STRONG_YEAR_IC = 0.015  # max(|best_year_ic|, |worst_year_ic|)
REGIME_MIN_SPREAD = 0.025      # best_year_ic - worst_year_ic


# --------------------------------------------------------------------------- #
#  Metrics
# --------------------------------------------------------------------------- #
def _rowwise_corr(a_df, b_df):
    """Per-row Pearson correlation over the commonly-valid columns."""
    mask = a_df.notna() & b_df.notna()
    a_v = a_df.where(mask)
    b_v = b_df.where(mask)
    n = mask.sum(axis=1)
    ma = a_v.mean(axis=1)
    mb = b_v.mean(axis=1)
    da = a_v.sub(ma, axis=0)
    db = b_v.sub(mb, axis=0)
    cov = (da * db).sum(axis=1)
    sa = (da ** 2).sum(axis=1).pow(0.5)
    sb = (db ** 2).sum(axis=1).pow(0.5)
    corr = cov / (sa * sb)
    return corr.where(n >= MIN_NAMES)


def compute_ic(signal, returns, universe, lag=2):
    """Daily rank-IC series under a T+lag execution lag."""
    sig = signal.replace([np.inf, -np.inf], np.nan)
    aligned = sig.shift(lag).where(universe.astype(bool))
    fwd = returns.where(universe.astype(bool))
    sr = aligned.rank(axis=1)
    rr = fwd.rank(axis=1)
    return _rowwise_corr(sr, rr).dropna()


def decile_backtest(signal, returns, universe, lag=2, q=0.1):
    """
    Dollar-neutral top/bottom-decile long-short backtest.
    Returns (daily pnl Series, gross annualised Sharpe).
    """
    sig = signal.replace([np.inf, -np.inf], np.nan).where(universe.astype(bool))
    r = sig.rank(axis=1, pct=True)
    longs = (r >= 1 - q).astype(float)
    shorts = (r <= q).astype(float)
    # equal-weight, each leg scaled to 0.5 gross
    lw = longs.div(longs.sum(axis=1), axis=0) * 0.5
    sw = shorts.div(shorts.sum(axis=1), axis=0) * 0.5
    weights = (lw - sw).shift(lag).fillna(0.0)
    pnl = (weights * returns.fillna(0.0)).sum(axis=1)
    pnl = pnl.loc[pnl.ne(0).cumsum() > 0]  # drop leading zeros
    sharpe = (pnl.mean() / pnl.std() * np.sqrt(252)) if pnl.std() > 0 else np.nan
    return pnl, sharpe


def yearly_table(ic, pnl):
    """Per-year mean IC, hit rate, IC count, and decile L/S Sharpe."""
    rows = []
    years = sorted(set(ic.index.year) | set(pnl.index.year))
    for y in years:
        ic_y = ic[ic.index.year == y]
        pnl_y = pnl[pnl.index.year == y]
        if len(ic_y) == 0:
            continue
        sharpe_y = (pnl_y.mean() / pnl_y.std() * np.sqrt(252)) if (len(pnl_y) and pnl_y.std() > 0) else np.nan
        rows.append({
            "year": y,
            "mean_ic": round(float(ic_y.mean()), 4),
            "hit_rate": round(float((ic_y > 0).mean()), 3),
            "n_days": int(len(ic_y)),
            "ls_sharpe": round(float(sharpe_y), 2) if pd.notna(sharpe_y) else None,
        })
    return rows


def evaluate_alpha(n, data, returns, universe, lag=2):
    """Compute the full metric bundle for alpha n."""
    fn = a.get_alpha(n)
    signal = fn(data)
    ic = compute_ic(signal, returns, universe, lag=lag)
    pnl, sharpe = decile_backtest(signal, returns, universe, lag=lag)

    mean_ic = float(ic.mean())
    std_ic = float(ic.std())
    hit = float((ic > 0).mean())
    icir = mean_ic / std_ic if std_ic > 0 else np.nan
    tstat = mean_ic / std_ic * np.sqrt(len(ic)) if std_ic > 0 else np.nan
    ytab = yearly_table(ic, pnl)
    yearly_ic = {r["year"]: r["mean_ic"] for r in ytab}

    # regime spread
    if yearly_ic:
        best_y = max(yearly_ic, key=yearly_ic.get)
        worst_y = min(yearly_ic, key=yearly_ic.get)
        best_ic, worst_ic = yearly_ic[best_y], yearly_ic[worst_y]
        spread = best_ic - worst_ic
    else:
        best_y = worst_y = None
        best_ic = worst_ic = spread = np.nan

    selected = (mean_ic > IC_THRESHOLD) and (hit > HIT_THRESHOLD)
    # bonus: strong *negative* IC alpha is tradeable by flipping the sign
    neg_selected = (mean_ic < -IC_THRESHOLD) and ((ic < 0).mean() > HIT_THRESHOLD)

    regime = (
        not selected and not neg_selected
        and pd.notna(spread)
        and max(abs(best_ic), abs(worst_ic)) > REGIME_STRONG_YEAR_IC
        and spread > REGIME_MIN_SPREAD
    )

    return {
        "alpha": f"alpha{n:03d}",
        "n": n,
        "mean_ic": round(mean_ic, 4),
        "std_ic": round(std_ic, 4),
        "ic_ir": round(float(icir), 3) if pd.notna(icir) else None,
        "t_stat": round(float(tstat), 2) if pd.notna(tstat) else None,
        "hit_rate": round(hit, 3),
        "ls_sharpe": round(float(sharpe), 2) if pd.notna(sharpe) else None,
        "n_days": int(len(ic)),
        "best_year": best_y, "best_year_ic": best_ic,
        "worst_year": worst_y, "worst_year_ic": worst_ic,
        "year_spread": round(float(spread), 4) if pd.notna(spread) else None,
        "selected": bool(selected),
        "neg_selected": bool(neg_selected),
        "regime_candidate": bool(regime),
        "indneutral_approx": n in a.INDNEUTRAL_ALPHAS,
        "uses_cap_proxy": n in a.CAP_ALPHAS,
        "yearly": ytab,
    }


# --------------------------------------------------------------------------- #
#  Reporting / persistence
# --------------------------------------------------------------------------- #
def print_report(res):
    flag = "  <== SELECTED" if res["selected"] else (
        "  <== SELECTED (flip sign)" if res["neg_selected"] else (
            "  <== regime candidate" if res["regime_candidate"] else ""))
    approx = " [indneut approx]" if res["indneutral_approx"] else ""
    approx += " [cap proxy]" if res["uses_cap_proxy"] else ""
    print(f"\n=== {res['alpha']}{approx}{flag} ===")
    print(f"  mean IC {res['mean_ic']:+.4f} | hit {res['hit_rate']:.1%} | "
          f"IC-IR {res['ic_ir']} | t {res['t_stat']} | L/S Sharpe {res['ls_sharpe']} | "
          f"{res['n_days']} days")
    print(f"  best {res['best_year']} (IC {res['best_year_ic']}) | "
          f"worst {res['worst_year']} (IC {res['worst_year_ic']}) | spread {res['year_spread']}")
    hdr = "    year   meanIC   hit   days  L/S Sh"
    print(hdr)
    for r in res["yearly"]:
        print(f"    {r['year']}  {r['mean_ic']:+.4f}  {r['hit_rate']:.0%}  "
              f"{r['n_days']:4d}   {r['ls_sharpe']}")


def _load_json(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def persist(results):
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) full metrics CSV (merge with prior runs, dedup on alpha)
    flat = [{k: v for k, v in r.items() if k != "yearly"} for r in results]
    csv_path = os.path.join(OUT_DIR, "all_metrics.csv")
    new_df = pd.DataFrame(flat)
    if os.path.exists(csv_path):
        old = pd.read_csv(csv_path)
        merged = pd.concat([old[~old["alpha"].isin(new_df["alpha"])], new_df])
    else:
        merged = new_df
    merged = merged.sort_values("n")
    merged.to_csv(csv_path, index=False)

    # 2) selected alphas (literal rule)  + sign-flip candidates
    sel_path = os.path.join(OUT_DIR, "selected_alphas.json")
    selected = _load_json(sel_path, {"criteria": f"mean_ic>{IC_THRESHOLD} and hit_rate>{HIT_THRESHOLD}",
                                     "alphas": {}})
    for r in results:
        if r["selected"] or r["neg_selected"]:
            selected["alphas"][r["alpha"]] = {
                "direction": "long" if r["selected"] else "short(flip)",
                "mean_ic": r["mean_ic"], "hit_rate": r["hit_rate"],
                "ic_ir": r["ic_ir"], "ls_sharpe": r["ls_sharpe"],
                "indneutral_approx": r["indneutral_approx"],
            }
    with open(sel_path, "w") as f:
        json.dump(selected, f, indent=2)

    # 3) regime-dependent candidates
    reg_path = os.path.join(OUT_DIR, "regime_alphas.json")
    regime = _load_json(reg_path, {"note": "high IC in some years, weak/negative in others",
                                   "alphas": {}})
    for r in results:
        if r["regime_candidate"]:
            regime["alphas"][r["alpha"]] = {
                "best_year": r["best_year"], "best_year_ic": r["best_year_ic"],
                "worst_year": r["worst_year"], "worst_year_ic": r["worst_year_ic"],
                "year_spread": r["year_spread"], "mean_ic": r["mean_ic"],
                "yearly_ic": {str(y["year"]): y["mean_ic"] for y in r["yearly"]},
            }
    with open(reg_path, "w") as f:
        json.dump(regime, f, indent=2)

    # 4) per-alpha yearly detail (json)
    detail_path = os.path.join(OUT_DIR, "yearly_detail.json")
    detail = _load_json(detail_path, {})
    for r in results:
        detail[r["alpha"]] = r["yearly"]
    with open(detail_path, "w") as f:
        json.dump(detail, f, indent=2)

    return csv_path, sel_path, reg_path


# --------------------------------------------------------------------------- #
def parse_alpha_arg(s):
    out = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return [n for n in out if 1 <= n <= 101]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default="1-10", help="e.g. '1-10' or '5,9,25'")
    ap.add_argument("--start", default=None, help="YYYY-MM-DD clip start")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD clip end")
    ap.add_argument("--lag", type=int, default=1, help="execution lag (T+lag); "
                    "1 = trade next session on yesterday's close (the live cadence)")
    args = ap.parse_args()

    nums = parse_alpha_arg(args.alphas)
    print(f"Evaluating alphas: {nums}")

    data, returns, universe = a.load_panel(start=args.start, end=args.end)

    results = []
    for n in nums:
        try:
            res = evaluate_alpha(n, data, returns, universe, lag=args.lag)
            print_report(res)
            results.append(res)
        except Exception as e:
            import traceback
            print(f"\n!!! alpha{n:03d} FAILED: {e}")
            traceback.print_exc()

    csv_path, sel_path, reg_path = persist(results)

    sel = [r["alpha"] for r in results if r["selected"]]
    neg = [r["alpha"] for r in results if r["neg_selected"]]
    reg = [r["alpha"] for r in results if r["regime_candidate"]]
    print("\n" + "=" * 60)
    print(f"SUMMARY ({len(results)} alphas)")
    print(f"  Selected (IC>{IC_THRESHOLD}, hit>{HIT_THRESHOLD:.0%}): {sel or 'none'}")
    print(f"  Selected if sign-flipped:                {neg or 'none'}")
    print(f"  Regime-dependent candidates:             {reg or 'none'}")
    print(f"  Metrics  -> {csv_path}")
    print(f"  Selected -> {sel_path}")
    print(f"  Regime   -> {reg_path}")


if __name__ == "__main__":
    main()
