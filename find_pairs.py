"""find_pairs.py — Within-industry cointegration scan for stat-arb pair expansion.

Standalone offline script (run once, then point generate_submission.py at the new
config CSV). Scans every same-industry pair of liquid stocks for Engle-Granger
cointegration, filters by ADF p-value, half-life, spread vol and beta magnitude.

Outputs a CSV in the same schema as kalman_universe_config.csv. By default merges
with the existing 106 hand-picked pairs (no overlap) to produce
kalman_universe_config_expanded.csv.

Usage:
    python find_pairs.py            # full scan + merge with existing
    python find_pairs.py --replace  # only output new pairs, don't merge
    python find_pairs.py --industry "Major Banks"   # scan one industry only
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_PICKLE       = "top_5000_yf_data.pkl"
META_CSV          = "top_5000_us_by_marketcap.csv"
EXISTING_CONFIG   = "kalman_universe_config.csv"
OUTPUT_CSV        = "kalman_universe_config_expanded.csv"

LOOKBACK_START    = "2018-01-01"   # IN-SAMPLE selection period
LOOKBACK_END      = "2024-12-31"   # leaves 2025+ as OOS validation

MIN_OBS           = 504            # ≥ 2 years of overlapping non-NaN data
ADV_MIN           = 5_000_000      # 60d ADV ≥ $5M
ADV_LOOKBACK      = 60
MIN_UNIV_FRACTION = 0.70           # both stocks must be in universe ≥70% of lookback

PRE_FILTER_CORR   = 0.40           # log-price correlation pre-filter (was 0.50)
ADF_P_MAX         = 0.05           # cointegration significance (classic threshold, was 0.01)
HL_MIN, HL_MAX    = 3.0, 60.0      # half-life range in days (was 3-20)
VOL_MIN, VOL_MAX  = 0.003, 0.100   # spread vol (was 0.005-0.080; captures dual-class pairs)
ABS_BETA_MIN      = 0.30
ABS_BETA_MAX      = 3.00

# ============================================================================
# HELPERS
# ============================================================================

def load_data():
    print(f"Loading {DATA_PICKLE}...")
    df = pd.read_pickle(DATA_PICKLE)
    df = df.loc[LOOKBACK_START:LOOKBACK_END]
    print(f"  Lookback: {df.index.min().date()} -> {df.index.max().date()} ({len(df)} days)")

    prices = df["Adj Close"]
    close = df["Close"]
    volume = df["Volume"]
    # dedup duplicate column labels (yfinance occasionally returns duplicates)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    close  = close.loc[:,  ~close.columns.duplicated()]
    volume = volume.loc[:, ~volume.columns.duplicated()]

    # Universe mask
    dollar_vol = (close * volume).fillna(0)
    adv_60 = dollar_vol.rolling(ADV_LOOKBACK, min_periods=ADV_LOOKBACK).mean()
    universe = (adv_60 >= ADV_MIN).astype(int)

    # Industry mapping
    meta = pd.read_csv(META_CSV)
    meta["symbol"] = meta["symbol"].str.replace("/", "-")
    industry_map = meta.set_index("symbol")["industry"].to_dict()

    return prices, universe, industry_map


def filter_candidates(prices, universe, industry_map):
    """Group liquid stocks by industry, only keep those with enough universe membership."""
    industries: dict[str, list[str]] = {}
    for stock in prices.columns:
        ind = industry_map.get(stock)
        if not ind or pd.isna(ind):
            continue
        if stock not in universe.columns:
            continue
        # Check liquidity throughout lookback
        univ_fraction = universe[stock].mean()
        if univ_fraction < MIN_UNIV_FRACTION:
            continue
        n_obs = prices[stock].notna().sum()
        if n_obs < MIN_OBS:
            continue
        industries.setdefault(ind, []).append(stock)

    industries = {k: sorted(v) for k, v in industries.items() if len(v) >= 2}
    return industries


def test_pair(y_logpx: np.ndarray, x_logpx: np.ndarray):
    """Engle-Granger cointegration test on aligned log prices.

    Returns dict with stats if pair passes all filters, else None.
    """
    # OLS: y = alpha + beta * x + epsilon
    X = sm.add_constant(x_logpx)
    ols = sm.OLS(y_logpx, X).fit()
    alpha, beta = ols.params
    if not (ABS_BETA_MIN <= abs(beta) <= ABS_BETA_MAX):
        return None

    resid = ols.resid

    # ADF on residuals
    try:
        adf = adfuller(resid, regression="c", autolag="AIC", maxlag=10)
    except Exception:
        return None
    adf_p = adf[1]
    if adf_p > ADF_P_MAX:
        return None

    # Half-life from AR(1) on first-differenced residuals
    de = np.diff(resid)
    lag_e = resid[:-1]
    ar = sm.OLS(de, sm.add_constant(lag_e)).fit()
    phi = ar.params[1]
    if phi >= 0:                  # non-mean-reverting
        return None
    half_life = -np.log(2) / np.log1p(phi)
    if not (HL_MIN <= half_life <= HL_MAX):
        return None

    spread_vol = float(np.std(resid))
    if not (VOL_MIN <= spread_vol <= VOL_MAX):
        return None

    return {
        "tls_alpha":      float(alpha),
        "tls_beta":       float(beta),
        "adf_p_value":    float(adf_p),
        "half_life_days": float(half_life),
        "spread_vol":     spread_vol,
        "n_obs":          len(y_logpx),
    }


def scan_industry(industry: str, stocks: list[str], log_prices: pd.DataFrame):
    """Run the full pair-finding pipeline for one industry."""
    n = len(stocks)
    n_cand = n * (n - 1) // 2

    # Sub-select log price subframe + correlation matrix
    sub = log_prices[stocks]
    corr = sub.corr()

    found = []
    n_corr_filtered = 0
    n_eg_filtered = 0
    n_passed = 0

    for i in range(n):
        si = stocks[i]
        y_full = sub[si]
        for j in range(i + 1, n):
            sj = stocks[j]
            c = corr.iloc[i, j]
            if pd.isna(c) or abs(c) < PRE_FILTER_CORR:
                n_corr_filtered += 1
                continue

            x_full = sub[sj]
            both = pd.concat([y_full, x_full], axis=1).dropna()
            if len(both) < MIN_OBS:
                n_corr_filtered += 1
                continue

            stats = test_pair(both.iloc[:, 0].values, both.iloc[:, 1].values)
            if stats is None:
                n_eg_filtered += 1
                continue

            # Convention: sort by ticker alphabetically (asset_y = lex-smaller)
            ay, ax = (si, sj) if si < sj else (sj, si)
            # If we swapped, the OLS we ran was y_si = alpha + beta * x_sj, so we
            # need to re-derive in the right direction. Cheapest: re-run with swap.
            if ay != si:
                stats = test_pair(both.iloc[:, 1].values, both.iloc[:, 0].values)
                if stats is None:
                    n_eg_filtered += 1
                    continue

            stats.update({
                "industry": industry,
                "asset_y":  ay,
                "asset_x":  ax,
                "corr_log_prices": float(c),
            })
            found.append(stats)
            n_passed += 1

    return found, dict(n_candidates=n_cand, n_corr_filtered=n_corr_filtered,
                       n_eg_filtered=n_eg_filtered, n_passed=n_passed)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replace", action="store_true",
                        help="Output only newly-found pairs (don't merge with existing 106).")
    parser.add_argument("--industry", type=str, default=None,
                        help="Scan only one industry (for debugging/quick test).")
    parser.add_argument("--min-stocks", type=int, default=4,
                        help="Skip industries with fewer than this many liquid stocks.")
    parser.add_argument("--top-k-per-industry", type=int, default=15,
                        help="Per-industry cap: keep top K pairs by ADF p-value (default 15).")
    args = parser.parse_args()

    t0 = time.time()
    prices, universe, industry_map = load_data()
    log_prices = np.log(prices)

    industries = filter_candidates(prices, universe, industry_map)
    industries = {k: v for k, v in industries.items() if len(v) >= args.min_stocks}
    print(f"Liquid industries to scan: {len(industries)} "
          f"(>= {args.min_stocks} stocks, >= {MIN_UNIV_FRACTION*100:.0f}% univ membership)")

    if args.industry:
        if args.industry not in industries:
            print(f"  ERROR: industry '{args.industry}' not in candidate set.")
            print(f"  Available: {list(industries.keys())[:20]}...")
            sys.exit(1)
        industries = {args.industry: industries[args.industry]}

    # Sort largest-first so we get progress signal early
    industries_sorted = sorted(industries.items(), key=lambda x: -len(x[1]))
    total_cands = sum(len(v) * (len(v) - 1) // 2 for _, v in industries_sorted)
    print(f"Total candidate pairs (pre-filter): {total_cands:,}")

    all_pairs = []
    totals = {"n_candidates": 0, "n_corr_filtered": 0, "n_eg_filtered": 0, "n_passed": 0}

    for idx, (ind, stocks) in enumerate(industries_sorted, 1):
        n = len(stocks)
        n_cand = n * (n - 1) // 2
        t_ind = time.time()
        found, stats = scan_industry(ind, stocks, log_prices)
        elapsed = time.time() - t_ind
        all_pairs.extend(found)
        for k, v in stats.items():
            totals[k] = totals.get(k, 0) + v
        print(f"[{idx:3d}/{len(industries_sorted)}] {ind[:50]:50s} "
              f"stocks={n:3d}  cand={n_cand:6,d}  passed={stats['n_passed']:3d}  "
              f"({elapsed:.1f}s)", flush=True)

    print()
    print("=" * 60)
    print(f"SCAN COMPLETE in {time.time()-t0:.1f}s")
    print("=" * 60)
    print(f"Total candidates:           {totals['n_candidates']:,}")
    print(f"Filtered by correlation:    {totals['n_corr_filtered']:,}")
    print(f"Filtered by EG/HL/vol/beta: {totals['n_eg_filtered']:,}")
    print(f"Passed all filters:         {totals['n_passed']:,}")

    if not all_pairs:
        print("\nNo pairs found. Loosen filters.")
        return

    new_df = pd.DataFrame(all_pairs)
    print(f"\nFound {len(new_df)} candidate pairs (pre-cap) across {new_df['industry'].nunique()} industries")

    # Per-industry cap: keep top K by ADF p-value (most-cointegrated first)
    if args.top_k_per_industry and args.top_k_per_industry > 0:
        before = len(new_df)
        new_df = (new_df.sort_values("adf_p_value")
                          .groupby("industry", group_keys=False)
                          .head(args.top_k_per_industry)
                          .reset_index(drop=True))
        print(f"Applied per-industry cap (top {args.top_k_per_industry} by ADF p): "
              f"{before} -> {len(new_df)} pairs")

    print("\nDistribution by industry (top 20):")
    print(new_df["industry"].value_counts().head(20).to_string())

    # Optional merge with existing 106
    if not args.replace and os.path.exists(EXISTING_CONFIG):
        old_df = pd.read_csv(EXISTING_CONFIG)
        existing_keys = set(zip(old_df["asset_y"], old_df["asset_x"]))
        new_keys = set(zip(new_df["asset_y"], new_df["asset_x"]))
        truly_new = new_keys - existing_keys
        added = new_df[new_df.apply(lambda r: (r["asset_y"], r["asset_x"]) in truly_new, axis=1)]
        print(f"\nMerging with existing {len(old_df)} pairs:")
        print(f"  Overlap with existing: {len(new_keys & existing_keys)}")
        print(f"  New pairs added:       {len(added)}")
        # Keep schema parity with original file when concatenating
        for col in old_df.columns:
            if col not in added.columns:
                added[col] = np.nan
        added = added[old_df.columns.tolist() + [c for c in added.columns if c not in old_df.columns]]
        merged = pd.concat([old_df, added.reindex(columns=added.columns)], ignore_index=True)
        merged.to_csv(OUTPUT_CSV, index=False)
        print(f"\nWrote {OUTPUT_CSV} with {len(merged)} total pairs ({len(old_df)} existing + {len(added)} new)")
    else:
        new_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nWrote {OUTPUT_CSV} with {len(new_df)} pairs (replace mode)")


if __name__ == "__main__":
    main()
