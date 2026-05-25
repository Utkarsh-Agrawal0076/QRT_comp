"""compare_stat_arb_configs.py — Backtest stat-arb on two pair configs head-to-head.

For each config:
  1. Cold-restart the Kalman pipeline (delete state files so no warm-start leakage)
  2. Build full historical weights_sa
  3. Compute gross + net daily PnL (TC = 5 bps/turnover)
  4. Report Sharpe, max DD, turnover, ann return, hit rate

Original config is kept as the baseline. Live file kalman_universe_config.csv is
unchanged at the end of the run (restored from backup).
"""
import os
import sys
import shutil
import time
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import generate_submission as gs

ORIGINAL_CFG = "kalman_universe_config_original.csv"
CURATED_CFG  = "kalman_universe_config_curated.csv"
LIVE_CFG     = "kalman_universe_config.csv"
TC_BPS       = 0.0005   # 5 bps per unit of turnover (round-trip)

# ---- 1. Load data once ----
print("=" * 60)
print("STAT-ARB CONFIG COMPARISON")
print("=" * 60)
df_hist, returns, universe, df_adv_60 = gs.load_data()

# Clean returns to avoid inf/-inf poisoning (same as in cell 6f)
returns_clean = returns.replace([np.inf, -np.inf], np.nan).clip(-0.5, 2.0).fillna(0)

# ---- 2. Helper to run + score one config ----
def run_and_score(name, cfg_path):
    print(f"\n{'='*60}\n>>> CONFIG: {name}  ({cfg_path})\n{'='*60}")
    cfg = pd.read_csv(cfg_path)
    print(f"  Pairs in config: {len(cfg)}, industries: {cfg['industry'].nunique()}")

    # Swap live config + delete state files for clean cold-start
    shutil.copy(cfg_path, LIVE_CFG)
    for f in [gs.KALMAN_STATE_PATH, gs.SA_WEIGHTS_CACHE]:
        if os.path.exists(f):
            os.remove(f)

    # Build weights for full history
    t0 = time.time()
    weights = gs.run_stat_arb(df_hist, universe)
    elapsed = time.time() - t0

    # Daily PnL = w_t * r_t (weights already shifted T+1 upstream)
    ar = returns_clean.reindex(index=weights.index, columns=weights.columns).fillna(0)
    pnl_g = (weights * ar).sum(axis=1)

    # Turnover proxy: sum of absolute weight changes per day
    turnover = weights.diff().abs().sum(axis=1).fillna(0)
    tc_cost = turnover * TC_BPS                    # per-day TC
    pnl_n   = pnl_g - tc_cost

    # Metrics
    eq    = (1 + pnl_g).cumprod()
    dd    = (eq / eq.cummax() - 1).min()
    ann_g = pnl_g.mean() * 252
    ann_n = pnl_n.mean() * 252
    vol   = pnl_g.std() * np.sqrt(252)
    sh_g  = ann_g / vol if vol > 0 else 0
    sh_n  = ann_n / vol if vol > 0 else 0
    hit   = (pnl_g > 0).mean()

    print(f"  Run time: {elapsed:.1f}s")
    print(f"  Annualised return: gross {ann_g*100:+.2f}%  net {ann_n*100:+.2f}%")
    print(f"  Annualised vol:    {vol*100:.2f}%")
    print(f"  Sharpe:            gross {sh_g:+.3f}  net {sh_n:+.3f}")
    print(f"  Max drawdown:      {dd*100:.1f}%")
    print(f"  Avg turnover:      {turnover.mean()*100:.2f}% / day")
    print(f"  Hit rate:          {hit*100:.1f}%")

    return {
        "name": name, "cfg_path": cfg_path,
        "n_pairs": len(cfg),
        "elapsed_s": elapsed,
        "ann_ret_gross": ann_g, "ann_ret_net": ann_n,
        "ann_vol": vol,
        "sharpe_gross": sh_g, "sharpe_net": sh_n,
        "max_dd": dd, "avg_turnover": turnover.mean(),
        "hit_rate": hit,
        "weights": weights, "pnl_g": pnl_g, "pnl_n": pnl_n,
    }

# ---- 3. Run both configs ----
results = {}
for name, cfg in [("ORIGINAL", ORIGINAL_CFG), ("CURATED", CURATED_CFG)]:
    if not os.path.exists(cfg):
        print(f"\n  SKIP {name}: {cfg} not found"); continue
    results[name] = run_and_score(name, cfg)

# ---- 4. Restore live config to ORIGINAL (do not commit until user decides) ----
shutil.copy(ORIGINAL_CFG, LIVE_CFG)
for f in [gs.KALMAN_STATE_PATH, gs.SA_WEIGHTS_CACHE]:
    if os.path.exists(f):
        os.remove(f)
print(f"\n>>> Restored live config to ORIGINAL. State files deleted.")

# ---- 5. Comparison table ----
if len(results) == 2:
    print("\n" + "=" * 60)
    print("HEAD-TO-HEAD")
    print("=" * 60)
    o, c = results["ORIGINAL"], results["CURATED"]
    rows = [
        ("Pairs",              o["n_pairs"],                c["n_pairs"]),
        ("Ann return GROSS %", o["ann_ret_gross"]*100,      c["ann_ret_gross"]*100),
        ("Ann return NET   %", o["ann_ret_net"]*100,        c["ann_ret_net"]*100),
        ("Ann vol         %",  o["ann_vol"]*100,            c["ann_vol"]*100),
        ("Sharpe GROSS",       o["sharpe_gross"],           c["sharpe_gross"]),
        ("Sharpe NET",         o["sharpe_net"],             c["sharpe_net"]),
        ("Max drawdown   %",   o["max_dd"]*100,             c["max_dd"]*100),
        ("Avg turnover  %/d",  o["avg_turnover"]*100,       c["avg_turnover"]*100),
        ("Hit rate       %",   o["hit_rate"]*100,           c["hit_rate"]*100),
    ]
    print(f"\n{'Metric':<22s} {'ORIGINAL':>15s} {'CURATED':>15s} {'Diff':>12s}")
    for label, vo, vc in rows:
        delta = vc - vo
        print(f"{label:<22s} {vo:>15.3f} {vc:>15.3f} {delta:>+12.3f}")

    # Recent 1-yr OOS slice
    print("\n--- LAST 1Y (OOS) ---")
    one_y_ago = o["pnl_g"].index.max() - pd.DateOffset(years=1)
    for name, r in results.items():
        p = r["pnl_g"].loc[one_y_ago:]
        sh = p.mean() / p.std() * np.sqrt(252) if p.std() > 0 else 0
        print(f"  {name:8s}  ann_ret={p.mean()*252*100:+.2f}%  vol={p.std()*np.sqrt(252)*100:.2f}%  Sharpe={sh:+.3f}")

print("\nDone.")
