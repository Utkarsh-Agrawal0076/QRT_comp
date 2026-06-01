# 101 Formulaic Alphas — backtesting harness

Implementation + screening pipeline for the 101 alphas from
Kakushadze (2015), *"101 Formulaic Alphas"* (arXiv:1601.00991), evaluated on
this repo's **$5M / 60-day-ADV universe** (`universe_5m.parquet`, ~2,200
names/day) under the live **T+1 execution lag** (data through yesterday's close
→ weights traded the next session → earn that day's return).

## Layout
- `operators.py` — vectorised WorldQuant operators (`rank`, `ts_rank`,
  `correlation`, `decay_linear`, `ts_argmax`, `indneutralize`, …). Everything
  acts on `date × ticker` DataFrames; `correlation`/`covariance` use the rolling
  moment formula and `ts_rank` uses `bottleneck.move_rank`, so the whole
  universe is processed at once.
- `alphas.py` — `alpha001 … alpha101`, each a near-verbatim transcription of the
  paper. Registry: `ALL_ALPHAS` (dict) and `get_alpha(n)`. Metadata sets:
  `INDNEUTRAL_ALPHAS`, `CAP_ALPHAS`.
- `data.py` — `load_panel()` builds the aligned input panel.
- `../alpha101_pipeline.py` — the screening pipeline (CLI).

## Running
```bash
# in-sample screen (2010-2020), live T+1 lag; 2021+ held out for OOS
python alpha101_pipeline.py --alphas 1-101 --lag 1 --end 2020-12-31
python alpha101_pipeline.py --alphas 5,9,25 --start 2021-01-01   # OOS check
```
Default `--lag 1` matches the live cadence. 2021+ is reserved for out-of-sample
validation, so the screen is run with `--end 2020-12-31`.
Outputs land in `alpha101_results/`:
- `all_metrics.csv` — mean IC, hit rate, IC-IR, t-stat, decile L/S Sharpe, yearly spread.
- `selected_alphas.json` — alphas passing **mean IC > 0.02 AND hit rate > 50%**
  (plus strong-negative-IC alphas marked `short(flip)`).
- `regime_alphas.json` — strong-in-some-years / weak-in-others candidates.
- `yearly_detail.json` — full per-year IC / hit / L/S-Sharpe tables.

## Methodology
- **Signal → IC**: signal at day *t* is lagged by `--lag` (default 1) and
  rank-correlated cross-sectionally with the return realised that day, masked to
  the daily 5M-ADV universe — exactly the relationship a T+1 portfolio trades.
  IC = daily Spearman (verified against `scipy.stats.spearmanr`); hit rate =
  share of days with IC > 0. IC roughly halves from lag 1 → lag 2 (these are
  fast signals), so the lag choice matters a lot.
- **Decile L/S**: top vs bottom decile of the signal, equal-weight, dollar-neutral,
  lagged 2 days; gross annualised Sharpe overall and per year.

## Data assumptions / caveats
- **vwap** is a surrogate typical price `(high+low+close)/3` (no intraday data) —
  same surrogate used elsewhere in this repo.
- **cap** (Alpha 56) is a *static* market cap from `top_5000_us_by_marketcap.csv`
  (current snapshot, not point-in-time) — a proxy. Flagged via `CAP_ALPHAS`.
- **indneutralize** uses the CSV's `sector`/`industry`; there is no sub-industry
  classification, so `IndClass.subindustry` falls back to `industry`. The ~18
  alphas in `INDNEUTRAL_ALPHAS` are therefore **approximate** and flagged
  `indneutral_approx` in the metrics.
- `returns` is the official `stores/returns.parquet` with ±inf cleaned to NaN.

## Note on thresholds
On this broad ~1,800-name universe at T+2, full-sample mean ICs cluster around
0.002–0.005 — about 10× below the 0.02 selection bar (these alphas were designed
for next-day, single-name horizons). The 0.02 / 50% rule is applied literally as
requested; the regime thresholds (`REGIME_STRONG_YEAR_IC`, `REGIME_MIN_SPREAD`)
are calibrated to the observed IC scale and live at the top of the pipeline.
