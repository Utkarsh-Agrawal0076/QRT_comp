# Strategy Research Log — Risk-Adjusted Momentum / Volume-Anomaly / Range-Percentile Classifier

Research notes documenting the full investigation of a proposed 3-factor "price-spike" classifier strategy, the complete chain of experiments (10 stages), the empirical results, and the analysis that led to shelving it. All experiments use the same data (`top_5000_yf_data.pkl`, daily OHLCV, 4,999 US stocks 2010-01-04 → 2026-05-21), the same 5M ADV universe filter, and the same QRT backtest cost model (**2 bps execution + 0.5%/yr financing on GMV**).

> Companion to [RESEARCH.md](RESEARCH.md) (Squeeze / FFT / GAM) and [PRODUCTION_STRATEGIES.md](PRODUCTION_STRATEGIES.md) (MR / Momentum / Stat-Arb). The production baseline this is measured against is the IV-blended 3-sleeve ensemble at net Sharpe **1.058** (4-yr OOS), with a ResMR overlay variant at 1.187.

The OOS window is the **last 5 years (2021-05-21 → 2026-05-21)** unless a 4-yr window (2022-05 → 2026-05) is noted to reconcile with the production sleeve numbers.

---

## Outcome Summary

| Stage | Question | Headline result | Verdict |
|---|---|---|---|
| 1 | Do the 3 raw factors predict? | f2 clean (+3.5 bps/d), f1 wrong-sign, **f3 high-IC trap** | f3/f1 misleading |
| 2 | Does a classifier on f1/f2/f3 work? | IC 0.0098 (t=3.75) but **net SR −0.53**, turnover 109% | GAM repeat |
| 3 | Is the short side the loser? | Short −0.67 SR, long +0.39; **but long is pure market beta (β=0.84)** | beta, not alpha |
| 4 | Any market-neutral alpha after stripping beta? | beta-residual D10−D1 = +7.6%/yr gross; net only with turnover control | thin, real |
| 5 | Reweight buckets? | demeaned deciles **non-monotone** (D7 best, regime-unstable) | overfit trap |
| 6 | Is the model stale to the regime? (walk-fwd / era split) | walk-fwd net −0.23; "train 2010-15" +0.54 | **+0.54 = overfitting** |
| 7 | Construction-agnostic signal quality of each scheme | era2010-15 best, walk-fwd ≈ static, **era2016-21 inverted** | non-stationary |
| (7b) | Fair pre-specified portfolios, all schemes | walk-fwd **net-negative**; avg IC was time-clustered | honest re-baseline |
| 8 | Full-history walk-fwd + regime slicing | full ≈ 0; **dies calm-bull, earns in stress** | regime-conditional |
| 9 | HMM regime gate | "skip bull-dominant" **+0.21**; aggressive bear-gating −0.70 | HMM = wrong axis |
| 10 | Cross-sectional dispersion gate | **non-monotone**: mid +0.44, low −0.20, high −0.48 | moderate-dispersion |

**Final verdict: SHELF.** Thin real signal (IC ~0.01); the reported "25%/yr" of the source was long-only **market beta** that QRT's auto-hedge strips; best *deployable* (point-in-time) net Sharpe ≈ **+0.21** via a calm-bull regime gate. Not competitive with the production sleeves (0.49 / 0.58 / 0.89). Only merit: −0.08 correlation to the S&P (a marginal diversifier).

---

## The Original Proposal

A long-biased "price-spike" strategy. Adjusted prices, cross-sectional winsorization (2nd/98th pct daily), three standardized factors:

**Factor 1 — Risk-Adjusted Momentum:**
$$f_{1,i} = \frac{P_{i,t}/P_{i,t-14} - 1}{\sigma_{14}(R_i)}$$

**Factor 2 — Volume Anomaly:**
$$f_{2,i} = \frac{V_{i,t} - \text{SMA}(V_i, 20)}{\text{StdDev}(V_i, 20)}$$

**Factor 3 — Range Percentile (325-day):**
$$f_{3,i} = \frac{P_{i,t} - \min(P_i, 325)}{\max(P_i, 325) - \min(P_i, 325)}$$

The proposal fed these into an XGBoost/RandomForest classifier to output a probability $\hat Y_{i,t}$ of "sustained upward drift," with dynamic exits (alpha-decay below the cross-sectional median; ATR-3× trailing stop).

### Adaptation to the competition (decided up front)

The literal proposal is **long-only and event-driven** (stops, trailing exits). QRT's framework is **daily notional targets, dollar-neutral, auto-hedged residual beta, GMV-capped, 500k risk limit** — and `utils.backtest_portfolio` *enforces* dollar-neutrality + GMV=1. ATR trailing stops cannot be expressed (you submit targets, fills at next-minute mid). So the strategy was reformulated as a **cross-sectional L/S sleeve**: build the score, rank cross-sectionally, trade the tails, T+1 execution, water-fill to GMV=1. Exit mechanics become implicit in the daily rebalance. This makes it directly comparable to the production sleeves.

Classifier note: `xgboost` is not installed; used sklearn `HistGradientBoostingClassifier` (an XGBoost-equivalent gradient-booster). Label = 1 if forward 5-day cumulative return beats the cross-sectional median (balanced, aligned with an L/S ranking objective).

---

## Stage 1 — Raw Factor Diagnostics ([`range_momentum_pipeline.py`](range_momentum_pipeline.py))

Before any model: does each factor carry a tradable signal? Standard battery (IC, IR, t-stat, decile spread, monotonicity), OOS T+1, winsorized 2/98 daily.

| Factor | mean IC | t-stat | D10−D1 spread | decile monotonicity | read |
|---|---|---|---|---|---|
| f1 risk-adj momentum | +0.0012 | 0.3 | **−1.9 bps/d** | **−0.70** | wrong-sign / dead at T+1 |
| f2 volume anomaly | +0.0042 | 2.5 | **+3.5 bps/d** | +0.60 | clean, tradable |
| f3 range percentile | **+0.0234** | **4.1** | **−0.9 bps/d** | +0.38 | **high-IC trap** |
| combined (z-sum) | +0.0148 | 3.3 | **−1.4 bps/d** | +0.26 | IC high, L/S loses |

**Findings:**
- **f3 is a textbook GAM-style trap** — highest IC (t=4.1) but a *negative* decile spread (the L/S extremes don't pay) and IS→OOS sign instability (IS spread −9.2 bps/d vs OOS −0.9). A slow 325-day level factor behaving as a regime-cyclic factor.
- **f1 is wrong-sign at T+1** (decile corr −0.70 — high 14-day momentum mean-reverts next day). Its IC only turns positive at lag 10 (+0.006); the short-term reversal is already owned by the MR sleeve.
- **f2 is the only clean factor** — monotone deciles (+0.60), IR ~1.1, +3.5 bps/d, stable.
- **Combining hurts:** f3's high-IC/no-spread structure dominates the rank and drags f2's good structure into a non-tradable blend (combo spread −1.4 bps/d).

**Decision (per user):** despite the diagnostic warning, build the classifier anyway and let the net-Sharpe backtest be the judge.

---

## Stage 2 — Classifier L/S Sleeve ([`range_momentum_stage2.py`](range_momentum_stage2.py))

HistGradientBoosting on f1/f2/f3, trained on 700k subsample of IS (2010–2021), scored on OOS, ranked → D10 long / D1 short, T+1, water-fill GMV=1.

| Metric | Value |
|---|---|
| signal mean IC | +0.0098 (t=3.75, IR 1.68) |
| decile spread D10−D1 | +2.09 bps/d (corr +0.57) — classifier *fixed* f3's broken deciles |
| **Gross Sharpe** | **+0.407** |
| Turnover | **108.7%/day** |
| **Net Sharpe** | **−0.532** |
| Max DD | −8.4% |

PnL correlation vs sleeves: Mom +0.34, ResMR +0.33, MR +0.14, SA −0.12.

**Finding — the GAM result, reproduced.** The classifier turned f1/f2/f3 into a positive, monotone decile spread (better than raw factors), but the gross edge is only +0.41 SR and **108% daily turnover × 2 bps ≈ 10%/yr in costs** flips it to net −0.53. Loses in 2023 (the AI-rotation regime-flip year that broke Momentum and GAM). The signal "lives inside the cost barrier."

---

## Stage 3 — Long/Short Decomposition: the Beta Discovery ([`range_momentum_stage3.py`](range_momentum_stage3.py))

Hypothesis (user): the source was long-only and made ~25%/yr; is the **short side** the drag?

| Book | ann net | net SR | turnover | max DD |
|---|---|---|---|---|
| Long (D10) @ 0.5 GMV | +3.9% | +0.39 | 98% | −12.7% |
| Short (D1) @ 0.5 GMV | −7.2% | −0.67 | 119% | −37.5% |
| Long-only D10 @ GMV=1 | +7.8% | +0.39 | 98% | −24.1% |

**Findings:**
- Yes, on raw returns the short book is the proximate drag (−0.67 vs +0.39). But the deeper truth: **market drift over the window is +16%/yr** (equal-weight universe, SR 0.73). The short book fights it — D1 (the "worst" names) still rises +3.4 bps/day = +8.5%/yr; shorting them costs the drift (−4%/yr gross) plus 119% turnover (−3.3%/yr) = −7.2%/yr net.
- **The long book's profit is almost entirely market beta.** Long-only D10 makes +7.8%/yr net, but **market beta = 0.84**. Beta-hedged (what QRT's auto-hedge does): **ann gross −0.1%, net −5.6%/yr, SR −0.71.** Concentration sweep (top 1%→20%) is all beta — none survives hedging.

**Conclusion:** the source's "25%/yr" was long, unhedged, bull-market beta — not stock-selection alpha. QRT's dollar-neutral + auto-hedge mandate structurally strips exactly that exposure.

---

## Stage 4 — Market-Neutral Alpha & Turnover ([`range_momentum_stage4.py`](range_momentum_stage4.py))

Does any alpha survive after stripping drift/beta?

Per-decile next-day return (bps): the **beta-residual** D10−D1 spread is **+3.01 bps/d (+7.6%/yr), monotonicity +0.67** — a real but thin residual edge (caveat: not clean at the extremes; D10 is ~flat demeaned).

Turnover is the binding constraint, not the long/short split:

| Variant | gross SR | net SR | turnover |
|---|---|---|---|
| Baseline (raw rank, k=1) | 0.41 | −0.53 | 109% |
| Sector-neutral ranking | **0.84** | −0.38 | 112% |
| Smooth-10 (raw rank) | — | +0.03 | 24% |
| **SN + smooth-5 + hysteresis** | 0.58 | **+0.24** | 25% |

**Findings:** sector-neutral ranking **doubles gross SR (0.41→0.84)** (raw rank carried sector tilts the dollar-neutral book couldn't shed). Turnover reduction (smoothing + hysteresis) is what lets the thin gross edge clear the cost barrier. Best variant net **+0.24**.

---

## Stage 5 — Decile Bucket Design & the Non-Monotone Trap ([`range_momentum_stage5.py`](range_momentum_stage5.py))

Properly demeaned decile returns (what a dollar-neutral book actually earns):

| D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---|---|---|---|---|---|---|---|---|---|
| −0.85 | −0.91 | **−1.92** | −0.60 | −0.96 | −1.20 | **+4.19** | +0.28 | +0.68 | +1.26 |

**Findings:**
- The "short loses money" framing (Stage 3) was a *raw-return illusion*. Demeaned, **D1 (the short) earns +0.85 bps shorted — mildly profitable.** The weak side is actually the long (D10 only +1.26, vs the standout **D7 +4.19**).
- The signal is **non-monotone** — the classifier's highest-probability bucket (D10) is *not* the best; D7 is. Bucket enumeration: D7–D10 long / D3 short gives gross SR 0.86 vs current D10/D1 0.42.
- **But it's an overfit trap.** Half-split: 1st half D10 is best (+2.5), 2nd half D7 is best (+8.1). The D7 standout is entirely a 2nd-half artifact — picking it from the full OOS is selection bias.

---

## Stage 5b — The User's Reweighted Design + Correlation/Per-Year

Design: **long D7–D10, short D1–D4, 5-pt hysteresis** (long enter 0.60/exit 0.55; short enter 0.40/exit 0.45).

| Variant | gross | net SR | turnover |
|---|---|---|---|
| raw, smooth-3 | 0.69 | **+0.23** | 24% |
| raw, smooth-10 | 0.50 | +0.23 | 11% |
| SN, smooth-3 | 0.71 | +0.11 | 25% |

Net Sharpe ceiling ~+0.23. Sector-neutral does *not* help here (wide buckets already shed sector tilts). Smooth-5 collapses to gross +0.15 (isolated-window artifact — do not trust).

**Correlation vs production sleeves (4-yr OOS, cached sleeves are stored already-shifted — verified by reproducing MR 0.49 / Mom 0.58 / SA 0.89 / ResMR 0.53):**

```
         NEW     MR    Mom     SA  ResMR
NEW    1.000  0.217  0.406 -0.040  0.213
```

Per-year NET Sharpe (4-yr OOS):

| Year | NEW | MR | Mom | SA | ResMR |
|---|---|---|---|---|---|
| 2022 | +1.18 | −0.30 | −0.35 | +1.58 | +1.32 |
| 2023 | −0.31 | +1.63 | −0.66 | +1.90 | +0.04 |
| 2024 | +0.44 | −0.86 | +1.26 | −0.94 | −0.67 |
| 2025 | −0.10 | +1.27 | +0.43 | +1.32 | +0.97 |
| 2026 | +0.97 | +0.76 | +2.86 | −0.68 | +0.31 |
| **FULL** | **+0.31** | +0.49 | +0.58 | +0.89 | +0.53 |

**Correction logged:** the "positive every year" claim was on *gross* PnL; on *net*, NEW is negative in 2023 (−0.31) and 2025 (−0.10). The +0.41 correlation with Momentum is the main ensemble drawback; the 2022 complementarity (NEW +1.18 when Mom/MR both lost) and −0.04 correlation with SA are the merits.

---

## Stage 6 — Regime-Staleness Hypothesis ([`range_momentum_stage6.py`](range_momentum_stage6.py))

Hypothesis (user): the model trained on 2010–2021 learned a mean-reversion-era mapping and is stale for 2023–2026. If true, a walk-forward model (always trains on recent regime) should beat it.

| | static (2010-21) | walk-forward | era 2010-15 | era 2016-21 |
|---|---|---|---|---|
| chosen-design net SR | **+0.230** | **−0.226** | **+0.541** | **−1.024** |

**Findings:** the hypothesis is *rejected*. Walk-forward retraining (always-recent) **hurts** (−0.23) — reproducing the GAM finding that regime-matched retraining doesn't anticipate the realized OOS relationship. The **oldest 2010–15 data trains the best** (+0.54), the recent **2016–21 trains the worst** (−1.02). Implies the lever is training-data *cleanliness*, not recency.

---

## Stage 7 — Construction-Agnostic Signal Quality ([`range_momentum_stage7.py`](range_momentum_stage7.py))

The fixed D7–D10/D1–D4 portfolio was reverse-engineered from the *static* signal's deciles, so imposing it on other signals is unfair. Evaluate each scheme's signal directly.

| scheme | mean IC | IR | t | demeaned D10−D1 | monotonicity | best long/short |
|---|---|---|---|---|---|---|
| static | 0.0098 | 1.68 | 3.75 | +2.09 | +0.57 | D7 / D3 |
| walk-forward | 0.0134 | 1.71 | 3.81 | +2.29 | +0.71 | **D9 / D1** |
| era 2010-15 | 0.0160 | 2.07 | 4.60 | **+10.45** | **+0.83** | D10 / D1 |
| era 2016-21 | 0.0000 | 0.01 | 0.02 | **−7.61** | **−0.67** | D1 / D7 (inverted) |

**Findings:** each signal puts its alpha in a different decile — the fixed portfolio was genuinely unfair (validating the user's critique). Walk-forward IC (0.0134) ≥ static; era2016-21 is a **dead/inverted** signal (the f→return map is **non-stationary**). era2010-15 is cleanly monotone with a 5× spread.

### Stage 7b — Fair Pre-Specified Portfolios (the reckoning)

Apply *standard* designs (decile D10/D1, quintile D9–10/D1–2) identically to static and walk-forward:

| Design (5pt hyst) | static net | walk-fwd net |
|---|---|---|
| quintile D9–10/D1–2, smooth-3 | **+0.20** | −0.11 |
| decile D10/D1, smooth-5 | +0.06 | −0.45 |

**Honest correction:** giving walk-forward its own principled portfolio does *not* rescue it — it is **net-negative everywhere**. The high average IC was misleading: walk-forward's signal is **time-clustered** (IC 2021 −0.004 trained on noisy 2018–21; great 2024+ on clean recent data). The "+0.54" era2010-15 result was selection-on-the-test-set (an OOS-chosen training window with no a-priori thesis) and was dropped. Stripping both overfitting sources (the OOS window pick *and* the OOS-tuned D7 decile design), the honest deployable number is **~+0.20** (static, principled quintile).

---

## Stage 8 — Full-History Walk-Forward + Regime Conditioning ([`range_momentum_stage8.py`](range_momentum_stage8.py))

Point-in-time walk-forward (train trailing 3yr, step 126d) from the earliest possible date, **2013–2026**. Portfolio = principled quintile D9–10/D1–2, smooth-3, hysteresis.

Full-period net SR **+0.058** (≈ 0). But violently regime-dependent:

| | low-vol | mid-vol | high-vol |
|---|---|---|---|
| bull | **−0.97** | +0.07 | +0.20 |
| bear | — | +4.11¹ | +0.43 |

Single axes: bull −0.04 / **bear +0.67**; low-vol **−0.92** / mid +0.28 / high +0.24. Correlation to SPY = **−0.08**.

**Finding:** the strategy **dies in calm low-vol bull markets and earns in stress** — an economically coherent thesis (a dispersion strategy needs dislocations to predict). The full-period zero is the good and bad regimes canceling. Point-in-time gating (regime lagged 1d): "skip low-vol" +0.15, "trade bear OR high-vol only" **+0.21** — a ~3.6× lift.

---

## Stage 9 — HMM Regime Gate ([`range_momentum_stage9.py`](range_momentum_stage9.py))

Wired up the frozen 3-state Gaussian HMM ([`hidden_markov_model/hmm_model.py`](hidden_markov_model/hmm_model.py)) — causal filtered P(bull)/P(chop)/P(bear) on SPY, lagged 1 day. Gate is anti-bull (trade in stress).

| Variant | net SR | active days |
|---|---|---|
| ungated | +0.06 | 100% |
| **hard: skip bull-dominant** | **+0.21** | 50% |
| hard: bear-dominant only | **−0.70** | 9% |
| soft: GMV × [P(bear)+0.5·P(chop)] | +0.08 | 56% |
| soft: GMV × P(bear) | −0.21 | 13% |

**Finding:** the HMM gate equals the crude vol filter (+0.21), no better. Aggressive bear-gating *hurts* (−0.70) — the HMM's "bear" state (9% of days, laggy) is misaligned with the strategy's edge. **Root cause:** the HMM measures *index-level* regime, but this is a *cross-sectional dispersion* sleeve; SPY-bear ≠ high cross-sectional dispersion. Wrong regime axis.

---

## Stage 10 — Cross-Sectional Dispersion Gate ([`range_momentum_stage10.py`](range_momentum_stage10.py))

The theoretically-correct regime variable: daily cross-sectional std of returns across the universe (dispersion), smoothed 21d, ranked in a trailing-252d window (point-in-time), lagged 1d.

Strategy net SR by dispersion bucket:

| dispersion | net SR | ann % |
|---|---|---|
| low | −0.20 | −0.9% |
| **mid** | **+0.44** | +7.0% |
| high | **−0.48** | −3.1% |

Gates: skip bottom tercile +0.09; above-median +0.07; **top-tercile-only −0.53**; soft −0.05. None beats the HMM's +0.21.

**Finding — the corrected thesis.** The relationship is **non-monotone and inverted at the top**: the strategy is a **moderate-dispersion** strategy, not a stress harvester. It needs enough cross-sectional spread to have signal, but in genuine chaos (2020 COVID, 2023 banking) correlations spike, the momentum/volume/range features whipsaw, and the signal *inverts*. This corrects the coarser Stage-8 reading ("thrives in high-vol") — SPY-vol and cross-sectional dispersion are different axes (index high-vol often = everything-down = high correlation = *low* dispersion). A high-pass gate therefore concentrates into the worst regime. A band-pass gate (mid-dispersion only) is the implied next step but was *not* run, because selecting the [0.33, 0.67] band from this same slice repeats the Stage-6 selection-bias trap.

---

## Reproducibility — Script ↔ Result Map

| Script | Purpose | Key output |
|---|---|---|
| [`range_momentum_pipeline.py`](range_momentum_pipeline.py) | Stage 1 — raw factor diagnostics | `stores/range_mom/stage1_factor_diag.json` |
| [`range_momentum_stage2.py`](range_momentum_stage2.py) | Stage 2 — classifier L/S sleeve | `stores/range_mom/stage2_classifier.json`, `oos_score.pkl` |
| [`range_momentum_stage3.py`](range_momentum_stage3.py) | Stage 3 — long/short + beta decomposition | `stores/range_mom/stage3_decomp.json` |
| [`range_momentum_stage4.py`](range_momentum_stage4.py) | Stage 4 — market-neutral alpha, turnover, sector-neutral | `stores/range_mom/stage4_neutral.json` |
| [`range_momentum_stage5.py`](range_momentum_stage5.py) | Stage 5 — demeaned deciles + bucket enumeration | `stores/range_mom/stage5_deciles.json` |
| [`range_momentum_stage6.py`](range_momentum_stage6.py) | Stage 6 — walk-forward + era-split regime test | `stores/range_mom/stage6_regime.json` |
| [`range_momentum_stage7.py`](range_momentum_stage7.py) | Stage 7 — signal-quality battery across schemes | `stores/range_mom/stage7_signal_quality.json`, `scores_*.pkl` |
| [`range_momentum_stage8.py`](range_momentum_stage8.py) | Stage 8 — full-history walk-forward + regime slicing | `stores/range_mom/stage8_regime.json`, `scores_wf_full.pkl` |
| [`range_momentum_stage9.py`](range_momentum_stage9.py) | Stage 9 — HMM regime gate | (stdout) |
| [`range_momentum_stage10.py`](range_momentum_stage10.py) | Stage 10 — cross-sectional dispersion gate | (stdout) |

### Cached artifacts
- `stores/squeeze/ohlcv_cache.pkl` — pre-split OHLCV (reused from the squeeze pipeline)
- `stores/range_mom/oos_score.pkl` — static model OOS score matrix
- `stores/range_mom/scores_{static,walkfwd,era2010-15,era2016-21}.pkl` — Stage-7 scheme scores
- `stores/range_mom/scores_wf_full.pkl` — full-history (2013–2026) walk-forward scores
- `stores/range_mom/panel.pkl` — stacked (date,stock) → [f1,f2,f3,y] training panel (~6.8M rows)
- `stores/sharpe_blender/sleeves.pkl` — production sleeve weights (MR/Mom/SA/ResMR), **stored already T+1-shifted**
- `hidden_markov_model/hmm_params.pkl`, `spy_cache.parquet` — frozen HMM + SPY series

---

## Lessons Learned

1. **High IC ≠ tradable — confirmed yet again.** f3 (t=4.1) and the classifier (t=3.75) both had strong IC and lost money, because the alpha lived in the body of the distribution, not the L/S extremes. Always inspect the demeaned decile spread.

2. **Decompose raw vs market-neutral before blaming the short book.** On raw returns the short looked like the loser (−0.67); demeaned, the short was mildly *profitable* and the long was the weak side. A dollar-neutral book only earns the demeaned component — analyze it there.

3. **A long-only backtest's headline return can be ~entirely market beta.** The source's "25%/yr" was β=0.84 × 16%/yr drift. Beta-hedging (which QRT enforces) took it to −5.6%/yr. Always report the beta-hedged number for any long-biased idea destined for a market-neutral mandate.

4. **Turnover is often the real binding constraint, not signal sign.** The strategy's net SR moved from −0.53 (109% turnover) to +0.24 purely via smoothing + hysteresis. Gross edge of +0.4 SR is below the cost barrier at high turnover.

5. **Selection-on-the-test-set is the cardinal sin — and it is seductive.** "Train on 2010–2015" scored +0.54 and felt like a discovery; it was a hindsight pick with no point-in-time story and no economic thesis. Walk-forward is the only honest protocol. When the overfitting was stripped, the deployable number was ~+0.20.

6. **Average IC hides time-clustering.** Walk-forward's average IC (0.0134) beat static, yet it was net-negative because its signal quality was concentrated in 2024–2026 and negative in 2021–2023 (when its trailing window was the noisy 2018–21 data). Look at IC *by year*, not just the mean.

7. **The f→return relationship is non-stationary.** The 2016–2021 training era produced an *inverted* signal (monotonicity −0.67). Non-stationarity is an argument for adaptive estimation and against any single fixed training set — but it also caps what retraining can achieve, because the realized OOS relationship is unknowable in advance.

8. **Match the regime detector to the strategy's actual driver.** An index-level HMM (SPY return + vol) gated no better than a crude vol filter, because this is a *cross-sectional dispersion* strategy. SPY-vol ≠ cross-sectional dispersion (index crashes are high-correlation/low-dispersion events).

9. **Test the regime relationship for monotonicity before gating.** The intuitive "trade in high dispersion" was wrong: the strategy is a *moderate*-dispersion strategy (mid +0.44, low −0.20, high −0.48). A high-pass gate concentrated into its worst regime. The shape of the conditional relationship matters as much as its existence.

10. **A regime-conditional zero-Sharpe strategy is not automatically salvageable.** Gating lifted the full-period +0.06 to a deployable ~+0.21, but bad years persisted and the ceiling stayed well below the production sleeves. Regime-conditioning sharpened the edge; it did not manufacture one.
