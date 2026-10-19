# Trial 14 — the deployed live config, backtested as deployed

**Verdict: RETIRE.**

The book that has been paper-traded was never the book that was backtested. The pipeline fit every 126 sessions on prices from 2009 and held a decile book for 21 sessions, marked close to close. The deployed cycle loads prices from 2018-01-01, refits ridge every session, rebuilds the decile book every session, and fills at the next open. This trial is that cycle, on history, scored once.

## Pre-registered kill rule

Retire unless all three hold, at N=14 (this evaluation counts; it is not a free reproduction of trial 5):

- OOS rank-IC Newey-West t (lags = 21, scored on `pred_raw`) ≥ 3.48: **0.78** (fail)
- net Sharpe > 0: **-1.122** (fail)
- DSR ≥ 0.95: **0.000** (fail)

Ex-ante minimum detectable effect, stated before the download: **1.233 net annual Sharpe** at N=14, design n_obs=1900, symmetric returns (the DSR = 0.95 hurdle from `graduation_hurdle.required_ann_sharpe`). Realized n_obs=1913; the same function at the realized sample and the realized skew is 1.225. That reconciliation does not move the kill thresholds.

## Graded book (daily refit, daily decile, next open, 10 bps/side)

- Mean rank IC (`pred_raw` vs residual 21-day label): 0.0102 (Newey-West t 0.78)
- Gross Sharpe 0.130, net Sharpe -1.122, DSR 0.000 at N=14
- Net annual return -8.31%, ann. vol 7.41%, max drawdown -49.84%
- One-way turnover 46.38×/yr, cost drag 9.28%/yr
- 12-1 momentum baseline, same dates, same next-open fills, same 10 bps: net Sharpe -0.106 (model does not beat it). Equal-weight member book, open-to-open: Sharpe 0.767.
- Window: 2019-03-11 → 2026-10-16 (1913 sessions).

The momentum and equal-weight numbers are the law-5 baselines. They were not kill criteria. The 21-day book below is not one either.

## Turnover and cost: daily vs 21-session rebalance

Same scores, same next-open fills, same 10 bps/side. The 21-session arm is the old hold applied to these predictions. It is a decomposition of the cost, not a candidate, and it cannot flip a retire into a keep.

| book | one-way turnover | cost drag / yr | gross Sharpe | net Sharpe |
|---|---:|---:|---:|---:|
| daily (graded) | 46.38× | 9.28% | 0.130 | -1.122 |
| 21-session hold | 8.82× | 1.76% | -0.120 | -0.383 |

Turnover ratio, daily / 21-session: 5.26. Extra cost drag from rebuilding every day: 7.51%/yr.

## Beta after integer-share rounding

Continuous weights are projected to zero ex-ante beta against the equal-weight member market. The order path then clips at 5% and truncates to integer shares at $1,000,000 (the declared notional, not a broker read), using that session's close. `int(w · equity / price)` truncates toward zero, as `orders_from_weights` does.

- Ex-ante beta, continuous: mean -0.0000, mean |β| 0.0000
- Ex-ante beta, after rounding: mean -0.0002, mean |β| 0.0017, p95 |β| 0.0054
- Mean net weight after rounding (dollar-neutrality break): -0.0006 (mean absolute 0.0017)
- Mean L1 distance between continuous and rounded weights: 0.0074
- Realized full-sample beta of the gross next-open returns vs the member-mean open-to-open market: continuous 0.030, rounded 0.030
- Realized beta vs SPY open-to-open (descriptive, not a kill): continuous 0.029, rounded 0.029

## What this does not say

Adjusted Yahoo prices are revised when a dividend lands; the history is not point-in-time. Names that died often have no Yahoo history, and the delisting return is missing. Sectors are as-of-today. Costs are linear 10 bps per side: no borrow fee, no square-root impact. The rounding beta is computed at a fixed $1M, not along the paper account's equity path. The 21-session arm was not allowed to overturn the verdict.

Price panel fingerprint (adjusted closes): `8f0cd7aa5caf6667…` (642 names, 2210 rows, 2018-01-02 → 2026-10-16). Regenerate with `python scripts/run_trial14_live_config.py --hypothesis T14 --n-trials 14`.
