# Live paper-trading monitor — as of 2026-09-17

## Cycle continuity
- cycles logged: **40** (2026-06-10 → latest 2026-08-10)
- prediction logs: **39** of 40 cycles (weights-only cycles predate prediction logging and cannot yield live IC)
- weekdays in window with NO log: **32** — 2026-06-16, 2026-06-19, 2026-07-03, 2026-08-07, 2026-08-11, 2026-08-12, 2026-08-13, 2026-08-14, 2026-08-17, 2026-08-18, 2026-08-19, 2026-08-20, 2026-08-21, 2026-08-24, 2026-08-25, 2026-08-26, 2026-08-27, 2026-08-28, 2026-08-31, 2026-09-01, 2026-09-02, 2026-09-03, 2026-09-04, 2026-09-07, 2026-09-08, 2026-09-09, 2026-09-10, 2026-09-11, 2026-09-14, 2026-09-15, 2026-09-16, 2026-09-17  *(NYSE holidays are not modeled and appear here; anything else is a missed cycle and must be explained)*

## Live IC vs backtest IC
- measurable cycles: **39** of 39 logged (a cycle matures 21 trading days after its as-of date)
- live mean rank IC: **-0.0908** (t_NW = -2.13)
- backtest mean rank IC (same config, 2010→2026 OOS): **+0.0225** (t_NW = 1.91)

### Control arm (12-1 momentum baseline, shadow-logged — no orders)
- baseline live mean rank IC: **-0.2342** over 39 matured cycles
- purpose: if the model's live IC sags vs backtest, the baseline's own live-vs-backtest gap separates 'model decayed' from 'period was hostile to everything'

## Data revisions (vendor rewriting the shared past)
- snapshot pairs compared: **38**; latest (2026-08-07 → cycle): 4,055 of 226,245 shared price cells changed (1.7923%), **9 return cells** changed (max |Δreturn| 6.82e-05)
- price-level changes are mostly benign re-adjustments; *return* changes alter features/labels — they are why backtest and live model literally saw different versions of the same past

## Realized book P&L (public-price marks, gross, no costs)
- 67 trading days marked; cumulative -1.47%, ann. vol 17.61%
- cross-check only: fills, costs and shorts-availability live at the broker; the Alpaca equity curve is authoritative

## Standing limitations
- live IC residualizes vs the equal-weight mean of logged names, not the full PIT universe (close, not identical, market proxy)
- yfinance marks are split/dividend-adjusted closes; broker fills will differ
- this monitor is read-only: it never feeds back into the strategy
