"""Phase 6 monitoring: is the live experiment intact, and does live IC
match backtest IC?

The live logs in ``results/live/`` are append-only records (committed by CI
after each cycle). This module turns them into the three numbers that matter:

- **continuity** -- which expected trading days have no logged cycle. A hole
  in the record must be caught within days and explained in the write-up,
  not discovered when the experiment ends.
- **live IC** -- per-cycle Spearman rank correlation between logged
  predictions and the realized residualized label, computed with the same
  label machinery as the backtest (``features.build_labels``) so the
  live-vs-backtest comparison is apples-to-apples. This is the project's
  ultimate out-of-sample test.
- **realized book P&L** -- mark-to-market of the logged weights from public
  prices: an independent cross-check on the broker's equity curve, not a
  performance claim (costs and fills live at the broker).

Monitoring is read-only by design: nothing here feeds back into the
strategy, so it cannot become a leak vector.

Assumptions stated:
- A cycle's predictions mature ``horizon`` TRADING days after its as-of
  date; immature cycles are omitted from the IC series (the measurable-vs-
  logged count is itself a monitoring number).
- The label used for live IC residualizes against the equal-weight mean of
  the logged names (all index members at their as-of date), not the full
  PIT universe -- a close but not identical market proxy; stated in the
  report's limitations.
- NYSE holidays are not modeled (no new dependency): holidays show up as
  "missing" weekdays and are accepted by eye. The check exists to catch
  silent multi-day gaps, not to be a perfect calendar.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd

from quantlab import features, metrics

_PRED_RE = re.compile(r"predictions_(\d{4}-\d{2}-\d{2})\.csv")
_WEIGHTS_RE = re.compile(r"weights_(\d{4}-\d{2}-\d{2})\.csv")
_SUMMARY_RE = re.compile(r"summary_(\d{4}-\d{2}-\d{2})\.json")


def load_live_records(
    live_dir: str,
) -> tuple[dict[pd.Timestamp, pd.Series], dict[pd.Timestamp, pd.DataFrame]]:
    """Read all logged cycles: (weights_by_date, predictions_by_date).

    Weights logs exist for every cycle; prediction logs only for cycles run
    after prediction logging was added -- the report states both counts.
    """
    weights: dict[pd.Timestamp, pd.Series] = {}
    preds: dict[pd.Timestamp, pd.DataFrame] = {}
    for fn in sorted(os.listdir(live_dir)):
        if m := _WEIGHTS_RE.fullmatch(fn):
            df = pd.read_csv(os.path.join(live_dir, fn), index_col="ticker")
            weights[pd.Timestamp(m.group(1))] = df["weight"].astype(float)
        elif m := _PRED_RE.fullmatch(fn):
            df = pd.read_csv(os.path.join(live_dir, fn), index_col="ticker")
            preds[pd.Timestamp(m.group(1))] = df.astype(float)
    return weights, preds


def cycle_continuity(
    logged_dates: list[pd.Timestamp], through: pd.Timestamp
) -> pd.DataFrame:
    """Every weekday from the first logged cycle through ``through``, with a
    ``logged`` flag. Missing weekdays = no log (holiday or a real gap)."""
    logged = {pd.Timestamp(d).normalize() for d in logged_dates}
    days = pd.bdate_range(min(logged), pd.Timestamp(through).normalize())
    return pd.DataFrame({"date": days, "logged": [d in logged for d in days]})


def realized_live_ic(
    preds_by_date: dict[pd.Timestamp, pd.DataFrame],
    prices: pd.DataFrame,
    horizon: int = 21,
    col: str = "pred_raw",
    min_names: int = 30,
) -> pd.Series:
    """Per-cycle Spearman IC of logged predictions vs the realized
    residualized label (same definition as the backtest's, via
    ``features.build_labels``). Cycles without ``horizon`` further trading
    days of prices -- or with < ``min_names`` overlapping names -- are
    omitted, not NaN-padded."""
    labels = features.build_labels(prices, horizon=horizon, residualize=True)
    out: dict[pd.Timestamp, float] = {}
    for asof, p in sorted(preds_by_date.items()):
        if asof not in prices.index:
            continue
        if prices.index.get_loc(asof) + horizon > len(prices.index) - 1:
            continue  # label not yet realized
        pair = pd.concat(
            [p[col], labels.loc[asof]], axis=1, keys=["pred", "label"]
        ).dropna()
        if len(pair) < min_names:
            continue
        # Spearman as Pearson-of-ranks: identical tie handling to
        # models.information_coefficient, so the numbers are comparable.
        out[asof] = float(pair["pred"].rank().corr(pair["label"].rank()))
    return pd.Series(out, dtype=float).sort_index()


def live_vs_backtest(
    live_ic: pd.Series, backtest_metrics: dict, horizon: int = 21
) -> dict:
    """The headline comparison. The NW t-stat needs > lags+2 matured cycles
    and is NaN before then -- early numbers are descriptive, not evidence."""
    return {
        "n_cycles_measurable": int(live_ic.notna().sum()),
        "live_mean_ic": float(live_ic.mean()) if len(live_ic) else float("nan"),
        "live_ic_tstat_nw": float(metrics.newey_west_tstat(live_ic, lags=horizon)),
        "backtest_mean_ic": float(backtest_metrics["mean_rank_ic"]),
        "backtest_ic_tstat_nw": float(backtest_metrics["ic_tstat_newey_west"]),
    }


def load_orders_paused_dates(live_dir: str) -> set[pd.Timestamp]:
    """As-of dates whose summary says orders were not sent.

    A paused cycle still logs a target book. That target is not what the
    broker holds, so book P&L must keep the previous submitted book
    instead of marking the unsent weights.
    """
    paused: set[pd.Timestamp] = set()
    if not os.path.isdir(live_dir):
        return paused
    for fn in os.listdir(live_dir):
        m = _SUMMARY_RE.fullmatch(fn)
        if not m:
            continue
        try:
            with open(os.path.join(live_dir, fn)) as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("orders_status") == "paused" or payload.get("orders_paused") is True:
            paused.add(pd.Timestamp(m.group(1)))
    return paused


def realized_book_returns(
    weights_by_date: dict[pd.Timestamp, pd.Series],
    prices: pd.DataFrame,
    hold_dates: set[pd.Timestamp] | None = None,
) -> pd.Series:
    """Daily mark-to-market of the logged books, gross, from public prices.

    The book logged at t earns from t+1 until the next logged book takes
    over (the backtest convention). A held name with a missing price
    contributes nothing that day (limitation: a delisting between cycles is
    under-counted here; the broker record is authoritative).

    ``hold_dates`` are cycles that logged a target but did not send it
    (``orders_status=paused``). Those weights are skipped, so the previous
    submitted book stays in force. Predictions on those dates are a
    separate record and are not dropped from the IC series.
    """
    held = {pd.Timestamp(d) for d in (hold_dates or set())}
    rets = prices.pct_change(fill_method=None)
    book = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    for d, w in weights_by_date.items():
        if d not in book.index or d in held:
            continue
        cols = w.index.intersection(book.columns)
        book.loc[d, :] = 0.0
        book.loc[d, cols] = w[cols]
    book = book.ffill()
    pnl = (book.shift(1) * rets).sum(axis=1, min_count=1)
    return pnl.dropna()


def render_report(
    asof: str,
    continuity: pd.DataFrame,
    n_weights_logged: int,
    n_preds_logged: int,
    comparison: dict | None,
    live_ic: pd.Series,
    book_pnl: pd.Series,
    horizon: int = 21,
    baseline_live_ic: pd.Series | None = None,
    revisions: list[dict] | None = None,
    n_orders_paused: int = 0,
) -> str:
    """One-page markdown report. Purely descriptive: it never says the live
    experiment 'works', only what was measured and what cannot be measured
    yet."""
    gaps = continuity[~continuity["logged"]]["date"]
    lines = [
        f"# Live paper-trading monitor — as of {asof}",
        "",
        "## Cycle continuity",
        f"- cycles logged: **{n_weights_logged}** "
        f"({continuity['date'].iloc[0].date()} → latest "
        f"{max(d.date() for d in continuity[continuity['logged']]['date'])})",
        f"- prediction logs: **{n_preds_logged}** of {n_weights_logged} cycles "
        "(weights-only cycles predate prediction logging and cannot yield live IC)",
        f"- weekdays in window with NO log: **{len(gaps)}**"
        + (
            " — " + ", ".join(str(d.date()) for d in gaps)
            + "  *(NYSE holidays are not modeled and appear here; anything "
            "else is a missed cycle and must be explained)*"
            if len(gaps)
            else " — record is gap-free"
        ),
        "",
        "## Live IC vs backtest IC",
    ]
    if comparison is None:
        lines.append("- *(offline run: prices not fetched, IC not computed)*")
    else:
        n_meas = comparison["n_cycles_measurable"]
        lines += [
            f"- measurable cycles: **{n_meas}** of {n_preds_logged} logged "
            f"(a cycle matures {horizon} trading days after its as-of date)",
            f"- live mean rank IC: **{comparison['live_mean_ic']:+.4f}** "
            f"(t_NW = {comparison['live_ic_tstat_nw']:.2f})"
            if n_meas
            else "- live mean rank IC: *not yet measurable*",
            f"- backtest mean rank IC (same config, 2010→2026 OOS): "
            f"**{comparison['backtest_mean_ic']:+.4f}** "
            f"(t_NW = {comparison['backtest_ic_tstat_nw']:.2f})",
        ]
        if n_meas < horizon + 3:
            lines.append(
                f"- **do not interpret yet**: t_NW needs > {horizon + 2} matured "
                "cycles; early ICs are single noisy draws"
            )
    if baseline_live_ic is not None:
        n_base = int(baseline_live_ic.notna().sum())
        lines += [
            "",
            "### Control arm (12-1 momentum baseline, shadow-logged — no orders)",
            (
                f"- baseline live mean rank IC: **{baseline_live_ic.mean():+.4f}** "
                f"over {n_base} matured cycles"
                if n_base
                else "- baseline live IC: *not yet measurable*"
            ),
            "- purpose: if the model's live IC sags vs backtest, the baseline's "
            "own live-vs-backtest gap separates 'model decayed' from 'period "
            "was hostile to everything'",
        ]
    if revisions:
        latest = revisions[-1]
        lines += [
            "",
            "## Data revisions (vendor rewriting the shared past)",
            f"- snapshot pairs compared: **{len(revisions)}**; latest "
            f"({latest.get('compared_to', '?')} → cycle): "
            f"{latest['n_price_cells_changed']:,} of "
            f"{latest['n_cells_compared']:,} shared price cells changed "
            f"({latest['frac_price_cells_changed']:.4%}), "
            f"**{latest['n_return_cells_changed']:,} return cells** changed "
            f"(max |Δreturn| {latest['max_abs_return_change']:.2e})",
            "- price-level changes are mostly benign re-adjustments; *return* "
            "changes alter features/labels — they are why backtest and live "
            "model literally saw different versions of the same past",
        ]
    lines += ["", "## Realized book P&L (public-price marks, gross, no costs)"]
    if len(book_pnl):
        cum = float((1 + book_pnl).prod() - 1)
        vol = float(book_pnl.std() * np.sqrt(252)) if len(book_pnl) > 1 else float("nan")
        lines += [
            f"- {len(book_pnl)} trading days marked; cumulative {cum:+.2%}, "
            f"ann. vol {vol:.2%}",
            "- cross-check only: fills, costs and shorts-availability live at "
            "the broker; the Alpaca equity curve is authoritative",
        ]
        if n_orders_paused:
            lines.append(
                f"- **orders paused on {n_orders_paused} logged cycle(s):** "
                "those weight files are the unsent target. Book P&L keeps "
                "the last submitted book, so a pause is not a rebalance."
            )
    else:
        lines.append("- no marked days yet (first book earns from its t+1 open)")
    lines += [
        "",
        "## Standing limitations",
        "- live IC residualizes vs the equal-weight mean of logged names, "
        "not the full PIT universe (close, not identical, market proxy)",
        "- yfinance marks are split/dividend-adjusted closes; broker fills "
        "will differ",
        "- this monitor is read-only: it never feeds back into the strategy",
    ]
    return "\n".join(lines) + "\n"
