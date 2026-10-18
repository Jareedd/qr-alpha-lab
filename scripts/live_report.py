"""One-page Phase 6 monitoring report (roadmap week 11).

    python scripts/live_report.py             # fetch prices, full report
    python scripts/live_report.py --offline   # continuity check only

Reads results/live/, writes results/live/report.md plus an IC chart once
cycles mature. Read-only with respect to the strategy: run it as often as
you like, it cannot contaminate anything.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# The report uses unicode arrows; Windows consoles default to cp1252 and the
# echo-to-stdout would crash AFTER the file is written (report fine, exit 1).
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from quantlab import monitor


def report_cache_dir(now: dt.datetime | None = None, root: str = "data_cache") -> str:
    """Price cache for this report. Not ``data_cache/live_*``.

    From 2026-08-10 this report wrote its parquet into the live cycle's
    directory. The revision monitor then did ``sorted(prices_*.parquet)[-1]``
    and compared whichever digest sorted last — the report's shorter panel,
    not the panel the cycle traded. The report now has its own directory,
    keyed by the New York session date, same clock as the live cache.
    """
    from quantlab.live import ny_session_date

    return os.path.join(root, f"report_{ny_session_date(now).isoformat()}")


def panel_staleness_days(last_cycle: pd.Timestamp, price_index: pd.Index) -> int:
    """Days the price panel lags the last logged cycle (>0 = stale).

    IC maturity, book marks and the live-vs-backtest comparison all truncate
    SILENTLY when the panel is old -- the report must say so on its face
    rather than print healthy-looking zeros.
    """
    if len(price_index) == 0:
        return 10**6
    return int((pd.Timestamp(last_cycle) - pd.Timestamp(price_index.max())).days)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-dir", default="results/live")
    ap.add_argument(
        "--backtest-metrics",
        default="results/metrics_sp500_ridge_both_residlabel.json",
        help="metrics JSON of the backtest config the live run mirrors "
        "(trial #5: residual label, sector+beta neutral)",
    )
    ap.add_argument("--horizon", type=int, default=21)
    ap.add_argument("--offline", action="store_true", help="no price fetch")
    args = ap.parse_args()

    weights_by_date, preds_by_date = monitor.load_live_records(args.live_dir)
    if not weights_by_date:
        sys.exit(f"no live records found in {args.live_dir}")

    today = pd.Timestamp(dt.date.today())
    continuity = monitor.cycle_continuity(sorted(weights_by_date), today)

    comparison = None
    live_ic = pd.Series(dtype=float)
    baseline_ic = None
    book_pnl = pd.Series(dtype=float)
    if not args.offline:
        from quantlab.data import load_prices

        tickers = sorted(
            set().union(
                *(set(w.index) for w in weights_by_date.values()),
                *(set(p.index) for p in preds_by_date.values()),
            )
        )
        # 600 calendar days back: 252d beta warm-up + buffer before first cycle.
        start = (min(weights_by_date) - pd.Timedelta(days=600)).date().isoformat()
        # A "latest"-keyed parquet under the default cache dir is frozen at
        # first download, and the CI cache restores it every night -- this
        # report ran six weeks on a June price panel, saying "0 measurable
        # cycles" while ~17 had matured. Its own per-day directory (not the
        # live cycle's — sharing that folder made the revision monitor
        # compare the wrong panel) keeps the report's prices <= 1 day old.
        cache_dir = report_cache_dir()
        prices = load_prices(tickers, start=start, min_coverage=0.0, cache_dir=cache_dir)
        live_ic = monitor.realized_live_ic(
            preds_by_date, prices, horizon=args.horizon
        )
        book_pnl = monitor.realized_book_returns(weights_by_date, prices)
        with open(args.backtest_metrics) as f:
            comparison = monitor.live_vs_backtest(
                live_ic, json.load(f), horizon=args.horizon
            )
        # Control arm: cycles logged after the baseline column was added.
        with_baseline = {
            d: p for d, p in preds_by_date.items() if "baseline_mom_12_1" in p.columns
        }
        if with_baseline:
            baseline_ic = monitor.realized_live_ic(
                with_baseline, prices, horizon=args.horizon, col="baseline_mom_12_1"
            )

    # Data-revision records written by the live cycle (vendor drift log).
    revisions = []
    for fn in sorted(os.listdir(args.live_dir)):
        if fn.startswith("revisions_") and fn.endswith(".json"):
            with open(os.path.join(args.live_dir, fn)) as f:
                revisions.append(json.load(f))

    md = monitor.render_report(
        asof=str(today.date()),
        continuity=continuity,
        n_weights_logged=len(weights_by_date),
        n_preds_logged=len(preds_by_date),
        comparison=comparison,
        live_ic=live_ic,
        book_pnl=book_pnl,
        horizon=args.horizon,
        baseline_live_ic=baseline_ic,
        revisions=revisions,
    )
    if not args.offline:
        stale = panel_staleness_days(max(weights_by_date), prices.index)
        if stale > 0:
            md += (
                f"\n> **STALE PRICE PANEL:** prices end "
                f"{prices.index.max().date()}, {stale} days before the last "
                f"logged cycle — every maturity-dependent number above is "
                f"truncated. Fix the cache before reading this report.\n"
            )

    out_md = os.path.join(args.live_dir, "report.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    print(f"[report] written to {out_md}")

    if len(live_ic):
        fig, ax = plt.subplots(figsize=(10, 4))
        live_ic.plot(ax=ax, marker="o", lw=1, label="live per-cycle rank IC")
        if comparison is not None:
            ax.axhline(
                comparison["backtest_mean_ic"], ls="--", c="gray",
                label=f"backtest mean IC {comparison['backtest_mean_ic']:+.4f}",
            )
        ax.axhline(0, c="k", lw=0.5)
        ax.set_title("Live IC vs backtest IC (matured cycles only)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.live_dir, "live_ic.png"), dpi=120)
        print(f"[report] IC chart: {os.path.join(args.live_dir, 'live_ic.png')}")


if __name__ == "__main__":
    main()
