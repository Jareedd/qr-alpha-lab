"""Data-revision report: how much did the vendor rewrite the shared past?

    python scripts/data_revisions_report.py
    python scripts/data_revisions_report.py --seed 2026-06-10=data_cache/prices_c876e14c18_2009-01-01_latest_0.0.parquet

Compares every consecutive pair of live-cycle price snapshots
(data_cache/live_YYYY-MM-DD/prices_*.parquet) and writes
results/data_revisions.csv + a summary to stdout. ``--seed`` registers an
extra dated snapshot (e.g. the research download that predates the live
caches) so the series starts as early as possible.

Read-only with respect to data and strategy (research law #7: we measure the
vendor's rewrites, we never make our own).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import revisions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", default="data_cache")
    ap.add_argument("--out", default="results")
    ap.add_argument(
        "--seed",
        action="append",
        default=[],
        metavar="TAG=PATH",
        help="extra dated snapshot, e.g. 2026-06-10=data_cache/prices_...parquet "
        "(the download date is the tag; repeatable)",
    )
    args = ap.parse_args()

    snaps = revisions.list_snapshots(args.cache_root)
    for seed in args.seed:
        tag, _, path = seed.partition("=")
        if not path or not os.path.exists(path):
            sys.exit(f"--seed {seed}: file not found")
        snaps.setdefault(tag, path)  # a live snapshot for the same day wins

    if len(snaps) < 2:
        sys.exit(
            f"need >= 2 dated snapshots to compare, found {len(snaps)} "
            f"({', '.join(sorted(snaps)) or 'none'}) -- snapshots accumulate "
            "one per live cycle"
        )

    table = revisions.revision_table(snaps)
    os.makedirs(args.out, exist_ok=True)
    out_csv = os.path.join(args.out, "data_revisions.csv")
    table.to_csv(out_csv, index=False)

    print(f"[revisions] {len(table)} snapshot pair(s) compared -> {out_csv}")
    for _, r in table.iterrows():
        print(
            f"  {r['from']} -> {r['to']}: "
            f"{r['n_price_cells_changed']:,}/{r['n_cells_compared']:,} price cells "
            f"changed ({r['frac_price_cells_changed']:.4%}); "
            f"{r['n_return_cells_changed']:,} RETURN cells changed "
            f"(max |dR| {r['max_abs_return_change']:.2e}); "
            f"{r['n_tickers_affected']} tickers touched"
            + ("" if r["is_consecutive_cycle"] else
               f"  [!] NON-CONSECUTIVE: spans {r['gap_weekdays']} weekdays "
               "-- not a one-day revision measurement")
        )
    print(
        "  (price-level changes are usually whole-history re-adjustments; "
        "return changes are the ones that alter features/labels)"
    )


if __name__ == "__main__":
    main()
