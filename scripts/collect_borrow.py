"""H7 borrow-snapshot collector (collection-only; zero trials).

    python scripts/collect_borrow.py            # snapshot -> results/live/borrow_{date}.json
    python scripts/collect_borrow.py --allow-overwrite   # deliberate re-run

Runs as a NON-FATAL step in live.yml after the trading cycle (the
revisions.py operational pattern: a collection bug must never cost a
cycle). The snapshot universe is the live experiment's own scored
cross-section — the tickers in the latest predictions/weights logs — so
no extra network or membership scrape is needed.

Write-once like every live artifact: an unbackfillable dataset is only
evidence if the records cannot be silently regenerated.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from quantlab import borrow
from quantlab.live import assert_write_once


def snapshot_universe(live_dir: str) -> tuple[list[str], str | None]:
    """(tickers, as-of date) from the most recent predictions and weights logs.

    The as-of date is returned, and recorded in the snapshot, because this
    collector deliberately runs even when the day's trading cycle failed
    (live.yml, `if: always()`) -- so the universe can legitimately be older
    than the snapshot. A borrow record that does not say which cross-section
    it covers is not evidence; stating it costs one field.
    """
    names: set[str] = set()
    asof: str | None = None
    for pattern in ("predictions_*.csv", "weights_*.csv"):
        files = sorted(glob.glob(os.path.join(live_dir, pattern)))
        if files:
            names |= set(pd.read_csv(files[-1], index_col="ticker").index.astype(str))
            stamp = os.path.basename(files[-1]).split("_")[-1].removesuffix(".csv")
            asof = max(asof, stamp) if asof else stamp
    return sorted(names), asof


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-dir", default="results/live")
    ap.add_argument("--allow-overwrite", action="store_true")
    args = ap.parse_args()

    universe, universe_asof = snapshot_universe(args.live_dir)
    if not universe:
        sys.exit("no live records found -- nothing to snapshot against")

    raw = borrow.fetch_ibkr_short_file()
    file_stamp, frame = borrow.parse_ibkr_short_file(raw)
    if frame.empty:
        sys.exit("IBKR file parsed to zero rows -- format drift? raw head: " + raw[:200])

    snap = borrow.build_snapshot(file_stamp, frame, universe, universe_asof=universe_asof)
    asof = (file_stamp.split(" ")[0] if file_stamp
            else str(pd.Timestamp.utcnow().date()))
    out_path = os.path.join(args.live_dir, f"borrow_{asof}.json")
    assert_write_once([out_path], allow_overwrite=args.allow_overwrite)
    with open(out_path, "w") as f:
        json.dump(snap, f, indent=2)
    print(
        f"[borrow] {out_path}: {snap['universe_covered']}/{snap['universe_size']} "
        f"universe names found in {snap['n_symbols_in_file']:,}-symbol file "
        f"(stamp {snap['file_stamp']}, universe as of {snap['universe_asof']}); "
        f"fee p50/p90/p99 {snap['fee_rate_percentiles_full_file']}"
    )


if __name__ == "__main__":
    main()
