"""Daily live paper-trading entrypoint (Phase 6).

    python scripts/live_trade.py                 # predict + log; orders PAUSED
    python scripts/live_trade.py --dry-run       # same, marked dry-run not paused
    python scripts/live_trade.py --submit-orders # re-enable paper order submission

Run after US market close (the workflow cron is 22:30 UTC weekdays).
Predictions are logged to results/live/ BEFORE any order is sent -- the
prediction log is the artifact that later yields live IC vs backtest IC.

Order submission is paused by default (2026-10-18). A paused cycle still
downloads data, logs predictions and weights, and writes a summary with
``orders_status=paused``. Re-enable with ``--submit-orders`` or
``QR_SUBMIT_ORDERS=1``. The Alpaca client still refuses any non-paper
endpoint.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab.live import SUBMIT_ORDERS_ENV, run_daily, submission_inputs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ridge", choices=["ridge", "gbr", "mlp"])
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="log predictions, send no orders (recorded as dry_run, not as the operational pause)",
    )
    ap.add_argument(
        "--submit-orders",
        action="store_true",
        help="re-enable paper order submission for this run "
        f"(sets {SUBMIT_ORDERS_ENV}=1). Default is paused.",
    )
    ap.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="re-run a failed cycle: replace the (normally write-once) "
        "per-date prediction/weights logs",
    )
    ap.add_argument("--out", default="results/live")
    args = ap.parse_args()

    try:
        submit, override = submission_inputs(args.dry_run, args.submit_orders)
    except ValueError as exc:
        sys.exit(str(exc))
    if override is not None:
        os.environ[SUBMIT_ORDERS_ENV] = override

    summary = run_daily(
        model_name=args.model,
        out_dir=args.out,
        submit=submit,
        allow_overwrite=args.allow_overwrite,
    )
    for k, v in summary.items():
        print(f"  {k}: {v}")
    if summary.get("n_failed", 0) > 0:
        print("  (some orders failed -- typically hard-to-borrow shorts; logged in summary)")


if __name__ == "__main__":
    main()
