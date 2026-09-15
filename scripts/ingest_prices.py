#!/usr/bin/env python
"""Operate the price store: ingest, backfill, inspect, resume, audit.

    python scripts/ingest_prices.py --symbols AAPL MSFT --start 2015-01-01
    python scripts/ingest_prices.py --symbols AAPL --backfill --start 2005-01-01
    python scripts/ingest_prices.py --status
    python scripts/ingest_prices.py --resume
    python scripts/ingest_prices.py --audit

Exit code 0 only when every requested symbol landed. A partial run exits 1
with a report at ``<root>/_runs/`` naming exactly what failed and what is
still pending -- so the next run (or the next person) starts from a record
rather than from a guess.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from quantlab import ingest as ing
from quantlab.ingest import PriceStore


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join(REPO, "data_store"),
                    help="store root (default: data_store/)")
    ap.add_argument("--symbols", nargs="*", default=None,
                    help="symbols to ingest; default = whatever the store holds")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default=None, help="default: latest available")
    ap.add_argument("--backfill", action="store_true",
                    help="extend history EARLIER than the watermark")
    ap.add_argument("--full", action="store_true",
                    help="refetch the whole window (repairs a revised source)")
    ap.add_argument("--resume", action="store_true",
                    help="ingest exactly what the last run left unfinished")
    ap.add_argument("--status", action="store_true", help="print store state and exit")
    ap.add_argument("--audit", action="store_true",
                    help="check manifest against partitions and exit")
    ap.add_argument("--reconcile", action="store_true",
                    help="rebuild the manifest from partitions and exit")
    ap.add_argument("--fail-fast", action="store_true")
    args = ap.parse_args()

    store = PriceStore(args.root)

    if args.status:
        manifest = store.load_manifest().get("symbols", {})
        print(f"store: {args.root}")
        print(f"symbols: {len(store.symbols())}")
        for symbol in store.symbols():
            m = manifest.get(symbol, {})
            flag = "" if m.get("valid", True) else "  INVALID"
            print(f"  {symbol:<8} {m.get('rows', '?'):>6} rows  "
                  f"{m.get('first_date', '?')} .. {m.get('last_date', '?')}{flag}")
        report = ing.last_report(store)
        if report:
            print(f"\nlast run {report['run_id']} ({report['mode']}): "
                  f"{report['summary']}")
            if report["pending_after_run"]:
                print(f"  PENDING: {', '.join(report['pending_after_run'])}")
        return 0

    if args.audit or args.reconcile:
        if args.reconcile:
            store.reconcile()
            print("manifest rebuilt from partitions")
        audit = store.audit()
        print(json.dumps(audit, indent=2))
        return 0 if audit["consistent"] else 1

    if args.resume:
        symbols = ing.resume_set(store)
        if not symbols:
            print("nothing pending -- the last run finished cleanly")
            return 0
        print(f"resuming {len(symbols)} symbol(s): {', '.join(symbols)}")
    else:
        symbols = args.symbols or store.symbols()
        if not symbols:
            return int(bool(sys.stderr.write(
                "no symbols given and the store is empty; pass --symbols\n")))

    mode = "backfill" if args.backfill else ("full" if args.full else "incremental")
    report = ing.ingest(store, symbols, start=args.start, end=args.end,
                        mode=mode, fail_fast=args.fail_fast)
    print(report.render())
    if not report.ok:
        print("\nre-run with --resume to retry exactly the unfinished work")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
