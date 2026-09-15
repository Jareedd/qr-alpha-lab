#!/usr/bin/env python
"""The demonstration: a run dies halfway, gets diagnosed, and resumes safely.

    python scripts/ingest_demo.py

Offline and deterministic -- a synthetic vendor plays the data source,
including its bad days. Nothing here touches the network or the real store.
The transcript is written to results/ingest_demo_transcript.txt.

Why this exists: "supports incremental updates and backfills" is a claim
anyone can make about a pipeline. The claim worth demonstrating is the one
that is expensive to get right -- **a failure at symbol 3 of 6 leaves the
store consistent, the remaining work named, and the resumed result identical
to a clean single-pass run.**
"""

from __future__ import annotations

import io
import os
import shutil
import sys
import tempfile
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from quantlab import ingest as ing
from quantlab.ingest import PriceStore

SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"]
CAL = pd.bdate_range("2024-01-01", "2024-06-28")
STEP1_END = "2024-03-01"
STEP2_END = "2024-06-28"


def vendor(symbol: str, start, end) -> pd.DataFrame:
    """Deterministic synthetic bars. A bar is a function of (symbol, date)."""
    start, end = pd.Timestamp(start), pd.Timestamp(end or CAL[-1])
    dates = CAL[(CAL >= start) & (CAL <= end)]
    if len(dates) == 0:
        return pd.DataFrame(columns=["close", "volume"],
                            index=pd.DatetimeIndex([], name="date"))
    off = np.asarray(CAL.get_indexer(dates), dtype=float)
    base = 100 + (sum(map(ord, symbol)) % 50)
    return pd.DataFrame({"close": base + off * 0.1, "volume": 1e6 + off},
                        index=pd.DatetimeIndex(dates, name="date"))


def poisoned_vendor(symbol: str, start, end) -> pd.DataFrame:
    """The same vendor, having a bad day for one symbol."""
    df = vendor(symbol, start, end)
    if symbol == "META" and not df.empty:
        df = df.copy()
        df.iloc[3, df.columns.get_loc("close")] = -1.0     # impossible price
    return df


def rule(title: str) -> None:
    print("\n" + "=" * 72)
    print(f" {title}")
    print("=" * 72)


def run(root: str) -> int:
    store = PriceStore(root)
    failures = []

    # ---------------------------------------------------------------- 1
    rule("1. First load: six symbols, 2024-01-01 -> 2024-03-01")
    r = ing.ingest(store, SYMBOLS, start="2024-01-01", end=STEP1_END, fetcher=vendor)
    print(r.render())
    print(f"  store now holds {len(store.symbols())} symbols, "
          f"{sum(len(store.read(s)) for s in store.symbols())} rows")
    if not r.ok:
        failures.append("initial load should have succeeded")

    # ---------------------------------------------------------------- 2
    rule("2. Incremental update to 2024-06-28: only the gap is fetched")
    windows = []

    def watched(symbol, start, end):
        windows.append((symbol, str(start.date())))
        return vendor(symbol, start, end)

    r = ing.ingest(store, SYMBOLS, start="2024-01-01", end=STEP2_END, fetcher=watched)
    print(r.render())
    print(f"  every fetch started at the watermark, not at 2024-01-01: "
          f"{sorted({w[1] for w in windows})}")
    added = {o.symbol: o.rows_added for o in r.by_status('ok')}
    print(f"  rows added per symbol: {added}")
    if sorted({w[1] for w in windows}) != [STEP1_END]:
        failures.append("incremental run did not start from the watermark")

    # ---------------------------------------------------------------- 3
    rule("3. Rerun the same window: safe, and a no-op")
    before = {s: store.read(s).copy() for s in store.symbols()}
    hashes_before = {s: m["content_hash"]
                     for s, m in store.load_manifest()["symbols"].items()}
    ing.ingest(store, SYMBOLS, start="2024-01-01", end=STEP2_END, mode="full",
               fetcher=vendor)
    hashes_after = {s: m["content_hash"]
                    for s, m in store.load_manifest()["symbols"].items()}
    unchanged = all(store.read(s).equals(before[s]) for s in before)
    print(f"  every partition byte-identical after a full rerun: {unchanged}")
    print(f"  content hashes unchanged: {hashes_before == hashes_after}")
    if not unchanged or hashes_before != hashes_after:
        failures.append("rerun was not idempotent")

    # ---------------------------------------------------------------- 4
    rule("4. Backfill: extend history EARLIER without disturbing what exists")
    backfill_cal = pd.bdate_range("2023-06-01", "2024-06-28")

    def deeper_vendor(symbol, start, end):
        start, end = pd.Timestamp(start), pd.Timestamp(end or backfill_cal[-1])
        dates = backfill_cal[(backfill_cal >= start) & (backfill_cal <= end)]
        off = np.asarray(backfill_cal.get_indexer(dates), dtype=float)
        base = 100 + (sum(map(ord, symbol)) % 50)
        return pd.DataFrame({"close": base + off * 0.05, "volume": 1e6 + off},
                            index=pd.DatetimeIndex(dates, name="date"))

    pre = store.read("AAPL").copy()
    ing.ingest(store, ["AAPL"], start="2023-06-01", end=STEP2_END,
               mode="backfill", fetcher=deeper_vendor)
    post = store.read("AAPL")
    overlap_intact = post.loc[pre.index].equals(pre)
    print(f"  AAPL span {pre.index.min().date()}..{pre.index.max().date()}"
          f"  ->  {post.index.min().date()}..{post.index.max().date()}")
    print(f"  rows {len(pre)} -> {len(post)}; index still unique: {post.index.is_unique}")
    print(f"  previously-stored rows untouched by the backfill: {overlap_intact}")
    if not overlap_intact or post.index.min() >= pre.index.min():
        failures.append("backfill disturbed existing rows or did not extend history")

    # ---------------------------------------------------------------- 5
    rule("5. Schema validation: a poisoned batch is quarantined, not stored")
    meta_before = store.read("META").copy()
    r = ing.ingest(store, ["META"], start="2024-01-01", end=STEP2_END,
                   mode="full", fetcher=poisoned_vendor)
    print(r.render())
    print(f"  META partition unchanged after the rejection: "
          f"{store.read('META').equals(meta_before)}")
    if not r.by_status("invalid") or not store.read("META").equals(meta_before):
        failures.append("a schema-violating batch was not properly quarantined")

    # ---------------------------------------------------------------- 6
    rule("6. THE FAILURE: the run dies while processing symbol 3 of 6")
    shutil.rmtree(store.data_root)
    os.makedirs(store.data_root, exist_ok=True)
    store.reconcile()

    def evicted(symbol):
        if symbol == "GOOGL":
            raise KeyboardInterrupt("pod evicted mid-run")

    try:
        ing.ingest(store, SYMBOLS, start="2024-01-01", end=STEP2_END,
                   fetcher=vendor, on_symbol=evicted)
    except KeyboardInterrupt as exc:
        print(f"  run died: KeyboardInterrupt({exc})")

    # ---------------------------------------------------------------- 7
    rule("7. DIAGNOSE from the committed record, not from memory")
    audit = store.audit()
    saved = ing.last_report(store)
    pending = ing.resume_set(store)
    print(f"  symbols landed        : {store.symbols()}")
    print(f"  store consistent      : {audit['consistent']}  "
          f"(unindexed={audit['unindexed']}, drift={audit['content_drift']})")
    print(f"  run report on disk    : _runs/run_{saved['run_id']}.json")
    print(f"  completed in that run : "
          f"{[o['symbol'] for o in saved['outcomes']]}")
    print(f"  PENDING (resume plan) : {pending}")
    print("\n  Note: the store is PARTIAL but CONSISTENT. Partitions are written")
    print("  atomically and the manifest is derived from them, so a crash leaves")
    print("  a smaller store -- never a corrupt one.")
    if not audit["consistent"] or pending != ["GOOGL", "AMZN", "META", "NVDA"]:
        failures.append("diagnosis after the crash was wrong")

    # ---------------------------------------------------------------- 8
    rule("8. RESUME exactly the pending work")
    r = ing.ingest(store, pending, start="2024-01-01", end=STEP2_END, fetcher=vendor)
    print(r.render())
    print(f"  store now holds {len(store.symbols())} symbols")

    # ---------------------------------------------------------------- 9
    rule("9. VERIFY: identical to a clean single-pass run")
    with tempfile.TemporaryDirectory() as clean_root:
        clean = PriceStore(os.path.join(clean_root, "store"))
        ing.ingest(clean, SYMBOLS, start="2024-01-01", end=STEP2_END, fetcher=vendor)
        same = all(store.read(s).equals(clean.read(s)) for s in SYMBOLS)
        print(f"  symbols match     : {store.symbols() == clean.symbols()}")
        print(f"  every row matches : {same}")
    print(f"  store consistent  : {store.audit()['consistent']}")
    print(f"  nothing pending   : {ing.resume_set(store) == []}")
    if not same or not store.audit()["consistent"] or ing.resume_set(store):
        failures.append("the resumed store differs from a clean run")

    rule("RESULT")
    if failures:
        for f in failures:
            print(f"  DEMO FAILED: {f}")
        return 1
    print("  Crash at symbol 3 of 6 -> partial but consistent store -> resume plan")
    print("  read off the committed run report -> resumed store byte-identical to a")
    print("  clean single-pass run. Incremental, backfill, validation and rerun")
    print("  safety all demonstrated on the same store.")
    return 0


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = run(os.path.join(tmp, "demo_store"))
        text = buf.getvalue()
    print(text, end="")
    out = os.path.join(REPO, "results", "ingest_demo_transcript.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"\n  Wrote {os.path.relpath(out, REPO)}")
    return code


if __name__ == "__main__":
    sys.exit(main())
