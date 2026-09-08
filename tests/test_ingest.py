"""Ingest store: incremental updates, schema validation, safe reruns, resume.

No network. A deterministic synthetic fetcher plays the vendor, including its
bad days -- 404s, empty responses, duplicated bars, negative prices, and a
process that dies halfway through a run.

The claim these tests establish is operational, not statistical: **a run that
fails partway leaves the store in a state a later run can safely continue
from, and no rerun ever duplicates or corrupts a row.**
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import ingest
from quantlab.ingest import (
    IngestReport, PriceStore, SchemaViolation, merge_partition,
    validate_partition,
)

CAL = pd.bdate_range("2024-01-01", "2024-06-28")


def make_bars(symbol: str, start, end=None) -> pd.DataFrame:
    """A deterministic price series: symbol-specific, date-addressable.

    Values are a pure function of (symbol, date), so a bar fetched today and
    the same bar fetched next week are identical -- which is what makes
    'a rerun changes nothing' a testable claim rather than a hope.
    """
    start = pd.Timestamp(start)
    end = pd.Timestamp(end) if end is not None else CAL[-1]
    dates = CAL[(CAL >= start) & (CAL <= end)]
    if len(dates) == 0:
        return pd.DataFrame(columns=["close", "volume"],
                            index=pd.DatetimeIndex([], name="date"))
    base = 100 + (sum(map(ord, symbol)) % 50)
    # Offset by position in the CALENDAR, not in the returned slice: the bar
    # for a given date must be identical whether it arrives in a full pull or
    # in a one-day incremental top-up, or "a rerun changes nothing" is
    # untestable.
    offsets = np.asarray(CAL.get_indexer(dates), dtype=float)
    close = base + offsets * 0.1
    out = pd.DataFrame({"close": close, "volume": 1_000_000.0 + offsets},
                       index=pd.DatetimeIndex(dates, name="date"))
    return out


def good_fetcher(symbol, start, end):
    return make_bars(symbol, start, end)


@pytest.fixture
def store(tmp_path):
    return PriceStore(str(tmp_path / "store"))


# --------------------------------------------------------------------------
# Schema validation
# --------------------------------------------------------------------------
def test_validate_accepts_a_well_formed_partition():
    assert validate_partition(make_bars("AAPL", "2024-01-01"), "AAPL") == []


@pytest.mark.parametrize("mutate, expected", [
    (lambda df: df.drop(columns=["volume"]), "missing required column"),
    (lambda df: df.set_index(pd.RangeIndex(len(df))), "DatetimeIndex"),
    (lambda df: df.set_index(df.index.tz_localize("UTC")), "timezone-naive"),
    (lambda df: pd.concat([df, df.iloc[[0]]]).sort_index(), "duplicate timestamp"),
    (lambda df: df.iloc[::-1], "sorted ascending"),
    (lambda df: df.assign(close=-1.0), "non-positive"),
    (lambda df: df.assign(close=np.nan), "NaN"),
    (lambda df: df.assign(volume=-5.0), "negative"),
])
def test_validate_rejects_each_way_a_partition_can_be_wrong(mutate, expected):
    df = make_bars("AAPL", "2024-01-01", "2024-02-01")
    problems = validate_partition(mutate(df), "AAPL")
    assert any(expected in p for p in problems), problems


def test_store_write_refuses_an_invalid_partition(store):
    bad = make_bars("AAPL", "2024-01-01", "2024-02-01").assign(close=-1.0)
    with pytest.raises(SchemaViolation, match="non-positive"):
        store.write("AAPL", bad)
    # The refusal must be total: no partition, no empty directory with a
    # half-written file inside it.
    assert not os.path.exists(store.partition_path("AAPL"))


def test_an_invalid_batch_leaves_previously_good_data_untouched(store):
    ingest.ingest(store, ["AAPL"], start="2024-01-01", end="2024-02-01",
                  fetcher=good_fetcher)
    before = store.read("AAPL")

    def poisoned(symbol, start, end):
        return make_bars(symbol, start, end).assign(close=-1.0)

    report = ingest.ingest(store, ["AAPL"], start="2024-01-01", end="2024-06-01",
                           fetcher=poisoned, mode="full")
    assert report.by_status("invalid") and not report.ok
    pd.testing.assert_frame_equal(store.read("AAPL"), before)


# --------------------------------------------------------------------------
# Incremental updates
# --------------------------------------------------------------------------
def test_incremental_run_fetches_only_from_the_watermark(store):
    ingest.ingest(store, ["AAPL"], start="2024-01-01", end="2024-03-01",
                  fetcher=good_fetcher)
    assert store.watermark("AAPL") == pd.Timestamp("2024-03-01")

    seen = {}

    def spy(symbol, start, end):
        seen[symbol] = start
        return make_bars(symbol, start, end)

    ingest.ingest(store, ["AAPL"], start="2024-01-01", end="2024-05-01", fetcher=spy)
    # Starts at the watermark, not at the original start: the point of an
    # incremental pipeline is that night 200 costs the same as night 2.
    assert seen["AAPL"] == pd.Timestamp("2024-03-01")
    assert store.watermark("AAPL") == pd.Timestamp("2024-05-01")


def test_incremental_extends_without_gaps_or_duplicates(store):
    ingest.ingest(store, ["AAPL"], start="2024-01-01", end="2024-03-01",
                  fetcher=good_fetcher)
    ingest.ingest(store, ["AAPL"], start="2024-01-01", end="2024-06-01",
                  fetcher=good_fetcher)
    stored = store.read("AAPL")
    expected = make_bars("AAPL", "2024-01-01", "2024-06-01")
    assert stored.index.is_unique
    assert list(stored.index) == list(expected.index)      # no gap at the seam


def test_a_current_symbol_refetches_only_the_watermark_day_and_changes_nothing(store):
    """A same-day rerun is cheap, and deliberately not a no-op at the seam.

    The watermark day itself IS refetched: a bar pulled mid-session is
    partial, and the corrected close arrives later. Overlapping by one day is
    free (the merge deduplicates); a gap would be permanent.
    """
    ingest.ingest(store, ["AAPL"], start="2024-01-01", end="2024-03-01",
                  fetcher=good_fetcher)
    before = store.read("AAPL").copy()
    windows = []

    def spy(symbol, start, end):
        windows.append((start, end))
        return make_bars(symbol, start, end)

    ingest.ingest(store, ["AAPL"], start="2024-01-01", end="2024-03-01", fetcher=spy)
    assert windows == [(pd.Timestamp("2024-03-01"), pd.Timestamp("2024-03-01"))]
    pd.testing.assert_frame_equal(store.read("AAPL"), before)


def test_a_symbol_past_the_requested_end_is_skipped_without_a_fetch(store):
    ingest.ingest(store, ["AAPL"], start="2024-01-01", end="2024-04-01",
                  fetcher=good_fetcher)
    calls = []

    def spy(symbol, start, end):
        calls.append(symbol)
        return make_bars(symbol, start, end)

    report = ingest.ingest(store, ["AAPL"], start="2024-01-01", end="2024-02-01",
                           fetcher=spy)
    assert calls == []
    assert report.by_status("skipped") and report.ok


# --------------------------------------------------------------------------
# Safe reruns / idempotence
# --------------------------------------------------------------------------
def test_rerunning_the_same_window_changes_nothing(store):
    ingest.ingest(store, ["AAPL", "MSFT"], start="2024-01-01", end="2024-04-01",
                  fetcher=good_fetcher)
    first = {s: store.read(s).copy() for s in ("AAPL", "MSFT")}
    hash_before = store.load_manifest()["symbols"]["AAPL"]["content_hash"]

    for _ in range(3):
        ingest.ingest(store, ["AAPL", "MSFT"], start="2024-01-01", end="2024-04-01",
                      mode="full", fetcher=good_fetcher)

    for s in ("AAPL", "MSFT"):
        pd.testing.assert_frame_equal(store.read(s), first[s])
    assert store.load_manifest()["symbols"]["AAPL"]["content_hash"] == hash_before


def test_merge_is_idempotent_and_incoming_wins_on_collision():
    a = make_bars("AAPL", "2024-01-01", "2024-02-01")
    b = make_bars("AAPL", "2024-01-15", "2024-03-01").assign(close=999.0)
    merged = merge_partition(a, b)
    assert merged.index.is_unique and merged.index.is_monotonic_increasing
    assert merged.loc[pd.Timestamp("2024-01-15"), "close"] == 999.0   # incoming wins
    assert merged.loc[pd.Timestamp("2024-01-02"), "close"] == a.loc[
        pd.Timestamp("2024-01-02"), "close"]                          # untouched
    pd.testing.assert_frame_equal(merge_partition(merged, b), merged)  # idempotent


def test_full_mode_repairs_a_revised_history_without_growing_the_panel(store):
    ingest.ingest(store, ["AAPL"], start="2024-01-01", end="2024-03-01",
                  fetcher=good_fetcher)
    rows_before = len(store.read("AAPL"))

    def revised(symbol, start, end):
        return make_bars(symbol, start, end).assign(close=lambda d: d["close"] * 1.5)

    ingest.ingest(store, ["AAPL"], start="2024-01-01", end="2024-03-01",
                  mode="full", fetcher=revised)
    after = store.read("AAPL")
    assert len(after) == rows_before                    # corrected, not appended
    assert after["close"].iloc[0] == pytest.approx(
        make_bars("AAPL", "2024-01-01", "2024-03-01")["close"].iloc[0] * 1.5)


# --------------------------------------------------------------------------
# Backfill
# --------------------------------------------------------------------------
def test_backfill_extends_history_earlier_without_disturbing_existing_rows(store):
    ingest.ingest(store, ["AAPL"], start="2024-03-01", end="2024-06-01",
                  fetcher=good_fetcher)
    existing = store.read("AAPL").copy()

    ingest.ingest(store, ["AAPL"], start="2024-01-01", end="2024-06-01",
                  mode="backfill", fetcher=good_fetcher)
    after = store.read("AAPL")
    assert after.index.min() == pd.Timestamp("2024-01-01")
    assert after.index.max() == existing.index.max()
    assert after.index.is_unique
    # Backfill fills the past; where the two overlap, what was already stored
    # wins, so a backfill cannot silently rewrite validated history.
    pd.testing.assert_frame_equal(after.loc[existing.index], existing)


# --------------------------------------------------------------------------
# Partial failure and failure reporting
# --------------------------------------------------------------------------
def test_one_bad_symbol_does_not_stop_the_run(store):
    def flaky(symbol, start, end):
        if symbol == "DEAD":
            raise RuntimeError("404 no data for DEAD")
        return make_bars(symbol, start, end)

    report = ingest.ingest(store, ["AAPL", "DEAD", "MSFT"], start="2024-01-01",
                           end="2024-03-01", fetcher=flaky)
    assert [o.symbol for o in report.by_status("ok")] == ["AAPL", "MSFT"]
    assert [o.symbol for o in report.failed] == ["DEAD"]
    assert "404" in report.by_status("error")[0].detail
    assert not report.ok                       # the run is not a success
    assert store.symbols() == ["AAPL", "MSFT"]  # but the good data landed


def test_fail_fast_aborts_and_names_the_pending_work(store):
    def flaky(symbol, start, end):
        if symbol == "DEAD":
            raise RuntimeError("rate limited")
        return make_bars(symbol, start, end)

    report = ingest.ingest(store, ["AAPL", "DEAD", "MSFT"], start="2024-01-01",
                           end="2024-03-01", fetcher=flaky, fail_fast=True)
    assert report.aborted and "rate limited" in report.abort_reason
    assert report.pending == ["MSFT"]


def test_an_empty_response_is_reported_not_treated_as_success(store):
    def empty(symbol, start, end):
        return pd.DataFrame(columns=["close", "volume"],
                            index=pd.DatetimeIndex([], name="date"))

    report = ingest.ingest(store, ["AAPL"], start="2024-01-01", end="2024-03-01",
                           fetcher=empty)
    assert report.by_status("empty")
    assert store.symbols() == []               # nothing written from nothing


def test_every_run_writes_a_report_even_when_it_dies(store):
    def exploding(symbol):
        if symbol == "MSFT":
            raise KeyboardInterrupt("operator killed the job")

    with pytest.raises(KeyboardInterrupt):
        ingest.ingest(store, ["AAPL", "MSFT", "GOOGL"], start="2024-01-01",
                      end="2024-03-01", fetcher=good_fetcher, on_symbol=exploding)

    saved = ingest.last_report(store)
    assert saved is not None
    assert [o["symbol"] for o in saved["outcomes"]] == ["AAPL"]
    assert saved["pending_after_run"] == ["MSFT", "GOOGL"]


# --------------------------------------------------------------------------
# THE demonstration: die halfway, diagnose, resume, verify
# --------------------------------------------------------------------------
def test_crash_halfway_then_resume_leaves_a_complete_uncorrupted_store(store):
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

    # 1. The run dies while processing the third symbol.
    def die_on_googl(symbol):
        if symbol == "GOOGL":
            raise KeyboardInterrupt("pod evicted")

    with pytest.raises(KeyboardInterrupt):
        ingest.ingest(store, symbols, start="2024-01-01", end="2024-04-01",
                      fetcher=good_fetcher, on_symbol=die_on_googl)

    # 2. Diagnose from the committed record, not from memory.
    assert store.symbols() == ["AAPL", "MSFT"]
    assert store.audit()["consistent"]              # a partial store is CONSISTENT
    pending = ingest.resume_set(store)
    assert pending == ["GOOGL", "AMZN", "META"]

    # 3. Resume exactly the pending set.
    report = ingest.ingest(store, pending, start="2024-01-01", end="2024-04-01",
                           fetcher=good_fetcher)
    assert report.ok

    # 4. The store is now complete, and identical to one built in a single
    #    clean pass -- the crash left no trace in the data.
    assert store.symbols() == sorted(symbols)
    for s in symbols:
        pd.testing.assert_frame_equal(
            store.read(s), make_bars(s, "2024-01-01", "2024-04-01"), check_freq=False
        )
    assert store.audit()["consistent"]


def test_resume_also_retries_symbols_that_failed(store):
    def flaky(symbol, start, end):
        if symbol == "DEAD":
            raise RuntimeError("transient 503")
        return make_bars(symbol, start, end)

    ingest.ingest(store, ["AAPL", "DEAD"], start="2024-01-01", end="2024-03-01",
                  fetcher=flaky)
    assert ingest.resume_set(store) == ["DEAD"]     # failures are resume work too

    report = ingest.ingest(store, ingest.resume_set(store), start="2024-01-01",
                           end="2024-03-01", fetcher=good_fetcher)
    assert report.ok and store.symbols() == ["AAPL", "DEAD"]
    assert ingest.resume_set(store) == []


# --------------------------------------------------------------------------
# Manifest is derived, not authoritative
# --------------------------------------------------------------------------
def test_a_crash_between_the_write_and_the_manifest_update_is_healed(store):
    ingest.ingest(store, ["AAPL", "MSFT"], start="2024-01-01", end="2024-03-01",
                  fetcher=good_fetcher)
    # Simulate the crash window: the partition landed, the index did not.
    manifest = store.load_manifest()
    del manifest["symbols"]["MSFT"]
    store.save_manifest(manifest)

    assert store.audit()["unindexed"] == ["MSFT"]
    assert not store.audit()["consistent"]

    store.reconcile()                                # cheap, always safe
    assert store.audit()["consistent"]
    assert "MSFT" in store.load_manifest()["symbols"]


def test_a_destroyed_manifest_is_rebuilt_from_the_partitions(store):
    ingest.ingest(store, ["AAPL", "MSFT"], start="2024-01-01", end="2024-03-01",
                  fetcher=good_fetcher)
    rows = {s: len(store.read(s)) for s in ("AAPL", "MSFT")}
    with open(store.manifest_path, "w") as fh:
        fh.write("{ this is not json")

    rebuilt = store.reconcile()
    assert {s: rebuilt["symbols"][s]["rows"] for s in rows} == rows
    assert store.audit()["consistent"]


def test_audit_detects_content_drift_under_the_manifest(store):
    ingest.ingest(store, ["AAPL"], start="2024-01-01", end="2024-03-01",
                  fetcher=good_fetcher)
    # Someone edited a partition out of band.
    df = store.read("AAPL")
    df.iloc[0, df.columns.get_loc("close")] *= 2
    store.write("AAPL", df)
    manifest = store.load_manifest()
    manifest["symbols"]["AAPL"]["content_hash"] = "stale00000000000"
    store.save_manifest(manifest)

    audit = store.audit()
    assert audit["content_drift"] == ["AAPL"] and not audit["consistent"]


def test_partition_write_is_atomic_and_leaves_no_temp_files(store):
    ingest.ingest(store, ["AAPL"], start="2024-01-01", end="2024-03-01",
                  fetcher=good_fetcher)
    leftovers = [f for _, _, files in os.walk(store.root) for f in files
                 if ".tmp." in f]
    assert leftovers == []


def test_a_failed_partition_write_does_not_replace_the_existing_file(store, monkeypatch):
    ingest.ingest(store, ["AAPL"], start="2024-01-01", end="2024-03-01",
                  fetcher=good_fetcher)
    before = store.read("AAPL").copy()

    def boom(self, *a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
    with pytest.raises(OSError):
        store.write("AAPL", make_bars("AAPL", "2024-01-01", "2024-06-01"))
    monkeypatch.undo()
    pd.testing.assert_frame_equal(store.read("AAPL"), before)


# --------------------------------------------------------------------------
# Input validation and reporting shape
# --------------------------------------------------------------------------
def test_invalid_mode_and_inverted_window_are_rejected(store):
    with pytest.raises(ValueError, match="mode must be"):
        ingest.ingest(store, ["AAPL"], fetcher=good_fetcher, mode="sideways")
    with pytest.raises(ValueError, match="before start"):
        ingest.ingest(store, ["AAPL"], start="2024-06-01", end="2024-01-01",
                      fetcher=good_fetcher)


def test_report_serializes_and_renders(store):
    def flaky(symbol, start, end):
        if symbol == "DEAD":
            raise RuntimeError("boom")
        return make_bars(symbol, start, end)

    report = ingest.ingest(store, ["AAPL", "DEAD"], start="2024-01-01",
                           end="2024-03-01", fetcher=flaky)
    doc = json.loads(json.dumps(report.as_dict()))
    assert doc["summary"]["ok"] == 1 and doc["summary"]["error"] == 1
    assert "FAILED DEAD" in report.render()


def test_timezone_aware_fetcher_output_is_normalized_not_rejected(store):
    def tz_fetcher(symbol, start, end):
        df = make_bars(symbol, start, end)
        df.index = df.index.tz_localize("America/New_York")
        return df

    report = ingest.ingest(store, ["AAPL"], start="2024-01-01", end="2024-03-01",
                           fetcher=tz_fetcher)
    assert report.ok and store.read("AAPL").index.tz is None


def test_two_runs_in_the_same_second_keep_separate_reports(store):
    """A resume fires seconds after the failure it is resuming; both records
    must survive, or the audit trail loses the failure it was written for."""
    ingest.ingest(store, ["AAPL"], start="2024-01-01", end="2024-02-01",
                  fetcher=good_fetcher)
    ingest.ingest(store, ["MSFT"], start="2024-01-01", end="2024-02-01",
                  fetcher=good_fetcher)
    runs = [f for f in os.listdir(store.runs_dir) if f.startswith("run_")]
    assert len(runs) == 2
    assert ingest.last_report(store)["requested"] == 1
