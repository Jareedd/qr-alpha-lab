"""An operable price store: incremental, validated, resumable, safe to rerun.

The research pipeline elsewhere in this repo treats data as a thing you
download. This module treats it as a thing you *operate*: a partitioned store
with a watermark per symbol, a schema contract enforced before anything is
written, atomic partition writes, a derivable manifest, and a per-run failure
report.

The design constraint that shapes everything here: **a run that dies halfway
must leave the store in a state a later run can safely continue from.** That
rules out the obvious implementations -- an append-to-one-big-file store
corrupts on a partial write, and an in-memory concat-then-write loses every
symbol when the 400th one 404s.

Layout::

    <root>/
      prices/symbol=AAPL/data.parquet     one partition per symbol
      prices/symbol=MSFT/data.parquet
      _manifest.json                      DERIVED index (rebuildable)
      _runs/run_<timestamp>.json          one failure report per run

The manifest is deliberately *derived*, never authoritative. Partitions are
written atomically, so the parquet files on disk are always the truth;
``PriceStore.reconcile()`` rebuilds the manifest from them. A crash between
writing a partition and updating the manifest is therefore not corruption --
it is a stale index that heals itself.

Idempotence rule: ingesting a window that is already covered is a no-op.
Re-running never duplicates a row, because every merge deduplicates on the
date index (last write wins, matching the vendor-correction convention in
``quantlab.data``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

import pandas as pd

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# The schema contract. Every partition must satisfy all of it before it is
# allowed anywhere near the store -- validation happens on the candidate
# frame, not after the write, because "validate then repair" is how bad rows
# reach a dataset that everyone downstream trusts.
REQUIRED_COLUMNS = ("close", "volume")


class SchemaViolation(ValueError):
    """A candidate partition failed the schema contract. Nothing was written."""


@dataclass
class SymbolOutcome:
    """What happened to one symbol in one run."""

    symbol: str
    status: str                       # ok | skipped | empty | invalid | error
    rows_before: int = 0
    rows_after: int = 0
    rows_added: int = 0
    first_date: str | None = None
    last_date: str | None = None
    detail: str = ""
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol, "status": self.status,
            "rows_before": self.rows_before, "rows_after": self.rows_after,
            "rows_added": self.rows_added, "first_date": self.first_date,
            "last_date": self.last_date, "detail": self.detail,
            "seconds": round(self.seconds, 3),
        }


@dataclass
class IngestReport:
    """Per-run failure reporting. Written to ``<root>/_runs/`` on every run.

    Operability means the answer to "what happened last night?" is a file, not
    a scroll through CI logs.
    """

    run_id: str
    mode: str
    started_at: str
    requested: list[str] = field(default_factory=list)
    outcomes: list[SymbolOutcome] = field(default_factory=list)
    seconds: float = 0.0
    aborted: bool = False
    abort_reason: str = ""

    def by_status(self, status: str) -> list[SymbolOutcome]:
        return [o for o in self.outcomes if o.status == status]

    @property
    def failed(self) -> list[SymbolOutcome]:
        return [o for o in self.outcomes if o.status in ("error", "invalid")]

    @property
    def ok(self) -> bool:
        return not self.failed and not self.aborted

    @property
    def pending(self) -> list[str]:
        """Requested symbols this run never reached -- the resume set."""
        done = {o.symbol for o in self.outcomes}
        return [s for s in self.requested if s not in done]

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "started_at": self.started_at,
            "seconds": round(self.seconds, 3),
            "schema_version": SCHEMA_VERSION,
            "requested": len(self.requested),
            "aborted": self.aborted,
            "abort_reason": self.abort_reason,
            "summary": {
                s: len(self.by_status(s))
                for s in ("ok", "skipped", "empty", "invalid", "error")
            },
            "pending_after_run": self.pending,
            "outcomes": [o.as_dict() for o in self.outcomes],
        }

    def render(self) -> str:
        lines = [f"run {self.run_id} ({self.mode})  {self.seconds:.1f}s"]
        for s in ("ok", "skipped", "empty", "invalid", "error"):
            n = len(self.by_status(s))
            if n:
                lines.append(f"  {s:<8} {n}")
        for o in self.failed:
            lines.append(f"  FAILED {o.symbol}: {o.detail}")
        if self.aborted:
            lines.append(f"  ABORTED: {self.abort_reason}")
            lines.append(f"  pending ({len(self.pending)}): "
                         f"{', '.join(self.pending[:10])}"
                         + (" ..." if len(self.pending) > 10 else ""))
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Schema validation
# --------------------------------------------------------------------------
def validate_partition(df: pd.DataFrame, symbol: str) -> list[str]:
    """Return a list of schema violations. Empty list == the frame may be stored.

    Each rule below exists because violating it produces a *plausible-looking*
    downstream number rather than a crash:

    - a duplicated date silently double-counts a return;
    - an unsorted or tz-aware index misaligns a merge across symbols;
    - a non-positive price makes log-returns infinite;
    - a NaN close propagates into a feature and out into an IC.
    """
    problems: list[str] = []
    if df is None:
        return [f"{symbol}: frame is None"]
    if not isinstance(df, pd.DataFrame):
        return [f"{symbol}: expected DataFrame, got {type(df).__name__}"]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        problems.append(f"{symbol}: missing required column(s) {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        problems.append(f"{symbol}: index must be a DatetimeIndex, got "
                        f"{type(df.index).__name__}")
        return problems                       # further checks would be noise
    if df.index.tz is not None:
        problems.append(f"{symbol}: index must be timezone-naive")
    if not df.index.is_unique:
        dupes = int(df.index.duplicated().sum())
        problems.append(f"{symbol}: index has {dupes} duplicate timestamp(s)")
    if not df.index.is_monotonic_increasing:
        problems.append(f"{symbol}: index must be sorted ascending")
    if df.index.hasnans:
        problems.append(f"{symbol}: index contains NaT")
    if "close" in df.columns:
        close = pd.to_numeric(df["close"], errors="coerce")
        if close.isna().any():
            problems.append(f"{symbol}: close has {int(close.isna().sum())} "
                            f"non-numeric/NaN value(s)")
        if (close.dropna() <= 0).any():
            problems.append(f"{symbol}: close has non-positive value(s)")
    if "volume" in df.columns:
        vol = pd.to_numeric(df["volume"], errors="coerce")
        if (vol.dropna() < 0).any():
            problems.append(f"{symbol}: volume has negative value(s)")
    return problems


def _content_hash(df: pd.DataFrame) -> str:
    """Hash the CONTENT, not the file.

    Parquet bytes are not reproducible across library versions, so a file hash
    would report spurious drift. Hashing the values makes "did this partition
    change?" answerable across environments.
    """
    payload = df.to_csv(float_format="%.10g").encode()
    return hashlib.sha256(payload).hexdigest()[:16]


# --------------------------------------------------------------------------
# The store
# --------------------------------------------------------------------------
class PriceStore:
    """Partitioned, atomically-written price store with a derivable manifest."""

    def __init__(self, root: str):
        self.root = root
        self.data_root = os.path.join(root, "prices")
        self.manifest_path = os.path.join(root, "_manifest.json")
        self.runs_dir = os.path.join(root, "_runs")
        os.makedirs(self.data_root, exist_ok=True)
        os.makedirs(self.runs_dir, exist_ok=True)

    # -- paths ------------------------------------------------------------
    def partition_dir(self, symbol: str) -> str:
        return os.path.join(self.data_root, f"symbol={symbol}")

    def partition_path(self, symbol: str) -> str:
        return os.path.join(self.partition_dir(symbol), "data.parquet")

    def symbols(self) -> list[str]:
        if not os.path.isdir(self.data_root):
            return []
        return sorted(
            d.split("=", 1)[1]
            for d in os.listdir(self.data_root)
            if d.startswith("symbol=")
            and os.path.exists(os.path.join(self.data_root, d, "data.parquet"))
        )

    # -- read / write -----------------------------------------------------
    def read(self, symbol: str) -> pd.DataFrame:
        path = self.partition_path(symbol)
        if not os.path.exists(path):
            return pd.DataFrame(columns=list(REQUIRED_COLUMNS),
                                index=pd.DatetimeIndex([], name="date"))
        df = pd.read_parquet(path)
        df.index = pd.DatetimeIndex(df.index)
        df.index.name = "date"
        return df

    def write(self, symbol: str, df: pd.DataFrame) -> None:
        """Validate, then write atomically. An invalid frame is never stored."""
        problems = validate_partition(df, symbol)
        if problems:
            raise SchemaViolation("; ".join(problems))
        os.makedirs(self.partition_dir(symbol), exist_ok=True)
        path = self.partition_path(symbol)
        tmp = f"{path}.tmp.{os.getpid()}"
        try:
            df.to_parquet(tmp)
            os.replace(tmp, path)             # atomic on POSIX: readers see
        finally:                              # either the old file or the new
            if os.path.exists(tmp):
                os.remove(tmp)

    def watermark(self, symbol: str) -> pd.Timestamp | None:
        """Last date held for a symbol -- where an incremental run resumes."""
        df = self.read(symbol)
        return None if df.empty else pd.Timestamp(df.index.max())

    def coverage(self, symbol: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        df = self.read(symbol)
        if df.empty:
            return None
        return pd.Timestamp(df.index.min()), pd.Timestamp(df.index.max())

    # -- manifest ---------------------------------------------------------
    def load_manifest(self) -> dict:
        if not os.path.exists(self.manifest_path):
            return {"schema_version": SCHEMA_VERSION, "symbols": {}}
        try:
            with open(self.manifest_path) as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            # The manifest is derived, so a damaged one is an inconvenience,
            # not a data loss. Say so and rebuild.
            logger.warning("manifest unreadable (%s); rebuilding from partitions", exc)
            return {"schema_version": SCHEMA_VERSION, "symbols": {}}

    def save_manifest(self, manifest: dict) -> None:
        tmp = f"{self.manifest_path}.tmp.{os.getpid()}"
        try:
            with open(tmp, "w") as fh:
                json.dump(manifest, fh, indent=2, sort_keys=True)
                fh.write("\n")
            os.replace(tmp, self.manifest_path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def reconcile(self) -> dict:
        """Rebuild the manifest from what is actually on disk.

        This is what makes a mid-run crash survivable: partitions are the
        truth, the manifest is an index over them. Run it after any abnormal
        exit -- or just always; it is cheap and it cannot make things worse.
        """
        manifest = {"schema_version": SCHEMA_VERSION,
                    "reconciled_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "symbols": {}}
        for symbol in self.symbols():
            df = self.read(symbol)
            problems = validate_partition(df, symbol)
            span = self.coverage(symbol)
            manifest["symbols"][symbol] = {
                "rows": len(df),
                "first_date": None if span is None else span[0].strftime("%Y-%m-%d"),
                "last_date": None if span is None else span[1].strftime("%Y-%m-%d"),
                "content_hash": _content_hash(df),
                "valid": not problems,
                "problems": problems,
            }
        self.save_manifest(manifest)
        return manifest

    def audit(self) -> dict:
        """Compare the manifest against the partitions and report the drift."""
        manifest = self.load_manifest().get("symbols", {})
        on_disk = set(self.symbols())
        indexed = set(manifest)
        drifted = []
        for symbol in sorted(on_disk & indexed):
            if _content_hash(self.read(symbol)) != manifest[symbol].get("content_hash"):
                drifted.append(symbol)
        invalid = sorted(s for s, m in manifest.items() if not m.get("valid", True))
        return {
            "partitions_on_disk": len(on_disk),
            "symbols_in_manifest": len(indexed),
            "unindexed": sorted(on_disk - indexed),      # crash after write
            "missing_partitions": sorted(indexed - on_disk),
            "content_drift": drifted,
            "invalid": invalid,
            "consistent": not (on_disk ^ indexed) and not drifted and not invalid,
        }


# --------------------------------------------------------------------------
# Merge + ingest
# --------------------------------------------------------------------------
def merge_partition(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """Idempotent merge: union of dates, incoming wins on collision.

    "Incoming wins" matches the vendor-correction convention in
    ``quantlab.data``: when a source re-sends a bar, the later copy is the
    corrected one. Because the result is keyed on the date index, running the
    same ingest twice is a no-op rather than a doubling -- which is the whole
    point of a safe rerun.
    """
    if existing is None or existing.empty:
        out = incoming.copy()
    elif incoming is None or incoming.empty:
        out = existing.copy()
    else:
        out = pd.concat([existing, incoming])
        out = out[~out.index.duplicated(keep="last")]
    out = out.sort_index(kind="stable")
    out.index = pd.DatetimeIndex(out.index)
    out.index.name = "date"
    return out


# A fetcher takes (symbol, start, end) and returns a DataFrame indexed by date
# with at least the REQUIRED_COLUMNS. Injectable so tests, the demo and a real
# vendor all drive the same code path.
Fetcher = Callable[[str, pd.Timestamp, pd.Timestamp | None], pd.DataFrame]


def yfinance_fetcher(symbol: str, start: pd.Timestamp, end: pd.Timestamp | None) -> pd.DataFrame:
    """Default fetcher: one symbol via the hardened loader in quantlab.data."""
    from quantlab.data import load_prices, load_volumes

    kwargs = {
        "tickers": [symbol],
        "start": start.strftime("%Y-%m-%d"),
        "end": None if end is None else end.strftime("%Y-%m-%d"),
        "cache_dir": os.environ.get("QUANTLAB_CACHE_DIR", "data_cache"),
    }
    px = load_prices(min_coverage=0.0, **kwargs)
    vol = load_volumes(**kwargs)
    out = pd.DataFrame({"close": px[symbol]})
    out["volume"] = vol[symbol] if symbol in vol.columns else float("nan")
    out.index.name = "date"
    return out


def ingest(
    store: PriceStore,
    symbols: Iterable[str],
    start: str = "2010-01-01",
    end: str | None = None,
    fetcher: Fetcher = yfinance_fetcher,
    *,
    mode: str = "incremental",
    fail_fast: bool = False,
    on_symbol: Callable[[str], None] | None = None,
) -> IngestReport:
    """Ingest ``symbols`` into ``store`` and return a per-run report.

    ``mode``:

    - ``incremental`` -- fetch only from each symbol's watermark forward. The
      normal nightly path; a second run the same day is a no-op.
    - ``backfill``    -- fetch the full ``start``..``end`` window and merge it
      under the existing data, so history can be extended *earlier* without
      touching (or duplicating) what is already stored.
    - ``full``        -- refetch the whole window and let it overwrite on
      collision. The repair path for a source that revised its history.

    Symbol-level failures never abort the run unless ``fail_fast``; each one
    is recorded in the report with its reason. A symbol that fails leaves its
    existing partition untouched -- a bad fetch cannot damage good data.

    ``on_symbol`` is called before each symbol is processed. It exists so the
    demo (and the tests) can inject a mid-run crash at a known point, which is
    the only honest way to prove resumability.
    """
    if mode not in ("incremental", "backfill", "full"):
        raise ValueError(f"mode must be incremental|backfill|full, got {mode!r}")
    symbols = list(symbols)
    start_ts = pd.Timestamp(start)
    end_ts = None if end is None else pd.Timestamp(end)
    if end_ts is not None and end_ts < start_ts:
        raise ValueError(f"end ({end}) is before start ({start})")

    # Microsecond suffix: run ids are also filenames, and two runs inside the
    # same second (a resume immediately after a failure, or a test) must not
    # overwrite each other's report. Still lexically sortable = chronological.
    now = time.time()
    run_id = f"{time.strftime('%Y%m%dT%H%M%S', time.localtime(now))}.{int(now % 1 * 1e6):06d}"
    report = IngestReport(run_id=run_id, mode=mode,
                          started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                          requested=list(symbols))
    t_run = time.time()

    try:
        for symbol in symbols:
            if on_symbol is not None:
                on_symbol(symbol)             # may raise: simulated crash
            t0 = time.time()
            existing = store.read(symbol)
            outcome = SymbolOutcome(symbol=symbol, status="ok",
                                    rows_before=len(existing))

            fetch_from = start_ts
            if mode == "incremental":
                wm = None if existing.empty else pd.Timestamp(existing.index.max())
                if wm is not None:
                    # Re-fetch the watermark day itself: a partial final bar is
                    # common, and the merge deduplicates, so overlap is free
                    # while a gap is not.
                    fetch_from = max(start_ts, wm)
                if end_ts is not None and fetch_from > end_ts:
                    outcome.status = "skipped"
                    outcome.detail = (f"watermark {fetch_from.date()} already at or "
                                      f"past end {end_ts.date()}")
                    outcome.rows_after = len(existing)
                    outcome.seconds = time.time() - t0
                    report.outcomes.append(outcome)
                    continue

            try:
                incoming = fetcher(symbol, fetch_from, end_ts)
            except Exception as exc:
                outcome.status = "error"
                outcome.detail = f"{type(exc).__name__}: {exc}"
                outcome.rows_after = len(existing)
                outcome.seconds = time.time() - t0
                report.outcomes.append(outcome)
                logger.warning("ingest %s FAILED: %s", symbol, outcome.detail)
                if fail_fast:
                    report.aborted = True
                    report.abort_reason = f"fail_fast on {symbol}: {outcome.detail}"
                    break
                continue

            if incoming is None or len(incoming) == 0:
                outcome.status = "empty"
                outcome.detail = f"no rows returned from {fetch_from.date()}"
                outcome.rows_after = len(existing)
                outcome.seconds = time.time() - t0
                report.outcomes.append(outcome)
                continue

            incoming = _coerce(incoming)
            problems = validate_partition(incoming, symbol)
            if problems:
                # Reject BEFORE the merge. A quarantined bad batch leaves the
                # stored history exactly as it was.
                outcome.status = "invalid"
                outcome.detail = "; ".join(problems)
                outcome.rows_after = len(existing)
                outcome.seconds = time.time() - t0
                report.outcomes.append(outcome)
                logger.warning("ingest %s REJECTED: %s", symbol, outcome.detail)
                if fail_fast:
                    report.aborted = True
                    report.abort_reason = f"fail_fast on {symbol}: {outcome.detail}"
                    break
                continue

            merged = (merge_partition(incoming, existing) if mode == "backfill"
                      else merge_partition(existing, incoming))
            store.write(symbol, merged)
            outcome.rows_after = len(merged)
            outcome.rows_added = len(merged) - len(existing)
            outcome.first_date = merged.index.min().strftime("%Y-%m-%d")
            outcome.last_date = merged.index.max().strftime("%Y-%m-%d")
            outcome.seconds = time.time() - t0
            report.outcomes.append(outcome)
    finally:
        # The report and the manifest are written even when the run dies, so
        # the next operator starts from a record rather than from silence.
        report.seconds = time.time() - t_run
        store.reconcile()
        _write_report(store, report)

    return report


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort normalisation of a fetcher's frame BEFORE validation.

    Deliberately narrow: it fixes representation (index type, ordering, name),
    never content. Dropping bad rows here would turn a schema violation into a
    silent partial load, which is the failure this module is built to prevent.
    """
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.DatetimeIndex(pd.to_datetime(out.index, errors="coerce"))
    if out.index.tz is not None:
        out.index = out.index.tz_convert("UTC").tz_localize(None)
    out.index = pd.DatetimeIndex(out.index.normalize().values)
    out.index.name = "date"
    return out.sort_index(kind="stable")


def _write_report(store: PriceStore, report: IngestReport) -> str:
    path = os.path.join(store.runs_dir, f"run_{report.run_id}.json")
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w") as fh:
            json.dump(report.as_dict(), fh, indent=2)
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def last_report(store: PriceStore) -> dict | None:
    """The most recent run report -- 'what happened last night?' in one call."""
    if not os.path.isdir(store.runs_dir):
        return None
    runs = sorted(f for f in os.listdir(store.runs_dir) if f.startswith("run_"))
    if not runs:
        return None
    with open(os.path.join(store.runs_dir, runs[-1])) as fh:
        return json.load(fh)


def resume_set(store: PriceStore, symbols: Iterable[str] | None = None) -> list[str]:
    """Symbols the last run did not finish: never reached, failed, or rejected.

    This is the resume plan. It is computed from the committed run report, not
    from memory, so it survives the process that produced it.
    """
    report = last_report(store)
    if report is None:
        return list(symbols or [])
    pending = list(report.get("pending_after_run", []))
    retry = [o["symbol"] for o in report.get("outcomes", [])
             if o["status"] in ("error", "invalid")]
    seen, out = set(), []
    for s in pending + retry:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out
