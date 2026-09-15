"""Data loading: real prices via yfinance (with local cache) or synthetic panels.

Operational contract (the reason this module is more defensive than it looks):
a research pipeline that silently loads *wrong* data is worse than one that
crashes. Every failure mode below either raises with an actionable message or
is reported in ``df.attrs["download_report"]`` -- never swallowed.

- Every download empty  -> ``DataUnavailableError`` (never an empty-frame
  "result", never a cached empty file that poisons every later run).
- Partial download      -> the frame is returned, the missing symbols are
  named in the report and logged; ``min_symbols`` turns "too partial" into
  a hard failure.
- Stale ``end=None``    -> an open-ended window is keyed as "latest"; a cache
  file written before ``max_cache_age_days`` ago is refetched instead of
  served forever. (This bug froze the live monitor's price panel for six
  weeks -- see research_log.md 2026-08-06.)
- Duplicate timestamps  -> deduplicated (last wins) and counted, per chunk,
  BEFORE the concat that would otherwise raise on a non-unique index.
- Invalid inputs        -> ``ValueError`` before any network or disk access.
- Interrupted rerun     -> the cache is written atomically (temp + rename) and
  an unreadable cache file is refetched, so a rerun is idempotent rather than
  a corruption trap.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field as _dc_field

import pandas as pd

logger = logging.getLogger(__name__)

# How old an open-ended ("end=None", i.e. "through latest") cache may be before
# it is refetched. One day: the panel is daily, so a cache written during the
# same session is fine, but yesterday's "latest" is by definition not latest.
LATEST_CACHE_MAX_AGE_DAYS = 1.0

# A default universe of liquid US large caps + sector ETFs (free data, survivorship-
# biased by construction -- see README "Known limitations").
DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "MA",
    "UNH", "HD", "PG", "KO", "PEP", "MRK", "ABBV", "XOM", "CVX", "WMT",
    "BAC", "DIS", "CSCO", "ADBE", "CRM", "NFLX", "INTC", "AMD", "QCOM", "TXN",
    "HON", "CAT", "BA", "GE", "MMM", "UPS", "RTX", "LMT", "GS", "MS",
    "C", "WFC", "T", "VZ", "CMCSA", "PFE", "JNJ", "LLY", "TMO", "ABT",
    "XLE", "XLF", "XLK", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE",
]


class DataUnavailableError(RuntimeError):
    """No usable rows could be obtained for the requested field/window.

    Raised instead of returning an empty frame: downstream, an empty panel
    turns into an empty feature matrix and a meaningless "no signal" verdict,
    which is indistinguishable from a real null result. That confusion is
    exactly what this project exists to avoid.
    """


@dataclass
class DownloadReport:
    """What actually happened on a load. Attached as ``df.attrs['download_report']``.

    Kept as data (not just log lines) so the ingest layer and the tests can
    assert on partial-failure behaviour instead of scraping stderr.
    """

    field: str
    requested: tuple[str, ...] = ()
    returned: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    dropped_low_coverage: tuple[str, ...] = ()
    duplicate_rows_dropped: int = 0
    chunks_total: int = 0
    chunks_empty: int = 0
    chunks_unusable: int = 0
    rows: int = 0
    source: str = "download"          # "download" | "cache"
    cache_path: str = ""
    cache_age_days: float | None = None
    warnings: list[str] = _dc_field(default_factory=list)

    @property
    def coverage(self) -> float:
        """Fraction of requested symbols present in the returned panel."""
        return len(self.returned) / len(self.requested) if self.requested else 0.0

    def as_dict(self) -> dict:
        return {
            "field": self.field,
            "requested": len(self.requested),
            "returned": len(self.returned),
            "missing": list(self.missing),
            "dropped_low_coverage": list(self.dropped_low_coverage),
            "duplicate_rows_dropped": self.duplicate_rows_dropped,
            "chunks_total": self.chunks_total,
            "chunks_empty": self.chunks_empty,
            "chunks_unusable": self.chunks_unusable,
            "rows": self.rows,
            "coverage": round(self.coverage, 4),
            "source": self.source,
            "cache_path": self.cache_path,
            "cache_age_days": self.cache_age_days,
            "warnings": list(self.warnings),
        }


def _validate_request(
    tickers: list[str],
    start: str,
    end: str | None,
    min_coverage: float,
    chunk_size: int,
) -> tuple[list[str], pd.Timestamp, pd.Timestamp | None]:
    """Reject malformed requests before any disk or network access.

    Returns the de-duplicated ticker list (request order preserved, so the
    column order of the result is a function of the request alone) plus the
    parsed dates.
    """
    if isinstance(tickers, str):
        raise ValueError(
            f"tickers must be a list of symbols, got a bare string {tickers!r}; "
            f'pass ["{tickers}"]'
        )
    if not isinstance(tickers, (list, tuple, pd.Index)):
        raise ValueError(f"tickers must be a list of symbols, got {type(tickers).__name__}")
    cleaned: list[str] = []
    seen: set[str] = set()
    for t in tickers:
        if not isinstance(t, str):
            raise ValueError(f"ticker symbols must be strings, got {t!r}")
        s = t.strip()
        if not s:
            raise ValueError("ticker symbols must be non-empty (got a blank entry)")
        if s not in seen:            # a duplicate request is a caller bug, but a
            seen.add(s)              # silent one: it would double-weight a name.
            cleaned.append(s)
    if not cleaned:
        raise ValueError("tickers is empty: nothing to download")
    if len(cleaned) != len(list(tickers)):
        logger.warning(
            "dropped %d duplicate ticker(s) from the request",
            len(list(tickers)) - len(cleaned),
        )

    try:
        start_ts = pd.Timestamp(start)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"start is not a parseable date: {start!r}") from exc
    if start_ts is pd.NaT or pd.isna(start_ts):
        raise ValueError(f"start is not a parseable date: {start!r}")
    end_ts: pd.Timestamp | None = None
    if end is not None:
        try:
            end_ts = pd.Timestamp(end)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"end is not a parseable date: {end!r}") from exc
        if end_ts is pd.NaT or pd.isna(end_ts):
            raise ValueError(f"end is not a parseable date: {end!r}")
        if end_ts < start_ts:
            raise ValueError(f"end ({end}) is before start ({start})")

    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError(f"min_coverage must be in [0, 1], got {min_coverage}")
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError(f"chunk_size must be a positive int, got {chunk_size!r}")
    return cleaned, start_ts, end_ts


def _normalize_frame(part: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Sort, de-duplicate and tz-normalize one chunk's index.

    Duplicate timestamps must die here, not after the concat: ``pd.concat``
    along axis=1 raises on a non-unique index, so a single duplicated vendor
    row would take down an otherwise healthy 500-name download. Last wins --
    when a vendor re-sends a bar, the later copy is the corrected one.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(part.index, errors="coerce"))
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    part = part.copy()
    # Rebuild from values to drop any inferred `freq`, so a frame that came off
    # the wire and the same frame read back from parquet compare equal.
    part.index = pd.DatetimeIndex(idx.normalize().values)
    part = part[part.index.notna()]
    # Stable sort: with a duplicated timestamp, "last wins" has to mean "the
    # copy the vendor sent last", and quicksort (pandas' default) does not
    # preserve that order among ties.
    part = part.sort_index(kind="stable")
    dupes = int(part.index.duplicated(keep="last").sum())
    if dupes:
        part = part[~part.index.duplicated(keep="last")]
    return part, dupes


def _extract_field(raw: pd.DataFrame, field: str, chunk: list[str]) -> pd.DataFrame | None:
    """Pull one OHLCV field out of a yfinance response, or None if unusable."""
    if isinstance(raw.columns, pd.MultiIndex):
        if field not in raw.columns.get_level_values(0):
            return None
        return raw[field]
    if field not in raw.columns:
        return None
    if len(chunk) != 1:
        # A flat column index for a multi-ticker request gives us no way to
        # tell whose prices these are. Guessing would silently attach one
        # name's price history to another -- refuse, and let the caller see
        # it as a failed chunk in the report.
        return None
    part = raw[[field]].copy()
    part.columns = [chunk[0]]
    return part


def _cache_is_fresh(cache_path: str, open_ended: bool, max_age_days: float) -> tuple[bool, float]:
    """Age a cache file and decide whether it may be served.

    A closed window (a fixed past ``end``) can never go stale -- the answer is
    frozen. An open-ended window keyed as "latest" absolutely can, and did.
    """
    age_days = (time.time() - os.path.getmtime(cache_path)) / 86400.0
    if not open_ended:
        return True, age_days
    return age_days <= max_age_days, age_days


def _download_field(
    tickers: list[str],
    field: str,
    start: str,
    end: str | None,
    cache_dir: str,
    min_coverage: float,
    chunk_size: int,
    *,
    min_symbols: int = 1,
    refresh: bool = False,
    max_cache_age_days: float = LATEST_CACHE_MAX_AGE_DAYS,
) -> pd.DataFrame:
    """Chunked yfinance download of one OHLCV field, cached to parquet."""
    tickers, start_ts, end_ts = _validate_request(
        tickers, start, end, min_coverage, chunk_size
    )
    if min_symbols < 1:
        raise ValueError(f"min_symbols must be >= 1, got {min_symbols}")

    report = DownloadReport(field=field, requested=tuple(tickers))
    os.makedirs(cache_dir, exist_ok=True)
    # Key on ticker *content*, not count: two different universes of the same
    # size must never silently share a cache file.
    digest = hashlib.md5(",".join(sorted(tickers)).encode()).hexdigest()[:10]
    # "prices" kept as the Close prefix so pre-refactor caches stay valid.
    prefix = "prices" if field == "Close" else field.lower()
    key = f"{prefix}_{digest}_{start}_{end or 'latest'}_{min_coverage}.parquet"
    cache_path = os.path.join(cache_dir, key)
    report.cache_path = cache_path

    # "Open ended" is about the WINDOW, not the literal argument: an end date
    # in the future leaves the right edge just as unfinished as end=None.
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    open_ended = end_ts is None or end_ts >= today

    if os.path.exists(cache_path) and not refresh:
        fresh, age_days = _cache_is_fresh(cache_path, open_ended, max_cache_age_days)
        if fresh:
            try:
                cached = pd.read_parquet(cache_path)
            except Exception as exc:  # corrupt/half-written parquet
                logger.warning(
                    "cache %s is unreadable (%s); refetching", cache_path, exc
                )
                report.warnings.append(f"unreadable cache refetched: {exc}")
            else:
                report.source = "cache"
                report.cache_age_days = round(age_days, 4)
                report.returned = tuple(c for c in cached.columns)
                report.missing = tuple(t for t in tickers if t not in set(cached.columns))
                report.rows = len(cached)
                if report.missing:
                    logger.info(
                        "%s cache hit with %d/%d symbols (missing: %s)",
                        field, len(report.returned), len(tickers),
                        _preview(report.missing),
                    )
                cached.attrs["download_report"] = report
                return cached
        else:
            logger.info(
                "%s cache %s is %.2f days old and the window is open-ended; refetching",
                field, cache_path, age_days,
            )
            report.warnings.append(f"stale open-ended cache ({age_days:.2f}d) refetched")

    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "yfinance is required for real data: pip install yfinance. "
            "For offline testing use synthetic data (see quantlab.synthetic)."
        ) from exc

    frames = []
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        report.chunks_total += 1
        raw = yf.download(chunk, start=start, end=end, auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            report.chunks_empty += 1
            logger.warning(
                "%s: empty download for chunk %d (%s)",
                field, report.chunks_total, _preview(chunk),
            )
            continue
        part = _extract_field(raw, field, chunk)
        if part is None or part.empty:
            report.chunks_unusable += 1
            logger.warning(
                "%s: unusable response for chunk %d (%s): no identifiable '%s' column",
                field, report.chunks_total, _preview(chunk), field,
            )
            continue
        part, dupes = _normalize_frame(part)
        report.duplicate_rows_dropped += dupes
        if dupes:
            logger.warning(
                "%s: dropped %d duplicate timestamp(s) in chunk %d",
                field, dupes, report.chunks_total,
            )
        frames.append(part)

    if not frames:
        # THE failure this module exists for: every chunk came back empty and
        # the old code died in pd.concat with "No objects to concatenate" --
        # a stack trace that says nothing about tickers, dates or rate limits.
        raise DataUnavailableError(
            f"no data returned for field {field!r}: all {report.chunks_total} "
            f"download chunk(s) were empty or unusable "
            f"({report.chunks_empty} empty, {report.chunks_unusable} unusable). "
            f"tickers={_preview(tickers)} start={start} end={end or 'latest'}. "
            "Common causes: no network, a rate limit, delisted/renamed symbols, "
            "or a window with no trading days. Nothing was cached."
        )

    out = pd.concat(frames, axis=1).sort_index(kind="stable")
    out = out.loc[:, ~out.columns.duplicated()]
    out = out.dropna(how="all").dropna(axis=1, how="all")
    before_coverage = set(out.columns)
    if min_coverage > 0:
        out = out.dropna(axis=1, thresh=int(len(out) * min_coverage))
    report.dropped_low_coverage = tuple(
        t for t in tickers if t in before_coverage and t not in set(out.columns)
    )
    # Deterministic column order: a function of the request, not of the order
    # chunks happened to come back in. Reruns are then byte-comparable.
    present = [t for t in tickers if t in set(out.columns)]
    extra = [c for c in out.columns if c not in set(tickers)]  # defensive; normally none
    out = out[present + extra]

    if out.empty or not present:
        raise DataUnavailableError(
            f"field {field!r} downloaded but nothing survived cleaning "
            f"(rows={len(out)}, symbols={len(present)}); "
            f"min_coverage={min_coverage} may be too strict for this window. "
            "Nothing was cached."
        )

    report.returned = tuple(present)
    report.missing = tuple(t for t in tickers if t not in set(present))
    report.rows = len(out)
    if report.missing:
        logger.warning(
            "%s: %d/%d symbols returned; missing %s",
            field, len(present), len(tickers), _preview(report.missing),
        )
    if len(present) < min_symbols:
        raise DataUnavailableError(
            f"only {len(present)} of {len(tickers)} symbols returned usable "
            f"{field} data (min_symbols={min_symbols}). Missing: "
            f"{_preview(report.missing)}. Nothing was cached."
        )

    _atomic_to_parquet(out, cache_path)
    out.attrs["download_report"] = report
    return out


def _preview(items, limit: int = 8) -> str:
    """Render a symbol list for a log line without dumping 500 names."""
    items = list(items)
    head = ", ".join(items[:limit])
    return head if len(items) <= limit else f"{head}, ... (+{len(items) - limit} more)"


def _atomic_to_parquet(df: pd.DataFrame, path: str) -> None:
    """Write via temp file + rename so an interrupted run leaves no half file.

    A truncated parquet at a cache path is worse than no cache: the next run
    reads it and either crashes far from the cause or, worse, proceeds on a
    partial panel.
    """
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        df.to_parquet(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def load_prices(
    tickers: list[str] | None = None,
    start: str = "2010-01-01",
    end: str | None = None,
    cache_dir: str = "data_cache",
    min_coverage: float = 0.9,
    chunk_size: int = 100,
    *,
    min_symbols: int = 1,
    refresh: bool = False,
    max_cache_age_days: float = LATEST_CACHE_MAX_AGE_DAYS,
) -> pd.DataFrame:
    """Return a (date x ticker) DataFrame of adjusted close prices.

    Downloads via yfinance (in chunks, to be polite to the API at large
    universe sizes) and caches to parquet so repeated runs are offline.

    ``min_coverage``: drop columns with less than this fraction of non-NaN
    rows. The 0.9 default suits a static always-listed universe; pass 0.0 for
    point-in-time universes, where names that IPO'd or delisted mid-window
    are EXACTLY the ones survivorship-bias work needs to keep.

    ``min_symbols``: raise ``DataUnavailableError`` rather than return a panel
    thinner than this. A cross-sectional strategy silently run on 3 of 500
    names produces numbers, and all of them are lies.

    ``max_cache_age_days``: only bites when the window is open-ended
    (``end=None`` or an ``end`` in the future), where "latest" means something
    different tomorrow. Closed windows are served from cache forever.

    ``refresh``: bypass the cache and refetch (the cache is then rewritten).

    The result carries a :class:`DownloadReport` in ``df.attrs['download_report']``
    describing missing symbols, dropped columns and cache provenance. NOTE:
    ``attrs`` do not survive a parquet round-trip, so the report is rebuilt on
    a cache hit from what the cached frame actually contains.
    """
    # `None` means "use the default universe"; an explicit [] does NOT -- a
    # point-in-time universe query that came back empty must fail loudly, not
    # silently substitute 60 of today's mega-caps (survivorship bias by typo).
    tickers = DEFAULT_UNIVERSE if tickers is None else tickers
    return _download_field(
        tickers, "Close", start, end, cache_dir, min_coverage, chunk_size,
        min_symbols=min_symbols, refresh=refresh,
        max_cache_age_days=max_cache_age_days,
    )


def load_volumes(
    tickers: list[str] | None = None,
    start: str = "2010-01-01",
    end: str | None = None,
    cache_dir: str = "data_cache",
    chunk_size: int = 100,
    *,
    min_symbols: int = 1,
    refresh: bool = False,
    max_cache_age_days: float = LATEST_CACHE_MAX_AGE_DAYS,
) -> pd.DataFrame:
    """Share volumes (for dollar-ADV / impact modeling), cached like prices.

    No coverage filter: missing volume simply means a name falls back to the
    cross-sectional median ADV inside the impact model (and is counted in
    adv_coverage).
    """
    tickers = DEFAULT_UNIVERSE if tickers is None else tickers
    return _download_field(
        tickers, "Volume", start, end, cache_dir, 0.0, chunk_size,
        min_symbols=min_symbols, refresh=refresh,
        max_cache_age_days=max_cache_age_days,
    )
