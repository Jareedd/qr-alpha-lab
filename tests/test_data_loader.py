"""Data-loader failure modes: what happens when the vendor misbehaves.

No network. A fake ``yfinance`` module is installed into ``sys.modules`` for
the duration of each test, so every scenario below is a known-answer test of
OUR handling, not of yfinance's mood.

These tests exist because the loader is the one place where a silent wrong
answer is indistinguishable from a real research finding: an empty panel
becomes an empty feature matrix becomes "no edge found", which is exactly the
conclusion this project reports honestly elsewhere. It must never be an
artifact of a rate limit.
"""

import logging
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import data
from quantlab.data import DataUnavailableError


# --------------------------------------------------------------------------
# Fake vendor
# --------------------------------------------------------------------------
def _multi_frame(tickers, dates, fields=("Close", "Volume")):
    """A yfinance-shaped MultiIndex response: columns (field, ticker)."""
    cols = pd.MultiIndex.from_product([list(fields), list(tickers)])
    rng = np.random.default_rng(0)
    vals = rng.normal(100, 1, size=(len(dates), len(cols)))
    return pd.DataFrame(vals, index=pd.DatetimeIndex(dates), columns=cols)


class FakeYF:
    """Records calls and replays a scripted response per chunk."""

    def __init__(self, responder):
        self._responder = responder
        self.calls = []

    def download(self, chunk, start=None, end=None, auto_adjust=True, progress=False):
        chunk = [chunk] if isinstance(chunk, str) else list(chunk)
        self.calls.append(chunk)
        return self._responder(chunk, start, end)


@pytest.fixture
def fake_yf(monkeypatch):
    def _install(responder):
        mod = types.ModuleType("yfinance")
        fake = FakeYF(responder)
        mod.download = fake.download
        monkeypatch.setitem(sys.modules, "yfinance", mod)
        return fake

    return _install


DATES = pd.bdate_range("2024-01-01", periods=40)


def _all_good(chunk, start, end):
    return _multi_frame(chunk, DATES)


# --------------------------------------------------------------------------
# 1. Every download returns empty -- the bug that motivated this file
# --------------------------------------------------------------------------
def test_all_downloads_empty_raises_actionable_error_and_caches_nothing(fake_yf, tmp_path):
    fake_yf(lambda chunk, start, end: pd.DataFrame())
    with pytest.raises(DataUnavailableError) as exc:
        data.load_prices(["AAPL", "MSFT"], start="2024-01-01", end="2024-02-01",
                         cache_dir=str(tmp_path))
    msg = str(exc.value)
    # The message must name what was asked for; "No objects to concatenate"
    # (the old pandas failure) named nothing.
    assert "AAPL" in msg and "2024-01-01" in msg and "Close" in msg
    # A failed download must not leave a cache file behind: a cached empty
    # panel would make every later run fail in a *different*, quieter way.
    assert list(tmp_path.iterdir()) == []


def test_none_response_is_treated_as_empty(fake_yf, tmp_path):
    fake_yf(lambda chunk, start, end: None)
    with pytest.raises(DataUnavailableError):
        data.load_prices(["AAPL"], start="2024-01-01", end="2024-02-01",
                         cache_dir=str(tmp_path))


# --------------------------------------------------------------------------
# 2. Partial downloads and missing symbols
# --------------------------------------------------------------------------
def test_partial_chunk_failure_returns_survivors_and_reports_missing(fake_yf, tmp_path, caplog):
    def responder(chunk, start, end):
        if "MSFT" in chunk:                       # second chunk fails entirely
            return pd.DataFrame()
        return _multi_frame(chunk, DATES)

    fake_yf(responder)
    with caplog.at_level(logging.WARNING, logger="quantlab.data"):
        px = data.load_prices(["AAPL", "MSFT"], start="2024-01-01", end="2024-02-01",
                              cache_dir=str(tmp_path), chunk_size=1)
    assert list(px.columns) == ["AAPL"]
    rep = px.attrs["download_report"]
    assert rep.missing == ("MSFT",) and rep.returned == ("AAPL",)
    assert rep.chunks_empty == 1 and rep.chunks_total == 2
    assert rep.coverage == pytest.approx(0.5)
    assert "MSFT" in caplog.text                 # the loss is announced, not silent


def test_symbol_missing_from_an_otherwise_good_response_is_reported(fake_yf, tmp_path):
    # The vendor returns the chunk but simply omits a delisted/renamed name.
    fake_yf(lambda chunk, start, end: _multi_frame([t for t in chunk if t != "DEAD"], DATES))
    px = data.load_prices(["AAPL", "DEAD", "MSFT"], start="2024-01-01",
                          end="2024-02-01", cache_dir=str(tmp_path))
    assert list(px.columns) == ["AAPL", "MSFT"]
    assert px.attrs["download_report"].missing == ("DEAD",)


def test_min_symbols_turns_a_too_partial_panel_into_a_hard_failure(fake_yf, tmp_path):
    fake_yf(lambda chunk, start, end: _multi_frame([t for t in chunk if t == "AAPL"], DATES))
    with pytest.raises(DataUnavailableError, match="min_symbols"):
        data.load_prices(["AAPL", "MSFT", "GOOGL"], start="2024-01-01",
                         end="2024-02-01", cache_dir=str(tmp_path), min_symbols=3)
    assert list(tmp_path.iterdir()) == []        # and still caches nothing


def test_flat_columns_for_a_multi_ticker_chunk_are_refused_not_guessed(fake_yf, tmp_path):
    # A flat column index for a 3-ticker request gives no way to know whose
    # prices these are. Attaching them to the wrong name would be a silent
    # catastrophe, so the chunk is refused.
    def responder(chunk, start, end):
        return pd.DataFrame({"Close": np.arange(len(DATES), dtype=float)}, index=DATES)

    fake_yf(responder)
    with pytest.raises(DataUnavailableError) as exc:
        data.load_prices(["AAPL", "MSFT", "GOOGL"], start="2024-01-01",
                         end="2024-02-01", cache_dir=str(tmp_path))
    assert "unusable" in str(exc.value)


def test_flat_columns_for_a_single_ticker_chunk_are_labelled_correctly(fake_yf, tmp_path):
    def responder(chunk, start, end):
        return pd.DataFrame({"Close": np.arange(len(DATES), dtype=float)}, index=DATES)

    fake_yf(responder)
    px = data.load_prices(["AAPL", "MSFT"], start="2024-01-01", end="2024-02-01",
                          cache_dir=str(tmp_path), chunk_size=1)
    assert list(px.columns) == ["AAPL", "MSFT"]


# --------------------------------------------------------------------------
# 3. Stale caches when the requested end date is "latest"
# --------------------------------------------------------------------------
def test_open_ended_cache_older_than_the_window_is_refetched(fake_yf, tmp_path):
    """The six-week bug: end=None keys on 'latest' and was served forever."""
    fake = fake_yf(_all_good)
    px1 = data.load_prices(["AAPL"], start="2024-01-01", end=None, cache_dir=str(tmp_path))
    assert px1.attrs["download_report"].source == "download"
    cache_file = next(iter(tmp_path.iterdir()))

    # Same day -> cache is legitimately fresh, no second download.
    px2 = data.load_prices(["AAPL"], start="2024-01-01", end=None, cache_dir=str(tmp_path))
    assert px2.attrs["download_report"].source == "cache"
    assert len(fake.calls) == 1

    # Age the file by three days: "latest" now means something else.
    old = os.path.getmtime(cache_file) - 3 * 86400
    os.utime(cache_file, (old, old))
    px3 = data.load_prices(["AAPL"], start="2024-01-01", end=None, cache_dir=str(tmp_path))
    assert px3.attrs["download_report"].source == "download"
    assert len(fake.calls) == 2


def test_closed_window_cache_never_expires(fake_yf, tmp_path):
    # A fixed past end date is a frozen question; a year-old answer is still
    # the right answer, and refetching it would only burn API quota.
    fake = fake_yf(_all_good)
    data.load_prices(["AAPL"], start="2024-01-01", end="2024-02-01", cache_dir=str(tmp_path))
    cache_file = next(iter(tmp_path.iterdir()))
    old = os.path.getmtime(cache_file) - 400 * 86400
    os.utime(cache_file, (old, old))
    px = data.load_prices(["AAPL"], start="2024-01-01", end="2024-02-01",
                          cache_dir=str(tmp_path))
    assert px.attrs["download_report"].source == "cache"
    assert len(fake.calls) == 1


def test_future_end_date_is_treated_as_open_ended(fake_yf, tmp_path):
    # An 'end' past today leaves the right edge just as unfinished as end=None.
    future = (pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
              + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    fake = fake_yf(_all_good)
    data.load_prices(["AAPL"], start="2024-01-01", end=future, cache_dir=str(tmp_path))
    cache_file = next(iter(tmp_path.iterdir()))
    old = os.path.getmtime(cache_file) - 3 * 86400
    os.utime(cache_file, (old, old))
    px = data.load_prices(["AAPL"], start="2024-01-01", end=future, cache_dir=str(tmp_path))
    assert px.attrs["download_report"].source == "download"
    assert len(fake.calls) == 2


def test_refresh_bypasses_a_fresh_cache(fake_yf, tmp_path):
    fake = fake_yf(_all_good)
    data.load_prices(["AAPL"], start="2024-01-01", end="2024-02-01", cache_dir=str(tmp_path))
    data.load_prices(["AAPL"], start="2024-01-01", end="2024-02-01",
                     cache_dir=str(tmp_path), refresh=True)
    assert len(fake.calls) == 2


def test_cache_hit_reports_missing_symbols_and_age(fake_yf, tmp_path):
    # attrs do not survive parquet, so the report must be rebuilt from the
    # cached frame -- including the fact that a name is absent from it.
    fake_yf(lambda chunk, start, end: _multi_frame([t for t in chunk if t != "DEAD"], DATES))
    data.load_prices(["AAPL", "DEAD"], start="2024-01-01", end="2024-02-01",
                     cache_dir=str(tmp_path))
    px = data.load_prices(["AAPL", "DEAD"], start="2024-01-01", end="2024-02-01",
                          cache_dir=str(tmp_path))
    rep = px.attrs["download_report"]
    assert rep.source == "cache" and rep.missing == ("DEAD",)
    assert rep.cache_age_days is not None and rep.cache_age_days >= 0


# --------------------------------------------------------------------------
# 4. Duplicate timestamps and invalid inputs
# --------------------------------------------------------------------------
def test_duplicate_timestamps_are_deduplicated_last_wins_and_counted(fake_yf, tmp_path):
    def responder(chunk, start, end):
        frame = _multi_frame(chunk, DATES)
        dup = frame.iloc[[5]].copy()
        dup.iloc[0, :] = 999.0                    # the corrected re-send
        return pd.concat([frame, dup])            # appended last, as a vendor would

    fake_yf(responder)
    px = data.load_prices(["AAPL", "MSFT"], start="2024-01-01", end="2024-02-01",
                          cache_dir=str(tmp_path))
    assert px.index.is_unique and px.index.is_monotonic_increasing
    assert px.loc[DATES[5], "AAPL"] == 999.0      # last copy wins
    assert px.attrs["download_report"].duplicate_rows_dropped == 1


def test_duplicate_timestamps_across_chunks_do_not_break_the_concat(fake_yf, tmp_path):
    # Pre-fix, a duplicated index reached pd.concat(axis=1) and took down the
    # whole download with InvalidIndexError -- one bad chunk, zero data.
    def responder(chunk, start, end):
        frame = _multi_frame(chunk, DATES)
        if "MSFT" in chunk:
            frame = pd.concat([frame, frame.iloc[[3]]])
        return frame

    fake_yf(responder)
    px = data.load_prices(["AAPL", "MSFT"], start="2024-01-01", end="2024-02-01",
                          cache_dir=str(tmp_path), chunk_size=1)
    assert list(px.columns) == ["AAPL", "MSFT"] and px.index.is_unique


def test_timezone_aware_index_is_normalized_to_naive_dates(fake_yf, tmp_path):
    def responder(chunk, start, end):
        frame = _multi_frame(chunk, DATES)
        frame.index = frame.index.tz_localize("America/New_York")
        return frame

    fake_yf(responder)
    px = data.load_prices(["AAPL"], start="2024-01-01", end="2024-02-01",
                          cache_dir=str(tmp_path))
    assert px.index.tz is None
    assert (px.index == px.index.normalize()).all()


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"tickers": []}, "empty"),
        ({"tickers": "AAPL"}, "bare string"),
        ({"tickers": ["AAPL", ""]}, "non-empty"),
        ({"tickers": ["AAPL", 7]}, "strings"),
        ({"tickers": ["AAPL"], "start": "not-a-date"}, "start is not a parseable date"),
        ({"tickers": ["AAPL"], "end": "not-a-date"}, "end is not a parseable date"),
        ({"tickers": ["AAPL"], "start": "2024-06-01", "end": "2024-01-01"}, "before start"),
        ({"tickers": ["AAPL"], "min_coverage": 1.5}, "min_coverage"),
        ({"tickers": ["AAPL"], "chunk_size": 0}, "chunk_size"),
    ],
)
def test_invalid_inputs_raise_before_any_network_or_disk_access(fake_yf, tmp_path, kwargs, match):
    fake = fake_yf(_all_good)
    kwargs.setdefault("start", "2024-01-01")
    kwargs.setdefault("end", "2024-02-01")
    with pytest.raises(ValueError, match=match):
        data.load_prices(cache_dir=str(tmp_path), **kwargs)
    assert fake.calls == []                       # never hit the vendor
    assert not tmp_path.exists() or list(tmp_path.iterdir()) == []


def test_duplicate_tickers_in_the_request_are_collapsed(fake_yf, tmp_path):
    fake_yf(_all_good)
    px = data.load_prices(["AAPL", "AAPL", "MSFT"], start="2024-01-01",
                          end="2024-02-01", cache_dir=str(tmp_path))
    assert list(px.columns) == ["AAPL", "MSFT"]
    assert px.attrs["download_report"].requested == ("AAPL", "MSFT")


# --------------------------------------------------------------------------
# 5. Reruns: idempotent, not duplicating or corrupting
# --------------------------------------------------------------------------
def test_rerun_returns_identical_data_and_does_not_grow_the_panel(fake_yf, tmp_path):
    fake_yf(_all_good)
    first = data.load_prices(["AAPL", "MSFT"], start="2024-01-01", end="2024-02-01",
                             cache_dir=str(tmp_path))
    second = data.load_prices(["AAPL", "MSFT"], start="2024-01-01", end="2024-02-01",
                              cache_dir=str(tmp_path))
    pd.testing.assert_frame_equal(first, second)
    assert first.shape == second.shape            # no row/column accretion
    assert len(list(tmp_path.iterdir())) == 1     # one cache file, not two


def test_column_order_follows_the_request_not_the_chunking(fake_yf, tmp_path):
    # Deterministic column order is what makes two runs byte-comparable.
    fake_yf(_all_good)
    px_a = data.load_prices(["MSFT", "AAPL", "GOOGL"], start="2024-01-01",
                            end="2024-02-01", cache_dir=str(tmp_path), chunk_size=1)
    px_b = data.load_prices(["MSFT", "AAPL", "GOOGL"], start="2024-01-01",
                            end="2024-02-01", cache_dir=str(tmp_path / "b"), chunk_size=3)
    assert list(px_a.columns) == ["MSFT", "AAPL", "GOOGL"] == list(px_b.columns)


def test_a_corrupt_cache_file_is_refetched_not_fatal(fake_yf, tmp_path):
    fake = fake_yf(_all_good)
    data.load_prices(["AAPL"], start="2024-01-01", end="2024-02-01", cache_dir=str(tmp_path))
    cache_file = next(iter(tmp_path.iterdir()))
    cache_file.write_bytes(b"PAR1 truncated garbage")   # an interrupted write
    px = data.load_prices(["AAPL"], start="2024-01-01", end="2024-02-01",
                          cache_dir=str(tmp_path))
    assert len(fake.calls) == 2 and not px.empty
    assert px.attrs["download_report"].source == "download"


def test_no_temp_files_survive_a_successful_write(fake_yf, tmp_path):
    fake_yf(_all_good)
    data.load_prices(["AAPL"], start="2024-01-01", end="2024-02-01", cache_dir=str(tmp_path))
    assert [p.name for p in tmp_path.iterdir() if ".tmp." in p.name] == []


def test_a_failed_write_leaves_no_partial_cache(fake_yf, tmp_path, monkeypatch):
    fake_yf(_all_good)

    def boom(self, *a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
    with pytest.raises(OSError):
        data.load_prices(["AAPL"], start="2024-01-01", end="2024-02-01",
                         cache_dir=str(tmp_path))
    assert list(tmp_path.iterdir()) == []          # no .tmp, no half parquet


def test_volumes_take_the_same_hardened_path(fake_yf, tmp_path):
    fake_yf(lambda chunk, start, end: pd.DataFrame())
    with pytest.raises(DataUnavailableError, match="Volume"):
        data.load_volumes(["AAPL"], start="2024-01-01", end="2024-02-01",
                          cache_dir=str(tmp_path))


def test_download_report_serializes_for_failure_reporting(fake_yf, tmp_path):
    fake_yf(lambda chunk, start, end: _multi_frame([t for t in chunk if t != "DEAD"], DATES))
    px = data.load_prices(["AAPL", "DEAD"], start="2024-01-01", end="2024-02-01",
                          cache_dir=str(tmp_path))
    d = px.attrs["download_report"].as_dict()
    import json
    assert json.loads(json.dumps(d))["missing"] == ["DEAD"]
    assert d["requested"] == 2 and d["returned"] == 1
