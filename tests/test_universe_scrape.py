"""The Wikipedia scrape: parsed by column NAME, and loud when it cannot parse.

Why this file exists: on 2026-08-11 the live cycle started dying at
``fetch_sp500_tables`` with ``ValueError: Length mismatch: Expected axis has 2
elements, new values have 6 elements``, and it kept dying every weekday for
five weeks. Wikipedia had moved the component-changes table to its own page
and given it an extra ``Refs`` column; the parser addressed both tables by
POSITION (``tables[1]``) and asserted a fixed six-column shape. Twenty-five
live cycles, and the H7 borrow snapshots that ride the same job, were lost.

So these are known-answer tests for the parse itself -- no network, HTML
fixtures in both the old and new layouts -- plus the guards that matter more
than the parse: a half-parsed or absent changes table must RAISE. Falling
back to "today's members" would hand the point-in-time universe back the
survivorship bias it exists to remove, and it would do it silently, which is
the only way this project can lose.
"""

import io
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import universe


# ---------------------------------------------------------------------------
# HTML fixtures: the two layouts Wikipedia has actually served
# ---------------------------------------------------------------------------

def _constituents_html(n: int = 503) -> str:
    """The "List of S&P 500 companies" table (flat header, 8 columns)."""
    rows = "".join(
        f"<tr><td>{_nth_ticker(i)}</td><td>Company {i}</td><td>Industrials</td>"
        f"<td>Sub</td><td>Town, ST</td><td>1957-03-04</td><td>{i}</td><td>1902</td></tr>"
        for i in range(n)
    )
    return (
        "<table><tr><th>Symbol</th><th>Security</th><th>GICS Sector</th>"
        "<th>GICS Sub-Industry</th><th>Headquarters Location</th>"
        "<th>Date added</th><th>CIK</th><th>Founded</th></tr>" + rows + "</table>"
    )


def _nth_ticker(i: int) -> str:
    return "T" + chr(65 + i // 676) + chr(65 + (i // 26) % 26) + chr(65 + i % 26)


def _changes_html_new(n: int = 300) -> str:
    """The 2026 layout: own page, two-row header, extra ``Refs`` column."""
    rows = "".join(
        f"<tr><td>March {1 + i % 28}, {2011 + i % 15}</td>"
        f"<td>{_nth_ticker(600 + i)}</td><td>New {i}</td>"
        f"<td>{_nth_ticker(1200 + i)}</td><td>Old {i}</td>"
        f"<td>Market cap change.</td><td>[{i}]</td></tr>"
        for i in range(n)
    )
    return (
        '<table class="wikitable">'
        '<tr><th rowspan="2">Effective Date</th><th colspan="2">Added</th>'
        '<th colspan="2">Removed</th><th rowspan="2">Reason</th>'
        '<th rowspan="2">Refs</th></tr>'
        "<tr><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th></tr>"
        + rows
        + "</table>"
    )


def _changes_html_legacy(n: int = 300) -> str:
    """The pre-2026 layout: same page as the constituents, flat 6-column header."""
    rows = "".join(
        f"<tr><td>March {1 + i % 28}, {2011 + i % 15}</td>"
        f"<td>{_nth_ticker(600 + i)}</td><td>New {i}</td>"
        f"<td>{_nth_ticker(1200 + i)}</td><td>Old {i}</td>"
        f"<td>Market cap change.</td></tr>"
        for i in range(n)
    )
    return (
        "<table><tr><th>Date</th><th>Added Ticker</th><th>Added Security</th>"
        "<th>Removed Ticker</th><th>Removed Security</th><th>Reason</th></tr>"
        + rows
        + "</table>"
    )


_NAVBOX = "<table><tr><th>vteS&P 500 companies</th><th>x</th></tr><tr><td>a</td><td>b</td></tr></table>"


class _FakeResponse:
    def __init__(self, html: str):
        self._html = html

    def read(self) -> bytes:
        return self._html.encode("utf-8")


def _serve(monkeypatch, pages: dict[str, str], seen: list[str] | None = None):
    """Route universe's urlopen to canned HTML, recording the URLs requested."""

    def fake_urlopen(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if seen is not None:
            seen.append(url)
        if url not in pages:
            raise AssertionError(f"unexpected fetch: {url}")
        return _FakeResponse(pages[url])

    monkeypatch.setattr(universe.urllib.request, "urlopen", fake_urlopen)


# ---------------------------------------------------------------------------
# The parse
# ---------------------------------------------------------------------------

def test_new_two_page_layout_is_parsed(monkeypatch, tmp_path):
    # The layout that broke production: changes on their own page, two-row
    # header, and a Refs column the old positional parser never expected.
    seen: list[str] = []
    _serve(
        monkeypatch,
        {
            universe.WIKI_URL: _constituents_html() + _NAVBOX,
            universe.WIKI_CHANGES_URL: _changes_html_new() + _NAVBOX,
        },
        seen,
    )
    current, changes = universe.fetch_sp500_tables(cache_dir=str(tmp_path))

    assert list(current.columns) == ["ticker", "sector"]
    assert len(current) == 503
    assert list(changes.columns) == ["date", "added", "removed", "reason"]
    assert len(changes) == 300
    assert changes["date"].iloc[0] == pd.Timestamp("2011-03-01")
    assert changes["added"].iloc[0] == _nth_ticker(600)
    assert changes["removed"].iloc[0] == _nth_ticker(1200)
    assert changes["reason"].iloc[0] == "Market cap change."
    assert seen == [universe.WIKI_URL, universe.WIKI_CHANGES_URL]


def test_legacy_single_page_layout_still_works(monkeypatch, tmp_path):
    # If Wikipedia moves the table back, no code change should be needed --
    # and the second page must not be fetched at all.
    seen: list[str] = []
    _serve(
        monkeypatch,
        {universe.WIKI_URL: _constituents_html() + _changes_html_legacy() + _NAVBOX},
        seen,
    )
    current, changes = universe.fetch_sp500_tables(cache_dir=str(tmp_path))

    assert len(current) == 503 and len(changes) == 300
    assert seen == [universe.WIKI_URL]  # no fallback fetch


def test_unknown_extra_columns_are_ignored(monkeypatch, tmp_path):
    # The specific regression: a NEW column must not shift or break the parse.
    extra = _changes_html_new().replace(
        "<th rowspan=\"2\">Refs</th>",
        "<th rowspan=\"2\">Refs</th><th rowspan=\"2\">Notes</th>",
    ).replace("<td>[", "<td>note</td><td>[")
    _serve(
        monkeypatch,
        {
            universe.WIKI_URL: _constituents_html(),
            universe.WIKI_CHANGES_URL: extra,
        },
    )
    _, changes = universe.fetch_sp500_tables(cache_dir=str(tmp_path))
    assert list(changes.columns) == ["date", "added", "removed", "reason"]
    assert changes["added"].iloc[0] == _nth_ticker(600)


# ---------------------------------------------------------------------------
# The guards: a bad scrape must never become a quiet result
# ---------------------------------------------------------------------------

def test_missing_changes_table_raises_instead_of_using_todays_members(monkeypatch, tmp_path):
    # THE test in this file. With no changes table, membership reconstruction
    # degenerates to "whoever is in the index today" -- survivorship bias, and
    # invisible in every downstream metric. It must be a crash, not a default.
    _serve(
        monkeypatch,
        {
            universe.WIKI_URL: _constituents_html() + _NAVBOX,
            universe.WIKI_CHANGES_URL: _NAVBOX,
        },
    )
    with pytest.raises(ValueError, match="survivorship bias"):
        universe.fetch_sp500_tables(cache_dir=str(tmp_path))


def test_short_changes_table_raises(monkeypatch, tmp_path):
    # A half-served table narrows the universe toward today's survivors by
    # degrees, which is worse than losing it outright: the run still finishes.
    _serve(
        monkeypatch,
        {
            universe.WIKI_URL: _constituents_html(),
            universe.WIKI_CHANGES_URL: _changes_html_new(n=20),
        },
    )
    with pytest.raises(ValueError, match="rows since 2010"):
        universe.fetch_sp500_tables(cache_dir=str(tmp_path))


def test_absurd_member_count_raises(monkeypatch, tmp_path):
    _serve(
        monkeypatch,
        {
            universe.WIKI_URL: _constituents_html(n=12),
            universe.WIKI_CHANGES_URL: _changes_html_new(),
        },
    )
    with pytest.raises(ValueError, match="outside the sane"):
        universe.fetch_sp500_tables(cache_dir=str(tmp_path))


def test_unparseable_dates_raise(monkeypatch, tmp_path):
    broken = _changes_html_new().replace("March", "Marbleuary")
    _serve(
        monkeypatch,
        {universe.WIKI_URL: _constituents_html(), universe.WIKI_CHANGES_URL: broken},
    )
    with pytest.raises(ValueError, match="parsed as dates"):
        universe.fetch_sp500_tables(cache_dir=str(tmp_path))


def test_missing_constituents_table_raises(monkeypatch, tmp_path):
    _serve(monkeypatch, {universe.WIKI_URL: _NAVBOX})
    with pytest.raises(ValueError, match="no S&P 500 constituents table"):
        universe.fetch_sp500_tables(cache_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# Ticker cells
# ---------------------------------------------------------------------------

def test_ticker_normalization_and_wiki_markup_recovery():
    assert universe._normalize_ticker("BRK.B") == "BRK-B"
    assert universe._normalize_ticker(" aapl ") == "AAPL"
    # Real cell in the live changes table, from a malformed template.
    assert universe._normalize_ticker("ALLE |") == "ALLE"


def test_junk_ticker_cell_raises_rather_than_entering_the_universe():
    # A junk symbol would silently become a member name that matches no price
    # series -- i.e. a hole in the universe, reported as "no data".
    with pytest.raises(ValueError, match="unparseable ticker cell"):
        universe._normalize_ticker("— (see note)")


# ---------------------------------------------------------------------------
# Column selection helper
# ---------------------------------------------------------------------------

def test_flat_columns_dedupes_spanned_header_cells():
    df = pd.read_html(
        io.StringIO(
            '<table><tr><th rowspan="2">Effective Date</th>'
            '<th colspan="2">Added</th></tr>'
            "<tr><th>Ticker</th><th>Security</th></tr>"
            "<tr><td>x</td><td>y</td><td>z</td></tr></table>"
        )
    )[0]
    # pandas repeats the label of a spanned header cell; "effective date
    # effective date" would not match a plain "date" lookup by equality, so
    # the dedupe is load-bearing.
    assert universe._flat_columns(df) == ["effective date", "added ticker", "added security"]


# ---------------------------------------------------------------------------
# The second copy of the bug: the SEC name crosswalk
# ---------------------------------------------------------------------------

def test_crosswalk_names_come_from_both_pages(monkeypatch, tmp_path):
    # fetch_sp500_security_names carried its own positional copy of the same
    # parse. It feeds DEAD companies' names to the SEC CIK crosswalk -- the
    # machinery that unblocked H1's survivorship problem -- so the identical
    # crash was sitting one cold cache away from the fundamentals path.
    from quantlab import cik_crosswalk

    _serve(
        monkeypatch,
        {
            universe.WIKI_URL: _constituents_html(n=503) + _NAVBOX,
            universe.WIKI_CHANGES_URL: _changes_html_new(n=300),
        },
    )
    names = cik_crosswalk.fetch_sp500_security_names(cache_dir=str(tmp_path))

    assert names[_nth_ticker(0)] == "Company 0"        # a current member
    assert names[_nth_ticker(1200)] == "Old 0"          # a REMOVED (dead) name
    assert names[_nth_ticker(600)] == "New 0"           # an added name
    assert len(names) == 503 + 600


def test_crosswalk_refuses_current_members_only(monkeypatch, tmp_path):
    # Without the changes table there are no dead-company names, and a
    # crosswalk built from survivors alone is the bias it exists to avoid.
    from quantlab import cik_crosswalk

    _serve(
        monkeypatch,
        {
            universe.WIKI_URL: _constituents_html() + _NAVBOX,
            universe.WIKI_CHANGES_URL: _NAVBOX,
        },
    )
    with pytest.raises(ValueError, match="survivorship-safe"):
        cik_crosswalk.fetch_sp500_security_names(cache_dir=str(tmp_path))


def test_changes_frame_keeps_names_only_when_asked():
    tables = pd.read_html(io.StringIO(_changes_html_new(n=3)))
    plain = universe._find_changes_table(tables)
    named = universe._find_changes_table(tables, with_names=True)
    assert set(plain.columns) == {"date", "added", "removed", "reason"}
    assert set(named.columns) == {
        "date", "added", "removed", "reason", "added_name", "removed_name"
    }
