"""Point-in-time S&P 500 membership, reconstructed from Wikipedia.

Why this module exists: backtesting on *today's* index members lets the long
side free-ride on hindsight -- every name is known to have survived (and
usually thrived). That is survivorship bias, and it silently inflates returns
(McLean-Pontiff-scale effects). The fix is to know who was in the index ON
each historical date and only trade those names at that time.

Method: Wikipedia maintains (a) the current constituent table on "List of
S&P 500 companies" and (b) a changes table (effective date, ticker added,
ticker removed) going back decades -- since ~2026-08 on its own page,
"Historical components of the S&P 500". Starting from today's membership and
walking the changes BACKWARD (undo each addition, redo each removal) yields
the membership set over any past interval. Change effective dates are
announced in advance, so gating trades by effective date is point-in-time
safe.

Honest limitations (do not delete -- quantify):
- The changes table is community-maintained: dense and reliable for recent
  decades, sparser before ~2000. Keep backtest start >= 2005 (we use 2010).
- It is also not a point-in-time vendor feed: its layout moves under us (it
  has broken this parser twice), and departures can go unrecorded. Measured
  2026-09-15: of the 498 names the live book scored on 2026-08-10, exactly
  one (EQR, renamed in an acquisition) is absent from BOTH today's members
  and the changes table, so it vanishes from the reconstructed universe
  without a trace -- a 0.2% silent survivorship hole in this scrape. The
  parser refuses a half-parsed table (see ``_assert_tables_sane``) but it
  cannot conjure rows Wikipedia never had. A paid PIT membership vendor
  (CRSP) is the only real fix; this is why the write-up calls the universe
  "point-in-time-ish".
- A point-in-time *membership mask* does not conjure up price data for dead
  companies: names removed via bankruptcy or acquisition often have no Yahoo
  history. ``coverage_report`` counts exactly how many member-names lack
  price data so the residual bias is a number in the write-up, not a secret.
- Ticker reuse across decades (same symbol, different company) is not
  resolved; it is rare within a post-2010 window.
"""

from __future__ import annotations

import io
import os
import re
import urllib.request

import pandas as pd

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
# Wikipedia moved the component-changes table OUT of the constituents article
# (the live cycle started crashing on it 2026-08-11; the constituents page now
# serves only the current-membership table). The changes live in their own
# article, which the constituents page's editor notes point to. We still look
# for the changes table on the constituents page first, so an upstream revert
# needs no code change here.
WIKI_CHANGES_URL = "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"
_UA = "qr-alpha-lab/0.1 (student research project)"

# Sanity floors. A community-maintained page is not a point-in-time vendor:
# it can be re-sectioned, re-columned, or served half-parsed at any time, and
# the dangerous failure is the SILENT one -- a short changes table quietly
# reconstructs a membership set that is too close to today's, i.e. it puts
# survivorship bias back into the universe that exists to remove it. These
# floors turn that into a loud crash. They are lower bounds on quantities
# that only ever grow (observed 2026-09: 503 members, 354 post-2010 changes).
MIN_CURRENT_MEMBERS = 450
MAX_CURRENT_MEMBERS = 550
MIN_CHANGE_ROWS_SINCE_2010 = 250
MIN_PARSED_DATE_FRAC = 0.98

_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9-]{0,6}$")


def _normalize_ticker(t: str) -> str:
    """Wikipedia uses BRK.B / BF.B; yfinance wants BRK-B / BF-B.

    Also strips leftover wiki markup: the changes table carries occasional
    cells like ``"ALLE |"`` from a malformed template. Taking the first
    whitespace/pipe-delimited token recovers the symbol; anything that still
    fails ``_TICKER_RE`` is a parse error, not a ticker, and raises rather
    than entering the membership set as a junk name.
    """
    if t is None or (isinstance(t, float) and pd.isna(t)):
        raise ValueError("empty ticker cell from Wikipedia")
    raw = str(t).strip()
    token = re.split(r"[\s|]+", raw)[0].upper().replace(".", "-")
    if not _TICKER_RE.match(token):
        raise ValueError(f"unparseable ticker cell from Wikipedia: {raw!r}")
    return token


def _flat_columns(df: pd.DataFrame) -> list[str]:
    """Lowercased, whitespace-joined column labels (MultiIndex-safe).

    Wikipedia's changes table uses a two-row header (Added/Removed over
    Ticker/Security); the constituents table uses a flat one. Selecting
    columns by NAME rather than by position is what makes this parser
    survive the upstream edits that have already broken it twice.
    """
    out = []
    for col in df.columns:
        parts = [str(x) for x in col] if isinstance(col, tuple) else [str(col)]
        # A pandas-flattened MultiIndex repeats the label when the header
        # cell spans both rows ("Effective Date Effective Date").
        dedup = [p for i, p in enumerate(parts) if i == 0 or p != parts[i - 1]]
        out.append(" ".join(dedup).strip().lower())
    return out


def _read_tables(url: str) -> list[pd.DataFrame]:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8")
    return pd.read_html(io.StringIO(html))


def _find_current_table(tables: list[pd.DataFrame]) -> pd.DataFrame | None:
    """The constituents table: has a Symbol column and a GICS Sector column."""
    for tbl in tables:
        cols = _flat_columns(tbl)
        if any(c == "symbol" for c in cols) and any("gics sector" in c for c in cols):
            out = tbl.copy()
            out.columns = cols
            return out
    return None


def _find_changes_table(
    tables: list[pd.DataFrame], with_names: bool = False
) -> pd.DataFrame | None:
    """The changes table: a date column plus added/removed ticker columns.

    Tolerates extra columns (the 2026 layout added a ``Refs`` column, which
    is exactly what broke the old positional parser) and either header shape.
    ``with_names`` also keeps the company-name columns -- the free source of
    DEAD companies' names, which the SEC CIK crosswalk depends on.
    """
    for tbl in tables:
        cols = _flat_columns(tbl)
        date_c = next((c for c in cols if "date" in c), None)
        added_c = next((c for c in cols if c.startswith("added") and "ticker" in c), None)
        removed_c = next((c for c in cols if c.startswith("removed") and "ticker" in c), None)
        if date_c and added_c and removed_c:
            out = tbl.copy()
            out.columns = cols
            reason_c = next((c for c in cols if "reason" in c), None)
            keep = {date_c: "date", added_c: "added", removed_c: "removed"}
            if reason_c:
                keep[reason_c] = "reason"
            if with_names:
                for pre in ("added", "removed"):
                    nm = next(
                        (c for c in cols if c.startswith(pre) and "security" in c), None
                    )
                    if nm:
                        keep[nm] = f"{pre}_name"
            out = out[list(keep)].rename(columns=keep)
            for optional in ("reason", *(("added_name", "removed_name") if with_names else ())):
                if optional not in out.columns:
                    out[optional] = None
            return out
    return None


def fetch_changes_frame(
    with_names: bool = False, tables: list[pd.DataFrame] | None = None
) -> tuple[pd.DataFrame | None, str]:
    """(raw changes frame, the URL it came from), constituents page first.

    Shared by every caller that needs the changes table, so the two-page
    fallback and the name-based column selection have exactly one
    implementation. A second copy of this logic in ``cik_crosswalk`` is what
    left the SEC name-crosswalk carrying the same latent crash that took the
    live cycle down for five weeks. ``tables`` passes in an already-fetched
    constituents page so the caller does not request it twice.
    """
    raw = _find_changes_table(
        tables if tables is not None else _read_tables(WIKI_URL), with_names=with_names
    )
    if raw is not None:
        return raw, WIKI_URL
    return (
        _find_changes_table(_read_tables(WIKI_CHANGES_URL), with_names=with_names),
        WIKI_CHANGES_URL,
    )


def fetch_sp500_tables(cache_dir: str = "data_cache") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (current_members, changes) from Wikipedia, cached locally.

    current_members: columns [ticker]; changes: columns [date, added,
    removed, reason] with one row per ticker movement (a single
    announcement covering an add and a removal becomes one row with both
    filled). ``reason`` is Wikipedia's free-text change rationale --
    retained for the H8 removal-reason census; it is descriptive text and
    never feeds a feature.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cur_path = os.path.join(cache_dir, "sp500_current.parquet")
    chg_path = os.path.join(cache_dir, "sp500_changes.parquet")
    if os.path.exists(cur_path) and os.path.exists(chg_path):
        current = pd.read_parquet(cur_path)
        changes = pd.read_parquet(chg_path)
        # Stale-cache checks: pre-sector caches lack 'sector'; pre-H8
        # caches lack 'reason'. Either -> refetch.
        if "sector" in current.columns and "reason" in changes.columns:
            return current, changes

    tables = _read_tables(WIKI_URL)
    cur_raw = _find_current_table(tables)
    if cur_raw is None:
        raise ValueError(
            f"no S&P 500 constituents table (Symbol + GICS Sector) at {WIKI_URL}; "
            f"tables found: {[_flat_columns(t) for t in tables]}"
        )
    sector_col = next(c for c in cur_raw.columns if "gics sector" in c)
    current = pd.DataFrame(
        {
            "ticker": cur_raw["symbol"].map(_normalize_ticker),
            "sector": cur_raw[sector_col].astype(str),
        }
    )

    # The changes table used to live on the constituents page and now lives on
    # its own; look on both, in that order, before giving up.
    raw, changes_url = fetch_changes_frame(tables=tables)
    if raw is None:
        raise ValueError(
            "no S&P 500 changes table (date + added/removed tickers) at "
            f"{WIKI_URL} or {WIKI_CHANGES_URL}. The point-in-time universe "
            "cannot be rebuilt without it; refusing to fall back to today's "
            "members, which would be survivorship bias by construction."
        )

    dates = pd.to_datetime(raw["date"], format="%B %d, %Y", errors="coerce")
    unparsed = dates.isna() & raw["date"].notna()
    if unparsed.any():  # tolerate stray formats, but only via an explicit pass
        dates = dates.fillna(pd.to_datetime(raw["date"], errors="coerce", format="mixed"))
    parsed_frac = float(dates.notna().mean()) if len(raw) else 0.0
    if parsed_frac < MIN_PARSED_DATE_FRAC:
        raise ValueError(
            f"only {parsed_frac:.1%} of change dates at {changes_url} parsed as dates "
            f"(need >= {MIN_PARSED_DATE_FRAC:.0%}); the date column format changed. "
            f"Examples: {raw['date'][dates.isna()].head(3).tolist()}"
        )

    changes = pd.DataFrame(
        {
            "date": dates,
            "added": raw["added"].map(lambda t: _normalize_ticker(t) if pd.notna(t) else None),
            "removed": raw["removed"].map(lambda t: _normalize_ticker(t) if pd.notna(t) else None),
            "reason": raw["reason"].map(lambda r: str(r).strip() if pd.notna(r) else None),
        }
    ).dropna(subset=["date"])

    _assert_tables_sane(current, changes, changes_url)

    current.to_parquet(cur_path)
    changes.to_parquet(chg_path)
    return current, changes


def _assert_tables_sane(current: pd.DataFrame, changes: pd.DataFrame, changes_url: str) -> None:
    """Refuse a half-parsed scrape rather than silently narrowing the universe.

    Every check here guards a failure that would otherwise look like a
    result: too few members or too few changes both bias the reconstructed
    membership toward today's survivors, and a book built on that reports an
    edge that is hindsight.
    """
    n_cur = len(current)
    if not MIN_CURRENT_MEMBERS <= n_cur <= MAX_CURRENT_MEMBERS:
        raise ValueError(
            f"S&P 500 constituents scrape returned {n_cur} names, outside the sane "
            f"[{MIN_CURRENT_MEMBERS}, {MAX_CURRENT_MEMBERS}] band ({WIKI_URL})"
        )
    if current["ticker"].duplicated().any():
        dupes = sorted(current.loc[current["ticker"].duplicated(), "ticker"])
        raise ValueError(f"duplicate tickers in constituents scrape: {dupes}")

    n_recent = int((changes["date"] >= pd.Timestamp("2010-01-01")).sum())
    if n_recent < MIN_CHANGE_ROWS_SINCE_2010:
        raise ValueError(
            f"changes table at {changes_url} has only {n_recent} rows since 2010 "
            f"(need >= {MIN_CHANGE_ROWS_SINCE_2010}). A short changes table rebuilds a "
            "membership set too close to today's -- survivorship bias, silently."
        )
    if changes[["added", "removed"]].notna().sum().sum() == 0:
        raise ValueError(f"changes table at {changes_url} parsed no tickers at all")


def build_membership_intervals(
    current: pd.DataFrame,
    changes: pd.DataFrame,
    start: str = "2010-01-01",
) -> list[tuple[pd.Timestamp, pd.Timestamp, frozenset]]:
    """Reconstruct membership backward from today.

    Returns a list of (interval_start, interval_end, members) covering
    [start, far-future], where each interval has constant membership and
    interval_start is inclusive, interval_end exclusive. Walking backward:
    before a change date, the added ticker was NOT a member and the removed
    ticker WAS.
    """
    start_ts = pd.Timestamp(start)
    members = set(current["ticker"])
    chg = changes.sort_values("date", ascending=False)

    intervals = []
    upper = pd.Timestamp("2262-01-01")  # effectively +inf for daily data
    for date, grp in chg.groupby("date", sort=False):
        if date <= start_ts:
            break
        intervals.append((date, upper, frozenset(members)))
        upper = date
        for added in grp["added"].dropna():
            members.discard(added)
        for removed in grp["removed"].dropna():
            members.add(removed)
    intervals.append((start_ts, upper, frozenset(members)))
    return intervals[::-1]  # chronological order


def membership_mask(
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
    intervals: list[tuple[pd.Timestamp, pd.Timestamp, frozenset]],
) -> pd.DataFrame:
    """Boolean (date x ticker) frame: was this name in the index on this date?"""
    mask = pd.DataFrame(False, index=dates, columns=tickers)
    valid = set(tickers)
    for lo, hi, members in intervals:
        cols = sorted(members & valid)
        if cols:
            mask.loc[(dates >= lo) & (dates < hi), cols] = True
    return mask


def all_members_in_window(
    intervals: list[tuple[pd.Timestamp, pd.Timestamp, frozenset]],
) -> list[str]:
    """Every ticker that was a member at any point in the window."""
    names: set[str] = set()
    for _, _, members in intervals:
        names |= members
    return sorted(names)


def sector_map(current: pd.DataFrame, tickers: list[str]) -> dict[str, str]:
    """ticker -> GICS sector, 'UNKNOWN' for names not in the current table.

    Honest limitation: Wikipedia only carries sectors for *current* members,
    so departed names get UNKNOWN and form their own neutralization bucket.
    Sectors are also as-of-today (companies occasionally reclassify); a
    point-in-time GICS history needs paid data.
    """
    known = dict(zip(current["ticker"], current.get("sector", pd.Series(dtype=str))))
    return {t: known.get(t, "UNKNOWN") for t in tickers}


def classify_removal_reason(reason: str | None) -> str:
    """Bucket Wikipedia's free-text removal rationale for the H8 census.

    Buckets (H8's spec separates corporate actions and index migrations
    from genuinely discretionary committee deletions — Greenwood–Sammon's
    decomposition says migrations drove much of the index effect's
    'disappearance', so they must not be conflated):

    - ``corporate_action``: M&A, taken private, spin-off, restructuring —
      the name left because it stopped existing in its old form.
    - ``distress``: bankruptcy / delisting — the name left feet-first.
    - ``migration``: moved to another S&P index (MidCap/SmallCap swap).
    - ``discretionary``: market-cap / representativeness deletions by the
      committee — H8's actual object.
    - ``unknown``: unclassifiable text (shown, never silently dropped).

    This is a keyword screen for the CENSUS (zero trials, no price data).
    The H8 registration mandates reconciling a 20-event random sample
    against contemporaneous press releases before any run; the frozen
    classification methodology is set there, not here.
    """
    if not reason:
        return "unknown"
    r = reason.lower()
    if any(k in r for k in ("acquir", "merg", "taken private", "private equity",
                            "takeover", "taken over", "purchas", "bought",
                            "spun off", "spin-off", "spinoff", "spins off",
                            "spinning off", "split into", "split-off",
                            "separated into")):
        return "corporate_action"
    if any(k in r for k in ("bankrupt", "chapter 11", "delist", "liquidat",
                            "receivership")):
        return "distress"
    if any(k in r for k in ("midcap", "mid cap", "smallcap", "small cap",
                            "600", "constituent swap", "moved to")):
        return "migration"
    if any(k in r for k in ("market cap", "market capitalization",
                            "representat", "no longer", "committee",
                            "index balance", "eligib")):
        return "discretionary"
    return "unknown"


def coverage_report(member_tickers: list[str], prices: pd.DataFrame) -> dict:
    """Quantify the residual survivorship bias: members with no price data.

    Dead companies (bankruptcy, acquisition) often vanish from free data
    sources. We cannot trade what we cannot price, so those names drop out of
    the backtest -- this measures how big that hole is instead of hiding it.
    """
    have = [t for t in member_tickers if t in prices.columns and prices[t].notna().any()]
    missing = sorted(set(member_tickers) - set(have))
    return {
        "n_members_ever": len(member_tickers),
        "n_with_price_data": len(have),
        "n_missing_price_data": len(missing),
        "missing_tickers": missing,
        "pct_covered": round(len(have) / max(len(member_tickers), 1), 4),
    }
