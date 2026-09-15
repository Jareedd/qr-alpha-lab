"""Name-based ticker->CIK crosswalk — the free survivorship unlock for H1.

The wall (measured repeatedly): SEC's ticker->CIK map (company_tickers.json and
the per-CIK submissions ``tickers`` arrays) is FORWARD-ONLY — a company's ticker
association is dropped at delisting, so dead names (Celgene, Monsanto, Raytheon,
Xilinx...) don't resolve, and ticker reuse silently maps a dead name to a LIVING
company (the reassignment leak: MON now points at a SPAC).

The fix: resolve by COMPANY NAME, not ticker. SEC's ``cik-lookup-data.txt`` lists
EVERY entity that ever filed (1M+ names incl. dead ones), keyed by name. The dead
companies' names are free from the Wikipedia S&P changes table's "Removed
Security" column (the prior CIK-recovery work lacked names because it used the
ticker-only membership list). Name matching also AVOIDS ticker-reassignment by
construction (a name is far more unique than a recycled ticker), and a filing-
history gate confirms we picked the dead operating company, not a namesake.

Pure parsers (normalize_name, parse_cik_lookup, match_name) are network-free and
known-answer tested; NameCikResolver adds the cached download + the
operating/reassignment gate (one submissions fetch per candidate).
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

import pandas as pd

from quantlab.universe import (
    WIKI_URL,
    _UA,
    _find_current_table,
    _normalize_ticker,
    _read_tables,
    fetch_changes_frame,
)

_CIK_LOOKUP_URL = "https://www.sec.gov/Archives/edgar/cik-lookup-data.txt"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

# TRUE legal-form suffixes only — safe to peel from the END. Deliberately NOT
# included: AMERICA/AMERICAN/INTERNATIONAL/TRUST/REIT etc. (peeling those wrecks
# real names like "Bank of America" or "Northern Trust" and invites collisions).
_SUFFIXES = {
    "INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY", "COMPANIES",
    "LTD", "LIMITED", "PLC", "LLC", "LP", "LLP", "HOLDINGS", "HOLDING", "GROUP",
    "GP", "SA", "NV", "AG", "SE",
}


def normalize_name(name: str) -> str:
    """Canonicalize a company name for matching: drop any ``/QUALIFIER/`` tail
    (``/DE/``, ``/NEW/``, ``/MO/``), upper-case, strip non-alphanumerics, drop a
    leading article ('THE'), and peel trailing corporate-form suffixes. 'Celgene'
    and 'CELGENE CORP /DE/' both -> 'CELGENE'. Empty if nothing remains."""
    s = str(name).split("/")[0]                      # drop /DE/, /NEW/ qualifiers
    s = re.sub(r"[^A-Za-z0-9 ]", " ", s).upper()
    toks = [t for t in s.split() if t]
    if toks and toks[0] == "THE":                    # leading article
        toks = toks[1:]
    while toks and toks[-1] in _SUFFIXES:            # peel trailing suffix tokens
        toks.pop()
    return " ".join(toks)


def parse_cik_lookup(text: str) -> dict[str, list[str]]:
    """``cik-lookup-data.txt`` -> {normalized_name: sorted unique [CIK]}.

    Line format is ``COMPANY NAME:0000000000:`` (a name may repeat across CIKs,
    and one CIK may carry several names/former-names). Zero-pads CIKs to 10."""
    index: dict[str, set[str]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.endswith(":") or ":" not in line[:-1]:
            continue
        name_part, _, cik_part = line.rstrip(":").rpartition(":")
        if not name_part or not cik_part.isdigit():
            continue
        key = normalize_name(name_part)
        if key:
            index.setdefault(key, set()).add(cik_part.zfill(10))
    return {k: sorted(v) for k, v in index.items()}


def match_name(name: str, index: dict[str, list[str]]) -> list[str]:
    """Candidate CIKs for ``name``: exact normalized match first; else a unique
    prefix match (the query is a prefix of exactly one indexed name, or vice
    versa). Ambiguous prefix (>1 distinct name) -> no match (avoid false links)."""
    key = normalize_name(name)
    if not key:
        return []
    if key in index:
        return index[key]
    pref = {k: v for k, v in index.items() if k.startswith(key + " ") or key.startswith(k + " ")}
    if len(pref) == 1:
        return next(iter(pref.values()))
    return []


class NameCikResolver:
    """Loads cik-lookup-data.txt (cached) and resolves dead company names to the
    CIK of the dead OPERATING filer, gating out namesakes/reassignments via the
    filing history."""

    def __init__(self, cache_dir: str = os.path.join("data_cache", "fundamentals")):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._index: dict[str, list[str]] | None = None
        self._last = 0.0

    def _get(self, url: str, timeout: int = 120) -> bytes:
        wait = 0.12 - (time.monotonic() - self._last)   # SEC fair-access (<10/s)
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        try:
            return urllib.request.urlopen(req, timeout=timeout).read()
        finally:
            self._last = time.monotonic()

    def index(self) -> dict[str, list[str]]:
        if self._index is None:
            path = os.path.join(self.cache_dir, "cik-lookup-data.txt")
            if not os.path.exists(path):
                open(path, "wb").write(self._get(_CIK_LOOKUP_URL))
            text = open(path, encoding="latin-1").read()
            self._index = parse_cik_lookup(text)
        return self._index

    def submissions_meta(self, cik: str) -> dict:
        """Per-CIK submissions metadata (cached): name, forms, last filing date."""
        path = os.path.join(self.cache_dir, f"sub_{cik}.json")
        if os.path.exists(path):
            return json.loads(open(path, encoding="utf-8").read())
        try:
            raw = self._get(_SUBMISSIONS_URL.format(cik=cik), timeout=60)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return {}
            raise
        open(path, "wb").write(raw)
        return json.loads(raw)

    def operating_cik(self, name: str, dead_by: str | None = None) -> str | None:
        """Resolve ``name`` to the CIK of the dead operating company: among name
        matches, keep those that filed 10-Ks; if ``dead_by`` (the S&P removal
        date) is given, require the last filing to be at/after the company's life
        but not that of a still-active reuser (last filing not after dead_by +
        ~2yr grace). Returns the best (latest-filing) survivor, or None."""
        cands = match_name(name, self.index())
        scored = []
        for cik in cands:
            meta = self.submissions_meta(cik)
            forms = (meta.get("filings", {}).get("recent", {}).get("form", []))
            dates = (meta.get("filings", {}).get("recent", {}).get("filingDate", []))
            if "10-K" not in forms or not dates:
                continue
            last = max(dates)
            if dead_by is not None:
                grace = (pd.Timestamp(dead_by) + pd.Timedelta(days=730))
                if pd.Timestamp(last) > grace:
                    continue                          # still-active namesake/reuser
            scored.append((last, cik))
        if not scored:
            return None
        return sorted(scored)[-1][1]                  # latest 10-K filer among matches


def fetch_sp500_security_names(
    cache_dir: str = "data_cache",
) -> dict[str, str]:
    """{ticker: company name} from the Wikipedia S&P page — current constituents
    PLUS the added/removed names in the changes table (the free source of DEAD
    companies' names). Cached as parquet."""
    path = os.path.join(cache_dir, "sp500_names.parquet")
    if os.path.exists(path):
        df = pd.read_parquet(path)
        return dict(zip(df["ticker"], df["name"]))
    os.makedirs(cache_dir, exist_ok=True)

    # Both tables are selected by COLUMN NAME through quantlab.universe, which
    # also handles the changes table having moved to its own Wikipedia page.
    # This function used to carry its own positional copy of that parse
    # (``tables[1]`` + a fixed six-column assignment) -- the identical latent
    # crash that killed the live cycle for five weeks from 2026-08-11, sitting
    # in the path that supplies DEAD companies' names to the SEC crosswalk,
    # i.e. the thing that unblocked H1's survivorship problem.
    tables = _read_tables(WIKI_URL)
    cur = _find_current_table(tables)
    if cur is None:
        raise ValueError(f"no S&P 500 constituents table at {WIKI_URL}")
    names: dict[str, str] = {}
    for t, nm in zip(cur["symbol"], cur["security"]):
        names[_normalize_ticker(t)] = str(nm)

    chg, chg_url = fetch_changes_frame(with_names=True, tables=tables)
    if chg is None:
        raise ValueError(
            f"no S&P 500 changes table at {WIKI_URL} or the changes page; the "
            "dead-company names it supplies are what make this crosswalk "
            "survivorship-safe, so a current-members-only answer is refused"
        )
    for tcol, ncol in (("added", "added_name"), ("removed", "removed_name")):
        for t, nm in zip(chg[tcol], chg[ncol]):
            if pd.notna(t) and pd.notna(nm):
                names.setdefault(_normalize_ticker(t), str(nm))
    print(f"[crosswalk] {len(names)} ticker->name pairs (changes from {chg_url})")
    pd.DataFrame({"ticker": list(names), "name": list(names.values())}).to_parquet(path)
    return names
