"""Tiingo EOD price layer — survivorship-bias-free prices for H1's price leg.

Tiingo carries DELISTED tickers' full EOD history (verified: ABMD, acquired by
J&J in 2022, returns prices through 2023-01-03), which is exactly the dead-name
price coverage the SEC-fundamentals + Tiingo-prices architecture needs. Auth via
TIINGO_API_KEY in .env (never hard-coded, never logged).

The pure parsers (parse_eod_prices, parse_supported_tickers) are network-free and
known-answer tested; the live TiingoSource caches to parquet and rate-limits so
the audit/run is offline-reproducible after the first pull.
"""

from __future__ import annotations

import io
import json
import os
import time
import urllib.error
import urllib.request
import zipfile

import pandas as pd

from quantlab.env import load_env

_BASE = "https://api.tiingo.com"
_SUPPORTED_URL = "https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip"
_MIN_INTERVAL = 0.5  # conservative spacing to avoid 429 on large universe pulls
_US_EXCHANGES = {"NYSE", "NASDAQ", "NYSE ARCA", "AMEX", "BATS", "NYSE MKT", "NMS"}


# --------------------------------------------------------------------------- #
# Pure parsers (no network) — pinned by tests.
# --------------------------------------------------------------------------- #

def parse_eod_prices(rows: list[dict], field: str = "adjClose") -> pd.Series:
    """Tiingo ``/prices`` JSON list -> date-indexed Series of ``field`` (default
    ``adjClose``, split/dividend-adjusted — the right input for returns).
    Timezone-stripped, sorted. Empty/None -> empty Series."""
    if not rows:
        return pd.Series(dtype=float, name=field)
    df = pd.DataFrame(rows)
    idx = pd.to_datetime(df["date"]).dt.tz_localize(None)
    s = pd.Series(df[field].to_numpy(dtype=float), index=idx, name=field)
    s.index.name = "date"
    return s.sort_index()


def parse_supported_tickers(
    csv_text: str, us_equity_only: bool = True
) -> pd.DataFrame:
    """Tiingo ``supported_tickers.csv`` -> DataFrame [ticker, exchange, assetType,
    startDate, endDate]. This is the survivorship-bias-free universe WITH date
    ranges: a DELISTED name has a past ``endDate`` (the crosswalk anchor the
    Wikipedia ticker list lacks). ``us_equity_only`` keeps US-listed common
    stock. Rows with no usable date range are dropped."""
    df = pd.read_csv(io.StringIO(csv_text))
    df.columns = [c.strip().lower() for c in df.columns]
    df["ticker"] = df["ticker"].astype(str).str.upper()
    for c in ("startdate", "enddate"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    if us_equity_only:
        df = df[
            df["assettype"].astype(str).str.lower().eq("stock")
            & df["exchange"].astype(str).str.upper().isin(_US_EXCHANGES)
        ]
    df = df.dropna(subset=["startdate"]).reset_index(drop=True)
    return df[["ticker", "exchange", "assettype", "startdate", "enddate"]]


# --------------------------------------------------------------------------- #
# Live source (cached, rate-limited).
# --------------------------------------------------------------------------- #

class TiingoSource:
    """Survivorship-bias-free EOD prices from Tiingo. Caches every pull to parquet
    so the audit/run is reproducible offline after the first fetch."""

    def __init__(
        self, env_path: str = ".env",
        cache_dir: str = os.path.join("data_cache", "tiingo"),
    ):
        load_env(env_path)
        self.key = os.environ.get("TIINGO_API_KEY", "")
        if not self.key:
            raise RuntimeError(
                "TIINGO_API_KEY missing — add it to .env (see .env.example).")
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._last = 0.0

    def _get(self, url: str, timeout: int = 30, retries: int = 6) -> bytes:
        req = urllib.request.Request(
            url, headers={"Content-Type": "application/json",
                          "Authorization": f"Token {self.key}"})
        for attempt in range(retries):
            wait = _MIN_INTERVAL - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    body = r.read()
                self._last = time.monotonic()
                return body
            except urllib.error.HTTPError as e:
                self._last = time.monotonic()
                if e.code == 404:
                    raise
                if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                    time.sleep(min(60, 3 * 2 ** attempt))   # harder backoff for 429
                    continue
                raise
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                # Transient network faults — incl. a raw socket read timeout during
                # r.read() (NOT wrapped in URLError) which previously slipped both
                # handlers and crashed a whole multi-name pull. Back off and retry.
                self._last = time.monotonic()
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise
        raise RuntimeError("unreachable")

    def coverage(self, ticker: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        """(startDate, endDate) of Tiingo's daily history for ``ticker`` — the
        cheap survivorship probe (a delisted name has a real past endDate). A
        404 / missing dates -> (None, None) (ticker absent from Tiingo)."""
        try:
            meta = json.loads(self._get(f"{_BASE}/tiingo/daily/{ticker.lower()}"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return (None, None)
            raise
        s, e = meta.get("startDate"), meta.get("endDate")
        return (pd.to_datetime(s) if s else None, pd.to_datetime(e) if e else None)

    def eod(
        self, ticker: str, start: str, end: str, field: str = "adjClose"
    ) -> pd.Series:
        """Adjusted daily prices for ``ticker`` over [start, end], cached to
        parquet. Empty Series if Tiingo lacks the ticker (404)."""
        safe = ticker.upper().replace("/", "_")
        path = os.path.join(self.cache_dir, f"eod_{safe}_{start}_{end}_{field}.parquet")
        if os.path.exists(path):
            return pd.read_parquet(path)[field]
        url = (f"{_BASE}/tiingo/daily/{ticker.lower()}/prices"
               f"?startDate={start}&endDate={end}")
        try:
            rows = json.loads(self._get(url))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                rows = []                       # genuinely absent -> cache empty
            elif e.code == 429:
                # rate-limited after retries: SKIP without caching, so a later
                # resumable pass retries this name. Never crashes the pull.
                return pd.Series(dtype=float, name=field)
            else:
                raise
        s = parse_eod_prices(rows, field=field)
        s.to_frame().to_parquet(path)
        return s

    def supported_tickers(self, us_equity_only: bool = True) -> pd.DataFrame:
        """Tiingo's full supported-ticker list (cached). The survivorship-safe
        universe + date-range crosswalk anchor."""
        path = os.path.join(self.cache_dir, "supported_tickers.parquet")
        if os.path.exists(path):
            return pd.read_parquet(path)
        raw = urllib.request.urlopen(_SUPPORTED_URL, timeout=120).read()
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            csv_text = zf.read(zf.namelist()[0]).decode("utf-8", "replace")
        df = parse_supported_tickers(csv_text, us_equity_only=us_equity_only)
        df.to_parquet(path)
        return df

    def prices(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        """Wide (date x ticker) adjusted-price frame over [start, end], one cached
        EOD pull per ticker. Delisting-inclusive: dead names carry history to
        their final print, then NaN."""
        cols = {}
        for t in tickers:
            s = self.eod(t, start, end)
            if not s.empty:
                cols[t.upper()] = s
        if not cols:
            return pd.DataFrame()
        return pd.DataFrame(cols).sort_index()
