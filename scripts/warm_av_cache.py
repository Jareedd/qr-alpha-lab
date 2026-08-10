"""Resumable Alpha Vantage earnings cache-warmer for the H13 PEAD graded universe.

WHY: FMP's free tier serves only ~27 demo symbols, so it cannot supply the S&P
500 earnings universe a credible graded trial #14 needs. Alpha Vantage's free
tier covers ALL symbols (estimatedEPS + reportedEPS + reportedDate back ~25y) but
is throttled to ~25 requests/DAY and ~5/MINUTE. This script builds the universe
cache across multiple days: each run pulls the next batch of UN-cached current
S&P 500 names at >12s spacing (per-minute cap) and STOPS EARLY when the daily cap
is hit (AV returns an Information/Note payload instead of earnings). Re-run it
once per day until coverage is sufficient; cached names are free on re-runs.

Usage:
    python scripts/warm_av_cache.py            # pull until daily cap, then stop
    python scripts/warm_av_cache.py 40 13      # cap this run to 40 names @ 13s

No graded decision happens here — it only fills data_cache/alphavantage/. The
leakage gate + filter + cross-check still run at graded time (run_pead.py).
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import universe as U
from quantlab.env import load_env
from quantlab.fmp_earnings import parse_alphavantage_earnings

CACHE = os.path.join("data_cache", "alphavantage")
AV_MIN_INTERVAL = 13.0          # ~5 req/min cap -> >12s spacing
MAX_CONSEC_RATELIMIT = 2        # stop after this many consecutive cap responses


def _current_sp500() -> list[str]:
    cur, _ = U.fetch_sp500_tables()
    col = cur["ticker"] if hasattr(cur, "columns") else cur
    return sorted({str(t).strip().upper() for t in col if str(t).strip()})


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10**9
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else AV_MIN_INTERVAL

    load_env(".env")
    key = (os.environ.get("ALPHAVANTAGE_API_KEY")
           or os.environ.get("ALPHA_VANTAGE_API_KEY") or "")
    if not key:
        sys.exit("No ALPHA_VANTAGE_API_KEY / ALPHAVANTAGE_API_KEY in .env.")
    os.makedirs(CACHE, exist_ok=True)

    syms = _current_sp500()
    uncached = [s for s in syms if not os.path.exists(
        os.path.join(CACHE, f"earnings_{s.replace('/', '_')}.json"))]
    todo = uncached[:limit]
    n_have = len(syms) - len(uncached)
    print(f"current S&P 500: {len(syms)} | already cached: {n_have} | "
          f"uncached: {len(uncached)} | pulling up to {len(todo)} @ {interval}s",
          flush=True)

    ok = consec_cap = 0
    last = 0.0
    for i, s in enumerate(todo, 1):
        wait = interval - (time.monotonic() - last)
        if wait > 0:
            time.sleep(wait)
        q = urllib.parse.urlencode({"function": "EARNINGS", "symbol": s, "apikey": key})
        url = f"https://www.alphavantage.co/query?{q}"
        try:
            raw = urllib.request.urlopen(urllib.request.Request(url), timeout=30).read()
            last = time.monotonic()
            payload = json.loads(raw)
        except (urllib.error.URLError, TimeoutError, ConnectionError,
                json.JSONDecodeError) as e:
            print(f"  [{i}/{len(todo)}] {s}: network error ({e}); skip", flush=True)
            continue

        if "quarterlyEarnings" not in payload:
            # Daily cap / rate note (AV returns {"Information": ...} or {"Note": ...}).
            msg = (payload.get("Information") or payload.get("Note")
                   or payload.get("Error Message") or str(payload)[:120])
            consec_cap += 1
            print(f"  [{i}/{len(todo)}] {s}: NO earnings -> likely daily cap: {msg[:110]}",
                  flush=True)
            if consec_cap >= MAX_CONSEC_RATELIMIT:
                print(f"  daily cap reached after {ok} new names this run. "
                      "Re-run tomorrow to continue.", flush=True)
                break
            continue

        consec_cap = 0
        with open(os.path.join(CACHE, f"earnings_{s.replace('/', '_')}.json"),
                  "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        ev = parse_alphavantage_earnings(payload, s)
        ok += 1
        print(f"  [{i}/{len(todo)}] {s}: cached ({len(ev)} events)", flush=True)

    n_cache = len(glob.glob(os.path.join(CACHE, "earnings_*.json")))
    print(f"DONE: +{ok} new this run; {n_cache}/{len(syms)} current S&P 500 "
          f"now cached. Re-run daily until coverage is sufficient (~150 names "
          "gives deep monthly cohorts).", flush=True)


if __name__ == "__main__":
    main()
