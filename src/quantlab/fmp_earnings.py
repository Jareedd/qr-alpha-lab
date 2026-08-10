"""FMP earnings layer for H13 PEAD — a FREE alternative to the Bloomberg CSV,
behind a leakage GATE that fails CLOSED.

WHY this module exists. H13 (PEAD) needs the surprise relative to the consensus
that stood BEFORE the print (Bernard-Thomas). Bloomberg/IBES give a true
point-in-time (PIT) consensus; free sources usually carry only the realized
actual. Financial Modeling Prep's ``/stable/earnings`` endpoint DOES carry an
``epsEstimated`` back to ~1985 — but we have NO contract that it is genuinely
point-in-time (it may be backfilled to the actual, or revised long after the
fact). So this layer pairs a cached, resumable fetcher + a pure parser with a
hard VALIDATION GATE: ``validate_pead_events`` scores the feed for the gross
red flags of a non-PIT / broken feed and refuses (``passed=False``) if any HARD
check trips. A contaminated graded trial is worse than no trial.

CRITICAL HONESTY: ``passed=True`` means "no GROSS red flags in THIS feed" — it
does NOT prove the estimates are point-in-time. The gold-standard PIT
confirmation is a cross-source/IBES agreement check (the optional ``cross_check``
arg, or a paid IBES snapshot). That residual risk is stated in the report dict
and printed loudly. See ``validate_pead_events``.

PIT-safety of the layer itself: no network at import; the parser references no
price and no future information; FUTURE rows (epsActual is null — not yet
reported) are DROPPED, so nothing dated after its own announcement enters.

Mirrors the cached/rate-limited ``_get`` + per-symbol cache pattern in
``tiingo_data.py`` / ``fundamentals_data.py`` so a first pull is reproducible
offline and resumable across days (the free tier is ~250 req/day).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from quantlab.env import load_env

_BASE = "https://financialmodelingprep.com/stable/earnings"
_MIN_INTERVAL = 0.30  # conservative spacing; free tier ~250 req/day, ~5 req/s cap
# Alpha Vantage free tier throttles at ~5 requests/MINUTE (separate from the ~25/
# day cap). Space cross-check calls > 12s apart so all requested names land in one
# pass instead of calls 6+ returning an uncached rate-note (thinning the sample).
_AV_MIN_INTERVAL = 13.0
CACHE = os.path.join("data_cache", "fmp")

# The tidy events schema produced by pead.parse_pead_csv — the harness contract.
# parse_fmp_earnings MUST emit exactly these columns so compute_sue/event-study
# consume FMP and Bloomberg interchangeably.
_TIDY_COLS = ["ticker", "ann_date", "period", "actual_eps", "est_eps",
              "surprise_pct", "num_est", "std_est"]


# --------------------------------------------------------------------------- #
# Pure parser (no network) — pinned by tests.
# --------------------------------------------------------------------------- #

def parse_fmp_earnings(json_list, symbol: str | None = None) -> pd.DataFrame:
    """FMP ``/stable/earnings`` JSON list -> tidy PEAD events DataFrame.

    Input rows look like ``{symbol, date, epsActual, epsEstimated,
    revenueActual, revenueEstimated, lastUpdated}`` (newest-first). Mapping into
    the pead schema:

      ticker        <- symbol  (upper-cased; ``symbol`` arg overrides if a row
                                lacks one — single-symbol pulls sometimes omit it)
      ann_date      <- date
      actual_eps    <- epsActual
      est_eps       <- epsEstimated
      surprise_pct  =  100 * (actual - est) / |est|   (NaN if est == 0)
      period/num_est/std_est = NaN  (FMP carries no analyst dispersion; compute_sue
                                     falls back to surprise_pct / rel-est, which is
                                     fine — documented in pead.compute_sue)

    FUTURE rows (epsActual is null — not yet reported) are DROPPED: they have no
    realized return and a null actual is not an event. Rows missing est_eps are
    also dropped (no surprise computable). Output carries ALL ``_TIDY_COLS``,
    ``ann_date`` as datetime, sorted by (ann_date, ticker). An extra
    ``last_updated`` column rides along (datetime, NaT if absent) for the
    recency caveat in ``validate_pead_events`` — it is dropped before the
    harness consumes the frame, so it never reaches the compute layer.

    PIT-safety: a pure field map; references no price and no future bar. Dropping
    null-actual future rows is the only forward-looking filter and it REMOVES
    rows, never adds information.
    """
    rows = json_list or []
    recs = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        actual = r.get("epsActual")
        est = r.get("epsEstimated")
        if actual is None:          # FUTURE / not-yet-reported -> drop
            continue
        if est is None:             # no consensus -> no surprise -> drop
            continue
        tkr = r.get("symbol") or symbol
        if not tkr:
            continue
        recs.append({
            "ticker": str(tkr).strip().upper(),
            "ann_date": r.get("date"),
            "actual_eps": actual,
            "est_eps": est,
            "last_updated": r.get("lastUpdated"),
        })

    if not recs:
        out = pd.DataFrame(columns=_TIDY_COLS + ["last_updated"])
        out["ann_date"] = pd.to_datetime(out["ann_date"])
        out["last_updated"] = pd.to_datetime(out["last_updated"])
        return out

    df = pd.DataFrame(recs)
    df["ann_date"] = pd.to_datetime(df["ann_date"], errors="coerce")
    df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce")
    for c in ("actual_eps", "est_eps"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # An event needs a parseable date and both EPS numbers (the SUE numerator).
    df = df.dropna(subset=["ann_date", "actual_eps", "est_eps"])

    # surprise_pct as a PERCENT (matches Bloomberg's column convention; NaN where
    # est == 0 so we never divide by zero — compute_sue then falls back row-wise).
    est = df["est_eps"].to_numpy(dtype=float)
    actual = df["actual_eps"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        spct = np.where(np.abs(est) > 0,
                        100.0 * (actual - est) / np.abs(est), np.nan)
    df["surprise_pct"] = spct
    # FMP has no dispersion / fiscal-period label.
    df["period"] = np.nan
    df["num_est"] = np.nan
    df["std_est"] = np.nan

    df = df[_TIDY_COLS + ["last_updated"]].copy()
    return df.sort_values(["ann_date", "ticker"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Live fetcher (cached, rate-limited, resumable).
# --------------------------------------------------------------------------- #

class FMPEarningsSource:
    """Cached, rate-limited FMP earnings fetcher. Caches raw JSON per symbol so a
    universe pull is resumable across days within the free daily quota, and the
    parse step is reproducible offline after the first fetch.

    No network at import or __init__-without-use beyond loading the key from
    ``.env`` (never printed, never committed)."""

    def __init__(self, env_path: str = ".env", cache_dir: str = CACHE,
                 min_interval: float = _MIN_INTERVAL):
        load_env(env_path)
        self.key = os.environ.get("FMP_API_KEY", "")
        if not self.key:
            raise RuntimeError(
                "FMP_API_KEY missing — add it to .env (see .env.example). "
                "Free key from https://financialmodelingprep.com.")
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        # Spacing between live requests. The free tier burst-caps (HTTP 402) if
        # hit too fast — a bulk universe pull should pass a slower interval
        # (~1.5-2s) than the 0.30s default used for occasional/cached calls.
        self.min_interval = float(min_interval)
        self._last = 0.0

    def _get(self, url: str, timeout: int = 30, retries: int = 5) -> bytes:
        req = urllib.request.Request(
            url, headers={"Content-Type": "application/json"})
        for attempt in range(retries):
            wait = self.min_interval - (time.monotonic() - self._last)
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
                # 402 = free-tier BURST cap (transient — a single call succeeds
                # right after a burst), so back off and retry like 429 rather than
                # giving up; only a persistent 402 after all retries propagates.
                if e.code in (402, 429, 500, 502, 503, 504) and attempt < retries - 1:
                    time.sleep(min(60, 3 * 2 ** attempt))   # hard backoff
                    continue
                raise
            except urllib.error.URLError:
                self._last = time.monotonic()
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise
        raise RuntimeError("unreachable")

    def fetch_earnings(self, symbol: str) -> list:
        """Raw FMP earnings JSON list for ``symbol``, cached to JSON per symbol.

        On a 429 after retries we return ``[]`` WITHOUT caching so a later
        resumable pass retries this name (mirrors TiingoSource.eod). A genuine
        404 caches an empty list (the symbol is absent — no need to re-hit it)."""
        safe = symbol.upper().replace("/", "_")
        path = os.path.join(self.cache_dir, f"earnings_{safe}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        q = urllib.parse.urlencode({"symbol": symbol.upper(), "apikey": self.key})
        url = f"{_BASE}?{q}"
        try:
            raw = self._get(url)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                data: list = []                 # genuinely absent -> cache empty
            elif e.code in (402, 429):
                # 429 = rate limit; 402 = free-tier burst/quota cap. Both are
                # transient w.r.t. a later resumable pass -> skip WITHOUT caching
                # so this name is retried next run (never cache a quota error).
                return []
            else:
                raise
        else:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                return []                        # garbage body -> skip, no cache
            if not isinstance(data, list):
                # FMP returns {"Error Message": ...} on bad key/over-quota; never
                # cache that — surface as empty and let a later pass retry.
                return []
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        return data

    def events(self, symbol: str) -> pd.DataFrame:
        """Tidy PEAD events for one symbol (cached fetch -> pure parse)."""
        return parse_fmp_earnings(self.fetch_earnings(symbol), symbol=symbol)


def build_events(
    symbols, env_path: str = ".env", cache_dir: str = CACHE,
    max_names: int | None = None, source: "FMPEarningsSource | None" = None,
    min_interval: float = _MIN_INTERVAL,
) -> pd.DataFrame:
    """Concat tidy PEAD events across a universe (sequential, cached, resumable).

    ``symbols`` is the ticker universe (e.g. the PIT S&P 500). ``max_names`` caps
    how many distinct symbols are pulled THIS run so a first pass stays inside the
    FMP free daily quota (~250 req/day); cached symbols are free on re-runs, so a
    multi-day pull just bumps ``max_names`` and re-runs — already-cached names are
    served from disk, new ones are fetched until the cap. Symbols that return no
    events (absent / quota-skipped) are silently omitted; the caller's coverage
    check in ``validate_pead_events`` surfaces a thin pull.

    Returns the tidy events frame (``_TIDY_COLS`` only; the internal
    ``last_updated`` column is RETAINED here so ``validate_pead_events`` can run
    its recency caveat — strip it before handing the frame to compute_sue, which
    ignores unknown columns anyway). No network at import; network happens here.
    """
    src = source or FMPEarningsSource(env_path=env_path, cache_dir=cache_dir,
                                      min_interval=min_interval)
    syms = [str(s).strip().upper() for s in symbols if str(s).strip()]
    # De-dupe preserving order; cap the number of distinct names pulled this run.
    seen: set[str] = set()
    ordered = []
    for s in syms:
        if s not in seen:
            seen.add(s)
            ordered.append(s)
    if max_names is not None:
        ordered = ordered[:max_names]

    frames = []
    for s in ordered:
        ev = src.events(s)
        if not ev.empty:
            frames.append(ev)
    if not frames:
        out = pd.DataFrame(columns=_TIDY_COLS + ["last_updated"])
        out["ann_date"] = pd.to_datetime(out["ann_date"])
        out["last_updated"] = pd.to_datetime(out["last_updated"])
        return out
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["ann_date", "ticker"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# THE LEAKAGE GATE — the credibility core. (Pure; no network.)
# --------------------------------------------------------------------------- #

# Thresholds (each is a defensible default; documented in the report dict). They
# are deliberately LOOSE enough to pass a real, noisy PIT feed and STRICT enough
# to catch the failure modes that would silently contaminate a graded trial.
EXACT_MATCH_MAX = 0.20      # HARD: > this fraction of est == actual EXACTLY -> backfilled
MIN_SURPRISE_STD = 1e-4    # HARD: surprise distribution must be non-degenerate
MIN_NONZERO_FRAC = 0.50    # HARD: at least this fraction of surprises non-zero
MIN_SIGN_FRAC = 0.05       # HARD: each sign (+/-) must be >= this fraction
RECENCY_REPORT_DAYS = 7    # SOFT: lastUpdated more than this many days after
                            #       ann_date = a post-hoc revision candidate (report)


def _surprise(events: pd.DataFrame) -> np.ndarray:
    """Per-event relative surprise (actual-est)/|est|, the gate's distribution
    object. Uses surprise_pct/100 where present (already computed), else recomputes
    from actual/est; rows with est == 0 contribute NaN and are dropped."""
    spct = pd.to_numeric(events.get("surprise_pct"), errors="coerce")
    actual = pd.to_numeric(events["actual_eps"], errors="coerce").to_numpy(dtype=float)
    est = pd.to_numeric(events["est_eps"], errors="coerce").to_numpy(dtype=float)
    s = spct.to_numpy(dtype=float) / 100.0 if spct is not None else np.full(len(events), np.nan)
    need = ~np.isfinite(s)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(np.abs(est) > 0, (actual - est) / np.abs(est), np.nan)
    s = np.where(need, rel, s)
    return s[np.isfinite(s)]


def validate_pead_events(events: pd.DataFrame, cross_check: pd.DataFrame | None = None,
                         today: pd.Timestamp | None = None) -> dict:
    """The LEAKAGE GATE for a free PEAD events feed. Fails CLOSED.

    Returns a structured report dict with a top-level ``passed`` bool. ``passed``
    is True ONLY if every HARD check passes; any HARD failure sets it False and
    records the reason in ``report['reasons']``. SOFT checks are reported, never
    auto-fail. Checks:

      coverage          (SOFT): #tickers, #events, history span, median ev/ticker.
      date_semantics    (HARD): every event with a non-null actual has ann_date
                                <= today. A future-dated event WITH a realized
                                actual is impossible for a real feed -> fail.
      surprise_dist     (HARD): the surprise (actual-est)/|est| must be a real
                                distribution: std > MIN_SURPRISE_STD, >=
                                MIN_NONZERO_FRAC non-zero, BOTH signs present each
                                >= MIN_SIGN_FRAC. Degenerate/one-signed -> broken
                                or aligned feed -> fail.
      backfilled        (HARD): fraction of events with est == actual EXACTLY.
                                > EXACT_MATCH_MAX (20%) -> the estimate is likely
                                backfilled to the actual (destroys the signal and
                                signals non-PIT) -> fail.
      recency           (SOFT): distribution of (lastUpdated - ann_date); the
                                share of OLD announcements whose estimate was
                                updated long after the fact (post-hoc revision
                                candidates). Surfaced loudly; NOT auto-failed (it
                                is a heuristic — a vendor can re-touch a row's
                                metadata without changing the estimate).
      cross_source      (OPTIONAL): if ``cross_check`` events are given, agreement
                                of est_eps on the (ticker, ann_date) overlap. High
                                agreement = strong PIT evidence. Skipped if absent.

    THE HONEST CAVEAT (also in report['caveat'] and printed): passed == "no gross
    red flags", NOT "proven point-in-time". The gold-standard PIT confirmation is
    a cross-source / IBES agreement check; without it the residual risk is that
    epsEstimated is a quietly-backfilled or post-hoc-revised value that LOOKS like
    a clean distribution. Treat a passing gate as necessary, not sufficient.

    Thresholds (module constants, all defensible defaults):
      EXACT_MATCH_MAX={emm}, MIN_SURPRISE_STD={mss}, MIN_NONZERO_FRAC={mnz},
      MIN_SIGN_FRAC={msf}, RECENCY_REPORT_DAYS={rrd}.

    PIT-safety: a pure diagnostic over the events table; no price, no network, no
    mutation of the input.
    """
    # tz-NAIVE today: ann_date is parsed tz-naive, and comparing a tz-aware
    # Timestamp against a tz-naive datetime64 column raises in pandas. Strip any
    # tz the caller (or the default UTC now) carries.
    today = pd.Timestamp(today) if today is not None else pd.Timestamp.utcnow()
    if today.tz is not None:
        today = today.tz_convert(None)
    today = today.normalize()
    reasons: list[str] = []
    report: dict = {
        "passed": False,
        "thresholds": {
            "exact_match_max": EXACT_MATCH_MAX,
            "min_surprise_std": MIN_SURPRISE_STD,
            "min_nonzero_frac": MIN_NONZERO_FRAC,
            "min_sign_frac": MIN_SIGN_FRAC,
            "recency_report_days": RECENCY_REPORT_DAYS,
        },
        "reasons": reasons,
        "caveat": (
            "PASSED means NO GROSS RED FLAGS in this feed, NOT proven "
            "point-in-time. The gold-standard PIT confirmation is cross-source / "
            "IBES agreement on the consensus estimate; without it, the residual "
            "risk is a quietly backfilled or post-hoc-revised epsEstimated that "
            "still looks like a clean distribution. A passing gate is NECESSARY, "
            "NOT SUFFICIENT — confirm with a cross-source check before trusting "
            "any graded H13 number."),
    }

    ev = events.copy()
    ev["ann_date"] = pd.to_datetime(ev["ann_date"], errors="coerce")
    n = int(len(ev))

    # ---- coverage (SOFT / report) ------------------------------------------- #
    if n == 0:
        report["coverage"] = {"n_events": 0, "n_tickers": 0}
        reasons.append("HARD date_semantics/surprise_dist/backfilled: "
                       "EMPTY events frame — nothing to validate.")
        report["date_semantics"] = {"passed": False, "n_future_with_actual": 0,
                                    "note": "empty frame"}
        report["surprise_dist"] = {"passed": False, "note": "empty frame"}
        report["backfilled"] = {"passed": False, "note": "empty frame"}
        report["passed"] = False
        return report

    per_ticker = ev.groupby("ticker").size()
    report["coverage"] = {
        "n_events": n,
        "n_tickers": int(ev["ticker"].nunique()),
        "history_start": str(ev["ann_date"].min().date()) if ev["ann_date"].notna().any() else None,
        "history_end": str(ev["ann_date"].max().date()) if ev["ann_date"].notna().any() else None,
        "median_events_per_ticker": float(per_ticker.median()),
    }

    # ---- date semantics (HARD) ---------------------------------------------- #
    has_actual = pd.to_numeric(ev["actual_eps"], errors="coerce").notna()
    future = ev["ann_date"] > today
    future_with_actual = int((future & has_actual).sum())
    ds_pass = future_with_actual == 0
    report["date_semantics"] = {
        "passed": bool(ds_pass),
        "n_future_with_actual": future_with_actual,
        "today": str(today.date()),
    }
    if not ds_pass:
        reasons.append(
            f"HARD date_semantics: {future_with_actual} event(s) dated AFTER "
            f"today ({today.date()}) carry a realized actual — impossible for a "
            "PIT feed (look-ahead / mis-dated rows).")

    # ---- surprise distribution sanity (HARD) -------------------------------- #
    s = _surprise(ev)
    n_s = int(len(s))
    if n_s == 0:
        sd_pass = False
        report["surprise_dist"] = {"passed": False, "note": "no finite surprises"}
        reasons.append("HARD surprise_dist: no finite surprises (all est==0?).")
    else:
        std = float(np.std(s))
        nonzero_frac = float(np.mean(np.abs(s) > 1e-12))
        pos_frac = float(np.mean(s > 0))
        neg_frac = float(np.mean(s < 0))
        sd_pass = (std > MIN_SURPRISE_STD
                   and nonzero_frac >= MIN_NONZERO_FRAC
                   and pos_frac >= MIN_SIGN_FRAC
                   and neg_frac >= MIN_SIGN_FRAC)
        report["surprise_dist"] = {
            "passed": bool(sd_pass), "std": std, "nonzero_frac": nonzero_frac,
            "pos_frac": pos_frac, "neg_frac": neg_frac, "n": n_s,
        }
        if not sd_pass:
            why = []
            if std <= MIN_SURPRISE_STD:
                why.append(f"std {std:.2e} <= {MIN_SURPRISE_STD:.0e} (degenerate)")
            if nonzero_frac < MIN_NONZERO_FRAC:
                why.append(f"only {nonzero_frac:.0%} non-zero < {MIN_NONZERO_FRAC:.0%}")
            if pos_frac < MIN_SIGN_FRAC or neg_frac < MIN_SIGN_FRAC:
                why.append(f"one-signed (pos {pos_frac:.0%} / neg {neg_frac:.0%})")
            reasons.append("HARD surprise_dist: " + "; ".join(why)
                           + " — a broken / aligned feed, not a real surprise.")

    # ---- backfilled-to-actual flag (HARD) ----------------------------------- #
    actual = pd.to_numeric(ev["actual_eps"], errors="coerce").to_numpy(dtype=float)
    est = pd.to_numeric(ev["est_eps"], errors="coerce").to_numpy(dtype=float)
    both = np.isfinite(actual) & np.isfinite(est)
    if both.sum() == 0:
        exact_frac = 1.0
    else:
        exact_frac = float(np.mean(actual[both] == est[both]))
    bf_pass = exact_frac <= EXACT_MATCH_MAX
    report["backfilled"] = {
        "passed": bool(bf_pass), "exact_match_frac": exact_frac,
        "threshold": EXACT_MATCH_MAX,
    }
    if not bf_pass:
        reasons.append(
            f"HARD backfilled: {exact_frac:.0%} of events have est == actual "
            f"EXACTLY (> {EXACT_MATCH_MAX:.0%}) — the consensus may be backfilled "
            "to the realized actual (non-PIT; destroys the signal).")

    # ---- lastUpdated recency (SOFT / report) -------------------------------- #
    if "last_updated" in ev.columns and pd.to_datetime(ev["last_updated"], errors="coerce").notna().any():
        lu = pd.to_datetime(ev["last_updated"], errors="coerce")
        lag_days = (lu - ev["ann_date"]).dt.days
        valid_lag = lag_days.dropna()
        stale_frac = float(np.mean(valid_lag > RECENCY_REPORT_DAYS)) if len(valid_lag) else float("nan")
        report["recency"] = {
            "n_with_last_updated": int(valid_lag.shape[0]),
            "median_lag_days": float(valid_lag.median()) if len(valid_lag) else float("nan"),
            "frac_updated_gt_window": stale_frac,
            "window_days": RECENCY_REPORT_DAYS,
            "note": ("SOFT/report only — a high share of announcements re-touched "
                     "long after the fact is a post-hoc-revision RISK, not proof; "
                     "vendor metadata can change without the estimate changing."),
        }
    else:
        report["recency"] = {"n_with_last_updated": 0,
                             "note": "no lastUpdated column — recency caveat "
                                     "cannot be assessed (treat PIT as unconfirmed)."}

    # ---- optional cross-source PIT check ------------------------------------ #
    if cross_check is not None and not cross_check.empty:
        cc = cross_check.copy()
        cc["ann_date"] = pd.to_datetime(cc["ann_date"], errors="coerce")
        key = ["ticker", "ann_date"]
        merged = ev[key + ["est_eps"]].merge(
            cc[key + ["est_eps"]], on=key, suffixes=("_fmp", "_x"))
        n_overlap = int(len(merged))
        if n_overlap == 0:
            report["cross_source"] = {"n_overlap": 0,
                                      "note": "no overlapping (ticker, ann_date) "
                                              "events — cross-check inconclusive."}
        else:
            a = pd.to_numeric(merged["est_eps_fmp"], errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(merged["est_eps_x"], errors="coerce").to_numpy(dtype=float)
            denom = np.where(np.abs(b) > 0, np.abs(b), np.nan)
            rel_diff = np.abs(a - b) / denom
            agree_frac = float(np.nanmean(rel_diff < 0.01))   # within 1%
            report["cross_source"] = {
                "n_overlap": n_overlap,
                "agree_frac_within_1pct": agree_frac,
                "median_rel_diff": float(np.nanmedian(rel_diff)),
                "note": ("HIGH agreement is strong PIT evidence; reported, not "
                         "auto-graded (this build leaves the cross-source pass/"
                         "fail call to the operator's review)."),
            }
    else:
        report["cross_source"] = {"n_overlap": 0,
                                  "note": "no cross_check source supplied — "
                                          "the gold-standard PIT confirmation is "
                                          "ABSENT (residual risk per caveat)."}

    report["passed"] = bool(ds_pass and sd_pass and bf_pass)
    return report


validate_pead_events.__doc__ = (validate_pead_events.__doc__ or "").format(
    emm=EXACT_MATCH_MAX, mss=MIN_SURPRISE_STD, mnz=MIN_NONZERO_FRAC,
    msf=MIN_SIGN_FRAC, rrd=RECENCY_REPORT_DAYS)


def format_validation_report(report: dict) -> str:
    """Human-readable rendering of a ``validate_pead_events`` report (used by the
    runner's DATA GATE so the report prints on PASS and FAIL alike)."""
    cov = report.get("coverage", {})
    lines = ["=== FMP PEAD validation (leakage gate) ==="]
    lines.append(
        f"  coverage    : {cov.get('n_events','?')} events / "
        f"{cov.get('n_tickers','?')} tickers, "
        f"{cov.get('history_start','?')}..{cov.get('history_end','?')}, "
        f"median {cov.get('median_events_per_ticker','?')} ev/ticker")
    ds = report.get("date_semantics", {})
    lines.append(f"  date-seman. : {'PASS' if ds.get('passed') else 'FAIL'} "
                 f"({ds.get('n_future_with_actual','?')} future-dated w/ actual)")
    sd = report.get("surprise_dist", {})
    if "std" in sd:
        lines.append(f"  surprise    : {'PASS' if sd.get('passed') else 'FAIL'} "
                     f"(std {sd['std']:.3f}, nonzero {sd['nonzero_frac']:.0%}, "
                     f"pos {sd['pos_frac']:.0%}/neg {sd['neg_frac']:.0%})")
    else:
        lines.append(f"  surprise    : {'PASS' if sd.get('passed') else 'FAIL'} "
                     f"({sd.get('note','')})")
    bf = report.get("backfilled", {})
    lines.append(f"  backfilled  : {'PASS' if bf.get('passed') else 'FAIL'} "
                 f"(est==actual exactly {bf.get('exact_match_frac', float('nan')):.0%}, "
                 f"max {bf.get('threshold', EXACT_MATCH_MAX):.0%})")
    rc = report.get("recency", {})
    if rc.get("n_with_last_updated"):
        lines.append(f"  recency     : SOFT — median lag {rc.get('median_lag_days'):.0f}d, "
                     f"{rc.get('frac_updated_gt_window', float('nan')):.0%} updated "
                     f">{rc.get('window_days')}d after announcement (revision risk)")
    else:
        lines.append(f"  recency     : SOFT — {rc.get('note','')}")
    cs = report.get("cross_source", {})
    if cs.get("n_overlap"):
        lines.append(f"  cross-src.  : {cs['n_overlap']} overlap, "
                     f"{cs.get('agree_frac_within_1pct', float('nan')):.0%} agree within 1%")
    else:
        lines.append(f"  cross-src.  : {cs.get('note','')}")
    lines.append(f"  >>> GATE {'PASSED' if report.get('passed') else 'FAILED'}")
    if report.get("reasons"):
        for r in report["reasons"]:
            lines.append(f"      - {r}")
    lines.append(f"  CAVEAT: {report.get('caveat','')}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# OPTIONAL Alpha Vantage cross-source (used ONLY if ALPHAVANTAGE_API_KEY is set).
# --------------------------------------------------------------------------- #

def parse_alphavantage_earnings(payload: dict, symbol: str) -> pd.DataFrame:
    """Alpha Vantage ``EARNINGS`` JSON -> tidy events (ticker, ann_date, est_eps,
    actual_eps) for the cross-source check. ``quarterlyEarnings`` rows carry
    ``reportedDate``, ``reportedEPS``, ``estimatedEPS``. Pure; no network.

    Used ONLY as a SECOND opinion on est_eps in ``validate_pead_events`` — it
    never feeds the graded trial."""
    rows = (payload or {}).get("quarterlyEarnings", []) or []
    recs = []
    for r in rows:
        est = r.get("estimatedEPS")
        act = r.get("reportedEPS")
        # Drop FUTURE / not-yet-reported quarters (no realized actual) and rows
        # lacking an estimate or a date — matches parse_fmp_earnings' PIT discipline
        # so the AV graded path never admits a null-actual event.
        if est in (None, "None", "") or act in (None, "None", ""):
            continue
        if r.get("reportedDate") in (None, "None", ""):
            continue
        recs.append({
            "ticker": symbol.upper(),
            "ann_date": r.get("reportedDate"),
            "actual_eps": act,
            "est_eps": est,
        })
    if not recs:
        return pd.DataFrame(columns=["ticker", "ann_date", "actual_eps", "est_eps"])
    df = pd.DataFrame(recs)
    df["ann_date"] = pd.to_datetime(df["ann_date"], errors="coerce")
    for c in ("actual_eps", "est_eps"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["ann_date", "est_eps", "actual_eps"]).reset_index(drop=True)


def fetch_alphavantage_cross_check(symbols, env_path: str = ".env",
                                   cache_dir: str = os.path.join("data_cache", "alphavantage"),
                                   max_names: int | None = None) -> pd.DataFrame | None:
    """Build a cross-source events frame from Alpha Vantage IF ALPHAVANTAGE_API_KEY
    is present; return None otherwise (the cross-check is then skipped gracefully).

    Network at call-time only; cached per symbol. AV's free tier is very tight
    (~25 req/day AND ~5 req/min) so this is intended for a small ``max_names``
    sample — enough to establish PIT agreement on the overlap, not to mirror the
    whole universe. Calls are spaced ``_AV_MIN_INTERVAL`` seconds apart to respect
    the per-minute throttle (else calls 6+ in a minute return a rate-note that we
    must skip, silently thinning the sample)."""
    load_env(env_path)
    # Accept either spelling: ALPHAVANTAGE_API_KEY (no underscore, the canonical
    # name in our docs) or ALPHA_VANTAGE_API_KEY (the spelling AV's own dashboard
    # suggests). Reading both prevents a silently-skipped cross-check from a name
    # typo — the exact false-negative that would look like "ran, found nothing".
    key = (os.environ.get("ALPHAVANTAGE_API_KEY")
           or os.environ.get("ALPHA_VANTAGE_API_KEY") or "")
    if not key:
        return None
    os.makedirs(cache_dir, exist_ok=True)
    syms = [str(s).strip().upper() for s in symbols if str(s).strip()]
    if max_names is not None:
        syms = syms[:max_names]
    frames = []
    last = [0.0]
    for s in syms:
        path = os.path.join(cache_dir, f"earnings_{s.replace('/', '_')}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                payload = json.load(fh)
        else:
            wait = _AV_MIN_INTERVAL - (time.monotonic() - last[0])
            if wait > 0:
                time.sleep(wait)
            q = urllib.parse.urlencode({"function": "EARNINGS", "symbol": s, "apikey": key})
            url = f"https://www.alphavantage.co/query?{q}"
            try:
                raw = urllib.request.urlopen(
                    urllib.request.Request(url), timeout=30).read()
                last[0] = time.monotonic()
                payload = json.loads(raw)
            except (urllib.error.URLError, json.JSONDecodeError):
                continue
            if "quarterlyEarnings" not in payload:   # rate note / error -> don't cache
                continue
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
        ev = parse_alphavantage_earnings(payload, s)
        if not ev.empty:
            frames.append(ev)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def build_av_events(
    symbols, cache_dir: str = os.path.join("data_cache", "alphavantage"),
    max_names: int | None = None, allow_fetch: bool = False, env_path: str = ".env",
) -> pd.DataFrame:
    """Tidy PEAD events from Alpha Vantage as a GRADED-TRIAL source (the free path
    that — unlike FMP's 27-symbol demo tier — covers the whole universe). Reads the
    per-symbol AV cache warmed by ``scripts/warm_av_cache.py``; CACHED-ONLY by
    default (``allow_fetch=False``) so a graded run does no network and is fully
    reproducible. Emits the SAME ``_TIDY_COLS`` (+ ``last_updated``) as the FMP
    builder so the gate / filter / SUE / event-study consume AV and FMP
    interchangeably. AV carries no dispersion or lastUpdated, so ``std_est`` /
    ``num_est`` / ``surprise_pct`` / ``last_updated`` are NaN/NaT (compute_sue then
    uses the scale-invariant ``rel_est`` branch — identical value to surprise_pct).

    PIT-safety: a pure field map over cached JSON (parse_alphavantage_earnings
    already drops null-actual rows and references no price/future bar)."""
    syms = [str(s).strip().upper() for s in symbols if str(s).strip()]
    seen: set[str] = set()
    ordered = []
    for s in syms:
        if s not in seen:
            seen.add(s)
            ordered.append(s)
    if max_names is not None:
        ordered = ordered[:max_names]

    if allow_fetch:                     # warm-then-read (rarely used; warm script preferred)
        fetch_alphavantage_cross_check(ordered, env_path=env_path,
                                       cache_dir=cache_dir, max_names=max_names)

    frames = []
    for s in ordered:
        path = os.path.join(cache_dir, f"earnings_{s.replace('/', '_')}.json")
        if not os.path.exists(path):
            continue                    # not warmed yet -> skip (coverage reported upstream)
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        ev = parse_alphavantage_earnings(payload, s)
        if ev.empty:
            continue
        ev = ev.assign(period=np.nan, surprise_pct=np.nan, num_est=np.nan,
                       std_est=np.nan, last_updated=pd.NaT)
        frames.append(ev[_TIDY_COLS + ["last_updated"]])
    if not frames:
        out = pd.DataFrame(columns=_TIDY_COLS + ["last_updated"])
        out["ann_date"] = pd.to_datetime(out["ann_date"])
        out["last_updated"] = pd.to_datetime(out["last_updated"])
        return out
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["ann_date", "ticker"]).reset_index(drop=True)
