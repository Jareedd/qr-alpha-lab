"""Live paper-trading: turn today's predictions into Alpaca paper orders.

Why live paper trading when the backtest verdict was "no defensible edge"?
Because the live IC vs backtest IC comparison is the ultimate out-of-sample
test of the *pipeline* (and of the null result itself), and because running
research infrastructure against a real broker API for weeks is evidence of
engineering most candidates never produce. This is monitoring infrastructure
demonstrated on the best available config -- the write-up will say exactly
that.

Discipline carried over from the backtest, stated explicitly:
- The model trains only on rows whose labels are FULLY REALIZED today
  (dates <= today - horizon). Training on partial labels would quietly use
  prices that don't exist yet.
- Predictions are logged to disk BEFORE any order is sent: the prediction
  record is the primary artifact (it cannot be revised), orders are side
  effects.
- Orders are integer-share, dollar-neutral-ish (rounding breaks exactness;
  the residual is reported, not hidden), and capped per-name.
- Paper endpoint only. The client refuses any base URL that does not
  contain 'paper'.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from quantlab import backtest, features, risk
from quantlab.env import load_env
from quantlab.models import make_model

HORIZON = 21  # must match the research config being monitored

# Longest feature lookback (mom_12_1, pct_52w_high). A name with fewer
# finite bars before the as-of date cannot have those features; that is
# warm-up, not a missing close.
WARMUP_BARS = 252
# Hold the cycle (no orders, no prediction log) when the scored
# cross-section falls below this fraction of the prior cycle, or of the
# members who should have been scoreable today. 64 names dropped on
# 2026-10-14 (496 → 432, fraction 0.87) and the book traded anyway.
UNIVERSE_MIN_FRAC = 0.95
# load_prices start for the live panel. Also the token the revision
# monitor uses to recognize this panel in an old, unmarked cache dir.
LIVE_PRICE_START = "2018-01-01"
# The download asks for every member since 2010, including names the
# vendor cannot price. The floor is a fraction of *today's* index, not
# of that longer list: a panel thinner than the current membership is
# not the cross-section we think we are about to score. This does not
# catch a missing as-of bar — those names are still columns. The
# universe gate does.
LIVE_MIN_SYMBOL_FRAC = 0.90

_PRED_NAME = re.compile(r"predictions_(\d{4}-\d{2}-\d{2})\.csv$")


class LiveCycleHeld(RuntimeError):
    """The cycle stopped before any order and before the prediction log.

    A held cycle must not write ``predictions_`` / ``weights_`` (those are
    the write-once IC record) and must not call the broker. The diagnostic
    JSON under ``results/live/`` is the explanation.
    """


def ny_session_date(now: dt.datetime | None = None) -> dt.date:
    """Calendar date in America/New_York.

    The runner clock is UTC. After midnight UTC it is still the previous
    evening in New York (EDT is UTC−4, EST is UTC−5), so a delayed cycle
    that starts at 00:30 UTC belongs to the session that just closed.
    Keying the price cache on the UTC date put that run in the next day's
    directory. The following night's on-time run then shared the directory,
    could be served the previous session's panel (mtime under one day),
    and collided with the write-once record of that session.
    """
    if now is None:
        now = dt.datetime.now(dt.timezone.utc)
    elif now.tzinfo is None:
        raise ValueError("ny_session_date requires a timezone-aware datetime")
    return now.astimezone(ZoneInfo("America/New_York")).date()


def live_cache_dir(now: dt.datetime | None = None, root: str = "data_cache") -> str:
    """``data_cache/live_{New York session date}``."""
    return os.path.join(root, f"live_{ny_session_date(now).isoformat()}")


def live_min_symbols(n_current: int) -> int:
    """Download floor for the live cycle.

    ``n_current`` is today's index membership, not every historical member.
    Dead names in the 2010-onward request often have no vendor history, so
    a floor against that longer list would fail a healthy download.
    """
    return max(1, int(LIVE_MIN_SYMBOL_FRAC * n_current))


def price_panel_fingerprint(prices: pd.DataFrame) -> dict:
    """Digest of the panel this cycle saw.

    The parquet cache is not committed, and ``attrs`` do not survive it.
    The summary has to be enough to tell two downloads of the same names
    apart later. NaNs are canonicalized to 0 before hashing so a missing
    print is part of the digest rather than a platform-dependent payload.
    """
    ordered = prices.sort_index().sort_index(axis=1)
    vals = ordered.to_numpy(dtype=np.float64, copy=True)
    np.nan_to_num(vals, copy=False, nan=0.0)
    last = ordered.index[-1]
    return {
        "n_rows": int(ordered.shape[0]),
        "n_cols": int(ordered.shape[1]),
        "first": str(pd.Timestamp(ordered.index[0]).date()),
        "last": str(pd.Timestamp(last).date()),
        "asof_closes": int(ordered.loc[last].notna().sum()),
        "sha256": hashlib.sha256(vals.tobytes()).hexdigest(),
    }


def _cell(prices: pd.DataFrame, today, ticker: str):
    value = prices.loc[today, ticker]
    if isinstance(value, pd.Series):
        value = value.iloc[-1]
    return value


def classify_asof_universe(
    prices: pd.DataFrame,
    member_mask: pd.DataFrame | None,
    *,
    warmup_bars: int = WARMUP_BARS,
) -> dict:
    """Why each as-of index member is scored or dropped.

    ``live_target_weights`` keeps a name only when every feature on the
    as-of row is finite (``DataFrame.dropna``). One missing close is
    enough to drop the name, and nothing else was counting those drops.
    This classifier explains that row. It does not decide the book.

    Point-in-time: features are ``features.build_features``, which reads
    prices at or before the as-of date only. Reasons, first match wins:

    - ``missing_from_panel``: an as-of member with no price column
    - ``missing_close``: the as-of bar itself is missing (the 2026-10-14
      failure mode — the name has history, just not tonight's print)
    - ``warmup``: the as-of close exists but fewer than ``warmup_bars``
      finite prices precede it, so the 12-1 / 52-week features cannot exist
    - ``feature_gap``: close and history exist, but some feature is NaN
      (an interior hole)
    """
    today = prices.index[-1]
    feats = features.build_features(prices, member_mask=member_mask)
    if member_mask is not None:
        flags = member_mask.loc[today]
        if isinstance(flags, pd.DataFrame):
            flags = flags.iloc[-1]
        members = [t for t in flags.index if bool(flags[t])]
    else:
        members = list(prices.columns)

    scored: list[str] = []
    dropped: list[dict] = []
    for ticker in members:
        if ticker not in prices.columns:
            dropped.append({"ticker": ticker, "reason": "missing_from_panel"})
            continue
        px = _cell(prices, today, ticker)
        if not np.isfinite(px):
            dropped.append({"ticker": ticker, "reason": "missing_close"})
            continue
        hist = prices[ticker].loc[:today]
        if isinstance(hist, pd.DataFrame):
            hist = hist.iloc[:, 0]
        n_hist = int(hist.iloc[:-1].notna().sum()) if len(hist) else 0
        if n_hist < warmup_bars:
            dropped.append({"ticker": ticker, "reason": "warmup", "n_history": n_hist})
            continue
        missing = []
        for name, frame in feats.items():
            if ticker not in frame.columns or not np.isfinite(_cell(frame, today, ticker)):
                missing.append(name)
        if missing:
            dropped.append(
                {"ticker": ticker, "reason": "feature_gap", "missing_features": missing}
            )
            continue
        scored.append(ticker)
    return {
        "asof": str(pd.Timestamp(today).date()),
        "scored": scored,
        "dropped": dropped,
    }


def prior_prediction_count(out_dir: str, asof: str) -> tuple[str | None, int | None]:
    """(date, row count) of the latest predictions log strictly before ``asof``."""
    if not os.path.isdir(out_dir):
        return None, None
    best: str | None = None
    for fn in os.listdir(out_dir):
        m = _PRED_NAME.fullmatch(fn)
        if m and m.group(1) < asof and (best is None or m.group(1) > best):
            best = m.group(1)
    if best is None:
        return None, None
    path = os.path.join(out_dir, f"predictions_{best}.csv")
    with open(path) as fh:
        n = max(sum(1 for _ in fh) - 1, 0)
    return best, n


def universe_hold_decision(
    n_scored: int,
    report: dict,
    prior_n: int | None,
    *,
    prior_asof: str | None = None,
    min_frac: float = UNIVERSE_MIN_FRAC,
    scored_names: set[str] | None = None,
) -> dict:
    """Whether the scored cross-section shrank enough to hold the cycle.

    Reference 1, when a prior prediction log exists: that cycle's name
    count. Reference 2, always: members who should have been scoreable
    today (scored + missing close + feature gap). Warm-up and names the
    vendor has no history for are excluded from reference 2, because they
    are not a silent shrink of a previously traded book.

    A disagreement between this classifier and the model's ``dropna`` row
    also holds. Trading an unexplained cross-section is the failure mode
    this gate exists to stop.
    """
    counts: dict[str, int] = {}
    for row in report["dropped"]:
        counts[row["reason"]] = counts.get(row["reason"], 0) + 1
    n_missing_close = counts.get("missing_close", 0)
    n_gap = counts.get("feature_gap", 0)
    n_warm = counts.get("warmup", 0)
    n_absent = counts.get("missing_from_panel", 0)
    eligible = n_scored + n_missing_close + n_gap
    reasons: list[str] = []
    if scored_names is not None and set(report["scored"]) != set(scored_names):
        reasons.append(
            "classifier scored names disagree with the model's dropna row; "
            "holding rather than trading an unexplained cross-section"
        )
    if prior_n is not None and n_scored < min_frac * prior_n:
        reasons.append(
            f"scored {n_scored} < {min_frac:.0%} of prior cycle "
            f"{prior_asof or '?'} ({prior_n} names)"
        )
    if eligible and n_scored < min_frac * eligible:
        reasons.append(
            f"scored {n_scored} < {min_frac:.0%} of scoreable members "
            f"({eligible} = scored + {n_missing_close} missing_close + "
            f"{n_gap} feature_gap; {n_warm} warmup and {n_absent} "
            "missing_from_panel excluded)"
        )
    return {
        "hold": bool(reasons),
        "reason": "; ".join(reasons),
        "n_scored": int(n_scored),
        "n_prior": None if prior_n is None else int(prior_n),
        "prior_asof": prior_asof,
        "n_eligible": int(eligible),
        "n_missing_close": n_missing_close,
        "n_feature_gap": n_gap,
        "n_warmup": n_warm,
        "n_missing_from_panel": n_absent,
        "min_frac": min_frac,
    }


def _write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)


def _download_report_dict(prices: pd.DataFrame) -> dict | None:
    report = prices.attrs.get("download_report")
    if report is None:
        return None
    if hasattr(report, "as_dict"):
        return report.as_dict()
    return None


def _dropped_detail(report: dict, decision: dict) -> str:
    """One log line: the decision, then the names, capped so a traceback fits."""
    parts = [decision["reason"]] if decision["reason"] else []
    for reason, limit in (("missing_close", 40), ("feature_gap", 20)):
        names = [d["ticker"] for d in report["dropped"] if d["reason"] == reason]
        if not names:
            continue
        shown = ", ".join(names[:limit])
        extra = f", ... (+{len(names) - limit} more)" if len(names) > limit else ""
        parts.append(f"{reason}: {shown}{extra}")
    parts.append("No orders sent; predictions were not logged.")
    return ". ".join(parts)


def apply_session_gate(out_dir: str, prices: pd.DataFrame, session: dt.date) -> None:
    """Hold when the panel does not contain the New York session.

    Writing a record under the panel's last date would either collide
    with that session's write-once files or silently re-trade yesterday's
    close. A market holiday fails here on purpose: there is no new bar,
    and the previous session's record already exists.
    """
    last = pd.Timestamp(prices.index[-1]).date()
    if last >= session:
        return
    payload = {
        "expected_session": session.isoformat(),
        "panel_last_bar": str(last),
        "held": True,
        "orders_sent": False,
        "predictions_logged": False,
        "price_panel": price_panel_fingerprint(prices),
        "download_report": _download_report_dict(prices),
        "reason": (
            f"price panel ends {last}, before the New York session "
            f"{session.isoformat()}. Refusing to reuse that bar: it would "
            "collide with the write-once record or trade the previous close."
        ),
    }
    path = os.path.join(out_dir, f"session_hold_{session.isoformat()}.json")
    _write_json(path, payload)
    raise LiveCycleHeld(payload["reason"])


def apply_universe_gate(
    out_dir: str,
    prices: pd.DataFrame,
    member_mask: pd.DataFrame | None,
    predictions: pd.DataFrame,
) -> dict:
    """Hold when the scored cross-section dropped sharply.

    Returns the decision record on a pass (the caller stores it on the
    summary). On a hold, writes ``universe_hold_{asof}.json`` with every
    dropped name and reason, then raises ``LiveCycleHeld`` before the
    prediction log and before any order.
    """
    report = classify_asof_universe(prices, member_mask)
    asof = str(prices.index[-1].date())
    prior_asof, prior_n = prior_prediction_count(out_dir, asof)
    decision = universe_hold_decision(
        len(predictions),
        report,
        prior_n,
        prior_asof=prior_asof,
        scored_names=set(predictions.index),
    )
    record = {
        "asof": asof,
        "held": decision["hold"],
        "orders_sent": False,
        **{k: decision[k] for k in (
            "n_scored", "n_prior", "prior_asof", "n_eligible",
            "n_missing_close", "n_feature_gap", "n_warmup",
            "n_missing_from_panel", "min_frac", "reason",
        )},
        "dropped": report["dropped"],
        "price_panel": price_panel_fingerprint(prices),
        "download_report": _download_report_dict(prices),
    }
    if decision["hold"]:
        record["predictions_logged"] = False
        path = os.path.join(out_dir, f"universe_hold_{asof}.json")
        _write_json(path, record)
        raise LiveCycleHeld(_dropped_detail(report, decision))
    return record


# ---------------------------------------------------------------------------
# Pure logic (tested offline)
# ---------------------------------------------------------------------------

def live_target_weights(
    prices: pd.DataFrame,
    member_mask: pd.DataFrame | None,
    sectors: dict[str, str],
    model_name: str = "ridge",
    horizon: int = HORIZON,
    quantile: float = 0.1,
    min_names: int = 50,
) -> tuple[pd.Series, pd.DataFrame]:
    """Fit on fully-labeled history, predict today's cross-section, build
    decile weights with sector demean + beta projection (the trial-#5-style
    construction).

    Returns ``(weights, predictions)`` for the LAST date in ``prices``:
    weights drive orders; predictions (columns ``pred_raw``,
    ``pred_sector_neutral`` and ``baseline_mom_12_1``, indexed by ticker)
    are the IC-bearing artifact. The backtest's ``mean_rank_ic`` is computed
    on PRE-demean predictions, so live IC must be measured on ``pred_raw``
    to be comparable; the neutralized column documents what actually drove
    the book.

    ``baseline_mom_12_1`` is the live experiment's CONTROL ARM: the raw
    12-1 momentum feature (the law-#5 baseline) logged on the same names
    the same day. If the model's live IC degrades vs backtest, the
    baseline's own live-vs-backtest gap tells us whether the model decayed
    or the period was hostile to everything. Z-scoring is monotone within a
    date, so its rank IC is identical to the backtest baseline's. No orders
    are ever sent for it — shadow-logged only, so it cannot leak anywhere.
    """
    feats = features.build_features(prices, member_mask=member_mask)
    labels = features.build_labels(
        prices, horizon=horizon, residualize=True, member_mask=member_mask
    )
    panel = features.stack_panel(feats, labels)

    today = prices.index[-1]
    cutoff = prices.index[-(horizon + 1)]  # last date with a complete label
    train = panel[panel.index.get_level_values("date") <= cutoff]
    if len(train) < 10_000:
        raise RuntimeError(f"training panel too small ({len(train)} rows)")

    feature_cols = [c for c in train.columns if c != "label"]
    model = make_model(model_name)
    model.fit(train[feature_cols].to_numpy(), train["label"].to_numpy())

    # Today's feature row: build directly from the feature frames (the panel
    # drops today because its label is NaN -- by design).
    row = pd.DataFrame({n: f.loc[today] for n, f in feats.items()}).dropna()
    if member_mask is not None:
        row = row[member_mask.loc[today].reindex(row.index).fillna(False)]
    if len(row) < min_names:
        raise RuntimeError(f"only {len(row)} tradable names today (< {min_names})")

    raw = pd.Series(model.predict(row[feature_cols].to_numpy()), index=row.index)
    raw.index = pd.MultiIndex.from_product(
        [[today], raw.index], names=["date", "ticker"]
    )
    neutral = risk.neutralize_predictions_by_sector(raw, sectors)

    weights = backtest.predictions_to_weights(neutral, quantile=quantile, rebalance_every=1)
    rets = prices.pct_change(fill_method=None)
    mkt = rets.where(member_mask).mean(axis=1) if member_mask is not None else rets.mean(axis=1)
    betas = risk.rolling_beta(rets, mkt)
    weights = risk.beta_neutralize_weights(weights, betas)
    predictions = pd.DataFrame(
        {
            "pred_raw": raw.droplevel("date"),
            "pred_sector_neutral": neutral.droplevel("date"),
            "baseline_mom_12_1": row["mom_12_1"],
        }
    )
    return weights.iloc[0], predictions


def assert_write_once(paths: list[str], allow_overwrite: bool = False) -> None:
    """Refuse to clobber existing per-date live logs.

    The prediction log is only evidence if it is write-once: a record that
    can be silently regenerated later (same-day re-run, CI retry, a stray
    --dry-run) is a revisable prediction, which is no prediction at all.
    """
    if allow_overwrite:
        return
    existing = [p for p in paths if os.path.exists(p)]
    if existing:
        raise RuntimeError(
            f"refusing to overwrite immutable live record(s): {existing} "
            "(pass allow_overwrite=True / --allow-overwrite only for a "
            "deliberate re-run of a failed cycle)"
        )


def orders_from_weights(
    target_w: pd.Series,
    current_qty: dict[str, float],
    last_prices: pd.Series,
    equity: float,
    max_name_frac: float = 0.05,
) -> list[dict]:
    """Integer-share order list moving current positions to target weights.

    Pure function: no IO, fully testable. Per-name notional is capped at
    ``max_name_frac`` of equity (belt and braces against a bad weight).
    """
    orders = []
    tickers = set(target_w[target_w != 0].index) | set(current_qty)
    for t in sorted(tickers):
        px = float(last_prices.get(t, np.nan))
        if not np.isfinite(px) or px <= 0:
            continue
        w = float(target_w.get(t, 0.0))
        w = float(np.clip(w, -max_name_frac, max_name_frac))
        target_shares = int(w * equity / px)  # truncates toward zero
        delta = target_shares - int(current_qty.get(t, 0))
        if delta != 0:
            orders.append(
                {"symbol": t, "qty": abs(delta), "side": "buy" if delta > 0 else "sell"}
            )
    return orders


# ---------------------------------------------------------------------------
# Alpaca paper REST client (thin, urllib-only, refuses non-paper endpoints)
# ---------------------------------------------------------------------------

class AlpacaPaper:
    def __init__(self, env_path: str = ".env"):
        load_env(env_path)
        self.key = os.environ.get("ALPACA_API_KEY_ID", "")
        self.secret = os.environ.get("ALPACA_API_SECRET_KEY", "")
        base = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        self.base = base.rstrip("/").removesuffix("/v2")
        if "paper" not in self.base:
            raise RuntimeError(f"refusing non-paper endpoint: {self.base}")
        if not self.key or not self.secret:
            raise RuntimeError("Alpaca credentials missing (see .env.example)")

    def _req(self, path: str, payload: dict | None = None, method: str | None = None):
        req = urllib.request.Request(
            f"{self.base}/v2{path}",
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={
                "APCA-API-KEY-ID": self.key,
                "APCA-API-SECRET-KEY": self.secret,
                "Content-Type": "application/json",
            },
            method=method or ("POST" if payload is not None else "GET"),
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            body = r.read().decode()
            return json.loads(body) if body else {}

    def account(self) -> dict:
        return self._req("/account")

    def positions(self) -> dict[str, float]:
        return {p["symbol"]: float(p["qty"]) for p in self._req("/positions")}

    def submit_order(self, symbol: str, qty: int, side: str) -> dict:
        return self._req(
            "/orders",
            {
                "symbol": symbol,
                "qty": str(qty),
                "side": side,
                "type": "market",
                "time_in_force": "day",
            },
        )


# ---------------------------------------------------------------------------
# Daily entrypoint
# ---------------------------------------------------------------------------

def run_daily(
    model_name: str = "ridge",
    out_dir: str = "results/live",
    env_path: str = ".env",
    submit: bool = True,
    allow_overwrite: bool = False,
) -> dict:
    """One live cycle: data -> predictions (logged first) -> orders.

    The per-date logs are write-once: a second run on the same as-of date
    raises instead of silently replacing the record (a revisable prediction
    log is worthless as evidence). ``allow_overwrite=True`` is the explicit,
    deliberate escape hatch for re-running a cycle that failed mid-way.
    """
    from quantlab import revisions, universe as univ
    from quantlab.data import DataUnavailableError, load_prices

    # Cache directory and the "is this tonight's close" check both use the
    # New York session date, not the runner's UTC date. See ny_session_date.
    session = ny_session_date()
    today_tag = session.isoformat()
    cache_dir = os.path.join("data_cache", f"live_{today_tag}")
    os.makedirs(out_dir, exist_ok=True)

    current, changes = univ.fetch_sp500_tables(cache_dir=cache_dir)
    intervals = univ.build_membership_intervals(current, changes, start="2010-01-01")
    members = univ.all_members_in_window(intervals)
    # Floor vs today's index, not vs every historical member (many are
    # unpriceable). A download that thin is not a tradeable cross-section.
    min_symbols = live_min_symbols(len(current))
    try:
        prices = load_prices(
            members,
            start=LIVE_PRICE_START,
            cache_dir=cache_dir,
            min_coverage=0.0,
            min_symbols=min_symbols,
            min_last_bar=today_tag,
        )
    except DataUnavailableError as exc:
        path = os.path.join(out_dir, f"download_hold_{today_tag}.json")
        _write_json(path, {
            "expected_session": today_tag,
            "held": True,
            "orders_sent": False,
            "predictions_logged": False,
            "min_symbols": min_symbols,
            "reason": str(exc),
        })
        raise LiveCycleHeld(str(exc)) from exc

    report = prices.attrs.get("download_report")
    cache_path = getattr(report, "cache_path", "") if report is not None else ""
    if cache_path:
        # Explicit file for the revision monitor. The report used to write
        # a second parquet into this same directory, and the monitor took
        # whichever name sorted last.
        revisions.write_live_panel_marker(cache_dir, cache_path)

    apply_session_gate(out_dir, prices, session)
    member_mask = univ.membership_mask(prices.index, prices.columns, intervals)
    sectors = univ.sector_map(current, list(prices.columns))

    target, predictions = live_target_weights(
        prices, member_mask, sectors, model_name=model_name
    )
    universe_record = apply_universe_gate(out_dir, prices, member_mask, predictions)

    # Primary artifacts first, immutable record before any order exists:
    # the full prediction cross-section (live IC is measured on pred_raw,
    # the same pre-demean object the backtest's mean_rank_ic uses), then
    # the order-bearing weights.
    asof = str(prices.index[-1].date())
    pred_path = os.path.join(out_dir, f"predictions_{asof}.csv")
    record_path = os.path.join(out_dir, f"weights_{asof}.csv")
    assert_write_once([pred_path, record_path], allow_overwrite)
    predictions.rename_axis("ticker").to_csv(pred_path)
    target[target != 0].rename("weight").to_csv(record_path)

    summary: dict = {
        "asof": asof,
        "n_names": int((target != 0).sum()),
        "submitted": False,
        "session": today_tag,
        "price_panel": universe_record["price_panel"],
        "download_report": universe_record["download_report"],
        "universe": {
            k: universe_record[k]
            for k in (
                "n_scored", "n_prior", "prior_asof", "n_eligible",
                "n_missing_close", "n_feature_gap", "n_warmup",
                "n_missing_from_panel", "min_frac", "held",
            )
        },
        "dropped": universe_record["dropped"],
    }
    if submit:
        broker = AlpacaPaper(env_path)
        acct = broker.account()
        equity = float(acct["equity"])
        orders = orders_from_weights(
            target, broker.positions(), prices.iloc[-1], equity
        )
        sent, failed = 0, []
        for o in orders:
            try:
                broker.submit_order(o["symbol"], o["qty"], o["side"])
                sent += 1
            except urllib.error.HTTPError as e:  # e.g. not shortable today
                failed.append({**o, "error": e.read().decode()[:120]})
        summary.update(
            {"submitted": True, "equity": equity, "orders_sent": sent,
             "orders_failed": failed[:20], "n_failed": len(failed)}
        )

    # Data-revision fingerprint: compare today's freshly downloaded history
    # against the previous cycle's snapshot of the SAME past (see
    # quantlab.revisions). Strictly after trading and wrapped: a measurement
    # bug must never cost a cycle. Read-only -- it observes vendor drift,
    # never adjusts for it.
    try:
        from quantlab import revisions

        rev = revisions.snapshot_revision_summary("data_cache", today_tag, prices)
        if rev is not None:
            with open(os.path.join(out_dir, f"revisions_{asof}.json"), "w") as f:
                json.dump(rev, f, indent=2)
            summary["revisions"] = {
                k: rev[k]
                for k in (
                    "compared_to", "gap_weekdays", "is_consecutive_cycle",
                    "frac_price_cells_changed", "n_return_cells_changed",
                    "comparison_file", "comparison_selection",
                    "prior_bar", "today_bar", "gap_weekdays_bars",
                )
                if k in rev
            }
    except Exception as exc:  # noqa: BLE001 -- monitoring is best-effort
        summary["revisions_error"] = str(exc)[:200]

    with open(os.path.join(out_dir, f"summary_{asof}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary
