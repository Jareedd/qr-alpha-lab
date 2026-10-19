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
- Paper endpoint only. The client accepts a base URL only when it is
  https and its hostname is exactly paper-api.alpaca.markets (optional
  /v2 path). A substring match is not a guard: ``?paper`` and
  ``paper.evil.example`` both contain the letters and both are refused.
- Order submission is paused unless ``QR_SUBMIT_ORDERS`` is explicitly
  turned on (or the CLI is passed ``--submit-orders``). A paused cycle
  still logs predictions; the summary says ``orders_status=paused`` so
  later P&L does not treat the unsent target as a rebalance.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from quantlab import backtest, features, risk
from quantlab.env import load_env
from quantlab.models import make_model

HORIZON = 21  # must match the research config being monitored

# Operational switch for the nightly job. Unset, empty, or a falsy value
# pauses broker submission. Only an explicit truthy value re-enables it.
# Default-paused is deliberate: a missing env var must not start sending
# orders again after this was turned off.
SUBMIT_ORDERS_ENV = "QR_SUBMIT_ORDERS"
_SUBMIT_TRUTHY = frozenset({"1", "true", "yes", "on"})
_SUBMIT_FALSY = frozenset({"0", "false", "no", "off", ""})
PAUSED_ORDERS_NOTE = (
    "ORDERS PAUSED: broker submission skipped. Predictions and weights "
    "were logged; no order was sent. Re-enable paper submission with "
    "QR_SUBMIT_ORDERS=1 or --submit-orders."
)


def resolve_orders_status(submit: bool, env_value: str | None) -> str:
    """Decide what this cycle does with the broker.

    Returns ``"dry_run"``, ``"paused"``, or ``"submit"``.

    ``submit=False`` is the explicit dry-run (local rehearsal). It is not
    the operational pause: the record is marked ``dry_run`` so a later
    reader can tell a one-off rehearsal from a halted live book.

    ``submit=True`` is what the nightly job asks for. Submission then
    happens only when ``QR_SUBMIT_ORDERS`` is an explicit truthy value.
    Unset or falsy pauses. An unrecognized value raises rather than
    guessing — a typo must not silently resume trading.
    """
    if not submit:
        return "dry_run"
    raw = "" if env_value is None else str(env_value).strip().lower()
    if raw in _SUBMIT_TRUTHY:
        return "submit"
    if raw in _SUBMIT_FALSY:
        return "paused"
    raise ValueError(
        f"{SUBMIT_ORDERS_ENV}={env_value!r} is not a recognized switch. "
        "Use 1 to submit paper orders or 0 to keep them paused."
    )


def submission_inputs(dry_run: bool, submit_orders: bool) -> tuple[bool, str | None]:
    """Map CLI flags onto ``run_daily(submit=...)`` plus an env override.

    ``--submit-orders`` re-enables submission for this process even if
    the environment says paused, by returning an override of ``"1"``.
    ``--dry-run`` passes ``submit=False``, which wins over a truthy env
    var. The two flags together are an error. With neither flag, the
    env var is left alone and ``run_daily`` applies the default (paused).
    """
    if dry_run and submit_orders:
        raise ValueError("pass only one of --dry-run and --submit-orders")
    if dry_run:
        return False, None
    if submit_orders:
        return True, "1"
    return True, None


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


def record_order_submission(
    summary: dict,
    status: str,
    *,
    broker_factory,
    target_weights: pd.Series,
    last_prices: pd.Series,
) -> dict:
    """Attach the order outcome to a cycle summary.

    ``status="paused"`` and ``status="dry_run"`` never call
    ``broker_factory``. The paused record is how later IC/P&L code tells
    an unsent target book from a book the broker actually holds.
    """
    if status not in {"paused", "dry_run", "submit"}:
        raise ValueError(f"unknown orders status: {status!r}")
    out = dict(summary)
    out["orders_status"] = status
    out["orders_paused"] = status == "paused"
    out["submitted"] = False
    if status == "paused":
        out["orders_note"] = PAUSED_ORDERS_NOTE
        logger.warning(PAUSED_ORDERS_NOTE)
        return out
    if status == "dry_run":
        out["orders_note"] = "dry-run: predictions logged, no orders sent"
        return out

    broker = broker_factory()
    acct = broker.account()
    equity = float(acct["equity"])
    orders = orders_from_weights(
        target_weights, broker.positions(), last_prices, equity
    )
    sent, failed = 0, []
    for o in orders:
        try:
            broker.submit_order(o["symbol"], o["qty"], o["side"])
            sent += 1
        except urllib.error.HTTPError as e:  # e.g. not shortable today
            failed.append({**o, "error": e.read().decode()[:120]})
    out.update(
        {
            "submitted": True,
            "equity": equity,
            "orders_sent": sent,
            "orders_failed": failed[:20],
            "n_failed": len(failed),
        }
    )
    return out


# ---------------------------------------------------------------------------
# Alpaca paper REST client (thin, urllib-only, refuses non-paper endpoints)
# ---------------------------------------------------------------------------

# Alpaca's paper trading API is this host and no other. The live host is
# api.alpaca.markets; a guard that searches the raw string for "paper"
# accepts that host with ``?paper`` bolted on, and any lookalike whose
# name merely contains the substring.
PAPER_API_HOST = "paper-api.alpaca.markets"
_PAPER_API_ORIGIN = f"https://{PAPER_API_HOST}"


def paper_base_url(raw: str) -> str:
    """Return the canonical paper origin, or raise.

    Fail closed. The URL must be ``https`` and its parsed hostname must be
    exactly ``paper-api.alpaca.markets``. An optional ``/v2`` path is
    allowed because ``_req`` appends ``/v2`` itself; the stored base is
    always the canonical origin, so a query, fragment, userinfo, explicit
    port, or any other path cannot change where orders go.

    Assumption: Alpaca serves paper trading only at that hostname over
    HTTPS. Anything else — including the live host — is refused.
    """
    try:
        parsed = urllib.parse.urlparse(raw.strip())
        host = parsed.hostname
        port = parsed.port
        has_userinfo = parsed.username is not None or parsed.password is not None
    except ValueError:
        # Malformed host or port. urlparse is permissive; the accessors are not.
        raise RuntimeError(f"refusing non-paper endpoint: {raw}") from None
    path = parsed.path.rstrip("/")
    if (
        parsed.scheme != "https"
        or host != PAPER_API_HOST
        or port is not None
        or has_userinfo
        or path not in ("", "/v2")
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError(f"refusing non-paper endpoint: {raw}")
    return _PAPER_API_ORIGIN


class AlpacaPaper:
    def __init__(self, env_path: str = ".env"):
        load_env(env_path)
        self.key = os.environ.get("ALPACA_API_KEY_ID", "")
        self.secret = os.environ.get("ALPACA_API_SECRET_KEY", "")
        base = os.environ.get("ALPACA_BASE_URL", _PAPER_API_ORIGIN)
        self.base = paper_base_url(base)
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

    ``submit=True`` does not by itself send orders. Broker submission is
    paused unless ``QR_SUBMIT_ORDERS`` is explicitly truthy (see
    ``resolve_orders_status``). Paused cycles still write the prediction
    and weight logs, and the summary is marked ``orders_status="paused"``.
    """
    from quantlab import universe as univ
    from quantlab.data import load_prices

    today_tag = dt.date.today().isoformat()
    cache_dir = os.path.join("data_cache", f"live_{today_tag}")
    os.makedirs(out_dir, exist_ok=True)

    current, changes = univ.fetch_sp500_tables(cache_dir=cache_dir)
    intervals = univ.build_membership_intervals(current, changes, start="2010-01-01")
    members = univ.all_members_in_window(intervals)
    prices = load_prices(members, start="2018-01-01", cache_dir=cache_dir, min_coverage=0.0)
    member_mask = univ.membership_mask(prices.index, prices.columns, intervals)
    sectors = univ.sector_map(current, list(prices.columns))

    target, predictions = live_target_weights(
        prices, member_mask, sectors, model_name=model_name
    )

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

    # Default is paused. ``submit=True`` (the nightly job) still does not
    # touch the broker unless QR_SUBMIT_ORDERS is explicitly truthy.
    status = resolve_orders_status(submit, os.environ.get(SUBMIT_ORDERS_ENV))
    summary = record_order_submission(
        {"asof": asof, "n_names": int((target != 0).sum()), "submitted": False},
        status,
        broker_factory=lambda: AlpacaPaper(env_path),
        target_weights=target,
        last_prices=prices.iloc[-1],
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
                for k in ("compared_to", "gap_weekdays", "is_consecutive_cycle",
                          "frac_price_cells_changed", "n_return_cells_changed")
            }
    except Exception as exc:  # noqa: BLE001 -- monitoring is best-effort
        summary["revisions_error"] = str(exc)[:200]

    with open(os.path.join(out_dir, f"summary_{asof}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary
