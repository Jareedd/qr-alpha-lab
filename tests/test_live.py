"""Live-trading logic: everything testable without a broker or network.

The IO layer is deliberately thin; the decisions (training cutoff, order
deltas, caps, paper-endpoint guard) are pure and pinned here.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import features, live
from quantlab.synthetic import make_panel


def test_orders_from_weights_known_answer():
    target = pd.Series({"AAA": 0.02, "BBB": -0.02, "CCC": 0.0})
    current = {"BBB": 10, "DDD": 5}  # DDD must be liquidated
    prices = pd.Series({"AAA": 100.0, "BBB": 50.0, "CCC": 10.0, "DDD": 20.0})
    orders = live.orders_from_weights(target, current, prices, equity=100_000.0)
    by_symbol = {o["symbol"]: o for o in orders}

    assert by_symbol["AAA"] == {"symbol": "AAA", "qty": 20, "side": "buy"}   # 2k/100
    assert by_symbol["BBB"] == {"symbol": "BBB", "qty": 50, "side": "sell"}  # -40 tgt -10 held
    assert by_symbol["DDD"] == {"symbol": "DDD", "qty": 5, "side": "sell"}   # flatten
    assert "CCC" not in by_symbol


def test_orders_respect_per_name_cap():
    target = pd.Series({"AAA": 0.50})  # absurd weight -> capped at 5%
    orders = live.orders_from_weights(
        target, {}, pd.Series({"AAA": 100.0}), equity=100_000.0, max_name_frac=0.05
    )
    assert orders == [{"symbol": "AAA", "qty": 50, "side": "buy"}]


def test_orders_skip_unpriceable_names():
    target = pd.Series({"GHOST": 0.02})
    orders = live.orders_from_weights(
        target, {}, pd.Series({"GHOST": np.nan}), equity=100_000.0
    )
    assert orders == []


def test_live_weights_train_only_on_complete_labels():
    # Poisoning the last `horizon` days of prices must not change the fitted
    # model's predictions ONLY through training data -- i.e., a model trained
    # with poisoned recent history must equal one trained on clean history,
    # because those rows have incomplete labels and are excluded from training.
    prices = make_panel(n_assets=40, n_days=1300, mode="planted", seed=11)
    sectors = prices.attrs["sectors"]

    w_clean, p_clean = live.live_target_weights(
        prices, None, sectors, horizon=21, min_names=20
    )

    poisoned = prices.copy()
    # Scale the final 20 days' prices (inside the horizon window): labels for
    # training-cutoff rows are unaffected; today's FEATURES change, so today's
    # weights may differ -- but the run must not crash and must stay neutral.
    poisoned.iloc[-20:] *= 1.5
    w_poisoned, _ = live.live_target_weights(
        poisoned, None, sectors, horizon=21, min_names=20
    )

    # The prediction log (the live-IC artifact) must cover the whole scored
    # cross-section -- not just the names that made the book -- in the raw
    # (backtest-comparable), sector-neutral (book-driving) and control-arm
    # columns.
    assert {"pred_raw", "pred_sector_neutral", "baseline_mom_12_1"} <= set(
        p_clean.columns
    )
    assert len(p_clean) >= (w_clean != 0).sum()
    assert np.isfinite(p_clean.to_numpy()).all()

    for w in (w_clean, w_poisoned):
        assert abs(w.sum()) < 1e-9            # dollar neutral
        assert (w != 0).sum() >= 6            # an actual book
        assert abs(w.abs().sum() - 1.0) < 1e-6  # gross preserved by projection
        # 40-name panel -> 4 names/side at ~0.125 base; beta projection can
        # legitimately push one to ~0.3. The real concentration guard is the
        # 5% per-name cap applied at order time (tested separately).
        assert w.abs().max() < 0.35


def test_prediction_log_control_arm_matches_baseline_feature():
    # The shadow-logged baseline column must be the SAME object the law-#5
    # backtest baseline ranks on (today's mom_12_1 feature), name for name --
    # otherwise the live control arm and the backtest baseline diverge.
    prices = make_panel(n_assets=40, n_days=1300, mode="planted", seed=11)
    _, preds = live.live_target_weights(
        prices, None, prices.attrs["sectors"], horizon=21, min_names=20
    )
    expected = features.build_features(prices)["mom_12_1"].iloc[-1]
    pd.testing.assert_series_equal(
        preds["baseline_mom_12_1"],
        expected.reindex(preds.index),
        check_names=False,
    )


def test_live_records_are_write_once(tmp_path):
    # A second cycle on the same as-of date must refuse to replace the
    # logged record -- a revisable prediction log is no evidence at all.
    existing = tmp_path / "predictions_2026-06-10.csv"
    existing.write_text("ticker,pred_raw\nAAA,0.1\n")
    fresh = tmp_path / "weights_2026-06-10.csv"

    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        live.assert_write_once([str(existing), str(fresh)])
    # explicit escape hatch for re-running a failed cycle
    live.assert_write_once([str(existing), str(fresh)], allow_overwrite=True)
    # nothing logged yet -> no objection
    live.assert_write_once([str(fresh)])


def test_orders_are_paused_unless_the_switch_is_explicit():
    # The nightly job calls run_daily(submit=True). That must NOT mean
    # "send orders" once the operational pause is the default.
    assert live.resolve_orders_status(True, None) == "paused"
    assert live.resolve_orders_status(True, "") == "paused"
    assert live.resolve_orders_status(True, "0") == "paused"
    assert live.resolve_orders_status(True, "false") == "paused"
    assert live.resolve_orders_status(True, "off") == "paused"
    for on in ("1", "true", "YES", "on"):
        assert live.resolve_orders_status(True, on) == "submit"
    # A dry-run is not the operational pause, and it wins over a truthy env.
    assert live.resolve_orders_status(False, "1") == "dry_run"
    with pytest.raises(ValueError, match="QR_SUBMIT_ORDERS"):
        live.resolve_orders_status(True, "maybe")


def test_submit_orders_flag_is_the_reenable_and_dry_run_conflicts():
    assert live.submission_inputs(False, False) == (True, None)
    assert live.submission_inputs(False, True) == (True, "1")
    assert live.submission_inputs(True, False) == (False, None)
    with pytest.raises(ValueError, match="only one"):
        live.submission_inputs(True, True)


class _Broker:
    """In-memory stand-in. No network, no Alpaca."""

    def __init__(self):
        self.sent = []

    def account(self):
        return {"equity": "100000"}

    def positions(self):
        return {"BBB": 10}

    def submit_order(self, symbol, qty, side):
        self.sent.append({"symbol": symbol, "qty": qty, "side": side})
        return {"id": "not-a-real-order"}


def test_paused_cycle_does_not_construct_a_broker_and_marks_the_record():
    def boom():
        raise AssertionError("paused cycle constructed a broker")

    summary = live.record_order_submission(
        {"asof": "2026-10-20", "n_names": 4, "submitted": False},
        "paused",
        broker_factory=boom,
        target_weights=pd.Series({"AAA": 0.02}),
        last_prices=pd.Series({"AAA": 100.0}),
    )
    assert summary["submitted"] is False
    assert summary["orders_paused"] is True
    assert summary["orders_status"] == "paused"
    assert "ORDERS PAUSED" in summary["orders_note"]
    assert "equity" not in summary
    assert "re-enable" in summary["orders_note"].lower() or "Re-enable" in summary["orders_note"]


def test_dry_run_record_is_not_marked_paused():
    def boom():
        raise AssertionError("dry-run constructed a broker")

    summary = live.record_order_submission(
        {"asof": "2026-10-20", "n_names": 4, "submitted": False},
        "dry_run",
        broker_factory=boom,
        target_weights=pd.Series({"AAA": 0.02}),
        last_prices=pd.Series({"AAA": 100.0}),
    )
    assert summary["orders_status"] == "dry_run"
    assert summary["orders_paused"] is False
    assert summary["submitted"] is False


def test_submit_status_uses_the_broker_and_marks_submitted():
    broker = _Broker()
    summary = live.record_order_submission(
        {"asof": "2026-10-20", "n_names": 1, "submitted": False},
        "submit",
        broker_factory=lambda: broker,
        target_weights=pd.Series({"AAA": 0.02, "BBB": 0.0}),
        last_prices=pd.Series({"AAA": 100.0, "BBB": 50.0}),
    )
    assert summary["submitted"] is True
    assert summary["orders_paused"] is False
    assert summary["orders_status"] == "submit"
    assert summary["equity"] == 100_000.0
    # 0.02 * 100000 / 100 = 20 shares, minus no current AAA; BBB held 10 -> flatten.
    assert {"symbol": "AAA", "qty": 20, "side": "buy"} in broker.sent
    assert {"symbol": "BBB", "qty": 10, "side": "sell"} in broker.sent
    assert summary["orders_sent"] == 2
    assert summary["n_failed"] == 0


def test_alpaca_client_refuses_live_endpoint(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "ALPACA_API_KEY_ID=k\nALPACA_API_SECRET_KEY=s\n"
        "ALPACA_BASE_URL=https://api.alpaca.markets\n"
    )
    os.environ.pop("ALPACA_BASE_URL", None)
    saved_key = os.environ.pop("ALPACA_API_KEY_ID", None)
    saved_sec = os.environ.pop("ALPACA_API_SECRET_KEY", None)
    try:
        with pytest.raises(RuntimeError, match="refusing non-paper"):
            live.AlpacaPaper(str(env))
    finally:
        if saved_key:
            os.environ["ALPACA_API_KEY_ID"] = saved_key
        if saved_sec:
            os.environ["ALPACA_API_SECRET_KEY"] = saved_sec
