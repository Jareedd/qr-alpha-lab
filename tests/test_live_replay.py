"""Offline replay of the deployed live config. No network, no broker."""

import datetime as dt
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import live
from quantlab.live_replay import (
    IC_T_MIN,
    drop_incomplete_session,
    exante_beta,
    integer_share_weights,
    kill_verdict,
    render_report,
    replay_books,
    ridge_from_sums,
    run_next_open_backtest,
    training_cutoff_position,
)
from quantlab.synthetic import make_panel
from sklearn.linear_model import Ridge


def test_ridge_sufficient_stats_match_sklearn():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 5))
    y = x @ np.array([0.2, -0.1, 0.0, 0.05, 0.3]) + rng.normal(size=400)
    model = Ridge(alpha=10.0).fit(x, y)
    coef, intercept = ridge_from_sums(
        len(x), x.sum(axis=0), float(y.sum()), x.T @ x, x.T @ y, 10.0
    )
    np.testing.assert_allclose(coef, model.coef_, rtol=1e-8, atol=1e-8)
    assert intercept == pytest.approx(float(model.intercept_), rel=1e-8, abs=1e-8)


def test_cutoff_is_the_live_index_rule():
    idx = pd.bdate_range("2020-01-01", periods=100)
    i = len(idx) - 1
    assert idx[training_cutoff_position(i, 21)] == idx[-(21 + 1)]


def test_next_open_earns_the_interval_after_the_fill():
    # Decision at the first close. Fill at the second open (110). The book
    # earns 121/110 - 1, not the 110/100 gap that happened before the fill.
    idx = pd.bdate_range("2024-01-02", periods=4)
    opens = pd.DataFrame({"AAA": [100.0, 110.0, 121.0, 133.1]}, index=idx)
    weights = pd.DataFrame({"AAA": [1.0, 0.0, 0.0, 0.0]}, index=idx)
    result = run_next_open_backtest(weights, opens, cost_bps=10.0)
    assert result["gross"].loc[idx[2]] == pytest.approx(0.10)
    assert idx[1] not in result["gross"].index or result["gross"].loc[idx[1]] != pytest.approx(0.10)
    # Entry cost is |Δw| * 10bps = 10 bps, booked on the first interval earned.
    assert result["net"].loc[idx[2]] == pytest.approx(0.10 - 10 / 1e4)


def test_integer_shares_match_live_orders_and_move_beta():
    idx = pd.bdate_range("2024-01-02", periods=2)
    weights = pd.DataFrame({"AAA": [0.5, 0.5], "BBB": [-0.5, -0.5]}, index=idx)
    closes = pd.DataFrame({"AAA": [3.0, 3.0], "BBB": [7.0, 7.0]}, index=idx)
    rounded = integer_share_weights(weights, closes, equity=100.0, max_name_frac=0.5)
    # int(0.5 * 100 / 3) = 16 → weight 0.48; int(-0.5 * 100 / 7) = -7 → -0.49.
    assert rounded.loc[idx[0], "AAA"] == pytest.approx(0.48)
    assert rounded.loc[idx[0], "BBB"] == pytest.approx(-0.49)
    orders = live.orders_from_weights(
        weights.loc[idx[0]], {}, closes.loc[idx[0]], equity=100.0, max_name_frac=0.5
    )
    by_symbol = {o["symbol"]: o for o in orders}
    assert by_symbol["AAA"] == {"symbol": "AAA", "qty": 16, "side": "buy"}
    assert by_symbol["BBB"] == {"symbol": "BBB", "qty": 7, "side": "sell"}
    betas = pd.DataFrame({"AAA": [1.0, 1.0], "BBB": [1.0, 1.0]}, index=idx)
    # Continuous book is dollar-neutral and beta-neutral against a common beta.
    assert exante_beta(weights, betas).loc[idx[0]] == pytest.approx(0.0)
    assert exante_beta(rounded, betas).loc[idx[0]] == pytest.approx(-0.01)


def test_kill_verdict_boundaries():
    assert kill_verdict(IC_T_MIN, 0.01, 0.95)["verdict"] == "KEEP"
    assert kill_verdict(IC_T_MIN - 1e-9, 0.01, 0.95)["verdict"] == "RETIRE"
    assert kill_verdict(IC_T_MIN, 0.0, 0.95)["verdict"] == "RETIRE"
    assert kill_verdict(IC_T_MIN, 0.01, 0.95 - 1e-12)["verdict"] == "RETIRE"
    assert kill_verdict(float("nan"), 1.0, 1.0)["verdict"] == "RETIRE"


def test_incomplete_session_is_dropped_before_the_close():
    idx = pd.to_datetime(["2026-10-15", "2026-10-16", "2026-10-19"])
    frame = pd.DataFrame({"AAA": [1.0, 2.0, 3.0]}, index=idx)
    # 14:24 UTC on 2026-10-19 is 10:24 in New York. The session is open.
    during = dt.datetime(2026, 10, 19, 14, 24, tzinfo=dt.timezone.utc)
    dropped = drop_incomplete_session(frame, now=during)
    assert list(dropped.index) == list(idx[:2])
    after_close = dt.datetime(2026, 10, 19, 20, 30, tzinfo=dt.timezone.utc)
    kept = drop_incomplete_session(frame, now=after_close)
    assert len(kept) == 3
    next_day = dt.datetime(2026, 10, 20, 14, 0, tzinfo=dt.timezone.utc)
    still = drop_incomplete_session(frame, now=next_day)
    assert len(still) == 3


def test_daily_refit_matches_live_on_the_last_session():
    prices = make_panel(n_assets=40, n_days=900, mode="planted", seed=3)
    sectors = prices.attrs["sectors"]
    books = replay_books(prices, None, sectors, min_names=15)
    _w, preds = live.live_target_weights(
        prices, None, sectors, horizon=21, min_names=15
    )
    last = prices.index[-1]
    got = books["pred_raw"].xs(last, level="date").sort_index()
    exp = preds["pred_raw"].sort_index()
    pd.testing.assert_series_equal(got, exp, check_names=False, atol=1e-6, rtol=1e-5)
    both = _w.index.union(books["weights_daily"].columns)
    np.testing.assert_allclose(
        _w.reindex(both).fillna(0.0).to_numpy(),
        books["weights_daily"].loc[last].reindex(both).fillna(0.0).to_numpy(),
        atol=1e-6,
        rtol=1e-5,
    )


def test_prediction_ignores_prices_after_the_decision():
    prices = make_panel(n_assets=40, n_days=900, mode="noise", seed=5)
    sectors = prices.attrs["sectors"]
    clean = replay_books(prices, None, sectors, min_names=15)
    dates = clean["pred_raw"].index.get_level_values("date").unique()
    # A middle prediction date, far enough from the sample end that the
    # poisoned tail is long, and far enough from the start that the
    # 10,000-row gate has opened.
    today = pd.Timestamp(dates[len(dates) // 2])
    pos = prices.index.get_loc(today)
    poisoned = prices.copy()
    poisoned.iloc[pos + 1 :] *= 1.7
    after = replay_books(poisoned, None, sectors, min_names=15)
    a = clean["pred_raw"].xs(today, level="date").sort_index()
    b = after["pred_raw"].xs(today, level="date").sort_index()
    pd.testing.assert_series_equal(a, b, check_names=False, atol=1e-8, rtol=1e-8)

    touched = prices.copy()
    touched.iloc[pos] *= 1.7
    changed = replay_books(touched, None, sectors, min_names=15)
    c = changed["pred_raw"].xs(today, level="date").sort_index()
    assert not np.allclose(a.to_numpy(), c.reindex(a.index).to_numpy())


def test_runner_does_not_call_the_broker():
    path = os.path.join(
        os.path.dirname(__file__), "..", "scripts", "run_trial14_live_config.py"
    )
    text = open(path, encoding="utf-8").read()
    for banned in ("live.run_daily(", "AlpacaPaper(", "load_env(", "submit_order("):
        assert banned not in text


def test_report_prints_the_payload_verdict():
    payload = {
        "verdict": {
            "verdict": "RETIRE",
            "legs": {
                "ic_t_nw": {"value": 0.4, "pass": False},
                "sharpe_net": {"value": -0.2, "pass": False},
                "dsr": {"value": 0.01, "pass": False},
            },
        },
        "config": {
            "train_start": "2018-01-01",
            "horizon": 21,
            "n_trials": 14,
            "max_name_frac": 0.05,
            "rounding_equity": 1_000_000.0,
        },
        "mde_realized": {"n_obs": 1800, "net_ann_sharpe": 1.27},
        "oos_daily": {
            "mean_rank_ic": 0.001,
            "ic_tstat_newey_west": 0.4,
            "sharpe_gross": -0.1,
            "sharpe_net": -0.2,
            "dsr": 0.01,
            "ann_return_net": -0.02,
            "ann_vol": 0.08,
            "max_drawdown": -0.15,
            "annual_turnover": 40.0,
            "ann_cost_drag": 0.08,
            "baseline_mom_sharpe_net": 0.1,
            "beats_mom_baseline": False,
            "baseline_ew_sharpe": 0.6,
            "first": "2019-03-01",
            "last": "2026-10-16",
            "n_days": 1800,
        },
        "cost_decomposition": {
            "daily": {
                "annual_turnover": 40.0,
                "ann_cost_drag": 0.08,
                "sharpe_gross": -0.1,
                "sharpe_net": -0.2,
            },
            "rebalance_21d": {
                "annual_turnover": 4.0,
                "ann_cost_drag": 0.008,
                "sharpe_gross": -0.05,
                "sharpe_net": -0.08,
            },
            "turnover_ratio_daily_over_21d": 10.0,
            "extra_cost_drag": 0.072,
        },
        "beta_after_rounding": {
            "continuous": {"mean": 0.0, "mean_abs": 0.0, "p95_abs": 0.0},
            "rounded": {"mean": 0.01, "mean_abs": 0.02, "p95_abs": 0.05},
            "rounded_net_exposure_mean": 0.001,
            "rounded_net_exposure_mean_abs": 0.004,
            "mean_l1_distortion": 0.01,
            "realized_member_continuous": 0.05,
            "realized_member_rounded": 0.06,
            "realized_spy_continuous": 0.3,
            "realized_spy_rounded": 0.31,
        },
        "data": {
            "sha256": "abc123",
            "n_cols": 10,
            "n_rows": 100,
            "first": "2018-01-02",
            "last": "2026-10-16",
        },
    }
    text = render_report(payload)
    assert text.startswith("# Trial 14")
    assert "**Verdict: RETIRE.**" in text
    assert "1.233" in text
    assert "cannot flip a retire into a keep" in text
