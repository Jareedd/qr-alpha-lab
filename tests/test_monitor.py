"""Phase 6 monitoring logic: known answers, no network, no broker.

The monitor is the instrument that will judge the live experiment -- it
gets the same known-answer discipline as the backtester, because a broken
monitor would quietly corrupt the project's headline comparison.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import features, monitor
from quantlab.synthetic import make_panel

HORIZON = 21


def test_realized_live_ic_perfect_foresight_is_one():
    # Predictions equal to the realized label must give IC exactly +1
    # (and the negation -1): pins both the label construction and the
    # rank-correlation convention against models.information_coefficient.
    prices = make_panel(n_assets=40, n_days=420, mode="noise", seed=5)
    labels = features.build_labels(prices, horizon=HORIZON, residualize=True)
    asof = prices.index[-60]  # > horizon days of future prices exist
    lab = labels.loc[asof].dropna()
    assert len(lab) >= 30, "test setup: need enough labeled names"

    ic = monitor.realized_live_ic(
        {asof: pd.DataFrame({"pred_raw": lab})}, prices, horizon=HORIZON
    )
    assert ic.loc[asof] == pytest.approx(1.0)

    ic_neg = monitor.realized_live_ic(
        {asof: pd.DataFrame({"pred_raw": -lab})}, prices, horizon=HORIZON
    )
    assert ic_neg.loc[asof] == pytest.approx(-1.0)


def test_realized_live_ic_omits_immature_cycles():
    # A cycle whose horizon has not elapsed must be omitted entirely --
    # a NaN or partial-window IC would quietly dilute the live record.
    prices = make_panel(n_assets=40, n_days=420, mode="noise", seed=5)
    asof = prices.index[-5]  # only 4 future days < horizon
    preds = {asof: pd.DataFrame({"pred_raw": pd.Series(1.0, index=prices.columns)})}
    ic = monitor.realized_live_ic(preds, prices, horizon=HORIZON)
    assert len(ic) == 0


def test_realized_live_ic_requires_min_names():
    prices = make_panel(n_assets=40, n_days=420, mode="noise", seed=5)
    asof = prices.index[-60]
    few = pd.DataFrame({"pred_raw": pd.Series(1.0, index=prices.columns[:5])})
    ic = monitor.realized_live_ic({asof: few}, prices, horizon=HORIZON, min_names=30)
    assert len(ic) == 0


def test_cycle_continuity_flags_gap_weekdays():
    logged = [pd.Timestamp("2026-06-08"), pd.Timestamp("2026-06-10")]  # Mon, Wed
    cont = monitor.cycle_continuity(logged, pd.Timestamp("2026-06-12"))  # thru Fri
    missing = [str(d.date()) for d in cont[~cont["logged"]]["date"]]
    assert missing == ["2026-06-09", "2026-06-11", "2026-06-12"]
    assert int(cont["logged"].sum()) == 2


def test_realized_book_returns_known_answer():
    idx = pd.bdate_range("2026-01-05", periods=4)
    prices = pd.DataFrame(
        {"A": [100.0, 110.0, 121.0, 121.0], "B": [50.0, 45.0, 40.5, 40.5]},
        index=idx,
    )
    book = {idx[0]: pd.Series({"A": 0.5, "B": -0.5})}
    pnl = monitor.realized_book_returns(book, prices)

    # Book logged at day0 earns from day1 (the backtest convention):
    # day1: 0.5*(+10%) - 0.5*(-10%) = +10%; day2 same; day3 flat.
    assert idx[0] not in pnl.index
    assert pnl.loc[idx[1]] == pytest.approx(0.10)
    assert pnl.loc[idx[2]] == pytest.approx(0.10)
    assert pnl.loc[idx[3]] == pytest.approx(0.0)


def test_book_holds_until_next_logged_book():
    idx = pd.bdate_range("2026-01-05", periods=4)
    prices = pd.DataFrame(
        {"A": [100.0, 110.0, 121.0, 133.1], "B": [50.0, 50.0, 50.0, 50.0]},
        index=idx,
    )
    books = {
        idx[0]: pd.Series({"A": 1.0}),   # long A...
        idx[2]: pd.Series({"A": -1.0}),  # ...flip short at day2
    }
    pnl = monitor.realized_book_returns(books, prices)
    assert pnl.loc[idx[1]] == pytest.approx(0.10)   # old book
    assert pnl.loc[idx[2]] == pytest.approx(0.10)   # old book still earns day2
    assert pnl.loc[idx[3]] == pytest.approx(-0.10)  # new book from day3


def test_paused_cycle_does_not_replace_the_submitted_book():
    # The weight file on a paused day is the unsent target. Marking it
    # would book a rebalance the broker never did.
    idx = pd.bdate_range("2026-01-05", periods=4)
    prices = pd.DataFrame(
        {"A": [100.0, 110.0, 121.0, 133.1], "B": [50.0, 50.0, 50.0, 50.0]},
        index=idx,
    )
    books = {
        idx[0]: pd.Series({"A": 1.0}),
        idx[2]: pd.Series({"A": -1.0}),  # logged, but orders were paused
    }
    pnl = monitor.realized_book_returns(books, prices, hold_dates={idx[2]})
    assert pnl.loc[idx[1]] == pytest.approx(0.10)
    assert pnl.loc[idx[2]] == pytest.approx(0.10)
    assert pnl.loc[idx[3]] == pytest.approx(0.10)  # old long book still held


def test_orders_paused_dates_come_from_the_summary(tmp_path):
    (tmp_path / "summary_2026-10-20.json").write_text(
        '{"asof": "2026-10-20", "orders_status": "paused", "orders_paused": true}'
    )
    (tmp_path / "summary_2026-10-19.json").write_text(
        '{"asof": "2026-10-19", "submitted": true, "equity": 1}'
    )
    paused = monitor.load_orders_paused_dates(str(tmp_path))
    assert paused == {pd.Timestamp("2026-10-20")}


def test_render_report_says_paused_weights_are_not_a_rebalance():
    cont = monitor.cycle_continuity([pd.Timestamp("2026-06-10")], "2026-06-10")
    md = monitor.render_report(
        asof="2026-06-11",
        continuity=cont,
        n_weights_logged=1,
        n_preds_logged=1,
        comparison=None,
        live_ic=pd.Series(dtype=float),
        book_pnl=pd.Series({pd.Timestamp("2026-06-11"): 0.0}),
        n_orders_paused=2,
    )
    assert "orders paused on 2" in md
    assert "unsent target" in md


def test_load_live_records_round_trip(tmp_path):
    w = pd.Series({"AAA": 0.5, "BBB": -0.5}, name="weight").rename_axis("ticker")
    p = pd.DataFrame(
        {"pred_raw": [0.1, -0.2], "pred_sector_neutral": [0.15, -0.15]},
        index=pd.Index(["AAA", "BBB"], name="ticker"),
    )
    w.to_csv(tmp_path / "weights_2026-06-10.csv")
    p.to_csv(tmp_path / "predictions_2026-06-10.csv")
    (tmp_path / "summary_2026-06-10.json").write_text("{}")  # must be ignored

    weights, preds = monitor.load_live_records(str(tmp_path))
    d = pd.Timestamp("2026-06-10")
    assert list(weights) == [d] and list(preds) == [d]
    pd.testing.assert_series_equal(weights[d], w.astype(float))
    pd.testing.assert_frame_equal(preds[d], p.astype(float))


def test_render_report_states_unmeasurable_honestly():
    cont = monitor.cycle_continuity([pd.Timestamp("2026-06-10")], "2026-06-10")
    comparison = {
        "n_cycles_measurable": 0,
        "live_mean_ic": float("nan"),
        "live_ic_tstat_nw": float("nan"),
        "backtest_mean_ic": 0.0225,
        "backtest_ic_tstat_nw": 1.91,
    }
    md = monitor.render_report(
        asof="2026-06-10",
        continuity=cont,
        n_weights_logged=1,
        n_preds_logged=0,
        comparison=comparison,
        live_ic=pd.Series(dtype=float),
        book_pnl=pd.Series(dtype=float),
    )
    assert "not yet measurable" in md
    assert "do not interpret yet" in md
    assert "record is gap-free" in md
    # optional sections absent when their inputs are absent
    assert "Control arm" not in md
    assert "Data revisions" not in md


def test_render_report_control_arm_and_revisions_sections():
    cont = monitor.cycle_continuity([pd.Timestamp("2026-06-11")], "2026-06-11")
    md = monitor.render_report(
        asof="2026-06-11",
        continuity=cont,
        n_weights_logged=1,
        n_preds_logged=1,
        comparison=None,
        live_ic=pd.Series(dtype=float),
        book_pnl=pd.Series(dtype=float),
        baseline_live_ic=pd.Series({pd.Timestamp("2026-06-11"): 0.0312}),
        revisions=[
            {
                "compared_to": "2026-06-10",
                "n_cells_compared": 1_000_000,
                "n_price_cells_changed": 2100,
                "frac_price_cells_changed": 0.0021,
                "n_return_cells_changed": 3,
                "max_abs_return_change": 1.7e-4,
            }
        ],
    )
    assert "Control arm" in md
    assert "+0.0312" in md
    assert "Data revisions" in md
    assert "2,100" in md and "3 return cells" in md
