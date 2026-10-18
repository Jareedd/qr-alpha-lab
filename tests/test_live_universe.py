"""Live-cycle data gates: a shrinking cross-section and a stale session.

No network and no broker. The 2026-10-14 cycle scored 432 names after 496
the cycle before, because a missing as-of close fails ``dropna`` and nothing
stopped the orders. These tests pin the hold, the reasons, and the New York
session date the price cache is keyed on.
"""

import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import live
from quantlab.synthetic import make_panel


def _prices(n_days=300, n_names=8, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=n_days)
    rets = rng.normal(0.0005, 0.01, size=(n_days, n_names))
    px = 100 * np.exp(np.cumsum(rets, axis=0))
    cols = [f"N{i:02d}" for i in range(n_names)]
    return pd.DataFrame(px, index=idx, columns=cols)


def _predictions_for(prices: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"pred_raw": np.linspace(-1, 1, len(names))}, index=names)


def test_ny_session_date_is_not_the_utc_date_after_midnight():
    # Tuesday 00:30 UTC is Monday 20:30 EDT. A delayed Monday cycle must
    # not share Tuesday's cache directory with Tuesday's on-time run.
    late = dt.datetime(2026, 10, 20, 0, 30, tzinfo=dt.timezone.utc)
    assert live.ny_session_date(late) == dt.date(2026, 10, 19)
    assert live.live_cache_dir(late).endswith("live_2026-10-19")
    on_time = dt.datetime(2026, 10, 19, 22, 30, tzinfo=dt.timezone.utc)
    assert live.ny_session_date(on_time) == dt.date(2026, 10, 19)
    # EST, not just EDT: 00:30 UTC Tuesday is Monday 19:30 Eastern.
    winter = dt.datetime(2026, 1, 6, 0, 30, tzinfo=dt.timezone.utc)
    assert live.ny_session_date(winter) == dt.date(2026, 1, 5)
    with pytest.raises(ValueError):
        live.ny_session_date(dt.datetime(2026, 10, 20, 0, 30))


def test_live_min_symbols_is_a_fraction_of_todays_index_not_one():
    # ~503 current members → floor 452. The historical request is longer
    # and includes unpriceable dead names; that longer count is not the floor.
    assert live.live_min_symbols(503) == int(0.90 * 503)
    assert live.live_min_symbols(503) > 400
    assert live.live_min_symbols(0) == 1


def test_missing_close_warmup_and_feature_gap_are_different_reasons():
    prices = _prices()
    prices.iloc[-1, 0] = np.nan                          # N00: no as-of close
    prices.iloc[:-30, 1] = np.nan                        # N01: only 30 bars
    prices.iloc[-40, 2] = np.nan                         # N02: interior hole
    mask = pd.DataFrame(True, index=prices.index, columns=list(prices.columns) + ["GHOST"])
    report = live.classify_asof_universe(prices, mask)
    by_reason = {}
    for row in report["dropped"]:
        by_reason.setdefault(row["reason"], []).append(row["ticker"])
    assert by_reason["missing_close"] == ["N00"]
    assert by_reason["warmup"] == ["N01"]
    assert "N02" in by_reason["feature_gap"]
    assert by_reason["missing_from_panel"] == ["GHOST"]
    assert "N00" not in report["scored"]
    assert "N03" in report["scored"]


def test_classifier_matches_the_dropna_row_the_model_scores():
    # If these two sets diverge, the gate holds every cycle. Pin them on a
    # clean panel and on a panel with one missing as-of close.
    # 1300 days: the training-panel floor inside live_target_weights is 10_000
    # rows, and the first ~252 days are feature warm-up.
    prices = make_panel(n_assets=40, n_days=1300, mode="noise", seed=3)
    _, preds = live.live_target_weights(prices, None, prices.attrs["sectors"], min_names=10)
    report = live.classify_asof_universe(prices, None)
    assert set(report["scored"]) == set(preds.index)

    holed = prices.copy()
    holed.iloc[-1, 0] = np.nan
    _, preds_holed = live.live_target_weights(
        holed, None, holed.attrs["sectors"], min_names=10
    )
    report_holed = live.classify_asof_universe(holed, None)
    assert set(report_holed["scored"]) == set(preds_holed.index)
    assert holed.columns[0] not in preds_holed.index
    assert any(
        d["ticker"] == holed.columns[0] and d["reason"] == "missing_close"
        for d in report_holed["dropped"]
    )


def test_the_october_shrink_holds_and_names_the_missing_closes():
    # 496 → 432 is the 2026-10-14 book. 432/496 = 0.87, under 0.95.
    dropped = [{"ticker": f"T{i:02d}", "reason": "missing_close"} for i in range(64)]
    decision = live.universe_hold_decision(
        432, {"dropped": dropped}, 496, prior_asof="2026-10-12"
    )
    assert decision["hold"] is True
    assert decision["n_missing_close"] == 64
    assert "496" in decision["reason"]
    # One name out of a hundred is not a sharp drop.
    small = live.universe_hold_decision(
        99, {"dropped": [{"ticker": "ZZZ", "reason": "missing_close"}]}, 100,
        prior_asof="2026-10-12",
    )
    assert small["hold"] is False


def test_warmup_names_do_not_trip_the_membership_gate():
    # Recent listings without 252 bars are expected. They are not the
    # silent shrink: the membership reference excludes them.
    dropped = [{"ticker": f"W{i}", "reason": "warmup", "n_history": 40} for i in range(30)]
    decision = live.universe_hold_decision(100, {"dropped": dropped}, None)
    assert decision["hold"] is False
    assert decision["n_eligible"] == 100
    assert decision["n_warmup"] == 30


def test_universe_hold_writes_the_names_and_sends_nothing(tmp_path):
    prices = _prices()
    gone = ["N00", "N01", "N02"]
    prices.loc[prices.index[-1], gone] = np.nan
    report = live.classify_asof_universe(prices, None)
    preds = _predictions_for(prices, report["scored"])
    # Prior cycle scored everyone. Three missing closes is 5/8 = 0.625.
    asof = str(prices.index[-1].date())
    prior = (prices.index[-1] - pd.Timedelta(days=1)).date().isoformat()
    pd.DataFrame({"pred_raw": range(8)}).to_csv(tmp_path / f"predictions_{prior}.csv")

    with pytest.raises(live.LiveCycleHeld, match="missing_close") as exc:
        live.apply_universe_gate(str(tmp_path), prices, None, preds)
    assert "N00" in str(exc.value)
    assert "No orders sent" in str(exc.value)
    assert list(tmp_path.glob("predictions_*.csv")) == [
        tmp_path / f"predictions_{prior}.csv"
    ]
    assert list(tmp_path.glob("weights_*.csv")) == []
    hold = json.loads((tmp_path / f"universe_hold_{asof}.json").read_text())
    assert hold["held"] is True and hold["orders_sent"] is False
    assert hold["predictions_logged"] is False
    reasons = {d["ticker"]: d["reason"] for d in hold["dropped"]}
    assert reasons["N00"] == "missing_close"
    assert hold["price_panel"]["asof_closes"] == 5
    assert "sha256" in hold["price_panel"]


def test_a_full_cross_section_is_not_held(tmp_path):
    prices = _prices()
    report = live.classify_asof_universe(prices, None)
    preds = _predictions_for(prices, report["scored"])
    prior = "2024-01-02"
    pd.DataFrame({"pred_raw": range(len(report["scored"]))}).to_csv(
        tmp_path / f"predictions_{prior}.csv"
    )
    record = live.apply_universe_gate(str(tmp_path), prices, None, preds)
    assert record["held"] is False
    assert list(tmp_path.glob("universe_hold_*.json")) == []


def test_session_gate_refuses_the_previous_close(tmp_path):
    prices = _prices(n_days=5)
    session = prices.index[-1].date() + dt.timedelta(days=1)
    with pytest.raises(live.LiveCycleHeld, match="write-once"):
        live.apply_session_gate(str(tmp_path), prices, session)
    hold = json.loads(
        (tmp_path / f"session_hold_{session.isoformat()}.json").read_text()
    )
    assert hold["orders_sent"] is False
    assert hold["predictions_logged"] is False
    assert hold["panel_last_bar"] == str(prices.index[-1].date())
    # The session that is actually in the panel is not a hold.
    live.apply_session_gate(str(tmp_path), prices, prices.index[-1].date())


def test_panel_fingerprint_changes_when_the_asof_close_changes():
    prices = _prices(n_days=5)
    a = live.price_panel_fingerprint(prices)
    other = prices.copy()
    other.iloc[-1, 0] *= 1.01
    b = live.price_panel_fingerprint(other)
    assert a["sha256"] != b["sha256"]
    assert a["n_cols"] == b["n_cols"]
    same = live.price_panel_fingerprint(prices.copy())
    assert same["sha256"] == a["sha256"]
