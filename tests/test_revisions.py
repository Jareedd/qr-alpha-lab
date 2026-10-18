"""Data-revision monitor: known-answer tests on hand-built snapshots.

The invariant that makes the monitor useful: a whole-history re-adjustment
(constant multiplier) changes every PRICE cell but zero RETURN cells; a
splice-point rewrite changes returns exactly at the splice. Both are pinned
here so the report's price-vs-return distinction stays trustworthy.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import revisions


def _panel(n_days=10, tickers=("AAA", "BBB", "CCC"), seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2026-01-05", periods=n_days)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n_days, len(tickers))), axis=0))
    return pd.DataFrame(prices, index=dates, columns=list(tickers))


def test_identical_snapshots_report_zero_changes():
    old = _panel()
    stats = revisions.compare_price_snapshots(old, old.copy())
    assert stats["n_price_cells_changed"] == 0
    assert stats["n_return_cells_changed"] == 0
    assert stats["n_price_cells_noise_band"] == 0
    assert stats["n_tickers_affected"] == 0
    assert stats["n_cells_compared"] == old.size


def test_serving_noise_lands_in_the_noise_band_not_in_changes():
    # Calibrated against the first real measurement (2026-06-12 -> -13):
    # yfinance re-serves history with ~1e-7 relative wobble on most cells.
    # That is the vendor's float noise, not a revision -- it must be
    # counted (the phenomenon is real and worth tracking) but never
    # conflated with actual adjustments (which start ~1e-3).
    old = _panel()
    new = old * (1.0 + 1e-7)  # uniform serving wobble
    stats = revisions.compare_price_snapshots(old, new)
    assert stats["n_price_cells_changed"] == 0
    assert stats["n_price_cells_noise_band"] == old.size
    # a uniform rescale leaves returns untouched at any threshold
    assert stats["n_return_cells_changed"] == 0


def test_full_history_readjustment_changes_prices_but_not_returns():
    # The benign (and most common) revision: a new dividend re-scales AAA's
    # ENTIRE history. Every price cell moves; no return moves.
    old = _panel()
    new = old.copy()
    new["AAA"] *= 1.02
    stats = revisions.compare_price_snapshots(old, new)
    assert stats["n_price_cells_changed"] == len(old)
    assert stats["n_return_cells_changed"] == 0
    assert stats["n_tickers_affected"] == 1
    assert stats["top_affected_tickers"][0]["ticker"] == "AAA"
    assert np.isclose(stats["max_abs_rel_price_change"], 0.02)


def test_splice_point_rewrite_changes_exactly_one_return():
    # The dangerous revision: history is re-scaled only AFTER day k, so one
    # return (the splice) is rewritten -- the kind that alters features.
    old = _panel()
    new = old.copy()
    k = 4
    new.iloc[k:, new.columns.get_loc("BBB")] *= 1.05
    stats = revisions.compare_price_snapshots(old, new)
    assert stats["n_price_cells_changed"] == len(old) - k
    assert stats["n_return_cells_changed"] == 1
    assert np.isclose(stats["max_abs_return_change"], 0.05, rtol=0.2)


def test_only_shared_past_is_compared_and_holes_are_counted():
    old = _panel(n_days=10)
    new = _panel(n_days=12)  # two extra days = new info, not revision
    new.loc[new.index[:10], :] = old.values  # shared past identical...
    new.iloc[2, new.columns.get_loc("CCC")] = np.nan  # ...one print vanished
    old.iloc[5, old.columns.get_loc("AAA")] = np.nan  # one print appeared
    stats = revisions.compare_price_snapshots(old, new)
    assert stats["n_shared_dates"] == 10
    assert stats["n_price_cells_changed"] == 0
    assert stats["n_cells_vanished"] == 1
    assert stats["n_cells_appeared"] == 1


def test_snapshot_discovery_and_pairing(tmp_path):
    for tag, scale in [("2026-06-10", 1.0), ("2026-06-11", 1.0), ("2026-06-12", 1.01)]:
        d = tmp_path / f"live_{tag}"
        d.mkdir()
        (_panel() * scale).to_parquet(d / "prices_abc123_2018-01-01_latest_0.0.parquet")
    (tmp_path / "live_not-a-date").mkdir()  # ignored: no snapshot inside

    snaps = revisions.list_snapshots(str(tmp_path))
    assert sorted(snaps) == ["2026-06-10", "2026-06-11", "2026-06-12"]

    tag, _ = revisions.latest_snapshot_before(str(tmp_path), "2026-06-12")
    assert tag == "2026-06-11"
    assert revisions.latest_snapshot_before(str(tmp_path), "2026-06-10") is None

    table = revisions.revision_table(snaps)
    assert list(table["from"]) == ["2026-06-10", "2026-06-11"]
    assert table["n_price_cells_changed"].tolist()[0] == 0  # 10 -> 11 identical
    assert table["n_price_cells_changed"].tolist()[1] > 0   # 11 -> 12 rescaled
    assert table["n_return_cells_changed"].tolist()[1] == 0  # but benign


def test_live_cycle_summary_needs_a_prior_snapshot(tmp_path):
    today = _panel()
    assert (
        revisions.snapshot_revision_summary(str(tmp_path), "2026-06-11", today)
        is None
    )
    d = tmp_path / "live_2026-06-10"
    d.mkdir()
    (today * 1.03).to_parquet(d / "prices_x_2018-01-01_latest_0.0.parquet")
    out = revisions.snapshot_revision_summary(str(tmp_path), "2026-06-11", today)
    assert out["compared_to"] == "2026-06-10"
    assert out["n_price_cells_changed"] == today.size
    assert out["n_return_cells_changed"] == 0


# ---------------------------------------------------------------------------
# Gap provenance (added 2026-09-15, after the five-week collection outage)
# ---------------------------------------------------------------------------

def test_snapshot_gap_counts_weekdays_not_calendar_days():
    # Fri -> Mon is one weekday of vendor rewriting, not three days of it.
    g = revisions.snapshot_gap("2026-08-07", "2026-08-10")
    assert g == {"gap_calendar_days": 3, "gap_weekdays": 1, "is_consecutive_cycle": True}


def test_the_outage_record_is_marked_non_consecutive():
    # Verified from results/live filenames (2026-10-18): last weights file
    # before the gap is 2026-08-10, first one back is 2026-09-28. That record
    # spans the whole outage and is a different quantity from a one-day
    # revision. Pooling it with daily records would report the outage as a
    # vendor-revision spike. (An older note used 2026-09-16 and "26 weekdays";
    # that date is not a logged cycle, and 26 counts only through 2026-09-15.)
    g = revisions.snapshot_gap("2026-08-10", "2026-09-28")
    assert g["gap_weekdays"] == 35
    assert g["is_consecutive_cycle"] is False


def test_live_summary_carries_the_gap(tmp_path):
    today = _panel()
    d = tmp_path / "live_2026-06-10"
    d.mkdir()
    (today * 1.03).to_parquet(d / "prices_x_2018-01-01_latest_0.0.parquet")

    out = revisions.snapshot_revision_summary(str(tmp_path), "2026-06-11", today)
    assert out["gap_weekdays"] == 1 and out["is_consecutive_cycle"] is True

    # Same prior snapshot, but the cycle resumes a month later.
    out = revisions.snapshot_revision_summary(str(tmp_path), "2026-07-10", today)
    assert out["compared_to"] == "2026-06-10"
    assert out["gap_calendar_days"] == 30
    assert out["is_consecutive_cycle"] is False


def test_subthreshold_price_wobble_is_not_a_return_revision():
    # Absolute |Δr| > 1e-6 used to flag serving noise. Two adjacent prices
    # each move by just under REL_TOL in opposite directions: no price cell
    # is over the price threshold, but |Δr| ≈ 2*REL_TOL used to count.
    old = _panel(n_days=8)
    eps = 0.9 * revisions.REL_TOL
    wiggle = np.where(np.arange(len(old)) % 2 == 0, 1.0 + eps, 1.0 - eps)
    new = old.copy()
    new["AAA"] = old["AAA"].to_numpy() * wiggle
    stats = revisions.compare_price_snapshots(old, new)
    assert stats["n_price_cells_changed"] == 0
    assert stats["n_return_cells_changed"] == 0
    assert stats["n_return_cells_noise_band"] > 0


def test_a_real_splice_still_counts_as_one_return_revision():
    old = _panel()
    new = old.copy()
    new.iloc[4:, new.columns.get_loc("BBB")] *= 1.05
    stats = revisions.compare_price_snapshots(old, new)
    assert stats["n_return_cells_changed"] == 1
    assert stats["n_price_cells_changed"] > 0


def test_explicit_marker_beats_whichever_filename_sorts_last(tmp_path):
    # The bug: sorted(prices_*.parquet)[-1] picked the report panel once
    # both files shared the live directory. zzz sorts after aaa.
    d = tmp_path / "live_2026-08-10"
    d.mkdir()
    live = _panel()
    report = _panel(n_days=4) * 2
    live_name = "prices_aaa_2018-01-01_latest_0.0.parquet"
    report_name = "prices_zzz_2024-10-18_latest_0.0.parquet"
    live.to_parquet(d / live_name)
    report.to_parquet(d / report_name)

    # No marker: the 2018-01-01 token is the live panel, not the sort-last file.
    chosen = revisions.find_price_snapshot(str(d))
    assert chosen.endswith(live_name)
    assert revisions.snapshot_selection(str(d)) == "live_start_fallback"

    # Marker naming the sort-FIRST file is explicit and wins over both heuristics.
    # Point it at the report name on purpose: explicit means that file, not a guess.
    revisions.write_live_panel_marker(str(d), str(d / report_name))
    assert revisions.find_price_snapshot(str(d)).endswith(report_name)
    assert revisions.snapshot_selection(str(d)) == "explicit"

    summary_dir = tmp_path / "live_2026-08-07"
    summary_dir.mkdir()
    live.to_parquet(summary_dir / "prices_only_2018-01-01_latest_0.0.parquet")
    out = revisions.snapshot_revision_summary(str(tmp_path), "2026-08-10", live)
    assert out["comparison_selection"] == "sole_parquet"
    assert out["comparison_file"].startswith("prices_only_")
    assert "prior_bar" in out and "today_bar" in out


def test_two_unmarked_live_panels_are_not_guessed(tmp_path):
    d = tmp_path / "live_2026-08-10"
    d.mkdir()
    panel = _panel()
    panel.to_parquet(d / "prices_aaa_2018-01-01_latest_0.0.parquet")
    panel.to_parquet(d / "prices_zzz_2018-01-01_latest_0.0.parquet")
    with pytest.raises(revisions.AmbiguousPriceSnapshot, match="sorts last"):
        revisions.find_price_snapshot(str(d))
