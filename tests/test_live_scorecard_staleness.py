"""The scorecard must report its own staleness (found 2026-09-15).

The nightly trading job failed every weekday from 2026-08-11 to 2026-09-15
(25 cycles), and nothing in the repo said so: the scorecard workflow kept
exiting 0, the committed SVG kept rendering, and the artifact a reader sees
still claimed "nightly". A monitoring artifact that cannot go red is
decoration. This pins the arithmetic and the alarm.
"""

import datetime as dt
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from live_scorecard import (  # noqa: E402
    STALE_AFTER_WEEKDAYS,
    build_svg,
    staleness,
    weekdays_between,
)


def _cycles(*dates: str) -> list[dict]:
    return [{"asof": d, "equity": 1_000_000.0, "n_names": 95} for d in dates]


def test_weekdays_between_skips_the_weekend():
    # Fri 2026-08-07 -> Mon 2026-08-10 is one weekday, not three.
    assert weekdays_between(dt.date(2026, 8, 7), dt.date(2026, 8, 10)) == 1
    assert weekdays_between(dt.date(2026, 8, 10), dt.date(2026, 8, 11)) == 1
    assert weekdays_between(dt.date(2026, 8, 10), dt.date(2026, 8, 10)) == 0


def test_the_observed_outage_is_measured():
    # The real one: last cycle 2026-08-10, checked 2026-09-15.
    days, last = staleness(_cycles("2026-08-10"), today=dt.date(2026, 9, 15))
    assert last == "2026-08-10"
    assert days == 26
    assert days > STALE_AFTER_WEEKDAYS


def test_next_morning_is_not_stale():
    # A cycle logged yesterday must not trip the alarm -- a monitor that cries
    # wolf nightly gets muted, and then it is the 2026-08-11 outage again.
    days, _ = staleness(_cycles("2026-09-14"), today=dt.date(2026, 9, 15))
    assert days == 1
    assert days <= STALE_AFTER_WEEKDAYS


def test_long_weekend_does_not_trip_the_alarm():
    # Thu cycle, checked the following Tue (Mon a market holiday): 3 weekdays
    # by this deliberately crude count, which is exactly at the threshold.
    days, _ = staleness(_cycles("2026-09-10"), today=dt.date(2026, 9, 15))
    assert days == 3
    assert days <= STALE_AFTER_WEEKDAYS


def test_svg_carries_the_stall_banner_when_stale():
    svg = build_svg(_cycles("2026-06-11", "2026-08-10"), today=dt.date(2026, 9, 15))
    assert "STALLED" in svg
    assert "no cycle logged since 2026-08-10" in svg
    assert "26 weekdays" in svg


def test_svg_has_no_banner_when_current():
    svg = build_svg(_cycles("2026-09-11", "2026-09-14"), today=dt.date(2026, 9, 15))
    assert "STALLED" not in svg


@pytest.mark.parametrize("stale", [True, False])
def test_svg_geometry_is_identical_either_way(stale):
    # The banner's space is reserved unconditionally, so a stalled card cannot
    # overlap the equity panel (and a fixed layout keeps the diff readable).
    today = dt.date(2026, 9, 15) if stale else dt.date(2026, 6, 12)
    svg = build_svg(_cycles("2026-06-11"), today=today)
    assert 'height="310"' in svg
    assert 'y="100"' in svg or 'y="90"' in svg
