"""Regression pin for the frozen-price-panel monitor bug (found 2026-08-06).

The nightly report ran six weeks on a June price cache ("latest"-keyed
parquet, restored by the CI cache every night), reporting "0 measurable
cycles" while ~17 had matured. The fix is a per-day snapshot cache dir plus
a staleness flag written into the report itself; this file pins the flag's
arithmetic so the alarm cannot silently rot.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from live_report import panel_staleness_days, report_cache_dir  # noqa: E402


def test_fresh_panel_is_not_stale():
    idx = pd.bdate_range("2026-07-01", "2026-08-05")
    assert panel_staleness_days(pd.Timestamp("2026-08-05"), idx) == 0
    # panel may even end after the last cycle (weekend fetch): negative, not stale
    assert panel_staleness_days(pd.Timestamp("2026-08-04"), idx) < 0


def test_june_frozen_panel_flags_the_observed_bug():
    idx = pd.bdate_range("2026-05-01", "2026-06-26")  # the frozen cache
    days = panel_staleness_days(pd.Timestamp("2026-08-05"), idx)
    assert days == 40  # six silent weeks


def test_empty_panel_is_maximally_stale():
    assert panel_staleness_days(pd.Timestamp("2026-08-05"), pd.Index([])) >= 10**5


def test_report_cache_is_not_the_live_cycle_directory():
    # 00:30 UTC Tuesday is still Monday evening in New York. The report
    # must not land in data_cache/live_* either way: sharing that folder
    # is how the revision monitor started comparing the wrong panel.
    import datetime as dt

    now = dt.datetime(2026, 10, 20, 0, 30, tzinfo=dt.timezone.utc)
    path = report_cache_dir(now)
    assert path.endswith("report_2026-10-19")
    assert "live_" not in path.replace("report_", "")
