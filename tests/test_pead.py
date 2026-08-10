"""H13 PEAD harness: offline known-answer tests + the falsification gate.

NO network: every test runs on a synthetic PEAD world (synthetic.make_pead_panel)
or a tiny in-memory CSV blob. The cardinal pins:
  * CSV parsing (a Bloomberg-style fixture blob; missing optional cols; bad rows).
  * SUE math, BOTH branches (std_est primary; surprise_pct / rel-est fallback).
  * the T+2 PIT entry — a poison-the-future pin: corrupting prices ON/BEFORE the
    announcement bar must NOT change the post-T+2 event-study result.
  * drift-vs-reaction logic (true drift persists; collapse = reaction artifact).
  * surprise-shuffle placebo ~0.
  * the machinery gate (planted beats null, paired per seed).
  * a full end-to-end offline run of run_pead._run_trial on a synthetic world +
    a WRITTEN fixture CSV (60d drift recovered in planted, ~0 in null; the
    verdict is the exact success-criteria conjunction).
"""

import io
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from quantlab import metrics, pead
from quantlab.synthetic import make_pead_panel


# --------------------------------------------------------------------------- #
# CSV parsing.
# --------------------------------------------------------------------------- #

def test_parse_pead_csv_full_blob():
    blob = (
        "ticker,ann_date,period,actual_eps,est_eps,surprise_pct,num_est,std_est\n"
        "AAPL,2023-02-02,2023Q1,1.88,1.94,-3.1,30,0.05\n"
        "msft ,2023-01-24,2023Q2,2.32,2.30,0.9,28,0.04\n"   # lowercase + space
        "GOOG,2023-02-02,2023Q1,1.05,1.18,-11.0,25,0.07\n"
    )
    df = pead.parse_pead_csv(io.StringIO(blob))
    assert list(df.columns) == [
        "ticker", "ann_date", "period", "actual_eps", "est_eps",
        "surprise_pct", "num_est", "std_est"]
    assert set(df["ticker"]) == {"AAPL", "MSFT", "GOOG"}      # upper + stripped
    assert df["ann_date"].dtype.kind == "M"                   # datetime
    assert df.loc[df["ticker"] == "AAPL", "std_est"].iloc[0] == 0.05


def test_parse_pead_csv_missing_optional_cols_and_bad_rows():
    # Only the four GUARANTEED columns; plus malformed rows that must be dropped.
    blob = (
        "ticker,ann_date,actual_eps,est_eps\n"
        "AAPL,2023-02-02,1.88,1.94\n"
        ",2023-02-02,1.0,1.1\n"            # no ticker -> drop
        "BAD,not-a-date,1.0,1.1\n"          # unparseable date -> drop
        "NOEPS,2023-02-02,,1.1\n"           # missing actual_eps -> drop
        "GOOD,2023-03-01,2.0,1.5\n"
    )
    df = pead.parse_pead_csv(io.StringIO(blob))
    # optional columns materialized as NaN, present in the schema
    for c in ("period", "surprise_pct", "num_est", "std_est"):
        assert c in df.columns
        assert df[c].isna().all()
    assert set(df["ticker"]) == {"AAPL", "GOOD"}              # 2 bad rows dropped


# --------------------------------------------------------------------------- #
# SUE math — both branches.
# --------------------------------------------------------------------------- #

def test_compute_sue_primary_std_branch():
    df = pd.DataFrame({
        "ticker": ["A", "B"], "ann_date": pd.to_datetime(["2023-01-01"] * 2),
        "actual_eps": [1.10, 0.90], "est_eps": [1.00, 1.00],
        "std_est": [0.05, 0.10], "surprise_pct": [np.nan, np.nan],
    })
    out = pead.compute_sue(df)
    # SUE = (actual-est)/std_est
    assert out.loc[out["ticker"] == "A", "sue"].iloc[0] == pytest.approx(2.0)
    assert out.loc[out["ticker"] == "B", "sue"].iloc[0] == pytest.approx(-1.0)
    assert (out["sue_method"] == "std").all()


def test_compute_sue_fallback_branches_never_mixed():
    df = pd.DataFrame({
        "ticker": ["P", "R"], "ann_date": pd.to_datetime(["2023-01-01"] * 2),
        "actual_eps": [1.10, 1.20], "est_eps": [1.00, 1.00],
        "std_est": [np.nan, np.nan],          # no dispersion -> fallback
        "surprise_pct": [10.0, np.nan],       # P uses surprise_pct, R uses rel-est
    })
    out = pead.compute_sue(df)
    # P: surprise_pct 10% -> 0.10
    assert out.loc[out["ticker"] == "P", "sue"].iloc[0] == pytest.approx(0.10)
    assert out.loc[out["ticker"] == "P", "sue_method"].iloc[0] == "surprise_pct"
    # R: (1.20-1.00)/|1.00| = 0.20
    assert out.loc[out["ticker"] == "R", "sue"].iloc[0] == pytest.approx(0.20)
    assert out.loc[out["ticker"] == "R", "sue_method"].iloc[0] == "rel_est"
    # the two definitions are never averaged: each row has exactly one method
    assert set(out["sue_method"]) == {"surprise_pct", "rel_est"}


def test_drop_low_eps_filter_frozen_threshold():
    """The pre-registered H13 rounding filter drops |est_eps| <= $0.10 (where 2-dp
    EPS rounding quantizes SUE), keeps the rest, and reports the count. Boundary
    is strict-greater-than: est == 0.10 is DROPPED."""
    df = pd.DataFrame({
        "ticker": ["A", "B", "C", "D"],
        "ann_date": pd.to_datetime(["2023-01-01"] * 4),
        "actual_eps": [0.05, 0.20, 0.11, 1.50],
        "est_eps": [0.03, 0.10, 0.11, 1.40],   # A:0.03 drop, B:0.10 drop(boundary), C/D keep
    })
    kept, n_dropped = pead.drop_low_eps(df)
    assert n_dropped == 2
    assert list(kept["ticker"]) == ["C", "D"]
    # threshold is the frozen module constant, not a magic literal
    assert pead.MIN_ABS_EST == 0.10
    # negative estimates use |est|: -0.50 is kept, -0.07 dropped
    df2 = pd.DataFrame({
        "ticker": ["E", "F"], "ann_date": pd.to_datetime(["2023-01-01"] * 2),
        "actual_eps": [-0.40, -0.10], "est_eps": [-0.50, -0.07],
    })
    kept2, n2 = pead.drop_low_eps(df2)
    assert n2 == 1 and list(kept2["ticker"]) == ["E"]


# --------------------------------------------------------------------------- #
# T+2 PIT entry — the poison-the-future pin.
# --------------------------------------------------------------------------- #

def test_event_study_enters_at_t_plus_2_not_before_poison_pin():
    """Corrupting prices ON or BEFORE the announcement bar must NOT change the
    post-T+2 event-study result. The entry price is at announcement+enter_lag
    (T+2); the first held return is the bar AFTER that, so nothing the book earns
    depends on the announcement bar or earlier — the cardinal PIT guarantee."""
    panel = make_pead_panel(mode="planted_pead", seed=7)
    events = panel.attrs["events"]
    base = pead.pead_event_study(events, panel, enter_lag=2, hold=60)["net_sharpe"]

    poison = panel.copy()
    idx = poison.index
    for _, row in events.iterrows():
        # pos = last trading bar on/before the announcement = the announcement bar
        pos = int(idx.searchsorted(row["ann_date"], side="right")) - 1
        col = poison.columns.get_loc(row["ticker"])
        if pos >= 0:
            poison.iloc[:pos + 1, col] *= 1.5   # wreck announcement bar + history
    poisoned = pead.pead_event_study(events, poison, enter_lag=2, hold=60)["net_sharpe"]
    assert poisoned == pytest.approx(base, abs=1e-9), (
        "post-T+2 result must be invariant to prices on/before the announcement bar")


def test_event_study_recovers_drift_planted_vs_null():
    planted = make_pead_panel(mode="planted_pead", seed=7)
    null = make_pead_panel(mode="null_pead", seed=7)
    rp = pead.pead_event_study(planted.attrs["events"], planted)
    rn = pead.pead_event_study(null.attrs["events"], null)
    assert rp["net_sharpe"] > 1.0          # planted drift recovered
    assert abs(rn["net_sharpe"]) < 0.5     # null: no drift
    assert rp["n_long"] > 0 and rp["n_short"] > 0


# --------------------------------------------------------------------------- #
# Drift-vs-reaction control.
# --------------------------------------------------------------------------- #

def test_drift_vs_reaction_persists_on_planted_world():
    """A TRUE planted drift persists for weeks -> T+5 retains >=50% of T+2."""
    panel = make_pead_panel(mode="planted_pead", seed=11)
    dvr = pead.drift_vs_reaction(panel.attrs["events"], panel, lags=(2, 5, 10))
    sr = dvr["sharpe_by_lag"]
    assert sr[2] > 0
    assert dvr["retention_t5_over_t2"] >= 0.5     # drift, not a reaction artifact
    # all three lags positive (persistence), not a single-bar pop
    assert sr[5] > 0 and sr[10] > 0


# --------------------------------------------------------------------------- #
# Surprise-shuffle placebo.
# --------------------------------------------------------------------------- #

def test_surprise_shuffle_is_near_zero_on_planted_world():
    panel = make_pead_panel(mode="planted_pead", seed=7)
    sr = pead.surprise_shuffle_sr(panel.attrs["events"], panel)
    assert abs(sr) < 0.5, "shuffling SUE across events must destroy the edge"


# --------------------------------------------------------------------------- #
# Machinery gate (paired).
# --------------------------------------------------------------------------- #

def test_machinery_gate_planted_beats_null_paired():
    gate = pead.machinery_gate(seeds=(7, 11, 23))
    assert gate["passed"]
    assert min(gate["planted_sr"]) > 0.5
    assert max(gate["null_sr"]) < 0.5
    # paired: planted strictly beats null on EVERY seed
    assert all(d > 0 for d in gate["diffs"])


# --------------------------------------------------------------------------- #
# End-to-end offline run of run_pead._run_trial on a written fixture CSV.
# --------------------------------------------------------------------------- #

def _write_fixture_csv(panel: pd.DataFrame, path: str) -> None:
    """Write a small Bloomberg-style surprise CSV from a synthetic world's events
    (NOT in data_cache — a tmp fixture the harness can ingest in tests)."""
    ev = panel.attrs["events"]
    ev[["ticker", "ann_date", "period", "actual_eps", "est_eps", "std_est"]] \
        .to_csv(path, index=False)


def test_end_to_end_run_trial_planted_recovers_and_null_rejects(tmp_path):
    import run_pead

    # PLANTED world -> write fixture CSV -> parse -> SUE -> _run_trial
    planted = make_pead_panel(mode="planted_pead", seed=7)
    csv_p = tmp_path / "pead_planted.csv"
    _write_fixture_csv(planted, str(csv_p))
    parsed_p = pead.parse_pead_csv(str(csv_p))
    events_p = pead.compute_sue(parsed_p)
    assert (events_p["sue_method"] == "std").all()           # std_est present
    res_p = run_pead._run_trial(events_p, planted, n_trials=14)

    # the 60-day drift is recovered in the planted world
    assert res_p["net_sharpe"] > 1.0
    assert res_p["event_study"]["n_events"] > 50
    # drift persists; shuffle ~0; effect not confined to the illiquid tercile
    assert res_p["drift_vs_reaction"]["retention_t5_over_t2"] >= 0.5
    assert abs(res_p["shuffle_sr"]) < 0.5
    # verdict is EXACTLY the success-criteria conjunction of the printed gates
    g = res_p["gates"]
    assert res_p["graduate"] == bool(
        g["t_nw"] and g["sr_pos"] and g["beats_baselines"] and g["dsr"]
        and g["drift_vs_reaction"] and g["not_illiquid_only"])

    # NULL world -> the same harness must find ~nothing
    null = make_pead_panel(mode="null_pead", seed=7)
    csv_n = tmp_path / "pead_null.csv"
    _write_fixture_csv(null, str(csv_n))
    events_n = pead.compute_sue(pead.parse_pead_csv(str(csv_n)))
    res_n = run_pead._run_trial(events_n, null, n_trials=14)
    assert abs(res_n["net_sharpe"]) < 0.5     # no drift
    assert not res_n["graduate"]               # cannot graduate on the null world


def test_run_trial_verdict_is_exact_conjunction(tmp_path):
    """The graduate flag is the AND of the six registered gates — nothing else."""
    import run_pead
    panel = make_pead_panel(mode="planted_pead", seed=23)
    events = pead.compute_sue(
        pead.parse_pead_csv(io.StringIO(
            panel.attrs["events"][["ticker", "ann_date", "period",
                                   "actual_eps", "est_eps", "std_est"]]
            .to_csv(index=False))))
    res = run_pead._run_trial(events, panel, n_trials=14)
    g = res["gates"]
    expected = (g["t_nw"] and g["sr_pos"] and g["beats_baselines"]
                and g["dsr"] and g["drift_vs_reaction"] and g["not_illiquid_only"])
    assert res["graduate"] == bool(expected)
