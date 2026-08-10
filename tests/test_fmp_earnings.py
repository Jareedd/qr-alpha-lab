"""H13 FMP earnings layer + leakage gate: offline known-answer tests. NO network.

Pins (the gate must NOT be a rubber stamp):
  * parse_fmp_earnings KAT — field map, drops FUTURE/null-actual rows, computes
    surprise_pct, materializes the tidy pead schema.
  * validate_pead_events: a GOOD frame PASSES; a BACKFILLED (est==actual) one
    FAILS; a DEGENERATE (all-zero surprise) one FAILS; a one-signed one FAILS; a
    FUTURE-dated-with-actual one FAILS the date check.
  * the cross-source check runs on an overlap and is reported (not auto-graded).
  * DATA-GATE wiring in run_pead: monkeypatch fmp.build_events to return a BAD
    frame -> the FMP path sys.exits spending NO trial; a GOOD frame -> proceeds
    to compute SUE + the price load.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from quantlab import fmp_earnings as fmp


# --------------------------------------------------------------------------- #
# parse_fmp_earnings — known-answer test.
# --------------------------------------------------------------------------- #

def test_parse_fmp_earnings_kat_maps_fields_drops_future_computes_surprise():
    # Newest-first, mixed: a future (null actual) row, a null-estimate row, two
    # good past rows. lastUpdated rides along; symbol upper-cased.
    payload = [
        {"symbol": "aapl", "date": "2026-10-30", "epsActual": None,
         "epsEstimated": 2.10, "lastUpdated": "2026-09-01"},          # FUTURE -> drop
        {"symbol": "AAPL", "date": "2025-07-31", "epsActual": 1.40,
         "epsEstimated": None, "lastUpdated": "2025-08-01"},          # no est -> drop
        {"symbol": "AAPL", "date": "2025-05-01", "epsActual": 1.65,
         "epsEstimated": 1.50, "lastUpdated": "2025-05-02"},          # good
        {"symbol": "AAPL", "date": "2025-02-01", "epsActual": 0.90,
         "epsEstimated": 1.00, "lastUpdated": "2025-02-02"},          # good (miss)
    ]
    df = fmp.parse_fmp_earnings(payload)

    # tidy pead schema present (+ internal last_updated)
    for c in ["ticker", "ann_date", "period", "actual_eps", "est_eps",
              "surprise_pct", "num_est", "std_est", "last_updated"]:
        assert c in df.columns
    # only the two good PAST rows survive, sorted ascending by date
    assert len(df) == 2
    assert list(df["ann_date"].dt.strftime("%Y-%m-%d")) == ["2025-02-01", "2025-05-01"]
    assert (df["ticker"] == "AAPL").all()
    # dispersion fields are NaN (FMP has none)
    assert df["std_est"].isna().all() and df["num_est"].isna().all()
    assert df["period"].isna().all()
    # surprise_pct = 100*(actual-est)/|est|
    beat = df[df["ann_date"] == "2025-05-01"].iloc[0]
    assert beat["surprise_pct"] == pytest.approx(100 * (1.65 - 1.50) / 1.50)   # +10%
    miss = df[df["ann_date"] == "2025-02-01"].iloc[0]
    assert miss["surprise_pct"] == pytest.approx(100 * (0.90 - 1.00) / 1.00)   # -10%


def test_parse_fmp_earnings_symbol_fallback_and_empty():
    # a row lacking 'symbol' uses the symbol= override
    df = fmp.parse_fmp_earnings(
        [{"date": "2025-05-01", "epsActual": 1.0, "epsEstimated": 0.9}],
        symbol="msft")
    assert df["ticker"].iloc[0] == "MSFT"
    # empty / None input -> empty tidy frame with datetime cols
    empty = fmp.parse_fmp_earnings([])
    assert empty.empty and "ann_date" in empty.columns
    assert empty["ann_date"].dtype.kind == "M"


# --------------------------------------------------------------------------- #
# Synthetic events frames for the gate.
# --------------------------------------------------------------------------- #

def _good_events(n_tickers: int = 8, per: int = 24, seed: int = 0) -> pd.DataFrame:
    """A realistic, non-degenerate events frame: both-signed, mostly-nonzero
    surprises, est != actual, lastUpdated ~1 day after the announcement."""
    rng = np.random.default_rng(seed)
    recs = []
    base = pd.Timestamp("2015-01-15")
    for ti in range(n_tickers):
        tkr = f"T{ti:02d}"
        for q in range(per):
            ann = base + pd.Timedelta(days=int(91 * q + ti))
            est = float(rng.uniform(0.5, 3.0))
            surprise = float(rng.normal(0.0, 0.08))     # both signs, real spread
            actual = est * (1.0 + surprise)
            recs.append({
                "ticker": tkr, "ann_date": ann, "period": np.nan,
                "actual_eps": round(actual, 4), "est_eps": round(est, 4),
                "surprise_pct": 100 * (actual - est) / abs(est),
                "num_est": np.nan, "std_est": np.nan,
                "last_updated": ann + pd.Timedelta(days=1),
            })
    return pd.DataFrame(recs)


def test_validate_good_frame_passes():
    rep = fmp.validate_pead_events(_good_events(), today=pd.Timestamp("2026-01-01"))
    assert rep["passed"] is True
    assert rep["date_semantics"]["passed"]
    assert rep["surprise_dist"]["passed"]
    assert rep["backfilled"]["passed"]
    # honest caveat is always present, even on a pass
    assert "NOT proven" in rep["caveat"]


def test_validate_backfilled_frame_fails():
    ev = _good_events()
    ev["actual_eps"] = ev["est_eps"]                 # est == actual EVERYWHERE
    ev["surprise_pct"] = 0.0
    rep = fmp.validate_pead_events(ev, today=pd.Timestamp("2026-01-01"))
    assert rep["passed"] is False
    assert rep["backfilled"]["passed"] is False
    assert rep["backfilled"]["exact_match_frac"] == pytest.approx(1.0)
    assert any("backfilled" in r for r in rep["reasons"])


def test_validate_degenerate_all_zero_surprise_fails():
    # est == actual gives zero surprise AND 100% exact-match; both HARD checks trip.
    ev = _good_events()
    ev["actual_eps"] = ev["est_eps"]
    ev["surprise_pct"] = 0.0
    rep = fmp.validate_pead_events(ev, today=pd.Timestamp("2026-01-01"))
    assert rep["passed"] is False
    assert rep["surprise_dist"]["passed"] is False


def test_validate_one_signed_surprise_fails():
    # All-positive surprises (a feed where actual always >= est) -> one-signed.
    rng = np.random.default_rng(1)
    ev = _good_events()
    pos = np.abs(rng.normal(0.0, 0.08, size=len(ev))) + 0.01
    ev["actual_eps"] = ev["est_eps"] * (1.0 + pos)
    ev["surprise_pct"] = 100 * (ev["actual_eps"] - ev["est_eps"]) / ev["est_eps"].abs()
    rep = fmp.validate_pead_events(ev, today=pd.Timestamp("2026-01-01"))
    assert rep["passed"] is False
    assert rep["surprise_dist"]["passed"] is False
    assert any("one-signed" in r for r in rep["reasons"])


def test_validate_future_dated_with_actual_fails_date_check():
    ev = _good_events()
    # plant one event dated AFTER 'today' that still carries a realized actual
    ev.loc[ev.index[0], "ann_date"] = pd.Timestamp("2030-01-01")
    rep = fmp.validate_pead_events(ev, today=pd.Timestamp("2026-01-01"))
    assert rep["passed"] is False
    assert rep["date_semantics"]["passed"] is False
    assert rep["date_semantics"]["n_future_with_actual"] >= 1
    assert any("date_semantics" in r for r in rep["reasons"])


def test_validate_empty_frame_fails_closed():
    rep = fmp.validate_pead_events(pd.DataFrame(
        columns=["ticker", "ann_date", "actual_eps", "est_eps"]))
    assert rep["passed"] is False


def test_validate_cross_source_reported_on_overlap():
    ev = _good_events(n_tickers=3, per=8)
    # an agreeing second source on the same (ticker, ann_date) keys
    cross = ev[["ticker", "ann_date", "est_eps"]].copy()
    rep = fmp.validate_pead_events(ev, cross_check=cross,
                                   today=pd.Timestamp("2026-01-01"))
    assert rep["cross_source"]["n_overlap"] > 0
    assert rep["cross_source"]["agree_frac_within_1pct"] == pytest.approx(1.0)
    # cross-source is REPORTED, not part of the pass/fail gate
    assert rep["passed"] is True


# --------------------------------------------------------------------------- #
# Alpha Vantage parser (pure).
# --------------------------------------------------------------------------- #

def test_parse_alphavantage_earnings_kat():
    payload = {"quarterlyEarnings": [
        {"reportedDate": "2025-05-01", "reportedEPS": "1.65", "estimatedEPS": "1.50"},
        {"reportedDate": "2025-02-01", "reportedEPS": "0.90", "estimatedEPS": "None"},
    ]}
    df = fmp.parse_alphavantage_earnings(payload, "AAPL")
    assert len(df) == 1                              # the 'None' estimate row dropped
    assert df["est_eps"].iloc[0] == pytest.approx(1.50)


# --------------------------------------------------------------------------- #
# DATA-GATE wiring in run_pead — the gate spends no trial on a bad frame.
# --------------------------------------------------------------------------- #

def _patch_common(monkeypatch):
    """Stub registration + machinery gates + universe + price load so the test
    isolates the FMP DATA GATE. None of these touch the network."""
    import run_pead
    monkeypatch.setattr(run_pead, "require_runnable_registration", lambda *a, **k: None)
    monkeypatch.setattr(run_pead.pead, "machinery_gate",
                        lambda *a, **k: {"seeds": [7], "planted_sr": [1.0],
                                         "null_sr": [0.0], "diffs": [1.0],
                                         "passed": True})
    monkeypatch.setattr(run_pead, "_pit_sp500_universe", lambda: ["T00", "T01"])
    return run_pead


def test_run_pead_fmp_gate_bad_frame_aborts_no_trial(monkeypatch):
    run_pead = _patch_common(monkeypatch)
    from quantlab import fmp_earnings as fmpmod

    # build_events returns a BACKFILLED frame -> gate fails -> SystemExit, no trial.
    bad = _good_events()
    bad["actual_eps"] = bad["est_eps"]
    bad["surprise_pct"] = 0.0
    monkeypatch.setattr(fmpmod, "build_events", lambda *a, **k: bad)
    # if the trial ran, _run_trial would be reached; make it explode loudly.
    monkeypatch.setattr(run_pead, "_run_trial",
                        lambda *a, **k: pytest.fail("trial ran on a GATE-FAILED feed!"))
    monkeypatch.setattr(sys, "argv", ["run_pead.py", "--source", "fmp"])
    with pytest.raises(SystemExit) as exc:
        run_pead.main()
    assert "DATA GATE" in str(exc.value)


def test_run_pead_fmp_gate_good_frame_proceeds(monkeypatch):
    run_pead = _patch_common(monkeypatch)
    from quantlab import fmp_earnings as fmpmod

    good = _good_events()
    monkeypatch.setattr(fmpmod, "build_events", lambda *a, **k: good)
    # stub the price load + the trial: we only assert the gate PASSED and the
    # pipeline advanced to scoring (no network).
    seen = {}

    def _fake_prices(events, price_source):
        seen["events"] = events
        return pd.DataFrame()    # value irrelevant; trial is stubbed

    def _fake_trial(events, prices, n_trials, volumes=None):
        seen["ran"] = True
        return {"event_study": {"n_events": 1, "n_long": 1, "n_short": 0},
                "cross_sectional": {"net_sharpe": 0.0, "t_nw": float("nan"),
                                    "n_months": 0, "annual_turnover": 0.0},
                "net_sharpe": 0.0, "t_nw": float("nan"), "dsr": float("nan"),
                "n_obs": 0, "drift_vs_reaction": {"sharpe_by_lag": {},
                "retention_t5_over_t2": 0.0}, "shuffle_sr": 0.0,
                "size_tercile": {"tercile_sharpe": {}, "tercile_n": {}},
                "sr_ew": 0.0, "sr_mom": 0.0, "mde_ann": 0.0,
                "gates": {"t_nw": False, "sr_pos": False, "beats_baselines": False,
                          "dsr": False, "drift_vs_reaction": False,
                          "not_illiquid_only": False},
                "graduate": False}

    monkeypatch.setattr(run_pead, "_load_prices_for_events", _fake_prices)
    monkeypatch.setattr(run_pead, "_run_trial", _fake_trial)
    monkeypatch.setattr(run_pead, "_print_verdict", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["run_pead.py", "--source", "fmp"])

    run_pead.main()                       # must NOT raise
    assert seen.get("ran") is True
    # the events handed downstream carry a SUE column (gate -> compute_sue ran)
    assert "sue" in seen["events"].columns


def test_fetch_earnings_skips_402_burst_without_caching(monkeypatch, tmp_path):
    """A free-tier burst cap (HTTP 402) after retries must SKIP the name (return
    []) and NEVER cache the error — so a later resumable pass retries it. (429 is
    handled the same way; both are transient.)"""
    import urllib.error
    monkeypatch.setattr(fmp, "load_env", lambda *a, **k: None)
    monkeypatch.setenv("FMP_API_KEY", "testkey")
    src = fmp.FMPEarningsSource(cache_dir=str(tmp_path), min_interval=0.0)
    assert src.min_interval == 0.0          # spacing is settable for bulk pulls

    def boom(url, **k):
        raise urllib.error.HTTPError(url, 402, "Payment Required", {}, None)

    monkeypatch.setattr(src, "_get", boom)
    assert src.fetch_earnings("ZZZZ") == []
    assert not (tmp_path / "earnings_ZZZZ.json").exists()   # quota error not cached


def test_build_av_events_cached_only_emits_tidy_schema(tmp_path):
    """build_av_events reads the warmed AV cache (no network) and emits the SAME
    tidy schema as the FMP builder, so the gate/filter/SUE consume AV unchanged.
    AV carries no dispersion/lastUpdated -> those columns are NaN/NaT."""
    payload = {"symbol": "XYZ", "quarterlyEarnings": [
        {"reportedDate": "2023-02-01", "reportedEPS": "1.20", "estimatedEPS": "1.10"},
        {"reportedDate": "2023-05-01", "reportedEPS": "0.90", "estimatedEPS": "1.00"},
        {"reportedDate": "2099-09-09", "reportedEPS": "None", "estimatedEPS": "1.0"},
    ]}
    import json as _json
    (tmp_path / "earnings_XYZ.json").write_text(_json.dumps(payload), encoding="utf-8")
    out = fmp.build_av_events(["XYZ", "NOPE"], cache_dir=str(tmp_path))
    # uncached "NOPE" is skipped; future/null-actual row dropped -> 2 events
    assert list(out["ticker"].unique()) == ["XYZ"]
    assert len(out) == 2
    assert set(fmp._TIDY_COLS + ["last_updated"]).issubset(out.columns)
    assert out["std_est"].isna().all() and out["last_updated"].isna().all()
    # flows through the SUE compute on the scale-invariant rel_est branch
    from quantlab import pead
    sued = pead.compute_sue(out.drop(columns=["last_updated"]))
    assert set(sued["sue_method"]) == {"rel_est"}


def test_run_pead_applies_frozen_low_eps_filter(monkeypatch):
    """The runner drops |est_eps| <= $0.10 events (frozen H13 filter) BEFORE
    scoring: a sub-threshold row must not reach _run_trial."""
    run_pead = _patch_common(monkeypatch)
    from quantlab import fmp_earnings as fmpmod

    good = _good_events(n_tickers=4, per=8)
    good.loc[len(good)] = {                       # one quantization-prone row
        "ticker": "LOWX", "ann_date": pd.Timestamp("2016-02-01"), "period": np.nan,
        "actual_eps": 0.04, "est_eps": 0.03, "surprise_pct": 33.3,
        "num_est": np.nan, "std_est": np.nan,
        "last_updated": pd.Timestamp("2016-02-02"),
    }
    monkeypatch.setattr(fmpmod, "build_events", lambda *a, **k: good)
    seen = {}
    monkeypatch.setattr(run_pead, "_load_prices_for_events",
                        lambda events, ps: seen.update(events=events) or pd.DataFrame())
    monkeypatch.setattr(run_pead, "_run_trial",
                        lambda *a, **k: {"event_study": {"n_events": 1, "n_long": 1,
                        "n_short": 0}, "cross_sectional": {"net_sharpe": 0.0,
                        "t_nw": float("nan"), "n_months": 0, "annual_turnover": 0.0},
                        "net_sharpe": 0.0, "t_nw": float("nan"), "dsr": float("nan"),
                        "n_obs": 0, "drift_vs_reaction": {"sharpe_by_lag": {},
                        "retention_t5_over_t2": 0.0}, "shuffle_sr": 0.0,
                        "size_tercile": {"tercile_sharpe": {}, "tercile_n": {}},
                        "sr_ew": 0.0, "sr_mom": 0.0, "mde_ann": 0.0, "gates": {},
                        "graduate": False})
    monkeypatch.setattr(run_pead, "_print_verdict", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["run_pead.py", "--source", "fmp"])

    run_pead.main()
    assert "LOWX" not in set(seen["events"]["ticker"])   # the low-EPS row was dropped
