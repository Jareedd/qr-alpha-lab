"""H13 post-earnings-announcement drift (PEAD) — PURE harness functions.

PEAD (Bernard-Thomas 1989): prices DRIFT in the direction of an earnings
surprise for weeks after the announcement, because the market under-reacts to
the news. The load-bearing input is the surprise relative to the
PRE-ANNOUNCEMENT consensus — which free data lacks (free sources carry the
realized actual, not the at-the-time estimate); Bloomberg/IBES PIT estimates
are the unlock. This module is the COMPUTE layer: it has no network and does no
I/O except parsing a surprise CSV. Real prices are injected by the caller
(Tiingo / the survivorship-safe SEC source); tests inject a synthetic panel.

The cardinal PIT rule, enforced everywhere in this module: NOTHING uses prices
on or before the announcement bar; the position is entered strictly at
announcement + ``enter_lag`` trading days (T+2 by default — skip the
announcement-day jump so we measure DRIFT, not the immediate reaction, and
avoid same-bar leakage), and forward returns are strictly post-entry. Every
public function below restates its own one-line PIT argument.

Shape: PEAD is an EVENT study of the same shape as the H8 deletion study
(quantlab.events) — enter event + k, hold H days, long/short by a per-event
score, overlapping events averaged in event time. ``pead_event_study`` reuses
that event-time aggregation pattern (a deletion is one name long vs a matched
basket short; a PEAD book is the top-SUE quantile long vs the bottom-SUE
quantile short — the same overlapping-event averaging), specialized to a
quantile long-short keyed on SUE rather than a matched control.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Canonical tidy columns produced by parse_pead_csv (the harness contract).
_REQUIRED = ["ticker", "ann_date", "actual_eps", "est_eps"]
_OPTIONAL = ["period", "surprise_pct", "num_est", "std_est"]
_TIDY_COLS = ["ticker", "ann_date", "period", "actual_eps", "est_eps",
              "surprise_pct", "num_est", "std_est"]


# --------------------------------------------------------------------------- #
# CSV parsing (the ONLY I/O in this module; no network).
# --------------------------------------------------------------------------- #

def parse_pead_csv(path_or_buffer) -> pd.DataFrame:
    """Parse a Bloomberg PEAD surprise CSV into a tidy DataFrame.

    Input schema (writeup/bloomberg_pead_pull.md): columns ``ticker, ann_date,
    period, actual_eps, est_eps, surprise_pct, num_est, std_est``; only
    ``ticker, ann_date, actual_eps, est_eps`` are GUARANTEED, the rest may be
    absent or blank. Output: tidy DataFrame with ALL of ``_TIDY_COLS`` present
    (missing optional columns materialized as NaN), ``ann_date`` as datetime,
    numeric columns coerced, malformed rows (missing key fields / unparseable
    dates / non-numeric required EPS) dropped. Robust to header whitespace/case.

    PIT-safety: this is a pure parser — it reads the operator's file and assigns
    types; ``ann_date`` is the event timestamp the rest of the harness gates on,
    and nothing here references prices or the future.
    """
    df = pd.read_csv(path_or_buffer)
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Materialize every canonical column (optional ones as NaN if absent).
    for col in _TIDY_COLS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[_TIDY_COLS].copy()

    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["ann_date"] = pd.to_datetime(df["ann_date"], errors="coerce")
    df["period"] = df["period"].astype(str).str.strip()
    df.loc[df["period"].isin(["nan", "None", ""]), "period"] = np.nan
    for col in ("actual_eps", "est_eps", "surprise_pct", "num_est", "std_est"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop malformed rows: a usable event needs a ticker, a parseable
    # announcement date, and both EPS numbers (the SUE numerator).
    bad_ticker = df["ticker"].isin(["", "NAN", "NONE"])
    df = df[~bad_ticker]
    # pandas>=3.0 string dtype propagates NaN through .astype(str) instead of
    # stringifying it, so a missing ticker must be dropped as NaN too.
    df = df.dropna(subset=["ticker", "ann_date", "actual_eps", "est_eps"])
    return df.sort_values(["ann_date", "ticker"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# SUE (standardized unexpected earnings) — the per-event surprise score.
# --------------------------------------------------------------------------- #

def compute_sue(df: pd.DataFrame) -> pd.DataFrame:
    """Per-event SUE (standardized unexpected earnings).

    Primary definition (used wherever ``std_est`` is present and > 0): SUE =
    ``(actual_eps - est_eps) / std_est`` — the surprise standardized by analyst
    DISPERSION, the canonical Bernard-Thomas measure. Fallback, used row-wise
    ONLY where ``std_est`` is absent/<=0: Bloomberg's ``surprise_pct`` (treated
    as a percentage, divided by 100 to a fraction) if present, else
    ``(actual_eps - est_eps) / |est_eps|`` (a relative surprise). The two
    definitions are NEVER silently averaged: each row gets exactly one, and the
    chosen branch is recorded in a ``sue_method`` column ('std' / 'surprise_pct'
    / 'rel_est') so a downstream report can show the mix. Rows where no branch
    yields a finite SUE are dropped.

    PIT-safety: SUE is a function of the announcement's actual vs the
    at-the-announcement consensus only (Bloomberg computes ``surprise_pct``
    against the estimate standing BEFORE the print — PIT by construction); it
    references no price and no post-announcement information.
    """
    out = df.copy()
    actual = out["actual_eps"].astype(float)
    est = out["est_eps"].astype(float)
    std = out.get("std_est", pd.Series(np.nan, index=out.index)).astype(float)
    spct = out.get("surprise_pct", pd.Series(np.nan, index=out.index)).astype(float)

    sue = pd.Series(np.nan, index=out.index, dtype=float)
    method = pd.Series("none", index=out.index, dtype=object)

    # Primary: dispersion-standardized, where std_est is usable.
    has_std = std.notna() & (std > 0)
    sue[has_std] = (actual[has_std] - est[has_std]) / std[has_std]
    method[has_std] = "std"

    # Fallback 1: Bloomberg surprise_pct (a percent -> fraction).
    use_spct = (~has_std) & spct.notna()
    sue[use_spct] = spct[use_spct] / 100.0
    method[use_spct] = "surprise_pct"

    # Fallback 2: relative surprise vs |est|, where neither above is available.
    use_rel = (~has_std) & (~use_spct) & est.notna() & (est.abs() > 0)
    sue[use_rel] = (actual[use_rel] - est[use_rel]) / est[use_rel].abs()
    method[use_rel] = "rel_est"

    out["sue"] = sue
    out["sue_method"] = method
    out = out[out["sue"].notna() & np.isfinite(out["sue"])]
    return out.reset_index(drop=True)


# Pre-registered H13 data-quality filter (frozen 2026-06-26, before forward
# returns). EPS reported to 2 decimals quantizes the relative surprise on
# sub-$0.10-EPS quarters: a 1-cent rounding on a 3-cent estimate is a 33% swing,
# so SUE sign becomes unreliable on low-price (often post-split) names — the PIT
# cross-check found ~42% (NVDA) / ~35% (WMT) FMP-vs-AV SUE sign flips concentrated
# there. Because the trial RANKS on SUE and trades the extreme quintiles, those
# quantized rows can land in the wrong leg. We DROP them (root-cause removal of
# the rounding artifact) rather than winsorize (which caps magnitude but keeps a
# sign-flipped row in the extreme leg). Declared filter, count reported; NOT a
# tunable knob.
MIN_ABS_EST = 0.10


def drop_low_eps(events: pd.DataFrame, min_abs_est: float = MIN_ABS_EST
                 ) -> tuple[pd.DataFrame, int]:
    """Drop announcements whose |est_eps| <= ``min_abs_est`` (the frozen H13
    rounding filter). Returns ``(kept_events, n_dropped)``. Pure; no leakage
    (uses only the at-announcement estimate, references no price or future bar)."""
    est = pd.to_numeric(events["est_eps"], errors="coerce")
    keep = est.abs() > float(min_abs_est)
    n_dropped = int((~keep).sum())
    return events[keep].reset_index(drop=True), n_dropped


# --------------------------------------------------------------------------- #
# Event-time long/short PEAD portfolio (the core of H13).
# --------------------------------------------------------------------------- #

def _entry_position(idx: pd.DatetimeIndex, ann_date: pd.Timestamp,
                    enter_lag: int) -> int | None:
    """Trading-day position of the ENTRY bar = announcement + ``enter_lag``.

    ``pos`` is the last trading bar on or before the announcement (so prices on
    that bar are the most recent the market knew AT the announcement); the entry
    bar is ``pos + enter_lag``. With ``enter_lag >= 1`` the entry is strictly
    AFTER the announcement bar — the PIT guarantee. Returns None if the entry
    bar is outside the panel."""
    pos = int(idx.searchsorted(pd.Timestamp(ann_date), side="right")) - 1
    if pos < 0:
        return None
    entry = pos + enter_lag
    return entry


def _event_legs(
    events: pd.DataFrame, prices: pd.DataFrame, enter_lag: int, hold: int,
    quantile: float, cost_bps: float,
) -> tuple[list[pd.Series], int, int]:
    """Build per-event signed daily excess-return series for the long/short
    book, cross-sectionally assigning each event to LONG (top-SUE quantile of
    its announcement-month cohort) / SHORT (bottom) / neither.

    Cohort = all events announced in the same calendar month, so SUE is ranked
    against contemporaneous peers (a cross-sectional sort, not an absolute
    threshold). Each long/short event contributes a daily series over its hold
    window starting at entry + 1; a one-time round-trip cost (``cost_bps`` each
    way) is charged on the first and last held bar. Returns
    ``(daily_series, n_long, n_short)``."""
    rets = prices.pct_change(fill_method=None)
    idx = prices.index
    ev = events.copy()
    ev["ann_date"] = pd.to_datetime(ev["ann_date"])
    ev["cohort"] = ev["ann_date"].dt.to_period("M")

    daily: list[pd.Series] = []
    n_long = n_short = 0
    for _, cohort in ev.groupby("cohort"):
        valid = cohort[cohort["ticker"].isin(prices.columns)]
        if len(valid) < max(5, int(np.ceil(1.0 / quantile))):
            # Too few names this month to form stable quantiles -> skip cohort
            # (a thin month cannot define a top/bottom 20% honestly).
            continue
        lo = valid["sue"].quantile(quantile)
        hi = valid["sue"].quantile(1.0 - quantile)
        for _, row in valid.iterrows():
            sign = 0.0
            if row["sue"] >= hi:
                sign = 1.0
            elif row["sue"] <= lo:
                sign = -1.0
            if sign == 0.0:
                continue
            entry = _entry_position(idx, row["ann_date"], enter_lag)
            if entry is None or entry < 0 or entry + hold >= len(idx):
                continue
            win = slice(entry + 1, entry + hold + 1)   # earns strictly post-entry
            r = rets[row["ticker"]].iloc[win].to_numpy() * sign
            if len(r) == 0:
                continue
            r = r.copy()
            r[0] -= cost_bps / 1e4
            r[-1] -= cost_bps / 1e4
            daily.append(pd.Series(r, index=idx[win]))
            if sign > 0:
                n_long += 1
            else:
                n_short += 1
    return daily, n_long, n_short


def pead_event_study(
    events: pd.DataFrame, prices: pd.DataFrame, enter_lag: int = 2,
    hold: int = 60, quantile: float = 0.2, cost_bps: float = 10.0,
    periods: int = 252,
) -> dict:
    """Event-time PEAD long/short portfolio (the H13 primary book).

    LONG the top-SUE quantile / SHORT the bottom-SUE quantile of each
    announcement-month cohort, ENTER at announcement + ``enter_lag`` trading
    days (T+2 by default — never on/before the announcement bar), hold ``hold``
    trading days, overlapping events averaged per calendar day into one daily
    series (the same event-time aggregation as quantlab.events.event_study, with
    a SUE-quantile long/short replacing the deletion's matched-control hedge).
    Returns the net daily series plus a summary dict (net Sharpe, Newey-West t at
    ``hold`` lags, per-event turnover proxy, n_events, n_long, n_short).

    PIT-safety: each leg's returns begin at entry + 1 = announcement +
    ``enter_lag`` + 1; with ``enter_lag >= 1`` nothing touches the announcement
    bar or earlier. The SUE used to sort is the at-announcement surprise
    (compute_sue), which references no price.
    """
    daily, n_long, n_short = _event_legs(
        events, prices, enter_lag, hold, quantile, cost_bps)
    if daily:
        port = pd.concat(daily, axis=1).mean(axis=1).dropna()
    else:
        port = pd.Series(dtype=float)
    n_events = n_long + n_short
    from quantlab import metrics
    sr = metrics.sharpe(port, periods=periods) if len(port) else 0.0
    t_nw = metrics.newey_west_tstat(port, lags=hold) if len(port) > hold + 2 else np.nan
    # Turnover proxy: each event is a one-shot round trip (enter once, exit
    # once) -> ~2 side-trades per event over the hold window; reported for the
    # cost-as-headline discipline (CLAUDE.md law #4).
    turnover = float(2.0 * n_events)
    return {
        "daily_portfolio": port,
        "net_sharpe": sr,
        "t_nw": float(t_nw) if t_nw == t_nw else np.nan,
        "turnover_trades": turnover,
        "n_events": int(n_events),
        "n_long": int(n_long),
        "n_short": int(n_short),
        "enter_lag": int(enter_lag),
        "hold": int(hold),
    }


# --------------------------------------------------------------------------- #
# Monthly cross-sectional variant.
# --------------------------------------------------------------------------- #

def pead_cross_sectional(
    events: pd.DataFrame, prices: pd.DataFrame, window_days: int = 90,
    quantile: float = 0.2, hold_months: int = 1, cost_bps: float = 10.0,
    periods: int = 12,
) -> dict:
    """Monthly cross-sectional PEAD variant: rank names by their MOST-RECENT SUE
    within a trailing ``window_days`` window, form a dollar-neutral EW quantile
    long/short, hold one month, rebalance monthly.

    At each month-end ``t`` a name's signal is the SUE of its latest
    announcement with ``ann_date`` in ``(t - window_days, t]`` (most-recent-only;
    no averaging across quarters). Long the top quantile / short the bottom; the
    book earns the next month's total return. Returns net monthly series +
    summary (net Sharpe, NW t at 1 lag — non-overlapping monthly labels,
    turnover, n_months).

    PIT-safety: the signal at month-end ``t`` uses ONLY announcements with
    ``ann_date <= t`` (a name with no announcement in the trailing window is not
    scored), and the label is the strictly-forward next-month return; weights
    formed at ``t`` earn from ``t`` to ``t+1``. Nothing peeks past ``t``.
    """
    ev = events.copy()
    ev["ann_date"] = pd.to_datetime(ev["ann_date"])
    ev = ev[ev["ticker"].isin(prices.columns)].sort_values("ann_date")

    # Month-end grid spanning the priceable window.
    monthly_px = prices.resample("ME").last()
    asof = monthly_px.index
    fwd = monthly_px.pct_change(fill_method=None).shift(-1)  # next-month return

    weights = pd.DataFrame(0.0, index=asof, columns=prices.columns)
    win = pd.Timedelta(days=window_days)
    for t in asof:
        recent = ev[(ev["ann_date"] <= t) & (ev["ann_date"] > t - win)]
        if recent.empty:
            continue
        # most-recent SUE per name within the window
        sig = (recent.sort_values("ann_date")
               .groupby("ticker")["sue"].last())
        if sig.shape[0] < max(5, int(np.ceil(1.0 / quantile))):
            continue
        lo, hi = sig.quantile(quantile), sig.quantile(1.0 - quantile)
        longs = sig[sig >= hi].index
        shorts = sig[sig <= lo].index
        if len(longs) == 0 or len(shorts) == 0:
            continue
        weights.loc[t, longs] = 0.5 / len(longs)
        weights.loc[t, shorts] = -0.5 / len(shorts)

    gross = (weights * fwd.reindex_like(weights)).sum(axis=1, min_count=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    net = (gross - turnover * cost_bps / 1e4).dropna()

    from quantlab import metrics
    sr = metrics.sharpe(net, periods=periods) if len(net) else 0.0
    t_nw = metrics.newey_west_tstat(net, lags=1) if len(net) > 3 else np.nan
    n_active = int((weights.abs().sum(axis=1) > 0).sum())
    return {
        "net": net,
        "net_sharpe": sr,
        "t_nw": float(t_nw) if t_nw == t_nw else np.nan,
        "annual_turnover": float(turnover.sum() / max(len(weights), 1) * periods),
        "n_months": int(net.shape[0]),
        "n_active_months": n_active,
        "weights": weights,
    }


# --------------------------------------------------------------------------- #
# Registered paired controls.
# --------------------------------------------------------------------------- #

def drift_vs_reaction(
    events: pd.DataFrame, prices: pd.DataFrame, lags=(2, 5, 10),
    hold: int = 60, quantile: float = 0.2, cost_bps: float = 10.0,
    periods: int = 252,
) -> dict:
    """The PEAD-specific kill control: net Sharpe of the SAME event-study book
    entered at each lag in ``lags`` (T+2, T+5, T+10 by default).

    TRUE drift PERSISTS for weeks, so a real PEAD retains most of its Sharpe as
    the entry slips a few days; a COLLAPSE from T+2 to T+5 means the strategy
    captured the announcement-day REACTION (the immediate jump, untradable once
    you account for entry latency — the trial-#11 entry-lag lesson applied to
    events), not drift. Returns ``{lag: net_sharpe}`` plus the T+5/T+2 retention
    ratio (the registered >=50% gate's measured quantity).

    PIT-safety: identical to pead_event_study at each lag — every entry is
    announcement + lag (>=2) and returns start the bar after; later lags are
    STRICTLY more conservative (they touch even less near the announcement)."""
    out: dict = {"sharpe_by_lag": {}}
    for lag in lags:
        res = pead_event_study(
            events, prices, enter_lag=int(lag), hold=hold, quantile=quantile,
            cost_bps=cost_bps, periods=periods)
        out["sharpe_by_lag"][int(lag)] = res["net_sharpe"]
    base = out["sharpe_by_lag"].get(int(lags[0]), 0.0)
    second = out["sharpe_by_lag"].get(int(lags[1]), 0.0) if len(lags) > 1 else base
    out["retention_t5_over_t2"] = (second / base) if base > 0 else 0.0
    return out


def surprise_shuffle_sr(
    events: pd.DataFrame, prices: pd.DataFrame, enter_lag: int = 2,
    hold: int = 60, quantile: float = 0.2, cost_bps: float = 10.0,
    periods: int = 252, seed: int = 13,
) -> float:
    """Surprise-shuffle placebo: permute the SUE column ACROSS events (so the
    surprise no longer corresponds to its own announcement/return), then run the
    identical event study. A real drift is destroyed by the shuffle -> the book
    earns ~0 (|SR| small). A non-zero shuffled SR signals the 'edge' was an
    artifact of the construction, not the surprise.

    PIT-safety: only the SUE LABELS are permuted; announcement dates, tickers and
    prices are untouched, so entry is still T+``enter_lag`` and the placebo
    measures pure-luck dispersion of the book, not any real edge."""
    rng = np.random.default_rng(seed)
    shuffled = events.copy().reset_index(drop=True)
    shuffled["sue"] = rng.permutation(shuffled["sue"].to_numpy())
    res = pead_event_study(
        shuffled, prices, enter_lag=enter_lag, hold=hold, quantile=quantile,
        cost_bps=cost_bps, periods=periods)
    return res["net_sharpe"]


def _dollar_volume_proxy(prices: pd.DataFrame, volumes: pd.DataFrame | None,
                         asof: pd.Timestamp, lookback: int = 63) -> pd.Series:
    """Trailing liquidity/size proxy per name as of ``asof``, past-only.

    If a ``volumes`` panel is supplied, use trailing-``lookback`` mean dollar
    volume (price * volume); otherwise fall back to trailing price level (a
    coarse but PIT size proxy when no volume is available — the synthetic world
    has no volume). Uses only data on or before ``asof``."""
    px = prices.loc[:asof]
    if len(px) < 2:
        return pd.Series(dtype=float)
    if volumes is not None:
        dv = (px * volumes.reindex_like(px)).iloc[-lookback:].mean()
        return dv.replace(0, np.nan).dropna()
    return px.iloc[-lookback:].mean().dropna()


def by_size_tercile(
    events: pd.DataFrame, prices: pd.DataFrame, volumes: pd.DataFrame | None = None,
    enter_lag: int = 2, hold: int = 60, quantile: float = 0.2,
    cost_bps: float = 10.0, periods: int = 252, lookback: int = 63,
) -> dict:
    """Split events into liquidity/size TERCILES (by a trailing dollar-volume —
    or price-level — proxy measured AT each announcement) and run the event
    study within each tercile separately.

    PEAD's headline risk is that the effect lives ONLY in the smallest/least-
    liquid tercile — if so it is not tradable on a large-cap universe. This
    reports each tercile's net Sharpe and n_events so a single-tercile
    concentration is visible (and logged, not hidden). Returns
    ``{'tercile_sharpe': {0,1,2: SR}, 'tercile_n': {...}}`` (tercile 0 = smallest).

    PIT-safety: the size proxy at each announcement uses only prices/volumes ON
    OR BEFORE the announcement bar; the event study within each tercile keeps the
    T+``enter_lag`` entry. No future information enters the split."""
    ev = events.copy()
    ev["ann_date"] = pd.to_datetime(ev["ann_date"])
    ev = ev[ev["ticker"].isin(prices.columns)]
    # Per-event size proxy at its announcement.
    sizes = []
    for _, row in ev.iterrows():
        proxy = _dollar_volume_proxy(prices, volumes, row["ann_date"], lookback)
        sizes.append(proxy.get(row["ticker"], np.nan))
    ev = ev.assign(_size=sizes).dropna(subset=["_size"])
    if ev.empty:
        return {"tercile_sharpe": {}, "tercile_n": {}}
    # Tercile labels 0/1/2 by ascending size (0 = smallest/least liquid).
    try:
        ev["_terc"] = pd.qcut(ev["_size"].rank(method="first"), 3,
                              labels=[0, 1, 2]).astype(int)
    except ValueError:
        return {"tercile_sharpe": {}, "tercile_n": {}}
    terc_sr, terc_n = {}, {}
    for terc, grp in ev.groupby("_terc"):
        res = pead_event_study(
            grp.drop(columns=["_size", "_terc"]), prices, enter_lag=enter_lag,
            hold=hold, quantile=quantile, cost_bps=cost_bps, periods=periods)
        terc_sr[int(terc)] = res["net_sharpe"]
        terc_n[int(terc)] = res["n_events"]
    return {"tercile_sharpe": terc_sr, "tercile_n": terc_n}


# --------------------------------------------------------------------------- #
# Machinery gate (law #4) — synthetic two-world validation.
# --------------------------------------------------------------------------- #

def machinery_gate(
    seeds=(7, 11, 23), enter_lag: int = 2, hold: int = 60, quantile: float = 0.2,
    cost_bps: float = 10.0, threshold: float = 0.5,
) -> dict:
    """Synthetic two-world gate: a ``planted_pead`` world (forward drift injected
    ONLY after high-|SUE| events) must be RECOVERED (event-study net SR >
    ``threshold``) and a ``null_pead`` world (same events, no drift) REJECTED
    (net SR < ``threshold``), PAIRED per seed (planted SR - null SR > 0 for each
    seed, and planted clears the threshold while null does not).

    If the harness cannot tell planted PEAD from its absence TODAY, no real H13
    number is trustworthy -> the runner ABORTS before spending a trial.

    Imports ``synthetic`` lazily (no import-time dependency on the synthetic
    module from the compute layer)."""
    from quantlab.synthetic import make_pead_panel

    planted_sr, null_sr, diffs = [], [], []
    for s in seeds:
        planted = make_pead_panel(mode="planted_pead", seed=s)
        null = make_pead_panel(mode="null_pead", seed=s)
        # SUE rides in attrs["events"] already (consistent with the planted EPS).
        ev_p = planted.attrs["events"]
        ev_n = null.attrs["events"]
        rp = pead_event_study(ev_p, planted, enter_lag=enter_lag, hold=hold,
                              quantile=quantile, cost_bps=cost_bps)
        rn = pead_event_study(ev_n, null, enter_lag=enter_lag, hold=hold,
                              quantile=quantile, cost_bps=cost_bps)
        planted_sr.append(rp["net_sharpe"])
        null_sr.append(rn["net_sharpe"])
        diffs.append(rp["net_sharpe"] - rn["net_sharpe"])
    passed = (min(planted_sr) > threshold
              and max(null_sr) < threshold
              and min(diffs) > 0.0)
    return {
        "planted_sr": planted_sr, "null_sr": null_sr, "diffs": diffs,
        "seeds": list(seeds), "threshold": threshold, "passed": bool(passed),
    }
