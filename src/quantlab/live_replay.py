"""Offline replay of the book ``live.py`` actually trades.

The walk-forward backtest (``scripts/run_pipeline.py``) fits every 126
sessions on prices from 2009 and holds a decile book for 21 sessions,
marked close to close. The deployed cycle does none of those things. It
loads prices from ``LIVE_PRICE_START`` (2018-01-01), refits ridge every
session on rows whose 21-day label is fully realized, rebuilds the decile
book every session, and sends a market order that fills at the next open.

This module is that cycle, run on history. It does not call ``run_daily``,
does not construct a broker client, and does not read credentials.

Point-in-time: features are ``features.build_features`` (past prices only).
The fit that predicts date t trains on dates ``<= t - horizon`` trading
sessions, which is ``prices.index[-(horizon + 1)]`` when t is the last row
— the same cutoff ``live.live_target_weights`` uses. The label on that
cutoff date consumes the close of t and nothing later. Fills are the next
session's open: a weight chosen at the close of t earns the open-to-open
return from t+1 to t+2, never the close-to-open gap that preceded the fill.
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from quantlab import backtest, features, metrics, risk
from quantlab.live import HORIZON, LIVE_PRICE_START, MIN_TRAIN_ROWS

# The pipeline's linear cost, and the cost every prior trial of this book
# was graded on. live.py itself submits uncosted paper orders; it does not
# encode a different number. 10 bps per side is the cost this replay uses.
COST_BPS = 10.0
QUANTILE = 0.1
MIN_NAMES = 50
MAX_NAME_FRAC = 0.05
# Declared sizing notional for the integer-share diagnostic. The 2026-06-10
# log recorded a $1M paper account; this is that figure, not a live read.
ROUNDING_EQUITY = 1_000_000.0
RIDGE_ALPHA = 10.0
N_TRIALS = 14
# Retire unless the out-of-sample rank IC clears this Newey-West t.
IC_T_MIN = 3.48
DSR_MIN = 0.95

# Ex-ante power, computed from graduation_hurdle.required_ann_sharpe before
# any price was read. Weekday count 2018-01-02..2026-10-16 is 2,294; minus
# 252 sessions of feature warmup, 23 sessions to clear MIN_TRAIN_ROWS at a
# 450-name cross-section, and the 21-session label embargo leaves 1,998
# design days. NYSE holidays shorten that. The quoted hurdle uses 1,900
# days, symmetric returns (skew 0, raw kurtosis 3), N=14.
MDE_N_OBS_DESIGN = 1900
MDE_NET_ANN_SHARPE = 1.233


def ridge_from_sums(
    n: float,
    sum_x: np.ndarray,
    sum_y: float,
    xtx: np.ndarray,
    xty: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, float]:
    """Ridge with intercept, matching ``sklearn.linear_model.Ridge``.

    Sklearn centers X and y, then solves ``(Xc'Xc + alpha I) coef = Xc'yc``
    without penalizing the intercept. Sufficient statistics are enough
    because that normal equation does not need the rows. ``alpha`` is the
    penalty on the sum of squares, not the mean — the same convention as
    ``Ridge(alpha=10)`` in ``make_model``.
    """
    n = float(n)
    if n <= 0:
        raise ValueError("ridge_from_sums needs at least one row")
    x_mean = sum_x / n
    y_mean = float(sum_y) / n
    a = np.array(xtx, dtype=float, copy=True)
    a -= n * np.outer(x_mean, x_mean)
    p = a.shape[0]
    a.flat[:: p + 1] += alpha
    b = np.asarray(xty, dtype=float) - y_mean * (n * x_mean)
    coef = np.linalg.solve(a, b)
    intercept = float(y_mean - x_mean @ coef)
    return coef, intercept


def training_cutoff_position(i: int, horizon: int) -> int:
    """Position of the last trainable date when predicting position ``i``.

    ``prices.index[-(horizon + 1)]`` when ``i`` is the last row. The label
    on that date uses the close at ``i`` and no later close.
    """
    return i - horizon


def drop_incomplete_session(
    frame: pd.DataFrame, now: dt.datetime | None = None
) -> pd.DataFrame:
    """Drop a still-forming last bar.

    A partial session dated today in New York, observed before 16:00
    America/New_York, is not a close. Leaving it in would put a half-day
    price into features and into any label that completes on that bar.
    """
    if frame.empty:
        return frame
    if now is None:
        now = dt.datetime.now(dt.timezone.utc)
    elif now.tzinfo is None:
        raise ValueError("drop_incomplete_session requires a timezone-aware now")
    ny = now.astimezone(ZoneInfo("America/New_York"))
    close_time = ny.replace(hour=16, minute=0, second=0, microsecond=0)
    last = pd.Timestamp(frame.index[-1])
    if last.tzinfo is not None:
        last = last.tz_convert("UTC").tz_localize(None)
    session = pd.Timestamp(ny.date())
    if last.normalize() == session and ny < close_time:
        return frame.iloc[:-1].copy()
    return frame


def _feature_matrix(
    feats: dict[str, pd.DataFrame], member_mask: pd.DataFrame | None
) -> pd.DataFrame:
    """Stacked feature rows with a finite value in every column.

    Membership is applied after ``dropna``, matching ``live_target_weights``:
    a member with one missing feature is unscored, and a non-member never
    enters the cross-section the model ranks.
    """
    wide = pd.DataFrame({name: frame.stack() for name, frame in feats.items()})
    wide.index.names = ["date", "ticker"]
    wide = wide.dropna()
    if member_mask is not None:
        in_index = member_mask.stack()
        in_index.index.names = ["date", "ticker"]
        keep = in_index.reindex(wide.index).fillna(False).to_numpy(dtype=bool)
        wide = wide.iloc[np.flatnonzero(keep)]
    return wide


def _prefix_sums(
    panel: pd.DataFrame, feature_cols: list[str]
) -> tuple[pd.DatetimeIndex, dict[str, np.ndarray]]:
    """Cumulative sufficient statistics of the labeled training panel.

    ``prefix['n'][k]`` counts rows whose date is ``unique_dates[k]`` or
    earlier. Predicting a cutoff date uses the last entry with
    ``unique_dates <= cutoff``.
    """
    panel = panel.sort_index(level="date", sort_remaining=False)
    dates = pd.DatetimeIndex(panel.index.get_level_values("date"))
    unique, codes = np.unique(dates.to_numpy(), return_inverse=True)
    x = panel[feature_cols].to_numpy(dtype=np.float64)
    y = panel["label"].to_numpy(dtype=np.float64)
    n_dates, p = len(unique), x.shape[1]
    n = np.bincount(codes, minlength=n_dates).astype(np.float64)
    sum_y = np.bincount(codes, weights=y, minlength=n_dates)
    sum_x = np.column_stack(
        [np.bincount(codes, weights=x[:, j], minlength=n_dates) for j in range(p)]
    )
    xty = np.column_stack(
        [
            np.bincount(codes, weights=x[:, j] * y, minlength=n_dates)
            for j in range(p)
        ]
    )
    xtx = np.zeros((n_dates, p, p), dtype=np.float64)
    for j in range(p):
        for k in range(j, p):
            both = np.bincount(codes, weights=x[:, j] * x[:, k], minlength=n_dates)
            xtx[:, j, k] = both
            xtx[:, k, j] = both
    prefix = {
        "n": np.cumsum(n),
        "sum_x": np.cumsum(sum_x, axis=0),
        "sum_y": np.cumsum(sum_y),
        "xty": np.cumsum(xty, axis=0),
        "xtx": np.cumsum(xtx, axis=0),
    }
    return pd.DatetimeIndex(unique), prefix


def daily_refit_predictions(
    prices: pd.DataFrame,
    member_mask: pd.DataFrame | None = None,
    *,
    horizon: int = HORIZON,
    min_train_rows: int = MIN_TRAIN_ROWS,
    min_names: int = MIN_NAMES,
    alpha: float = RIDGE_ALPHA,
) -> dict:
    """One ridge fit per session, each using only labels complete that day.

    ``pred_raw`` is indexed by (date, ticker) and is the pre-demean score
    live logs for IC. Days with fewer than ``min_names`` finite
    member-features are skipped (live would hold the cycle; a historical
    replay leaves that session unscored and the book, if any, carries
    forward in the backtest). ``feats`` and ``panel`` are returned so the
    caller does not rebuild them.
    """
    feats = features.build_features(prices, member_mask=member_mask)
    labels = features.build_labels(
        prices, horizon=horizon, residualize=True, member_mask=member_mask
    )
    panel = features.stack_panel(feats, labels)
    feature_cols = [c for c in panel.columns if c != "label"]
    unique_dates, prefix = _prefix_sums(panel, feature_cols)
    wide = _feature_matrix(feats, member_mask)

    by_date: dict[pd.Timestamp, pd.DataFrame] = {}
    for date, group in wide.groupby(level="date", sort=False):
        by_date[pd.Timestamp(date)] = group

    pieces: list[pd.Series] = []
    n_skipped_thin = 0
    n_skipped_short = 0
    index = pd.DatetimeIndex(prices.index)
    for i, today in enumerate(index):
        cut_pos = training_cutoff_position(i, horizon)
        if cut_pos < 0:
            n_skipped_short += 1
            continue
        cutoff = pd.Timestamp(index[cut_pos])
        k = int(unique_dates.searchsorted(cutoff, side="right")) - 1
        if k < 0 or prefix["n"][k] < min_train_rows:
            n_skipped_short += 1
            continue
        coef, intercept = ridge_from_sums(
            prefix["n"][k],
            prefix["sum_x"][k],
            prefix["sum_y"][k],
            prefix["xtx"][k],
            prefix["xty"][k],
            alpha,
        )
        group = by_date.get(pd.Timestamp(today))
        if group is None or len(group) < min_names:
            n_skipped_thin += 1
            continue
        x_today = group[feature_cols].to_numpy(dtype=np.float64)
        pred = x_today @ coef + intercept
        tickers = group.index.get_level_values("ticker")
        pieces.append(
            pd.Series(
                pred,
                index=pd.MultiIndex.from_arrays(
                    [[today] * len(tickers), tickers], names=["date", "ticker"]
                ),
            )
        )
    if not pieces:
        raise RuntimeError(
            "daily refit produced no predictions "
            f"(short={n_skipped_short}, thin={n_skipped_thin})"
        )
    raw = pd.concat(pieces).sort_index()
    info = {
        "n_prediction_dates": int(raw.index.get_level_values("date").nunique()),
        "n_skipped_short_history": n_skipped_short,
        "n_skipped_thin_cross_section": n_skipped_thin,
        "first_prediction": str(pd.Timestamp(raw.index.get_level_values("date")[0]).date()),
        "last_prediction": str(pd.Timestamp(raw.index.get_level_values("date")[-1]).date()),
        "feature_cols": feature_cols,
        "horizon": horizon,
        "min_train_rows": min_train_rows,
        "alpha": alpha,
        "label": "residual",
    }
    return {"pred_raw": raw, "info": info, "feats": feats, "panel": panel}


def replay_books(
    prices: pd.DataFrame,
    member_mask: pd.DataFrame | None,
    sectors: dict[str, str],
    *,
    horizon: int = HORIZON,
    min_names: int = MIN_NAMES,
    quantile: float = QUANTILE,
) -> dict:
    """Daily-refit predictions and the two pre-declared weight books.

    Both books are built from the same sector-neutral scores. ``weights_daily``
    is the deployed cadence (``rebalance_every=1``). ``weights_21d`` is the
    old hold, kept as a cost diagnostic: it is not a second strategy and it
    cannot change the trial's verdict.
    """
    fitted = daily_refit_predictions(
        prices, member_mask, horizon=horizon, min_names=min_names
    )
    raw = fitted["pred_raw"]
    info = fitted["info"]
    neutral = risk.neutralize_predictions_by_sector(raw, sectors) if sectors else raw
    weights_daily = backtest.predictions_to_weights(
        neutral, quantile=quantile, rebalance_every=1
    )
    weights_21 = backtest.predictions_to_weights(
        neutral, quantile=quantile, rebalance_every=21
    )
    # Same market live_target_weights neutralizes against: equal-weight
    # member mean, asset returns unmasked (a non-member's beta is unused
    # because that name never gets a weight).
    rets = prices.pct_change(fill_method=None)
    mkt = rets.where(member_mask).mean(axis=1) if member_mask is not None else rets.mean(axis=1)
    betas = risk.rolling_beta(rets, mkt)
    weights_daily = risk.beta_neutralize_weights(weights_daily, betas)
    weights_21 = risk.beta_neutralize_weights(weights_21, betas)
    return {
        "pred_raw": raw,
        "pred_neutral": neutral,
        "weights_daily": weights_daily,
        "weights_21d": weights_21,
        "betas": betas,
        "market_close": mkt,
        "panel": fitted["panel"],
        "feats": fitted["feats"],
        "info": info,
    }


def run_next_open_backtest(
    weights: pd.DataFrame,
    opens: pd.DataFrame,
    cost_bps: float = COST_BPS,
) -> dict:
    """Mark a close-t decision at the next open.

    Open-to-open return on date d is ``Open[d] / Open[d-1] - 1``. The book
    chosen at the close of t is filled at the open of t+1 and held until the
    next fill, so it earns the open-to-open return on t+2 (two trading rows
    later). Costs use the same one-way linear charge as ``backtest.run_backtest``:
    ``|Δw| * cost_bps / 1e4``, with the initial build included and no
    terminal liquidation (``diff`` against a flat book the day before the
    first fill).
    """
    oo = opens.pct_change(fill_method=None).reindex(columns=weights.columns)
    daily_w = (
        weights.reindex(index=oo.index, columns=weights.columns).ffill().shift(2).fillna(0.0)
    )
    gross = (daily_w * oo).sum(axis=1)
    w_change = daily_w.diff().abs().sum(axis=1)
    costs = w_change * (cost_bps / 1e4)
    net = gross - costs
    held = daily_w.abs().sum(axis=1) > 0
    if not bool(held.any()):
        raise RuntimeError("next-open backtest: the book is never on")
    start = held[held].index[0]
    gross = gross.loc[start:]
    net = net.loc[start:]
    costs = costs.loc[start:]
    w_change = w_change.loc[start:]
    daily_w = daily_w.loc[start:]
    years = max(len(net) / 252.0, 1e-9)
    # sum skips a leading NaN if the slice starts on the first row.
    ann_turnover = float(w_change.sum() / 2.0 / years)
    return {
        "gross": gross,
        "net": net,
        "costs": costs,
        "annual_turnover": ann_turnover,
        "ann_cost_drag": float((gross - net).dropna().mean() * 252.0),
        "daily_weights": daily_w,
    }


def integer_share_weights(
    weights: pd.DataFrame,
    closes: pd.DataFrame,
    equity: float = ROUNDING_EQUITY,
    max_name_frac: float = MAX_NAME_FRAC,
) -> pd.DataFrame:
    """Decision weights after the live order's integer-share truncation.

    ``live.orders_from_weights`` clips each name at ``max_name_frac`` and
    sets the target share count to ``int(w * equity / price)``, which
    truncates toward zero. The weight actually held, if every order fills,
    is ``shares * price / equity``. Names without a positive price drop out.
    This is a sizing identity, not a broker call.
    """
    if equity <= 0:
        raise ValueError(f"equity must be positive, got {equity}")
    px = closes.reindex(index=weights.index, columns=weights.columns)
    clipped = weights.clip(lower=-max_name_frac, upper=max_name_frac)
    raw = clipped.to_numpy(dtype=float) * float(equity) / px.to_numpy(dtype=float)
    finite = np.isfinite(raw)
    shares = np.zeros(raw.shape, dtype=float)
    # np.trunc matches int() (toward zero) for finite values.
    shares[finite] = np.trunc(raw[finite])
    held = shares * px.to_numpy(dtype=float) / float(equity)
    held = np.where(np.isfinite(held), held, 0.0)
    return pd.DataFrame(held, index=weights.index, columns=weights.columns)


def exante_beta(weights: pd.DataFrame, betas: pd.DataFrame) -> pd.Series:
    """w · β at each decision date, with the neutralizer's beta fill.

    Latest beta row at or before the date; missing betas on active names
    filled with the cross-sectional mean of those names. On a book that
    just went through ``beta_neutralize_weights`` this is ~0. Rounding
    breaks that identity, which is the quantity the diagnostic reports.
    """
    out: dict[pd.Timestamp, float] = {}
    for date, row in weights.iterrows():
        active = row[row != 0.0]
        if active.empty:
            continue
        hist = betas.loc[:date]
        if hist.empty:
            continue
        b = hist.iloc[-1].reindex(active.index)
        b = b.fillna(b.mean())
        if b.notna().any():
            out[pd.Timestamp(date)] = float((active * b).sum())
    return pd.Series(out, dtype=float).sort_index()


def beta_summary(path: pd.Series) -> dict:
    """Mean, mean absolute, and 95th percentile absolute of a beta path."""
    x = path.dropna()
    if x.empty:
        return {"n": 0, "mean": None, "mean_abs": None, "p95_abs": None}
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "mean_abs": float(x.abs().mean()),
        "p95_abs": float(x.abs().quantile(0.95)),
    }


def kill_verdict(ic_t: float, sharpe_net: float, dsr: float) -> dict:
    """RETIRE unless every pre-registered leg passes.

    Legs, frozen before the run: Newey-West IC t >= 3.48, net Sharpe > 0,
    DSR >= 0.95. One failure retires the book. A non-finite number fails
    the leg it sits on.
    """
    ic_ok = bool(np.isfinite(ic_t) and ic_t >= IC_T_MIN)
    sr_ok = bool(np.isfinite(sharpe_net) and sharpe_net > 0.0)
    dsr_ok = bool(np.isfinite(dsr) and dsr >= DSR_MIN)
    return {
        "verdict": "KEEP" if ic_ok and sr_ok and dsr_ok else "RETIRE",
        "legs": {
            "ic_t_nw": {"value": None if not np.isfinite(ic_t) else float(ic_t),
                        "threshold": IC_T_MIN, "pass": ic_ok},
            "sharpe_net": {"value": None if not np.isfinite(sharpe_net) else float(sharpe_net),
                           "threshold": 0.0, "rule": "> 0", "pass": sr_ok},
            "dsr": {"value": None if not np.isfinite(dsr) else float(dsr),
                    "threshold": DSR_MIN, "pass": dsr_ok},
        },
    }


def pack_arm(result: dict, n_trials: int) -> dict:
    """Gross and net together. A gross-only number is not a result here."""
    stats = metrics.summary(
        result["net"], result["gross"], result["annual_turnover"], n_trials=n_trials
    )
    stats["ann_cost_drag"] = float(result["ann_cost_drag"])
    return stats


def render_report(payload: dict) -> str:
    """Short factual report. The verdict sentence is the payload's, not a rewrite."""
    oos = payload["oos_daily"]
    dec = payload["cost_decomposition"]
    beta = payload["beta_after_rounding"]
    verdict = payload["verdict"]["verdict"]
    legs = payload["verdict"]["legs"]
    d21 = dec["rebalance_21d"]
    daily = dec["daily"]

    def fmt(x, spec=".3f"):
        if x is None:
            return "n/a"
        return format(x, spec)

    lines = [
        "# Trial 14 — the deployed live config, backtested as deployed",
        "",
        f"**Verdict: {verdict}.**",
        "",
        "The book that has been paper-traded was never the book that was backtested. "
        "The pipeline fit every 126 sessions on prices from 2009 and held a decile "
        "book for 21 sessions, marked close to close. The deployed cycle loads prices "
        f"from {payload['config']['train_start']}, refits ridge every session, rebuilds "
        "the decile book every session, and fills at the next open. This trial is that "
        "cycle, on history, scored once.",
        "",
        "## Pre-registered kill rule",
        "",
        "Retire unless all three hold, at N=14 (this evaluation counts; it is not a "
        "free reproduction of trial 5):",
        "",
        f"- OOS rank-IC Newey-West t (lags = {payload['config']['horizon']}, "
        f"scored on `pred_raw`) ≥ {IC_T_MIN}: "
        f"**{fmt(legs['ic_t_nw']['value'], '.2f')}** "
        f"({'pass' if legs['ic_t_nw']['pass'] else 'fail'})",
        f"- net Sharpe > 0: **{fmt(legs['sharpe_net']['value'], '.3f')}** "
        f"({'pass' if legs['sharpe_net']['pass'] else 'fail'})",
        f"- DSR ≥ {DSR_MIN}: **{fmt(legs['dsr']['value'], '.3f')}** "
        f"({'pass' if legs['dsr']['pass'] else 'fail'})",
        "",
        f"Ex-ante minimum detectable effect, stated before the download: "
        f"**{MDE_NET_ANN_SHARPE:.3f} net annual Sharpe** at N=14, "
        f"design n_obs={MDE_N_OBS_DESIGN}, symmetric returns "
        f"(the DSR = 0.95 hurdle from `graduation_hurdle.required_ann_sharpe`). "
        f"Realized n_obs={payload['mde_realized']['n_obs']}; the same function at "
        f"the realized sample and the realized skew is "
        f"{fmt(payload['mde_realized']['net_ann_sharpe'], '.3f')}. "
        "That reconciliation does not move the kill thresholds.",
        "",
        "## Graded book (daily refit, daily decile, next open, 10 bps/side)",
        "",
        f"- Mean rank IC (`pred_raw` vs residual 21-day label): "
        f"{fmt(oos['mean_rank_ic'], '.4f')} "
        f"(Newey-West t {fmt(oos['ic_tstat_newey_west'], '.2f')})",
        f"- Gross Sharpe {fmt(oos['sharpe_gross'], '.3f')}, "
        f"net Sharpe {fmt(oos['sharpe_net'], '.3f')}, "
        f"DSR {fmt(oos['dsr'], '.3f')} at N={payload['config']['n_trials']}",
        f"- Net annual return {fmt(oos['ann_return_net'], '.2%')}, "
        f"ann. vol {fmt(oos['ann_vol'], '.2%')}, "
        f"max drawdown {fmt(oos['max_drawdown'], '.2%')}",
        f"- One-way turnover {fmt(oos['annual_turnover'], '.2f')}×/yr, "
        f"cost drag {fmt(oos['ann_cost_drag'], '.2%')}/yr",
        f"- 12-1 momentum baseline, same dates, same next-open fills, same 10 bps: "
        f"net Sharpe {fmt(oos['baseline_mom_sharpe_net'], '.3f')} "
        f"(model {'beats' if oos['beats_mom_baseline'] else 'does not beat'} it). "
        f"Equal-weight member book, open-to-open: Sharpe {fmt(oos['baseline_ew_sharpe'], '.3f')}.",
        f"- Window: {oos['first']} → {oos['last']} ({oos['n_days']} sessions).",
        "",
        "The momentum and equal-weight numbers are the law-5 baselines. They were "
        "not kill criteria. The 21-day book below is not one either.",
        "",
        "## Turnover and cost: daily vs 21-session rebalance",
        "",
        "Same scores, same next-open fills, same 10 bps/side. The 21-session arm "
        "is the old hold applied to these predictions. It is a decomposition of "
        "the cost, not a candidate, and it cannot flip a retire into a keep.",
        "",
        "| book | one-way turnover | cost drag / yr | gross Sharpe | net Sharpe |",
        "|---|---:|---:|---:|---:|",
        f"| daily (graded) | {daily['annual_turnover']:.2f}× | "
        f"{daily['ann_cost_drag']:.2%} | {daily['sharpe_gross']:.3f} | "
        f"{daily['sharpe_net']:.3f} |",
        f"| 21-session hold | {d21['annual_turnover']:.2f}× | "
        f"{d21['ann_cost_drag']:.2%} | {d21['sharpe_gross']:.3f} | "
        f"{d21['sharpe_net']:.3f} |",
        "",
        f"Turnover ratio, daily / 21-session: "
        f"{fmt(dec['turnover_ratio_daily_over_21d'], '.2f')}. "
        f"Extra cost drag from rebuilding every day: "
        f"{fmt(dec['extra_cost_drag'], '.2%')}/yr.",
        "",
        "## Beta after integer-share rounding",
        "",
        f"Continuous weights are projected to zero ex-ante beta against the "
        f"equal-weight member market. The order path then clips at "
        f"{payload['config']['max_name_frac']:.0%} and truncates to integer shares "
        f"at ${payload['config']['rounding_equity']:,.0f} "
        f"(the declared notional, not a broker read), using that session's close. "
        f"`int(w · equity / price)` truncates toward zero, as `orders_from_weights` does.",
        "",
        f"- Ex-ante beta, continuous: mean {fmt(beta['continuous']['mean'], '.4f')}, "
        f"mean |β| {fmt(beta['continuous']['mean_abs'], '.4f')}",
        f"- Ex-ante beta, after rounding: mean {fmt(beta['rounded']['mean'], '.4f')}, "
        f"mean |β| {fmt(beta['rounded']['mean_abs'], '.4f')}, "
        f"p95 |β| {fmt(beta['rounded']['p95_abs'], '.4f')}",
        f"- Mean net weight after rounding (dollar-neutrality break): "
        f"{fmt(beta['rounded_net_exposure_mean'], '.4f')} "
        f"(mean absolute {fmt(beta['rounded_net_exposure_mean_abs'], '.4f')})",
        f"- Mean L1 distance between continuous and rounded weights: "
        f"{fmt(beta['mean_l1_distortion'], '.4f')}",
        f"- Realized full-sample beta of the gross next-open returns vs the member-mean "
        f"open-to-open market: continuous {fmt(beta['realized_member_continuous'], '.3f')}, "
        f"rounded {fmt(beta['realized_member_rounded'], '.3f')}",
        f"- Realized beta vs SPY open-to-open (descriptive, not a kill): "
        f"continuous {fmt(beta.get('realized_spy_continuous'), '.3f')}, "
        f"rounded {fmt(beta.get('realized_spy_rounded'), '.3f')}",
        "",
        "## What this does not say",
        "",
        "Adjusted Yahoo prices are revised when a dividend lands; the history is "
        "not point-in-time. Names that died often have no Yahoo history, and the "
        "delisting return is missing. Sectors are as-of-today. Costs are linear "
        "10 bps per side: no borrow fee, no square-root impact. The rounding beta "
        "is computed at a fixed $1M, not along the paper account's equity path. "
        "The 21-session arm was not allowed to overturn the verdict.",
        "",
        f"Price panel fingerprint (adjusted closes): `{payload['data']['sha256'][:16]}…` "
        f"({payload['data']['n_cols']} names, {payload['data']['n_rows']} rows, "
        f"{payload['data']['first']} → {payload['data']['last']}). "
        "Regenerate with `python scripts/run_trial14_live_config.py --hypothesis T14 --n-trials 14`.",
        "",
    ]
    return "\n".join(lines)


def assert_ridge_matches_sklearn() -> None:
    """Import-time guard used by the test, not by the trial."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 5))
    y = rng.normal(size=400)
    model = Ridge(alpha=RIDGE_ALPHA).fit(x, y)
    coef, intercept = ridge_from_sums(
        len(x), x.sum(axis=0), float(y.sum()), x.T @ x, x.T @ y, RIDGE_ALPHA
    )
    if not np.allclose(coef, model.coef_, rtol=1e-8, atol=1e-8):
        raise AssertionError("ridge_from_sums coef drifted from sklearn")
    if not np.isclose(intercept, float(model.intercept_), rtol=1e-8, atol=1e-8):
        raise AssertionError("ridge_from_sums intercept drifted from sklearn")
