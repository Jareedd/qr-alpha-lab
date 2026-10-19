"""Trial 14: offline replay of the deployed live config.

The graded book is the one ``live.py`` trades: prices from 2018-01-01,
ridge refit every session on fully labeled rows, decile book rebuilt every
session, sector demean, beta projection, 10 bps/side, filled at the next
open. The 21-session hold and the integer-share beta are diagnostics
declared in the T14 registration. They do not get a vote.

This process does not call ``live.run_daily``, does not construct a broker
client, does not read ``.env``, and does not submit orders. Real prices
come from yfinance through ``quantlab.data`` (or the cache that loader
already wrote). Synthetic planted/noise runs are the falsification gate
for this path; they do not spend a trial.

    python scripts/run_trial14_live_config.py --hypothesis T14 --n-trials 14
    python scripts/run_trial14_live_config.py --data planted
    python scripts/run_trial14_live_config.py --data noise
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from graduation_hurdle import required_ann_sharpe  # noqa: E402
from quantlab import baselines, live, metrics, models  # noqa: E402
from quantlab.live_diagnosis import beta_decomposition  # noqa: E402
from quantlab.live_replay import (  # noqa: E402
    COST_BPS,
    MDE_N_OBS_DESIGN,
    MDE_NET_ANN_SHARPE,
    N_TRIALS,
    ROUNDING_EQUITY,
)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return x if np.isfinite(x) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return str(obj.date())
    return obj


def _realized_beta(strat: pd.Series, market: pd.Series) -> float | None:
    both = pd.concat([strat, market], axis=1, keys=["s", "m"], sort=True).dropna()
    if len(both) < 30:
        return None
    return float(beta_decomposition(both["s"], both["m"])["beta"])


def _member_open_market(opens: pd.DataFrame, member_mask: pd.DataFrame | None) -> pd.Series:
    oo = opens.pct_change(fill_method=None)
    if member_mask is not None:
        oo = oo.where(member_mask.reindex_like(oo))
    return oo.mean(axis=1)


def _score(books: dict, opens: pd.DataFrame, member_mask, n_trials: int, cost_bps: float) -> dict:
    """Next-open P&L for both cadences, IC, baselines, rounding beta."""
    from quantlab import live_replay as lr

    daily = lr.run_next_open_backtest(books["weights_daily"], opens, cost_bps=cost_bps)
    hold = lr.run_next_open_backtest(books["weights_21d"], opens, cost_bps=cost_bps)
    graded = lr.pack_arm(daily, n_trials=n_trials)
    hold_stats = lr.pack_arm(hold, n_trials=n_trials)

    ic = models.information_coefficient(books["pred_raw"], books["panel"])
    ic_t = metrics.newey_west_tstat(ic, lags=books["info"]["horizon"])
    graded["mean_rank_ic"] = float(ic.mean())
    graded["ic_tstat_newey_west"] = float(ic_t)
    graded["first"] = str(pd.Timestamp(daily["net"].index[0]).date())
    graded["last"] = str(pd.Timestamp(daily["net"].index[-1]).date())

    mom_w = baselines.momentum_baseline_weights(
        books["feats"], books["pred_raw"].index, rebalance_every=1
    )
    mom = lr.run_next_open_backtest(mom_w, opens, cost_bps=cost_bps)
    graded["baseline_mom_sharpe_net"] = metrics.sharpe(mom["net"])
    ew = _member_open_market(opens, member_mask).reindex(daily["net"].index)
    graded["baseline_ew_sharpe"] = metrics.sharpe(ew)
    graded["beats_mom_baseline"] = bool(
        graded["sharpe_net"] > graded["baseline_mom_sharpe_net"]
    )

    rounded = lr.integer_share_weights(books["weights_daily"], books["closes"])
    rounded_bt = lr.run_next_open_backtest(rounded, opens, cost_bps=cost_bps)
    b_cont = lr.exante_beta(books["weights_daily"], books["betas"])
    b_round = lr.exante_beta(rounded, books["betas"])
    active = books["weights_daily"].abs().sum(axis=1) > 0
    l1 = (
        rounded.reindex_like(books["weights_daily"]).fillna(0.0)
        - books["weights_daily"].fillna(0.0)
    ).abs().sum(axis=1)
    net_w = rounded.sum(axis=1)
    mkt_oo = _member_open_market(opens, member_mask)
    beta_block = {
        "continuous": lr.beta_summary(b_cont),
        "rounded": lr.beta_summary(b_round),
        "rounded_net_exposure_mean": float(net_w[active].mean()) if active.any() else None,
        "rounded_net_exposure_mean_abs": float(net_w[active].abs().mean()) if active.any() else None,
        "mean_l1_distortion": float(l1[active].mean()) if active.any() else None,
        "realized_member_continuous": _realized_beta(daily["gross"], mkt_oo),
        "realized_member_rounded": _realized_beta(rounded_bt["gross"], mkt_oo),
        "rounded_sharpe_net": metrics.sharpe(rounded_bt["net"]),
    }
    ratio = (
        daily["annual_turnover"] / hold["annual_turnover"]
        if hold["annual_turnover"]
        else None
    )
    decomp = {
        "daily": {
            "annual_turnover": daily["annual_turnover"],
            "ann_cost_drag": daily["ann_cost_drag"],
            "sharpe_gross": graded["sharpe_gross"],
            "sharpe_net": graded["sharpe_net"],
        },
        "rebalance_21d": {
            "annual_turnover": hold["annual_turnover"],
            "ann_cost_drag": hold["ann_cost_drag"],
            "sharpe_gross": hold_stats["sharpe_gross"],
            "sharpe_net": hold_stats["sharpe_net"],
            "dsr": hold_stats["dsr"],
            "note": "diagnostic only; not a kill input and not a second trial",
        },
        "turnover_ratio_daily_over_21d": None if ratio is None else float(ratio),
        "extra_cost_drag": float(daily["ann_cost_drag"] - hold["ann_cost_drag"]),
    }
    return {
        "daily_result": daily,
        "hold_result": hold,
        "oos_daily": graded,
        "cost_decomposition": decomp,
        "beta_after_rounding": beta_block,
        "ic": ic,
    }


def _attach_spy(beta_block: dict, daily_gross: pd.Series, rounded_gross: pd.Series,
                start: str, cache_dir: str) -> None:
    """Descriptive SPY beta. A failed download does not fail the trial."""
    try:
        from quantlab.data import load_opens

        spy = load_opens(
            ["SPY"], start=start, cache_dir=cache_dir, min_coverage=0.0
        )["SPY"].pct_change(fill_method=None)
    except Exception as exc:  # vendor hiccup: the member-mean beta still stands
        beta_block["spy_error"] = str(exc)
        beta_block["realized_spy_continuous"] = None
        beta_block["realized_spy_rounded"] = None
        return
    beta_block["realized_spy_continuous"] = _realized_beta(daily_gross, spy)
    beta_block["realized_spy_rounded"] = _realized_beta(rounded_gross, spy)


def _write_equity(path: str, daily: pd.Series, hold: pd.Series, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    (1 + daily.fillna(0)).cumprod().plot(ax=ax, label="daily rebalance, net")
    (1 + hold.reindex(daily.index).fillna(0)).cumprod().plot(
        ax=ax, label="21-session hold, net", alpha=0.75
    )
    ax.set_title(title)
    ax.set_ylabel("growth of $1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _write_returns(path: str, daily: dict, hold: dict) -> None:
    frame = pd.DataFrame(
        {
            "net_daily": daily["net"],
            "gross_daily": daily["gross"],
            "cost_daily": daily["costs"],
            "net_21d": hold["net"],
            "gross_21d": hold["gross"],
            "cost_21d": hold["costs"],
        }
    )
    frame.index.name = "date"
    frame.to_csv(path)


def run_synthetic(mode: str, seed: int, out_dir: str) -> dict:
    """Falsification gate for this path. Synthetic, labeled, not a trial."""
    from quantlab.live_replay import replay_books, run_next_open_backtest
    from quantlab.synthetic import make_panel

    prices = make_panel(mode=mode, seed=seed)
    # Open = previous close: the next-open fill earns the following
    # close-to-close return. Labeled synthetic. Not a market open.
    opens = prices.shift(1)
    books = replay_books(
        prices, None, prices.attrs.get("sectors", {}), min_names=10
    )
    books["closes"] = prices
    zero = run_next_open_backtest(books["weights_daily"], opens, cost_bps=0.0)
    costed = run_next_open_backtest(books["weights_daily"], opens, cost_bps=COST_BPS)
    ic = models.information_coefficient(books["pred_raw"], books["panel"])
    # n_trials=1 matches the original planted gate's default. The noise gate
    # uses 20, the CI threshold this repo already publishes.
    n_for_dsr = 1 if mode == "planted" else 20
    dsr_zero = metrics.deflated_sharpe_ratio(zero["net"], n_trials=n_for_dsr)
    dsr_cost = metrics.deflated_sharpe_ratio(costed["net"], n_trials=n_for_dsr)
    payload = {
        "synthetic": True,
        "mode": mode,
        "seed": seed,
        "opens": "previous close (synthetic; not a market open)",
        "mean_rank_ic": float(ic.mean()),
        "ic_t_nw": float(metrics.newey_west_tstat(ic, lags=21)),
        "sharpe_net_zero_cost": metrics.sharpe(zero["net"]),
        "dsr_zero_cost": float(dsr_zero),
        "sharpe_net_10bps": metrics.sharpe(costed["net"]),
        "dsr_10bps": float(dsr_cost),
        "annual_turnover_10bps": costed["annual_turnover"],
        "n_trials_for_dsr": n_for_dsr,
        "n_predictions": books["info"]["n_prediction_dates"],
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"trial14_gate_{mode}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(payload), fh, indent=2)
    print(f"[gate:{mode}] IC {payload['mean_rank_ic']:.4f}  "
          f"DSR@0bps {payload['dsr_zero_cost']:.4f}  "
          f"DSR@10bps {payload['dsr_10bps']:.4f}  "
          f"netSR@10bps {payload['sharpe_net_10bps']:.3f}  "
          f"turnover {payload['annual_turnover_10bps']:.1f}x")
    if mode == "planted":
        if not (payload["mean_rank_ic"] > 0.02 and payload["dsr_zero_cost"] > 0.95):
            sys.exit(
                "FALSIFICATION GATE FAILED: daily-refit path did not recover "
                "the planted signal at zero cost. Hunt the leak before any "
                "real-data number from this path is trusted."
            )
    elif mode == "noise":
        # Residual-label rank IC on this synthetic panel is not zero: vol_3m
        # predicts the compounded simple return (Jensen) even when the log
        # shock is unpredictable, and the existing walk-forward shows the
        # same IC. The gate is the book, net of costs. A DSR that clears
        # the noise ceiling is a leak; a nonzero IC alone is not.
        if payload["dsr_10bps"] > 0.5 or payload["dsr_zero_cost"] > 0.5:
            sys.exit(
                "FALSIFICATION GATE FAILED: daily-refit path found alpha in "
                "noise. Stop. Hunt the leak."
            )
    return payload


def run_real(out_dir: str) -> dict:
    from quantlab import live_replay as lr
    from quantlab import universe as univ
    from quantlab.data import load_opens, load_prices

    cache_dir = os.path.join("data_cache", "trial14")
    current, changes = univ.fetch_sp500_tables(cache_dir=cache_dir)
    intervals = univ.build_membership_intervals(current, changes, start="2010-01-01")
    members = univ.all_members_in_window(intervals)
    floor = live.live_min_symbols(len(current))
    print(f"[data] PIT members {len(members)}; download floor {floor} "
          f"(90% of today's {len(current)} names); start {live.LIVE_PRICE_START}")
    closes = load_prices(
        members, start=live.LIVE_PRICE_START, cache_dir=cache_dir,
        min_coverage=0.0, min_symbols=floor,
    )
    opens = load_opens(
        list(closes.columns), start=live.LIVE_PRICE_START, cache_dir=cache_dir,
        min_coverage=0.0, min_symbols=max(1, int(0.9 * closes.shape[1])),
    )
    closes = lr.drop_incomplete_session(closes)
    opens = lr.drop_incomplete_session(opens)
    cols = [c for c in closes.columns if c in opens.columns]
    idx = closes.index.intersection(opens.index).sort_values()
    closes = closes.loc[idx, cols].sort_index()
    opens = opens.loc[idx, cols].sort_index()
    print(f"[data] {closes.shape[1]} names x {closes.shape[0]} sessions "
          f"({closes.index[0].date()} → {closes.index[-1].date()})")

    member_mask = univ.membership_mask(closes.index, closes.columns, intervals)
    sectors = univ.sector_map(current, list(closes.columns))
    cov = univ.coverage_report(members, closes)
    print(f"[universe] {cov['n_with_price_data']}/{cov['n_members_ever']} "
          f"members priced ({cov['pct_covered']:.1%}); "
          f"{cov['n_missing_price_data']} unpriceable")

    print("[model] daily refit (this is the slow step)")
    books = lr.replay_books(closes, member_mask, sectors)
    books["closes"] = closes
    print(f"[model] predictions {books['info']['first_prediction']} → "
          f"{books['info']['last_prediction']} "
          f"({books['info']['n_prediction_dates']} sessions; "
          f"thin skips {books['info']['n_skipped_thin_cross_section']})")

    scored = _score(books, opens, member_mask, n_trials=N_TRIALS, cost_bps=COST_BPS)
    rounded = lr.integer_share_weights(books["weights_daily"], closes)
    rounded_bt = lr.run_next_open_backtest(rounded, opens, cost_bps=COST_BPS)
    _attach_spy(
        scored["beta_after_rounding"],
        scored["daily_result"]["gross"],
        rounded_bt["gross"],
        live.LIVE_PRICE_START,
        cache_dir,
    )

    oos = scored["oos_daily"]
    verdict = lr.kill_verdict(oos["ic_tstat_newey_west"], oos["sharpe_net"], oos["dsr"])
    n_obs = int(oos["n_days"])
    skew = float(oos["skew"])
    # Raw kurtosis for the reconciliation print. metrics.summary stores skew
    # but not kurtosis; recompute from the net series. This does not move
    # the pre-registered MDE.
    kurt = float(pd.Series(scored["daily_result"]["net"]).dropna().kurtosis() + 3.0)
    mde_real, _bench = required_ann_sharpe(N_TRIALS, n_obs, skew, kurt)
    mde_sym, _bench_s = required_ann_sharpe(N_TRIALS, n_obs, 0.0, 3.0)

    fingerprint = live.price_panel_fingerprint(closes)
    payload = {
        "trial": 14,
        "hypothesis": "T14",
        "synthetic": False,
        "verdict": verdict,
        "config": {
            "train_start": live.LIVE_PRICE_START,
            "horizon": live.HORIZON,
            "refit": "daily, expanding, labels complete through t-horizon",
            "rebalance": "daily decile, quantile 0.1",
            "fills": "next open (weight at close t earns open t+1 → open t+2)",
            "features": books["info"]["feature_cols"],
            "label": "residual 21-day, past-only beta",
            "neutralize": "sector demean of scores, then beta projection of weights",
            "universe": "point-in-time S&P 500, membership from 2010, prices from 2018",
            "model": "ridge alpha 10",
            "cost_bps": COST_BPS,
            "n_trials": N_TRIALS,
            "max_name_frac": lr.MAX_NAME_FRAC,
            "rounding_equity": ROUNDING_EQUITY,
            "ic_scored_on": "pred_raw",
            "mde_ex_ante_net_ann_sharpe": MDE_NET_ANN_SHARPE,
            "mde_ex_ante_n_obs": MDE_N_OBS_DESIGN,
        },
        "mde_realized": {
            "n_obs": n_obs,
            "net_ann_sharpe": float(mde_real),
            "net_ann_sharpe_symmetric": float(mde_sym),
            "skew": skew,
            "kurtosis_raw": kurt,
            "note": "reconciliation beside the result; kill thresholds unchanged",
        },
        "oos_daily": {k: oos[k] for k in oos if k != "config"},
        "cost_decomposition": scored["cost_decomposition"],
        "beta_after_rounding": scored["beta_after_rounding"],
        "replay_info": books["info"],
        "data": {
            **fingerprint,
            "n_members_ever": cov["n_members_ever"],
            "n_with_price_data": cov["n_with_price_data"],
            "n_missing_price_data": cov["n_missing_price_data"],
            "pct_covered": cov["pct_covered"],
            "missing_tickers": cov["missing_tickers"],
            "source": "yfinance adjusted open and close, auto_adjust=True",
        },
    }
    # metrics.summary nests nothing called config; the comprehension is safe
    # even if it doesn't strip. Drop non-scalars that render_report doesn't
    # need and that aren't JSON-friendly (none expected).
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join("writeup"), exist_ok=True)
    json_path = os.path.join(out_dir, "trial14_live_config.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(payload), fh, indent=2)
    csv_path = os.path.join(out_dir, "trial14_live_config_returns.csv")
    _write_returns(csv_path, scored["daily_result"], scored["hold_result"])
    png_path = os.path.join(out_dir, "trial14_live_config_equity.png")
    _write_equity(
        png_path,
        scored["daily_result"]["net"],
        scored["hold_result"]["net"],
        f"Trial 14 live-config replay | {verdict['verdict']} | "
        f"net SR {oos['sharpe_net']:.2f} DSR {oos['dsr']:.2f}",
    )
    report = lr.render_report(payload)
    report_path = os.path.join("writeup", "trial14_live_config.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(f"\n=== Trial 14 verdict: {verdict['verdict']} ===")
    for name, leg in verdict["legs"].items():
        print(f"  {name}: {leg['value']}  pass={leg['pass']}")
    print(f"  turnover daily {scored['cost_decomposition']['daily']['annual_turnover']:.2f}x  "
          f"21d {scored['cost_decomposition']['rebalance_21d']['annual_turnover']:.2f}x  "
          f"extra cost drag {scored['cost_decomposition']['extra_cost_drag']:.2%}/yr")
    rb = scored["beta_after_rounding"]["rounded"]
    print(f"  ex-ante beta after rounding: mean {rb['mean']:.4f}  "
          f"mean|β| {rb['mean_abs']:.4f}  p95|β| {rb['p95_abs']:.4f}")
    print(f"  wrote {json_path}")
    print(f"  wrote {report_path}")
    return payload


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", choices=["sp500", "planted", "noise"], default="sp500")
    ap.add_argument("--hypothesis", default=None)
    ap.add_argument("--n-trials", type=int, default=N_TRIALS)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    if args.data == "sp500":
        if args.hypothesis != "T14":
            sys.exit(
                "real-data run requires --hypothesis T14 "
                "(the pre-registered trial; see writeup/preregistered_hypotheses.md)"
            )
        if args.n_trials != N_TRIALS:
            sys.exit(
                f"--n-trials must be {N_TRIALS} for this trial "
                "(conservative accounting: the deployed config is a new evaluation)"
            )
        from quantlab.registry import require_runnable_registration

        try:
            require_runnable_registration("T14")
        except RuntimeError as exc:
            sys.exit(f"REGISTRATION GATE: {exc}")
        print("[registration] T14 is PROPOSED. This run spends trial #14 at N=14.")
        run_real(args.out)
        return 0

    run_synthetic(args.data, args.seed, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
