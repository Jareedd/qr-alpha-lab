"""End-to-end research pipeline run.

Examples:
    python scripts/run_pipeline.py --data planted          # sanity: must find signal
    python scripts/run_pipeline.py --data noise            # sanity: must find nothing
    python scripts/run_pipeline.py --data yfinance         # real data (needs internet)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quantlab import backtest, baselines, features, metrics, models, risk, validation


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        choices=["yfinance", "sp500", "planted", "noise"],
        default="planted",
        help="sp500 = point-in-time S&P 500 membership (survivorship-bias "
        "aware); yfinance = static present-day universe (biased, kept for "
        "comparison).",
    )
    ap.add_argument(
        "--model",
        choices=["ridge", "ridge_cv", "gbr", "mlp"],
        default="ridge",
        help="ridge_cv re-selects alpha per roll via nested walk-forward "
        "(in-sample only -- does not inflate the trial count). mlp is a "
        "small (16,8) net for the model-class ablation.",
    )
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--horizon", type=int, default=21)
    ap.add_argument(
        "--label",
        choices=["raw", "residual"],
        default="raw",
        help="residual: subtract beta_t * forward market return from the "
        "label (past-only betas) -- predict idiosyncratic return, the only "
        "part a dollar-neutral book can harvest.",
    )
    ap.add_argument(
        "--rebalance",
        type=int,
        default=None,
        help="Rebalance every N days (default: the label horizon). Longer "
        "rebalance = less turnover = less cost drag on a weak edge.",
    )
    ap.add_argument(
        "--n-trials",
        type=int,
        default=1,
        help="How many strategy variants you have tried IN TOTAL (be honest). "
        "Used by the Deflated Sharpe Ratio.",
    )
    ap.add_argument(
        "--neutralize",
        choices=["none", "sector", "beta", "both"],
        default="none",
        help="sector: demean predictions within (date, sector); beta: project "
        "rebalance weights to zero ex-ante market beta (rolling 252d, "
        "past-only). Factor exposure usually explains most of a naive "
        "signal -- this asks what is left.",
    )
    ap.add_argument(
        "--delisting-return",
        type=float,
        default=None,
        help="SCENARIO (real-data modes): inject a synthetic final return of "
        "this size into every name whose prices end mid-window, to BOUND the "
        "missing-delisting-return bias (Shumway 1997: -0.30 is the classic "
        "worst case). Same strategy under a data assumption -- not a new "
        "alpha trial. All artifacts get a _dlret tag.",
    )
    ap.add_argument(
        "--capacity",
        action="store_true",
        help="Run a square-root-impact capacity sweep (needs volume data; "
        "real-data modes only). Same strategy under execution scenarios -- "
        "not a new alpha trial.",
    )
    ap.add_argument(
        "--hypothesis",
        default=None,
        metavar="Hn",
        help="Real-data runs only: the pre-registered hypothesis this run "
        "spends a trial on (must exist in writeup/preregistered_hypotheses.md "
        "with status PROPOSED). Law #3, mechanized.",
    )
    ap.add_argument(
        "--reproduce",
        default=None,
        metavar="REASON",
        help="Real-data runs only: declare this run a REPRODUCTION of "
        "already-logged work (e.g. 'trial #5 artifacts for capacity sweep') "
        "-- not a new trial, no N increment.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=7,
        help="RNG seed for the synthetic panel generators (law #8: every "
        "result must be regenerable from a config + seed). Ignored for real "
        "data, where the 'seed' is the vendor snapshot. Recorded in the "
        "metrics JSON so an artifact names the config that produced it.",
    )
    ap.add_argument("--out", default="results")
    ap.add_argument(
        "--fail-if-dsr-below",
        type=float,
        default=None,
        help="Exit non-zero if DSR < this (CI gate: planted data must recover).",
    )
    ap.add_argument(
        "--fail-if-dsr-above",
        type=float,
        default=None,
        help="Exit non-zero if DSR > this (CI gate: noise data finding alpha = leakage).",
    )
    args = ap.parse_args()

    # Law #3, mechanized: a real-data run is either a registered new trial
    # or a declared reproduction -- never an unaccounted variant. Synthetic
    # modes are exempt (synthetic data is free by law).
    if args.data in ("yfinance", "sp500"):
        if bool(args.hypothesis) == bool(args.reproduce):
            sys.exit(
                "real-data runs require exactly ONE of:\n"
                "  --hypothesis Hn      (a registered, PROPOSED hypothesis: "
                "this run spends a trial; bump --n-trials and log it)\n"
                "  --reproduce REASON   (re-running already-logged work: no "
                "new trial)\n"
                "Synthetic modes (--data planted/noise) need neither."
            )
        if args.hypothesis:
            from quantlab.registry import require_runnable_registration

            try:
                require_runnable_registration(args.hypothesis)
            except RuntimeError as exc:
                sys.exit(f"REGISTRATION GATE: {exc}")
            print(f"[registration] {args.hypothesis} verified PROPOSED -- "
                  f"this run is a NEW TRIAL at --n-trials {args.n_trials}; "
                  "log it whatever it says.")
        else:
            print(f"[reproduce] declared reproduction: {args.reproduce}")

    member_mask = None
    sectors: dict[str, str] = {}
    if args.data == "yfinance":
        from quantlab.data import load_prices

        prices = load_prices()
        if args.neutralize in ("sector", "both"):
            try:
                from quantlab import universe as univ

                cur, _ = univ.fetch_sp500_tables()
                sectors = univ.sector_map(cur, list(prices.columns))
            except Exception as exc:  # offline: every name -> UNKNOWN bucket
                print(f"[warn] no sector data ({exc}); sector demean is a no-op")
    elif args.data == "sp500":
        from quantlab import universe as univ
        from quantlab.data import load_prices

        current, changes = univ.fetch_sp500_tables()
        intervals = univ.build_membership_intervals(current, changes, start="2010-01-01")
        members = univ.all_members_in_window(intervals)
        # Prices start a year before membership so features have warm-up
        # history; min_coverage=0 keeps mid-window IPOs and delistings.
        prices = load_prices(members, start="2009-01-01", min_coverage=0.0)
        cov = univ.coverage_report(members, prices)
        member_mask = univ.membership_mask(prices.index, prices.columns, intervals)
        sectors = univ.sector_map(current, list(prices.columns))
        print(
            f"[universe] point-in-time S&P 500: {cov['n_members_ever']} members "
            f"ever in window, {cov['n_with_price_data']} with price data "
            f"({cov['pct_covered']:.1%}); {cov['n_missing_price_data']} dead names "
            f"unpriceable -> residual bias, see coverage JSON"
        )
    else:
        from quantlab.synthetic import make_panel

        prices = make_panel(mode=args.data, seed=args.seed)
        sectors = prices.attrs.get("sectors", {})
    print(f"[data] {prices.shape[1]} assets x {prices.shape[0]} days ({args.data})")

    if args.delisting_return is not None:
        if args.data not in ("yfinance", "sp500"):
            sys.exit("--delisting-return is a real-data scenario (yfinance/sp500)")
        from quantlab.synthetic import inject_delisting_returns

        prices = inject_delisting_returns(prices, args.delisting_return)
        print(
            f"[scenario] SYNTHETIC delisting return {args.delisting_return:+.0%} "
            f"injected into {prices.attrs['delist_injected']} names whose "
            "series end mid-window -- bias BOUND, not data; artifacts tagged _dlret"
        )

    # Features/labels are masked to index members BEFORE z-scoring, so
    # non-members never shift the cross-sectional stats the model sees;
    # raw features still use each name's full (public) price history.
    feats = features.build_features(prices, member_mask=member_mask)
    labels = features.build_labels(
        prices,
        horizon=args.horizon,
        residualize=(args.label == "residual"),
        member_mask=member_mask,
    )
    panel = features.stack_panel(feats, labels)
    if member_mask is not None:
        # Defensive second gate: a (date, ticker) row survives only if the
        # name was in the index on that date.
        in_index = member_mask.stack()
        in_index.index.names = ["date", "ticker"]
        panel = panel[in_index.reindex(panel.index, fill_value=False)]
    print(f"[features] panel: {len(panel):,} rows, {len(feats)} features, "
          f"label={args.label}")

    splitter = validation.WalkForwardSplitter(embargo_days=args.horizon)
    preds = models.walk_forward_predict(panel, splitter, model_name=args.model)
    ic = models.information_coefficient(preds, panel)
    # Overlapping h-day labels autocorrelate daily ICs; the naive t assumes
    # independence. Quote the Newey-West one (lags = horizon).
    ic_t_nw = metrics.newey_west_tstat(ic, lags=args.horizon)
    print(
        f"[model] {args.model}: mean rank IC = {ic.mean():.4f} "
        f"(t_naive={ic.mean()/ic.sem():.2f}, t_NW={ic_t_nw:.2f})"
    )

    # Feature-stability diagnostic: univariate per-window ICs (overfitting tell).
    fw_ics = models.feature_window_ics(panel, splitter)
    stability = {
        f: f"{fw_ics[f].mean():+.4f} (sign-consistency "
        f"{(np.sign(fw_ics[f]) == np.sign(fw_ics[f].mean())).mean():.0%})"
        for f in fw_ics.columns
        if f != "test_start"
    }
    print("[features] per-window IC stability:")
    for f, s in stability.items():
        print(f"    {f:>14}: {s}")

    if args.neutralize in ("sector", "both") and sectors:
        preds = risk.neutralize_predictions_by_sector(preds, sectors)
        print(f"[risk] sector-demeaned predictions ({len(set(sectors.values()))} sectors)")

    rebalance_every = args.rebalance or args.horizon
    weights = backtest.predictions_to_weights(preds, rebalance_every=rebalance_every)

    # Market proxy + rolling betas (past-only) -- used by beta neutralization
    # and by the risk report either way, so exposure is always measured.
    mkt = baselines.equal_weight_returns(prices, member_mask=member_mask)
    asset_rets = prices.pct_change(fill_method=None)
    betas = risk.rolling_beta(asset_rets, mkt)
    if args.neutralize in ("beta", "both"):
        weights = risk.beta_neutralize_weights(weights, betas)
        print("[risk] weights projected to zero ex-ante beta at each rebalance")

    result = backtest.run_backtest(weights, prices, cost_bps=args.cost_bps)
    stats = metrics.summary(
        result["net"], result["gross"], result["annual_turnover"], n_trials=args.n_trials
    )
    stats["mean_rank_ic"] = float(ic.mean())
    stats["ic_tstat_newey_west"] = float(ic_t_nw)
    if args.delisting_return is not None:
        # Loud scenario stamp (law #7): these numbers contain synthetic prints.
        stats["scenario_delisting_return"] = float(args.delisting_return)
        stats["scenario_names_injected"] = int(prices.attrs["delist_injected"])
        stats["scenario_note"] = (
            "SYNTHETIC delisting returns injected to bound the missing-"
            "delisting-return bias -- not a tradable result"
        )

    # Baselines (law #5): same OOS dates, same backtester, same costs.
    mom_w = baselines.momentum_baseline_weights(
        feats, preds.index, rebalance_every=rebalance_every
    )
    mom_res = backtest.run_backtest(mom_w, prices, cost_bps=args.cost_bps)
    ew = mkt.loc[result["net"].index[0] :]
    stats["baseline_mom_sharpe_net"] = metrics.sharpe(mom_res["net"])
    stats["baseline_ew_sharpe"] = metrics.sharpe(ew)
    stats["beats_mom_baseline"] = bool(stats["sharpe_net"] > stats["baseline_mom_sharpe_net"])
    # Baseline IC on the same OOS (date, ticker) pairs: the backtest-side
    # anchor for the live control arm (live.py shadow-logs this feature).
    mom_sig = feats["mom_12_1"].stack()
    mom_sig.index.names = ["date", "ticker"]
    mom_ic = models.information_coefficient(mom_sig.reindex(preds.index), panel)
    stats["baseline_mom_ic"] = float(mom_ic.mean())
    stats["baseline_mom_ic_tstat_nw"] = float(
        metrics.newey_west_tstat(mom_ic, lags=args.horizon)
    )

    rr = risk.risk_report(result["net"], mkt, result["daily_weights"], betas, sectors)
    stats.update({f"risk_{k}": v for k, v in rr.items()})

    capacity = None
    if args.capacity:
        if args.data not in ("yfinance", "sp500"):
            print("[capacity] skipped: needs real volume data")
        else:
            from quantlab import impact
            from quantlab.data import load_volumes

            vol_tickers = list(prices.columns)
            volumes = load_volumes(
                vol_tickers, start=str(prices.index[0].date())
            )
            adv = impact.dollar_adv(prices, volumes)
            free = backtest.run_backtest(weights, prices, cost_bps=0.0)
            capacity = impact.capacity_curve(
                weights, prices, adv, free["gross"], spread_bps=args.cost_bps
            )
            cap_dead = impact.capacity_at_zero(capacity)
            print(f"[capacity] adv coverage {capacity.attrs['adv_coverage']:.1%}; "
                  f"net SR by AUM:")
            for aum, row in capacity.iterrows():
                print(f"    ${aum/1e6:>7,.0f}M: SR {row['sharpe_net']:+.2f} "
                      f"(cost drag {row['ann_cost_drag']:.2%}/yr)")
            print(f"    edge dies at: "
                  f"{'never within sweep' if cap_dead is None else f'${cap_dead/1e6:,.0f}M'}")

    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.data}_{args.model}"
    if args.neutralize != "none":
        tag += f"_{args.neutralize}"
    if args.label != "raw":
        tag += "_residlabel"
    if args.horizon != 21:
        tag += f"_h{args.horizon}"
    if args.rebalance and args.rebalance != args.horizon:
        tag += f"_r{args.rebalance}"
    if args.delisting_return is not None:
        tag += f"_dlret{round(args.delisting_return * 100):+d}"
    # Law #8: the artifact must name the config that produced it. Without
    # this the seed lived only in a default argument and no committed result
    # could be tied back to the run that made it.
    stats["config"] = {
        "data": args.data,
        "model": args.model,
        "seed": args.seed if args.data in ("planted", "noise", "planted_regime") else None,
        "horizon": args.horizon,
        "rebalance": args.rebalance or args.horizon,
        "label": args.label,
        "neutralize": args.neutralize,
        "cost_bps": args.cost_bps,
        "n_trials": args.n_trials,
    }
    with open(os.path.join(args.out, f"metrics_{tag}.json"), "w") as f:
        json.dump(stats, f, indent=2)
    if args.data == "sp500":
        with open(os.path.join(args.out, "sp500_pit_coverage.json"), "w") as f:
            json.dump(cov, f, indent=2)
    fw_ics.to_csv(os.path.join(args.out, f"feature_ics_{tag}.csv"))
    if capacity is not None:
        cap_payload = {
            "adv_coverage": capacity.attrs["adv_coverage"],
            "curve": capacity.reset_index().to_dict(orient="records"),
            "note": "square-root impact, k=1.0 (order of magnitude); "
            "one-day execution; spread included",
        }
        with open(os.path.join(args.out, f"capacity_{tag}.json"), "w") as f:
            json.dump(cap_payload, f, indent=2)

    fig, ax = plt.subplots(figsize=(10, 5))
    (1 + result["net"]).cumprod().plot(ax=ax, label=f"net ({args.cost_bps}bps)")
    (1 + result["gross"]).cumprod().plot(ax=ax, label="gross", alpha=0.6)
    ax.set_title(f"Long-short equity curve -- {tag} | net SR={stats['sharpe_net']:.2f} "
                 f"DSR={stats['dsr']:.2f}")
    ax.legend()
    ax.set_ylabel("growth of $1")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f"equity_{tag}.png"), dpi=120)

    print("\n=== Out-of-sample results (all predictions are walk-forward) ===")
    for k, v in stats.items():
        print(f"  {k:>18}: {v:.4f}" if isinstance(v, float) else f"  {k:>18}: {v}")
    verdict = (
        "PASS: signal recovered" if stats["dsr"] > 0.95
        else "No statistically defensible signal (as expected for noise / weak edges)"
    )
    print(f"\n  DSR verdict: {verdict}")
    print(
        f"  Baselines (same OOS window, net of costs): "
        f"12-1 momentum SR={stats['baseline_mom_sharpe_net']:.2f}, "
        f"equal-weight SR={stats['baseline_ew_sharpe']:.2f} -> model "
        f"{'beats' if stats['beats_mom_baseline'] else 'DOES NOT beat'} momentum baseline"
    )
    print(
        f"  Risk ({args.neutralize}): mkt corr={rr['market_corr']:.2f}, "
        f"realized beta={rr['realized_beta_mean']:.2f} "
        f"(p95 |beta|={rr['realized_beta_p95_abs']:.2f}), "
        f"sector net mean|max abs={rr['sector_net_mean_abs']:.3f}|{rr['sector_net_max_abs']:.3f}"
    )

    if args.fail_if_dsr_below is not None and stats["dsr"] < args.fail_if_dsr_below:
        sys.exit(
            f"FALSIFICATION GATE FAILED: DSR {stats['dsr']:.4f} < "
            f"{args.fail_if_dsr_below} -- pipeline failed to recover a planted signal."
        )
    if args.fail_if_dsr_above is not None and stats["dsr"] > args.fail_if_dsr_above:
        sys.exit(
            f"FALSIFICATION GATE FAILED: DSR {stats['dsr']:.4f} > "
            f"{args.fail_if_dsr_above} -- pipeline 'found alpha' in noise: hunt the leak."
        )


if __name__ == "__main__":
    main()
