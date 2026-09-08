#!/usr/bin/env python
"""Rebuild the survivorship-bias case-study chart from committed artifacts.

Offline and deterministic: every number is read out of a metrics JSON in
results/ that was written by a logged run. Nothing here recomputes a
backtest, so the FIGURE is reproducible even though the underlying vendor
pull is not -- which is the honest position for a real-data result.

    python scripts/case_study_chart.py

Writes results/case_study_survivorship.png.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Two series, so two categorical hues assigned to the ENTITY (which universe),
# fixed order, never cycled. Validated for CVD separation (worst adjacent
# protan dE 24.7, normal-vision dE 33.6) against a light surface.
BIASED = "#2a78d6"        # slot 1, blue  -- the flattering universe
HONEST = "#eb6834"        # slot 2, orange -- the point-in-time universe
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d8d7d2"
SURFACE = "#fcfcfb"

SOURCES = {
    "biased": "results/metrics_yfinance_ridge.json",
    "honest": "results/metrics_sp500_ridge.json",
}

# One panel per metric. Deliberately NOT one axis with four bars: net Sharpe,
# rank IC and a t-stat live on unrelated scales, and forcing them onto a
# shared axis would be a chart that lies about magnitude.
PANELS = [
    ("sharpe_net", "Net Sharpe", "{:+.2f}"),
    ("mean_rank_ic", "Mean rank IC (OOS)", "{:+.3f}"),
    ("ic_tstat_newey_west", "IC t-stat (Newey-West)", "{:+.2f}"),
    ("annual_turnover", "Turnover (x/yr)", "{:.1f}"),
]


def load(path: str) -> dict:
    full = os.path.join(REPO, path)
    if not os.path.exists(full):
        sys.exit(f"missing artifact {path}: the chart is built only from committed runs")
    with open(full) as fh:
        return json.load(fh)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "results",
                                                  "case_study_survivorship.png"))
    args = ap.parse_args()

    biased, honest = load(SOURCES["biased"]), load(SOURCES["honest"])
    labels = ["Static universe\n(today's members)", "Point-in-time\nS&P 500"]

    fig, axes = plt.subplots(1, len(PANELS), figsize=(11.0, 3.5))
    fig.patch.set_facecolor(SURFACE)

    for ax, (key, title, fmt) in zip(axes, PANELS):
        vals = [biased[key], honest[key]]
        bars = ax.bar([0, 1], vals, width=0.55, color=[BIASED, HONEST],
                      zorder=3, linewidth=0)
        ax.set_facecolor(SURFACE)
        ax.set_title(title, fontsize=10, color=INK, pad=10)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels, fontsize=8, color=INK_2)
        ax.axhline(0, color=INK_2, linewidth=1.0, zorder=4)
        ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="y", labelsize=8, colors=INK_2, length=0)
        ax.tick_params(axis="x", length=0)
        # Direct labels on every bar: only two per panel, so this is selective,
        # not a number on every point. Text wears ink, never the series colour.
        span = max(abs(v) for v in vals) or 1.0
        for bar, v in zip(bars, vals):
            off = 0.06 * span
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + (off if v >= 0 else -off), fmt.format(v),
                    ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=9, color=INK, fontweight="bold")
        lo, hi = min(vals + [0]), max(vals + [0])
        pad = 0.28 * (hi - lo or 1.0)
        # A strictly non-negative quantity (turnover) gets a zero baseline:
        # padding it below zero invents a region the metric cannot occupy.
        ax.set_ylim(lo if lo >= 0 else lo - pad, hi + pad)

    fig.suptitle(
        "The edge was the universe: same model, same features, same costs",
        fontsize=12, color=INK, y=1.0,
    )
    axes[0].set_ylabel("", color=INK_2)
    # Legend is redundant here -- the x-axis names both entities directly on
    # every panel -- so identity is never carried by colour alone.
    fig.text(
        0.5, -0.10,
        "Ridge, 5 price-only features, 21d horizon, 10 bps/side. Left bars: "
        "results/metrics_yfinance_ridge.json (trial #1). Right bars: "
        "results/metrics_sp500_ridge.json (trial #2).\n"
        "Turnover roughly doubles on the point-in-time universe because dead "
        "names enter and leave -- churn the static universe never sees.",
        ha="center", va="top", fontsize=7.5, color=INK_2,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {os.path.relpath(args.out, REPO)}")
    print(f"  static universe : net SR {biased['sharpe_net']:+.4f}  "
          f"IC {biased['mean_rank_ic']:+.4f}  t_NW {biased['ic_tstat_newey_west']:+.2f}")
    print(f"  point-in-time   : net SR {honest['sharpe_net']:+.4f}  "
          f"IC {honest['mean_rank_ic']:+.4f}  t_NW {honest['ic_tstat_newey_west']:+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
