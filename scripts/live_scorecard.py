#!/usr/bin/env python3
"""Render the live-experiment scorecard SVG from results/live/ logs.

Reads ONLY what the nightly cycle already committed (summary_*.json,
predictions_*.csv) — no network, no recomputation, stdlib only — and renders
a self-contained SVG for embedding in a profile/README:

  * paper-equity curve across every logged cycle (the number is what it is;
    the point of the experiment is the discipline, not the brag),
  * cycle continuity + last-cycle stats (names, orders, revision monitor),
  * the control-arm reminder: predictions are logged before any order
    exists, with a shadow 12-1 momentum arm for live-vs-backtest attribution.

Run:  python scripts/live_scorecard.py            # writes live-scorecard.svg
The scorecard workflow (.github/workflows/scorecard.yml) runs this nightly
after the trading job and commits the SVG when it changes.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re

LIVE_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "live")
OUT = os.path.join(os.path.dirname(__file__), "..", "live-scorecard.svg")

BG, PANEL, LINE = "#0d1117", "#161b22", "#30363d"
TXT, DIM, KEY = "#e6edf3", "#8b949e", "#58a6ff"
GRN, RED, GOLD = "#3fb950", "#f85149", "#ffd757"
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"

W, H = 820, 310

# A live experiment that has quietly stopped is worse than no live experiment:
# the scorecard keeps rendering, the workflow keeps going green, and the
# artifact a reader sees still says "nightly". That is exactly what happened
# between 2026-08-11 and 2026-09-15 -- 25 weekday cycles lost to a crash in the
# universe scrape, with every signal in the repo reporting success. So the
# renderer measures its own freshness and says so, on the artifact itself.
STALE_AFTER_WEEKDAYS = 3


def load_cycles() -> list[dict]:
    cycles = []
    for path in sorted(glob.glob(os.path.join(LIVE_DIR, "summary_*.json"))):
        m = re.search(r"summary_(\d{4}-\d{2}-\d{2})\.json$", path)
        if not m:
            continue
        try:
            with open(path) as fh:
                d = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        d.setdefault("asof", m.group(1))
        if isinstance(d.get("equity"), (int, float)):
            cycles.append(d)
    return cycles


def fmt_money(x: float) -> str:
    return f"${x:,.0f}"


def weekdays_between(start: dt.date, end: dt.date) -> int:
    """Weekdays strictly after ``start`` through ``end`` (holidays not modelled).

    Deliberately crude: it over-counts around market holidays, which is the
    safe direction for a staleness alarm -- a false "check this" costs a
    glance, a missed stall costs the dataset.
    """
    days, cur = 0, start
    while cur < end:
        cur += dt.timedelta(days=1)
        if cur.weekday() < 5:
            days += 1
    return days


def staleness(cycles: list[dict], today: dt.date | None = None) -> tuple[int, str]:
    """(weekdays since the last logged cycle, that cycle's date)."""
    last = cycles[-1]["asof"]
    today = today or dt.date.today()
    return weekdays_between(dt.date.fromisoformat(last), today), last


def build_svg(cycles: list[dict], today: dt.date | None = None) -> str:
    n = len(cycles)
    eq = [c["equity"] for c in cycles]
    dates = [c["asof"] for c in cycles]
    start_cap = 1_000_000.0
    last = cycles[-1]
    ret_pct = (eq[-1] / start_cap - 1.0) * 100.0
    ret_col = GRN if ret_pct >= 0 else RED
    n_rev = (last.get("revisions") or {}).get("n_return_cells_changed")

    # ---- equity curve geometry -------------------------------------------
    gx, gy, gw, gh = 30, 100, 560, 150
    lo = min(min(eq), start_cap)
    hi = max(max(eq), start_cap)
    pad = (hi - lo) * 0.10 or 1.0
    lo, hi = lo - pad, hi + pad

    def X(i: float) -> float:
        return gx + (i / max(n - 1, 1)) * gw

    def Y(v: float) -> float:
        return gy + gh - ((v - lo) / (hi - lo)) * gh

    pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(eq))
    area = f"{gx},{Y(eq[0]):.1f} {pts} {X(n - 1):.1f},{gy + gh} {gx},{gy + gh}"
    y_start = Y(start_cap)

    # sparse x labels: first, middle, last
    xlab = ""
    for i in (0, n // 2, n - 1):
        anchor = "start" if i == 0 else ("end" if i == n - 1 else "middle")
        xlab += (f'<text x="{X(i):.0f}" y="{gy + gh + 16}" font-family="{MONO}" '
                 f'font-size="10" fill="{DIM}" text-anchor="{anchor}">{dates[i]}</text>')

    # ---- right-hand stat tiles -------------------------------------------
    tiles = [
        ("CYCLES LOGGED", f"{n}", TXT),
        ("PAPER EQUITY", fmt_money(eq[-1]), TXT),
        ("SINCE START", f"{ret_pct:+.2f}%", ret_col),
        ("BOOK / ORDERS", f'{last.get("n_names", "—")} names · {last.get("orders_sent", "—")} orders', TXT),
        ("ORDER FAILURES", f'{last.get("n_failed", 0)}', GRN if not last.get("n_failed") else RED),
        ("RETURN-CELL REVISIONS", "—" if n_rev is None else f"{n_rev}", TXT),
    ]
    # Freshness strip: stated on the artifact, not just in a workflow log.
    stale_days, last_date = staleness(cycles, today)
    hdr_col = DIM
    stale_svg = ""
    if stale_days > STALE_AFTER_WEEKDAYS:
        hdr_col = RED
        stale_svg = (
            f'<rect x="24" y="70" width="{W - 48}" height="16" rx="4" fill="{RED}" opacity="0.18"/>'
            f'<text x="30" y="82" font-family="{MONO}" font-size="10" font-weight="700" fill="{RED}">'
            f'STALLED — no cycle logged since {last_date} ({stale_days} weekdays). '
            f'The curve below ends there; it is not current.</text>'
        )

    tx, ty, tw, th, gap = 610, 94, 186, 27, 4
    tiles_svg = ""
    for i, (lab, val, col) in enumerate(tiles):
        y0 = ty + i * (th + gap)
        tiles_svg += (
            f'<rect x="{tx}" y="{y0}" width="{tw}" height="{th}" rx="5" fill="{PANEL}" stroke="{LINE}"/>'
            f'<text x="{tx + 9}" y="{y0 + 11}" font-family="{MONO}" font-size="7.5" fill="{KEY}" letter-spacing="1">{lab}</text>'
            f'<text x="{tx + 9}" y="{y0 + 22}" font-family="{MONO}" font-size="11" fill="{col}">{val}</text>'
        )

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<rect width="{W}" height="{H}" rx="12" fill="{BG}" stroke="{LINE}" stroke-width="2"/>
<text x="24" y="30" font-family="{MONO}" font-size="15" font-weight="700" fill="{TXT}">LIVE PAPER-TRADING EXPERIMENT</text>
<text x="24" y="48" font-family="{MONO}" font-size="10.5" fill="{DIM}">qr-alpha-lab · nightly via GitHub Actions · predictions logged BEFORE any order exists · shadow 12-1 momentum control arm</text>
<text x="24" y="64" font-family="{MONO}" font-size="10.5" fill="{hdr_col}">cycle {n} · {dates[0]} → {dates[-1]} · Alpaca paper account (client refuses non-paper endpoints)</text>
{stale_svg}
<g>
  <rect x="{gx - 8}" y="{gy - 10}" width="{gw + 16}" height="{gh + 34}" rx="8" fill="{PANEL}" stroke="{LINE}"/>
  <line x1="{gx}" y1="{y_start:.1f}" x2="{gx + gw}" y2="{y_start:.1f}" stroke="{DIM}" stroke-width="1" stroke-dasharray="4 4" opacity=".7"/>
  <text x="{gx + gw - 4}" y="{y_start - 5:.1f}" font-family="{MONO}" font-size="9" fill="{DIM}" text-anchor="end">$1,000,000 start</text>
  <polygon points="{area}" fill="{ret_col}" opacity="0.12"/>
  <polyline points="{pts}" fill="none" stroke="{ret_col}" stroke-width="2" stroke-linejoin="round"/>
  <circle cx="{X(n - 1):.1f}" cy="{Y(eq[-1]):.1f}" r="3.5" fill="{ret_col}"/>
  <text x="{X(n - 1) - 8:.1f}" y="{Y(eq[-1]) - 9:.1f}" font-family="{MONO}" font-size="11" font-weight="700" fill="{ret_col}" text-anchor="end">{fmt_money(eq[-1])} ({ret_pct:+.2f}%)</text>
  {xlab}
</g>
{tiles_svg}
<text x="24" y="{H - 12}" font-family="{MONO}" font-size="9.5" fill="{DIM}">honest by construction: every cycle committed to the repo — including this drawdown. live IC vs backtest IC graded at 21-day label maturity.</text>
</svg>'''


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check-stale",
        action="store_true",
        help="render nothing; exit non-zero if the live experiment has stalled "
             "(run as a separate CI step so the stall is an alarm, not a quiet green check)",
    )
    args = ap.parse_args(argv)

    cycles = load_cycles()
    if not cycles:
        print("no live cycles found; nothing to render")
        return 1

    stale_days, last_date = staleness(cycles)
    if args.check_stale:
        if stale_days > STALE_AFTER_WEEKDAYS:
            print(
                f"::error title=Live experiment stalled::no cycle logged since {last_date} "
                f"({stale_days} weekdays). The nightly job is failing or disabled; every "
                f"weekday it stays down is an unbackfillable H5/H7 record lost."
            )
            return 1
        print(f"live cycles current: last {last_date} ({stale_days} weekdays ago)")
        return 0

    svg = build_svg(cycles)
    with open(OUT, "w") as fh:
        fh.write(svg)
    note = f" STALE: last cycle {last_date}, {stale_days} weekdays ago" if stale_days > STALE_AFTER_WEEKDAYS else ""
    print(f"wrote {os.path.relpath(OUT)} ({len(svg)} bytes, {len(cycles)} cycles).{note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
