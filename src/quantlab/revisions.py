"""Data-revision monitor: how much does the vendor rewrite the past?

Every live cycle downloads a fresh price history into its own dated cache
directory (``data_cache/live_YYYY-MM-DD``). Two snapshots taken on different
days claim to describe the SAME past — any overlapping cell where they
disagree is a retroactive data revision (dividend/split re-adjustment, a
correction, or a symbol-level rewrite). Free-data backtests silently assume
this never happens; here we measure it instead.

Why it matters (and why interviewers care): "point-in-time" is usually
discussed as a *universe* property (survivorship bias), but the data values
themselves are also not point-in-time — yfinance re-adjusts whole histories
when a dividend lands. A backtest run today and the live model trained
yesterday literally saw different versions of 2020. The distinction that
makes the measurement useful:

- **price-level revisions** are usually harmless: a constant re-scaling of a
  full history (new dividend adjustment) changes every price but almost no
  *returns*, and everything downstream of features consumes returns.
- **return revisions** are the dangerous kind: they change features, labels
  and marks. They cluster at adjustment splice points.

So every comparison reports both, separately.

Read-only by design: nothing here feeds back into data loading or the
strategy. It observes drift; it never "fixes" it (research law #7 — we do
not rewrite market data, including unwinding the vendor's rewrites).

Assumptions stated:
- Snapshots are adjusted closes from the same vendor; comparisons across
  vendors would conflate revision with methodology.
- Cells present in one snapshot and absent in the other ("appeared"/
  "vanished" history) are counted but not treated as numeric changes.
- TWO thresholds, calibrated by the first real measurement (2026-06-12 →
  -13 snapshots): the vendor re-SERVES history with ~1e-7 relative float
  wobble on a majority of cells (51% of price cells and 61% of return
  cells flagged at a 1e-9 tolerance, magnitudes ~1e-7 — the same drift
  the capacity re-run measured), while REAL adjustments sit at 1e-3 and
  above (dividend re-scalings ~0.3–1.1%, one split-factor rewrite at 90%
  on day one). So: changes are counted above REL_TOL = 1e-6 (above the
  serving-noise floor, three orders below real events), and cells in
  (NOISE_FLOOR, REL_TOL] are counted separately as the noise band — the
  serving-noise phenomenon is itself worth measuring, not hiding.
"""

from __future__ import annotations

import datetime as dt
import glob
import os
import re

import numpy as np
import pandas as pd

_LIVE_DIR_RE = re.compile(r"live_(\d{4}-\d{2}-\d{2})$")

# Above the vendor's ~1e-7 serving-noise floor; far below real adjustments.
REL_TOL = 1e-6
# Anything above this but below REL_TOL is counted as the noise band.
NOISE_FLOOR = 1e-9


def find_price_snapshot(cache_dir: str) -> str | None:
    """Path of the (single) prices parquet inside one cycle's cache dir."""
    hits = sorted(glob.glob(os.path.join(cache_dir, "prices_*.parquet")))
    return hits[-1] if hits else None


def list_snapshots(cache_root: str = "data_cache") -> dict[str, str]:
    """tag (YYYY-MM-DD) -> prices parquet path, for every live cycle cache."""
    out: dict[str, str] = {}
    if not os.path.isdir(cache_root):
        return out
    for name in sorted(os.listdir(cache_root)):
        m = _LIVE_DIR_RE.search(name)
        if not m:
            continue
        snap = find_price_snapshot(os.path.join(cache_root, name))
        if snap:
            out[m.group(1)] = snap
    return out


def latest_snapshot_before(cache_root: str, tag: str) -> tuple[str, str] | None:
    """(tag, path) of the most recent snapshot strictly older than ``tag``."""
    older = {t: p for t, p in list_snapshots(cache_root).items() if t < tag}
    if not older:
        return None
    t = max(older)
    return t, older[t]


def compare_price_snapshots(
    old: pd.DataFrame, new: pd.DataFrame, rel_tol: float = REL_TOL
) -> dict:
    """Quantify how the shared past differs between two price snapshots.

    Only the intersection of (dates x tickers) is compared — the new
    snapshot's extra trading day is new information, not a revision.
    Returns a JSON-ready dict; see module docstring for the price-level vs
    return distinction.
    """
    dates = old.index.intersection(new.index)
    cols = old.columns.intersection(new.columns)
    o = old.loc[dates, cols]
    n = new.loc[dates, cols]

    both = o.notna() & n.notna()
    rel = (n / o - 1.0).where(both)
    changed = rel.abs() > rel_tol
    price_noise = (rel.abs() > NOISE_FLOOR) & ~changed

    # Returns on the SHARED grid: the quantity features/labels actually eat.
    o_ret = o.pct_change(fill_method=None)
    n_ret = n.pct_change(fill_method=None)
    ret_both = o_ret.notna() & n_ret.notna()
    ret_diff = (n_ret - o_ret).abs().where(ret_both)
    ret_changed = ret_diff > rel_tol
    ret_noise = (ret_diff > NOISE_FLOOR) & ~ret_changed

    per_ticker = changed.sum()
    affected = per_ticker[per_ticker > 0].sort_values(ascending=False)
    top = [
        {
            "ticker": str(t),
            "n_price_cells": int(per_ticker[t]),
            "max_abs_rel_change": float(rel[t].abs().max()),
            "n_return_cells": int(ret_changed[t].sum()),
        }
        for t in affected.index[:10]
    ]

    n_cells = int(both.sum().sum())
    n_changed = int(changed.sum().sum())
    n_ret_cells = int(ret_both.sum().sum())
    n_ret_changed = int(ret_changed.sum().sum())
    return {
        "n_shared_dates": int(len(dates)),
        "n_shared_tickers": int(len(cols)),
        "n_cells_compared": n_cells,
        "n_price_cells_changed": n_changed,
        "frac_price_cells_changed": float(n_changed / n_cells) if n_cells else 0.0,
        "max_abs_rel_price_change": float(rel.abs().max().max()) if n_changed else 0.0,
        "n_return_cells_compared": n_ret_cells,
        "n_return_cells_changed": n_ret_changed,
        "frac_return_cells_changed": (
            float(n_ret_changed / n_ret_cells) if n_ret_cells else 0.0
        ),
        "max_abs_return_change": (
            float((n_ret - o_ret).abs().where(ret_both).max().max())
            if n_ret_changed
            else 0.0
        ),
        "n_tickers_affected": int((per_ticker > 0).sum()),
        # The vendor's float-serving wobble (~1e-7): measured, not hidden,
        # but never mistaken for a revision.
        "n_price_cells_noise_band": int(price_noise.sum().sum()),
        "n_return_cells_noise_band": int(ret_noise.sum().sum()),
        "n_cells_appeared": int((n.notna() & o.isna()).sum().sum()),
        "n_cells_vanished": int((o.notna() & n.isna()).sum().sum()),
        "top_affected_tickers": top,
    }


def revision_table(snapshots: dict[str, str]) -> pd.DataFrame:
    """One row per consecutive snapshot pair (sorted by tag), full stats.

    ``top_affected_tickers`` is dropped from the table (kept in the per-pair
    dicts) so the result is a flat, plottable frame.
    """
    tags = sorted(snapshots)
    rows = []
    for prev, cur in zip(tags, tags[1:]):
        stats = compare_price_snapshots(
            pd.read_parquet(snapshots[prev]), pd.read_parquet(snapshots[cur])
        )
        stats.pop("top_affected_tickers")
        rows.append({"from": prev, "to": cur, **snapshot_gap(prev, cur), **stats})
    return pd.DataFrame(rows)


def snapshot_gap(prior_tag: str, today_tag: str) -> dict:
    """Calendar and weekday distance between two snapshot tags.

    Recorded on EVERY comparison because the H5 series is only interpretable
    if each record says how much time it spans. A consecutive-cycle record
    covers one trading day of vendor rewriting; the record that follows an
    outage covers however long the outage lasted, and the two are not the
    same observation. The live job was down 2026-08-11 → 2026-09-15, so the
    first record after it spans 26 weekdays -- an outlier by construction,
    not by vendor behaviour. Stage 2 must be able to see that from the record
    alone rather than inferring it from a gap in filenames.
    """
    prior_d = dt.date.fromisoformat(prior_tag)
    today_d = dt.date.fromisoformat(today_tag)
    cal = (today_d - prior_d).days
    wd, cur = 0, prior_d
    while cur < today_d:
        cur += dt.timedelta(days=1)
        if cur.weekday() < 5:
            wd += 1
    return {
        "gap_calendar_days": cal,
        "gap_weekdays": wd,
        "is_consecutive_cycle": wd <= 1,
    }


def snapshot_revision_summary(
    cache_root: str, today_tag: str, today_prices: pd.DataFrame
) -> dict | None:
    """Compare today's freshly downloaded panel against the most recent prior
    snapshot. Returns None when there is no prior snapshot to compare against
    (first cycle on a machine). Used by the live cycle; failures there are
    swallowed by the caller — measurement must never block trading."""
    prior = latest_snapshot_before(cache_root, today_tag)
    if prior is None:
        return None
    prior_tag, prior_path = prior
    stats = compare_price_snapshots(pd.read_parquet(prior_path), today_prices)
    return {"compared_to": prior_tag, **snapshot_gap(prior_tag, today_tag), **stats}
