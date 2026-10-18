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
  on day one). Price changes are counted when the relative move exceeds
  REL_TOL = 1e-6. Return changes use the same relative object on the
  gross return, |(1+r_new)/(1+r_old) - 1|, with tolerance 2*REL_TOL.
  A return spans two prices, and each may move by just under REL_TOL
  without either price cell counting; absolute |Δr| > REL_TOL (the rule
  before 2026-10-18) flagged that serving noise as a return revision.
  Cells in (NOISE_FLOOR, tolerance] are the noise band — the
  serving-noise phenomenon is itself worth measuring, not hiding.

Which file is "the" snapshot is explicit. The live cycle writes
``live_panel.json`` naming the parquet it actually loaded. Before that
marker existed, ``find_price_snapshot`` took ``sorted(prices_*.parquet)[-1]``.
From 2026-08-10 the monitoring report wrote a second parquet into the
same ``data_cache/live_*`` directory, and whichever digest sorted last
became the comparison — the report's shorter panel, not the live panel.
Those committed revision JSONs are a different object from the records
before 2026-08-10 and are left as written. New records name the file
and the selection rule (``explicit``, ``sole_parquet``, or the
``live_start_fallback`` used only when an old directory has two files
and no marker).
"""

from __future__ import annotations

import datetime as dt
import glob
import json
import os
import re

import numpy as np
import pandas as pd

_LIVE_DIR_RE = re.compile(r"live_(\d{4}-\d{2}-\d{2})$")

# Above the vendor's ~1e-7 serving-noise floor; far below real adjustments.
REL_TOL = 1e-6
# Anything above this but below the change tolerance is the noise band.
NOISE_FLOOR = 1e-9
# A return is a ratio of two prices. Tolerance is two price-tolerances,
# so a pair of sub-threshold price wiggles cannot count as a revision.
RETURN_REL_FACTOR = 2.0

# The live cycle's cache filename contains this start date (see live.py).
# The report panel uses a much shorter window, so an unmarked directory
# that holds both files can still be told apart without sorting names.
LIVE_PANEL_MARKER = "live_panel.json"
LIVE_PANEL_START_TOKEN = "_2018-01-01_"


class AmbiguousPriceSnapshot(RuntimeError):
    """More than one prices parquet and no way to name the live panel.

    Sorting filenames and taking the last one is how the monitor started
    comparing the report panel on 2026-08-10. Refusing is the point.
    """


def write_live_panel_marker(cache_dir: str, parquet_path: str) -> str:
    """Name the parquet the live cycle actually loaded.

    The marker is the comparison file. A later reader does not have to
    guess among whatever else landed in the directory.
    """
    os.makedirs(cache_dir, exist_ok=True)
    marker = os.path.join(cache_dir, LIVE_PANEL_MARKER)
    payload = {"file": os.path.basename(parquet_path)}
    tmp = f"{marker}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(payload, fh)
    os.replace(tmp, marker)
    return marker


def find_price_snapshot(cache_dir: str) -> str | None:
    """Path of the live cycle's prices parquet inside one cache dir.

    Preference order, and nothing else:

    1. ``live_panel.json`` names the file (explicit).
    2. Exactly one ``prices_*.parquet`` (older cycles, before the marker).
    3. Several parquets and exactly one whose name contains the live
       start date ``2018-01-01`` (directories already polluted by the
       report before it moved to its own folder).

    Two unmarked files that both look like the live panel raise
    ``AmbiguousPriceSnapshot`` instead of picking a sort order.
    """
    marker = os.path.join(cache_dir, LIVE_PANEL_MARKER)
    if os.path.isfile(marker):
        with open(marker) as fh:
            name = json.load(fh)["file"]
        path = os.path.join(cache_dir, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{marker} names {name}, which is not in {cache_dir}"
            )
        return path
    hits = sorted(glob.glob(os.path.join(cache_dir, "prices_*.parquet")))
    if not hits:
        return None
    if len(hits) == 1:
        return hits[0]
    anchored = [h for h in hits if LIVE_PANEL_START_TOKEN in os.path.basename(h)]
    if len(anchored) == 1:
        return anchored[0]
    raise AmbiguousPriceSnapshot(
        f"{cache_dir} has {len(hits)} price files and no {LIVE_PANEL_MARKER}; "
        "refusing to pick whichever name sorts last. Files: "
        + ", ".join(os.path.basename(h) for h in hits)
    )


def snapshot_selection(cache_dir: str) -> str:
    """How ``find_price_snapshot`` would choose, for the revision record."""
    if os.path.isfile(os.path.join(cache_dir, LIVE_PANEL_MARKER)):
        return "explicit"
    hits = glob.glob(os.path.join(cache_dir, "prices_*.parquet"))
    if len(hits) <= 1:
        return "sole_parquet"
    return "live_start_fallback"


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
    # Relative change in the gross return, same object as |p_new/p_old - 1|.
    # Tolerance is 2*REL_TOL: each of the two prices may sit just under the
    # price threshold. Absolute |Δr| > REL_TOL counted that as a revision.
    denom = (1.0 + o_ret).where(ret_both)
    safe = ret_both & denom.notna() & (denom != 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ret_rel = ((1.0 + n_ret) / denom - 1.0).abs().where(safe)
    ret_tol = RETURN_REL_FACTOR * rel_tol
    ret_changed = ret_rel > ret_tol
    ret_noise = (ret_rel > NOISE_FLOOR) & ~ret_changed

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
    same observation. The live job logged no cycle from 2026-08-11 through
    2026-09-25 (34 weekdays, 33 NYSE sessions; the first cycle back is
    2026-09-28). The record that spans that gap is an outlier by
    construction, not by vendor behaviour. Older notes that say the outage
    ended 2026-09-15 counted only the first 26 weekdays of it. Stage 2
    must be able to see the span from the record alone rather than
    inferring it from a gap in filenames.

    ``gap_weekdays`` is the distance between the snapshot tags. Those tags
    are the New York session dates of the cache directories. ``prior_bar``
    and ``today_bar`` on the revision record are the panels' last indexes,
    which is what to use when a tag and a bar disagree.
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
    prior_prices = pd.read_parquet(prior_path)
    stats = compare_price_snapshots(prior_prices, today_prices)
    prior_bar = (
        str(pd.Timestamp(prior_prices.index[-1]).date()) if len(prior_prices) else None
    )
    today_bar = (
        str(pd.Timestamp(today_prices.index[-1]).date()) if len(today_prices) else None
    )
    bar_gap = (
        snapshot_gap(prior_bar, today_bar)["gap_weekdays"]
        if prior_bar and today_bar
        else None
    )
    return {
        "compared_to": prior_tag,
        **snapshot_gap(prior_tag, today_tag),
        "prior_bar": prior_bar,
        "today_bar": today_bar,
        "gap_weekdays_bars": bar_gap,
        "comparison_file": os.path.basename(prior_path),
        "comparison_selection": snapshot_selection(os.path.dirname(prior_path)),
        **stats,
    }
