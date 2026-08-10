"""H13 PEAD — cross-source PIT confirmation (FMP est_eps vs Alpha Vantage).

WHY this exists (standalone, not the graded runner). The FMP leakage gate
(`validate_pead_events`) passes the gross red-flag checks, but a passing gate is
NECESSARY, NOT SUFFICIENT: it cannot prove the `epsEstimated` is the genuine
point-in-time (PIT) consensus rather than a quietly backfilled / post-hoc value.
The gold-standard tie-breaker free data allows is a CROSS-SOURCE agreement check:
if two independent vendors (FMP and Alpha Vantage) report the SAME consensus
estimate for the same (ticker, ann_date), it is very unlikely BOTH backfilled to
the realized actual in the same way -> the estimate is the real consensus.

This script is deliberately TIGHT on quota: it reuses the FMP names ALREADY
cached on disk (zero new FMP calls) and fetches Alpha Vantage for exactly those
same names (AV free tier is ~25 req/day), maximizing the (ticker, ann_date)
overlap per call. It spends NO trial and grades nothing — it only produces the
evidence that informs whether a graded trial #14 may proceed.

Output: prints the leakage-gate report (with the cross_source block populated)
plus a richer per-name agreement breakdown, and writes the full comparison to
results/pead_cross_check.json for reproducibility (CLAUDE.md law #8).
"""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from quantlab import fmp_earnings as fmp


def cached_fmp_symbols(cache_dir: str = fmp.CACHE) -> list[str]:
    """Discover the FMP symbols already cached on disk (so we spend NO new FMP
    quota and the AV pull targets exactly the names with FMP history)."""
    out = []
    for p in sorted(glob.glob(os.path.join(cache_dir, "earnings_*.json"))):
        base = os.path.basename(p)
        sym = base[len("earnings_"):-len(".json")]
        if sym:
            out.append(sym)
    return out


def load_cached_fmp_events(symbols: list[str]) -> pd.DataFrame:
    """Tidy FMP events for the cached symbols, parsed from disk (no network)."""
    src = fmp.FMPEarningsSource()        # needs FMP_API_KEY but won't fetch cached
    frames = []
    for s in symbols:
        ev = src.events(s)               # cache hit -> pure parse, no network
        if not ev.empty:
            frames.append(ev)
    if not frames:
        return pd.DataFrame(columns=fmp._TIDY_COLS + ["last_updated"])
    return pd.concat(frames, ignore_index=True).sort_values(
        ["ann_date", "ticker"]).reset_index(drop=True)


def merge_sources(fmp_ev: pd.DataFrame, av_ev: pd.DataFrame) -> pd.DataFrame:
    """Inner-join FMP and AV events on (ticker, ann_date), carrying BOTH the
    estimate AND the actual from each source. The actual is what powers the
    decisive backfill-signature test (FMP est vs AV est vs AV actual)."""
    key = ["ticker", "ann_date"]
    a = fmp_ev[key + ["est_eps", "actual_eps"]].copy()
    b = av_ev[key + ["est_eps", "actual_eps"]].copy()
    a["ann_date"] = pd.to_datetime(a["ann_date"]).dt.normalize()
    b["ann_date"] = pd.to_datetime(b["ann_date"]).dt.normalize()
    merged = a.merge(b, on=key, suffixes=("_fmp", "_av"))
    if merged.empty:
        return merged
    for c in ("est_eps_fmp", "actual_eps_fmp", "est_eps_av", "actual_eps_av"):
        merged[c] = pd.to_numeric(merged[c], errors="coerce")
    return merged.dropna(subset=["est_eps_fmp", "est_eps_av",
                                 "actual_eps_av"]).reset_index(drop=True)


def diagnostics(merged: pd.DataFrame) -> dict:
    """The decisive cross-source diagnostics over the merged overlap.

    Three independent lenses on 'is FMP est a genuine PIT consensus?':
      1. estimate agreement   — |FMP_est - AV_est| / |AV_est|  (level similarity)
      2. backfill signature    — is FMP_est CLOSER to AV_est (good: independent
         PIT consensus) or to AV_actual (bad: FMP backfilled to the realized
         number)?  `frac_closer_to_est` should be HIGH if FMP is PIT.
      3. surprise correlation  — corr(FMP_actual-FMP_est, AV_actual-AV_est). Two
         genuine PIT feeds disagree a little on the estimate but agree strongly on
         the SURPRISE; a backfilled FMP feed has ~zero surprise and kills this corr.
    """
    if merged.empty:
        return {"n_overlap": 0}
    fe = merged["est_eps_fmp"].to_numpy(float)
    fa = merged["actual_eps_fmp"].to_numpy(float)
    ae = merged["est_eps_av"].to_numpy(float)
    aa = merged["actual_eps_av"].to_numpy(float)

    denom = np.where(np.abs(ae) > 0, np.abs(ae), np.nan)
    rel = np.abs(fe - ae) / denom

    d_est = np.abs(fe - ae)        # FMP est -> AV est
    d_act = np.abs(fe - aa)        # FMP est -> AV actual
    # Tie -> not counted as "closer to est"; strict inequality is the PIT vote.
    closer_to_est = d_est < d_act
    # Where AV est != AV actual (a real surprise quarter), the test is informative.
    informative = np.abs(ae - aa) > 1e-9
    frac_closer_est_inf = (float(np.mean(closer_to_est[informative]))
                           if informative.sum() else float("nan"))

    fmp_surp = fa - fe
    av_surp = aa - ae
    ok = np.isfinite(fmp_surp) & np.isfinite(av_surp)
    surp_corr = (float(np.corrcoef(fmp_surp[ok], av_surp[ok])[0, 1])
                 if ok.sum() > 2 else float("nan"))

    return {
        "n_overlap": int(len(merged)),
        "agree_frac_within_1pct": float(np.nanmean(rel < 0.01)),
        "median_rel_diff": float(np.nanmedian(rel)),
        "exact_frac": float(np.mean(fe == ae)),
        "n_informative": int(informative.sum()),
        "frac_fmp_closer_to_av_est": frac_closer_est_inf,
        "median_dist_to_av_est": float(np.nanmedian(d_est[informative])) if informative.sum() else float("nan"),
        "median_dist_to_av_actual": float(np.nanmedian(d_act[informative])) if informative.sum() else float("nan"),
        "surprise_corr": surp_corr,
    }


def per_name_table(merged: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker version of the diagnostics (so a single bad name like TSLA is
    visible rather than averaged away)."""
    rows = []
    for tkr, grp in merged.groupby("ticker"):
        rows.append({"ticker": tkr, **diagnostics(grp)})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("n_overlap", ascending=False)


def main() -> None:
    syms = cached_fmp_symbols()
    if not syms:
        sys.exit("No cached FMP names under data_cache/fmp/ — run an FMP pull "
                 "first (scripts/run_pead.py --source fmp --max-names N).")
    print(f"[fmp] reusing {len(syms)} cached names (no new FMP quota): "
          f"{', '.join(syms)}")

    fmp_ev = load_cached_fmp_events(syms)
    print(f"[fmp] {len(fmp_ev)} events across {fmp_ev['ticker'].nunique()} tickers, "
          f"{fmp_ev['ann_date'].min().date()}..{fmp_ev['ann_date'].max().date()}")

    print(f"[av ] fetching Alpha Vantage EARNINGS for the same {len(syms)} names "
          "(free tier ~25/day; cached per symbol on disk)...")
    av_ev = fmp.fetch_alphavantage_cross_check(syms, max_names=len(syms))
    if av_ev is None or av_ev.empty:
        sys.exit(
            "[av ] NO Alpha Vantage data returned. Either ALPHA_VANTAGE_API_KEY / "
            "ALPHAVANTAGE_API_KEY is missing from .env, or the daily quota (~25 "
            "req/day) is exhausted (AV returns a rate note, which we do NOT cache). "
            "Check the key and retry tomorrow if quota-limited. No trial spent.")
    print(f"[av ] {len(av_ev)} events across {av_ev['ticker'].nunique()} tickers, "
          f"{av_ev['ann_date'].min().date()}..{av_ev['ann_date'].max().date()}")

    # The leakage gate WITH the cross-source block populated.
    report = fmp.validate_pead_events(fmp_ev, cross_check=av_ev)
    print()
    print(fmp.format_validation_report(report))

    # Decisive diagnostics over the merged overlap (carries both actuals).
    merged = merge_sources(fmp_ev, av_ev)
    overall = diagnostics(merged)
    detail = per_name_table(merged)

    print("\n=== per-name cross-source diagnostics (FMP est vs AV est / AV actual) ===")
    if detail.empty:
        print("  (no overlapping (ticker, ann_date) events — inconclusive)")
    else:
        print(f"  {'ticker':<7} {'ovlp':>5} {'agree<1%':>9} {'exact':>6} "
              f"{'closer2est':>11} {'surp_corr':>10}")
        for _, r in detail.iterrows():
            print(f"  {r['ticker']:<7} {int(r['n_overlap']):>5} "
                  f"{r['agree_frac_within_1pct']:>8.0%} {r['exact_frac']:>6.0%} "
                  f"{r['frac_fmp_closer_to_av_est']:>10.0%} "
                  f"{r['surprise_corr']:>10.2f}")
        print("\n  --- OVERALL (pooled) ---")
        print(f"  overlap events ........... {overall['n_overlap']}")
        print(f"  est agree within 1% ...... {overall['agree_frac_within_1pct']:.0%} "
              f"(median rel-diff {overall['median_rel_diff']:.4f}, "
              f"exact {overall['exact_frac']:.0%})")
        print(f"  BACKFILL SIGNATURE: on {overall['n_informative']} surprise "
              "quarters, FMP est is closer to")
        print(f"     AV est than to AV actual {overall['frac_fmp_closer_to_av_est']:.0%} "
              "of the time  (HIGH = independent PIT consensus; LOW = backfilled)")
        print(f"     median |FMP_est - AV_est| = {overall['median_dist_to_av_est']:.4f} "
              f"vs |FMP_est - AV_actual| = {overall['median_dist_to_av_actual']:.4f}")
        print(f"  SURPRISE CORR (FMP vs AV)  {overall['surprise_corr']:.2f}  "
              "(HIGH = two genuine PIT feeds agree on the surprise; "
              "~0 = a backfilled/zero-surprise feed)")

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "pead_cross_check.json")
    payload = {
        "fmp_symbols": syms,
        "av_symbols": sorted(av_ev["ticker"].unique().tolist()),
        "n_fmp_events": int(len(fmp_ev)),
        "n_av_events": int(len(av_ev)),
        "report": report,
        "overall_diagnostics": overall,
        "per_name": detail.to_dict(orient="records"),
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"\n[out] wrote {out_path}")
    print("\nINTERPRETATION: the cross-source block is strong PIT evidence ONLY if "
          "BOTH (a) FMP est sits closer to AV est than to AV actual on surprise "
          "quarters AND (b) the FMP/AV surprise correlation is high. Level "
          "agreement alone is NOT enough (a backfilled feed also matches the level "
          "on no-surprise quarters). It remains evidence, not proof — shared "
          "upstream data is possible; IBES via WRDS is the gold standard. This run "
          "grades nothing and spends no trial.")


if __name__ == "__main__":
    main()
