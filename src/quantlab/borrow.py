"""H7: daily short-borrow snapshots — COLLECTION ONLY, zero trials.

Registered 2026-06-12 (see writeup/preregistered_hypotheses.md H7). The
borrow fee is the observable price of concentrated negative information;
its history cannot be honestly backfilled — whoever wants it must have
been snapshotting it, which is the moat (the H5 argument, second
instance). This module only COLLECTS: per the two-stage protocol, no
analysis of any kind is permitted before the Stage-2 registration is
written (≥ 60 cycles, ~September 2026). There is deliberately no function
here that joins borrow data to returns.

Source, verified at implementation (2026-06-12): IBKR's public
short-stock availability file — FTP host ftp2.interactivebrokers.com,
user 'shortstock' (no password), file 'usa.txt'. Pipe-delimited, with a
'#BOF|YYYY.MM.DD|HH:MM:SS' stamp line and a '#SYM|CUR|NAME|...' header:

    #SYM|CUR|NAME|CON|ISIN|REBATERATE|FEERATE|AVAILABLE|FIGI|

AVAILABLE is sometimes ">10000000" (capped); FEERATE/REBATERATE are
percents. ~20k instruments including bonds; we keep equity-like symbols
present in our own live universe plus aggregate stats over the full file.

Operational law (the revisions.py pattern): a collection bug must never
cost a trading cycle — the live.yml step is non-fatal, and this module
raises only inside its own scope.
"""

from __future__ import annotations

import datetime as dt
import io
from ftplib import FTP

import pandas as pd

FTP_HOST = "ftp2.interactivebrokers.com"
FTP_USER = "shortstock"
FILE = "usa.txt"


def fetch_ibkr_short_file(timeout: int = 60) -> str:
    """Download the raw availability file (network; CI-safe via the
    non-fatal wrapper in live.yml, never called from tests)."""
    ftp = FTP(FTP_HOST, timeout=timeout)
    try:
        ftp.login(FTP_USER, "")
        buf = io.BytesIO()
        ftp.retrbinary(f"RETR {FILE}", buf.write, blocksize=1 << 20)
    finally:
        try:
            ftp.quit()
        except Exception:  # noqa: BLE001 -- close must not mask the payload
            ftp.close()
    return buf.getvalue().decode("latin-1")


def parse_ibkr_short_file(text: str) -> tuple[str, pd.DataFrame]:
    """(file_stamp, frame) from the raw pipe-delimited text.

    file_stamp is the file's OWN '#BOF' timestamp ('YYYY-MM-DD HH:MM:SS')
    — the artifact carries the source's clock, not ours. Frame is indexed
    by symbol with columns rebate_rate, fee_rate, available (int; '>'
    caps stripped), available_capped (bool). Unparseable lines are
    counted, not fatal: the snapshot reports n_skipped so silent format
    drift is visible in the record.
    """
    stamp = ""
    rows: list[dict] = []
    n_skipped = 0
    for line in text.splitlines():
        if line.startswith("#BOF"):
            parts = line.split("|")
            if len(parts) >= 3:
                stamp = f"{parts[1].replace('.', '-')} {parts[2]}"
            continue
        if not line or line.startswith("#"):
            continue
        cells = line.split("|")
        if len(cells) < 8:
            n_skipped += 1
            continue
        sym, avail_raw = cells[0], cells[7].strip()
        capped = avail_raw.startswith(">")
        try:
            rows.append(
                {
                    "symbol": sym,
                    "rebate_rate": float(cells[5]) if cells[5].strip() else None,
                    "fee_rate": float(cells[6]) if cells[6].strip() else None,
                    "available": int(avail_raw.lstrip(">")) if avail_raw else 0,
                    "available_capped": capped,
                }
            )
        except ValueError:
            n_skipped += 1
    frame = pd.DataFrame(rows).set_index("symbol") if rows else pd.DataFrame()
    frame.attrs["n_skipped"] = n_skipped
    return stamp, frame


def build_snapshot(
    file_stamp: str,
    frame: pd.DataFrame,
    universe: list[str],
    universe_asof: str | None = None,
) -> dict:
    """JSON-ready snapshot: per-name records for OUR universe (the live
    book's scored cross-section), plus whole-file aggregates so format or
    coverage drift is visible without storing 20k rows a day.

    ``universe_asof`` dates the cross-section itself. It can lag the snapshot
    when the collector runs on a day whose trading cycle failed -- which is by
    design (an unbackfillable record should not stop because something else
    broke), and therefore has to be visible in the record."""
    in_univ = frame.loc[frame.index.intersection(universe)]
    fees = frame["fee_rate"].dropna()
    return {
        "source": f"{FTP_HOST}/{FILE}",
        "file_stamp": file_stamp,
        "universe_asof": universe_asof,
        "fetched_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "n_symbols_in_file": int(len(frame)),
        "n_skipped_lines": int(frame.attrs.get("n_skipped", 0)),
        "universe_size": len(universe),
        "universe_covered": int(len(in_univ)),
        "fee_rate_percentiles_full_file": {
            str(q): round(float(fees.quantile(q / 100)), 4)
            for q in (50, 90, 99)
        }
        if len(fees)
        else {},
        "names": {
            sym: {
                "fee_rate": (None if pd.isna(r["fee_rate"]) else float(r["fee_rate"])),
                "rebate_rate": (
                    None if pd.isna(r["rebate_rate"]) else float(r["rebate_rate"])
                ),
                "available": int(r["available"]),
                "available_capped": bool(r["available_capped"]),
            }
            for sym, r in in_univ.iterrows()
        },
    }
