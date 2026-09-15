#!/usr/bin/env python
"""Ten-minute reproduction: clone, install, run this, get a verdict.

The claim this script defends is narrow and checkable: **the falsification
harness and its headline numbers are reproducible offline, from a seed, with
no API keys.** It does NOT reproduce the real-data results -- those depend on
a vendor snapshot nobody else can obtain, and pretending otherwise would be
the kind of overclaim this project exists to avoid. The boundary is drawn
explicitly in REPRODUCE.md and in this script's own output.

Every check is a *documented command* run as a subprocess (or the documented
entry point imported), so what a reviewer types is what CI runs is what this
verifies. Expected values and tolerances live in ``repro/expected.json``.

    python scripts/reproduce.py              # verify against expected.json
    python scripts/reproduce.py --strict     # demand near-bitwise agreement
    python scripts/reproduce.py --update     # re-record expectations (logs it)

Exit code 0 = every check reproduced. Non-zero = something moved, and the
table says which number, by how much, and against what tolerance.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from contextlib import redirect_stdout

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

EXPECTED_PATH = os.path.join(REPO, "repro", "expected.json")

# Tolerances. Rationale, because a tolerance without one is a fudge factor:
# the pipeline is deterministic given a seed, so a rerun on THIS machine
# agrees to ~1e-14. Across a different CPU, BLAS or library patch version the
# accumulated float drift is larger but still tiny -- the observed
# machine-to-machine drift on this project has been ~1e-14 (research_log.md,
# 2026-06-10 PC migration). The default tolerances below are ~1e10 times that
# drift: loose enough that a reviewer on different hardware is not chasing
# ghosts, tight enough that a real behaviour change cannot hide inside them.
# The BINARY GATES (planted recovered, noise rejected, leak caught) are the
# actual scientific claim; the point estimates are the evidence for it.
DEFAULT_TOL = {
    "mean_rank_ic": 1e-3,
    "sharpe_net": 5e-3,
    "sharpe_gross": 5e-3,
    "dsr": 5e-3,
    "psr": 5e-3,
    "annual_turnover": 1e-2,
    "ic_tstat_newey_west": 5e-2,
    "max_drawdown": 5e-3,
    "n_days": 0,            # exact: a different row count is a different study
    "leak_dsr": 5e-3,
    "clean_noise_dsr": 5e-3,
    "planted_dsr": 5e-3,
}
STRICT_TOL = {k: (0 if v == 0 else 1e-10) for k, v in DEFAULT_TOL.items()}


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------
def _run_pipeline(args: list[str], out_dir: str) -> dict:
    """Run the documented CLI and return the metrics JSON it wrote."""
    cmd = [sys.executable, os.path.join(REPO, "scripts", "run_pipeline.py"),
           *args, "--out", out_dir]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )
    written = [f for f in os.listdir(out_dir) if f.startswith("metrics_")]
    if len(written) != 1:
        raise RuntimeError(f"expected exactly one metrics file, got {written}")
    with open(os.path.join(out_dir, written[0])) as fh:
        return json.load(fh)


def check_planted(tmp: str) -> dict:
    """The pipeline must RECOVER a known planted signal."""
    out = os.path.join(tmp, "planted")
    m = _run_pipeline(["--data", "planted", "--seed", "7",
                       "--fail-if-dsr-below", "0.95"], out)
    return {
        "mean_rank_ic": m["mean_rank_ic"],
        "sharpe_net": m["sharpe_net"],
        "dsr": m["dsr"],
        "annual_turnover": m["annual_turnover"],
        "ic_tstat_newey_west": m["ic_tstat_newey_west"],
        "n_days": m["n_days"],
        "_gate": ("planted DSR >= 0.95", m["dsr"] >= 0.95),
    }


def check_noise(tmp: str) -> dict:
    """The pipeline must REJECT pure noise."""
    out = os.path.join(tmp, "noise")
    m = _run_pipeline(["--data", "noise", "--seed", "7", "--n-trials", "20",
                       "--fail-if-dsr-above", "0.5"], out)
    return {
        "mean_rank_ic": m["mean_rank_ic"],
        "sharpe_net": m["sharpe_net"],
        "dsr": m["dsr"],
        "n_days": m["n_days"],
        "_gate": ("noise DSR <= 0.5", m["dsr"] <= 0.5),
    }


def check_leak(tmp: str) -> dict:
    """A one-line look-ahead leak must flip the noise verdict to 'alpha'."""
    import leak_demo

    buf = io.StringIO()
    with redirect_stdout(buf):
        code = leak_demo.run()
    text = buf.getvalue()
    vals = {}
    for line in text.splitlines():
        if "PLANTED signal" in line:
            vals["planted_dsr"] = float(line.split("DSR=")[1])
        elif "PURE NOISE" in line:
            vals["clean_noise_dsr"] = float(line.split("DSR=")[1])
        elif "look-ahead leak" in line and "DSR=" in line:
            vals["leak_dsr"] = float(line.split("DSR=")[1])
    vals["_gate"] = ("leak caught (demo exit 0)", code == 0)
    vals["_transcript"] = text
    return vals


def check_determinism(tmp: str) -> dict:
    """Same seed -> identical numbers; different seed -> different numbers.

    Both halves matter. The first is reproducibility. The second proves the
    seed is actually wired to the data and that "reproducible" is not just
    "the number is hardcoded somewhere".
    """
    a = _run_pipeline(["--data", "planted", "--seed", "7"], os.path.join(tmp, "det_a"))
    b = _run_pipeline(["--data", "planted", "--seed", "7"], os.path.join(tmp, "det_b"))
    c = _run_pipeline(["--data", "planted", "--seed", "99"], os.path.join(tmp, "det_c"))
    same_seed_identical = all(
        a[k] == b[k] for k in a if isinstance(a[k], (int, float))
    )
    seed_actually_bites = a["mean_rank_ic"] != c["mean_rank_ic"]
    return {
        "_gate": (
            "same seed bitwise identical AND a different seed moves the result",
            same_seed_identical and seed_actually_bites,
        ),
        "_note": (
            f"seed 7 IC={a['mean_rank_ic']:.6f} (twice, bitwise identical: "
            f"{same_seed_identical}); seed 99 IC={c['mean_rank_ic']:.6f}"
        ),
    }


CHECKS = [
    ("planted_recovery", "Planted signal is recovered", check_planted),
    ("noise_rejection", "Pure noise is rejected", check_noise),
    ("leak_detection", "A 1-line look-ahead leak is caught", check_leak),
    ("determinism", "Seeded runs are reproducible and seed-sensitive", check_determinism),
]


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def _compare(name: str, observed: dict, expected: dict, tol: dict) -> list[tuple]:
    rows = []
    for key, obs in observed.items():
        if key.startswith("_"):
            continue
        exp = expected.get(key)
        t = tol.get(key, 1e-3)
        if exp is None:
            rows.append((f"{name}.{key}", None, t, obs, "NEW"))
            continue
        delta = abs(obs - exp)
        rows.append((f"{name}.{key}", exp, t, obs, "ok" if delta <= t else "MISMATCH"))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strict", action="store_true",
                    help="near-bitwise tolerances (same machine, same libs)")
    ap.add_argument("--update", action="store_true",
                    help="re-record repro/expected.json (log the reason!)")
    ap.add_argument("--report", default=os.path.join(REPO, "results", "repro_report.json"))
    args = ap.parse_args()

    tol = STRICT_TOL if args.strict else DEFAULT_TOL
    expected_doc = {}
    if os.path.exists(EXPECTED_PATH):
        with open(EXPECTED_PATH) as fh:
            expected_doc = json.load(fh)
    expected_all = expected_doc.get("checks", {})

    print("=" * 78)
    print(" qr-alpha-lab REPRODUCTION -- offline, seeded, no API keys")
    print("=" * 78)
    print(f" python {platform.python_version()} on {platform.system()} "
          f"{platform.machine()}")
    try:
        import numpy, pandas, sklearn, scipy
        print(f" numpy {numpy.__version__}  pandas {pandas.__version__}  "
              f"scikit-learn {sklearn.__version__}  scipy {scipy.__version__}")
    except ImportError:  # pragma: no cover
        pass
    print(f" tolerance profile: {'STRICT' if args.strict else 'default'}")
    print()

    all_rows: list[tuple] = []
    gates: list[tuple[str, str, bool]] = []
    observed_all: dict = {}
    t0 = time.time()

    with tempfile.TemporaryDirectory() as tmp:
        for name, title, fn in CHECKS:
            t = time.time()
            try:
                obs = fn(tmp)
            except Exception as exc:
                print(f"[{name}] ERROR: {exc}")
                gates.append((name, "check ran", False))
                continue
            elapsed = time.time() - t
            gate_desc, gate_ok = obs.get("_gate", ("(no gate)", True))
            gates.append((name, gate_desc, gate_ok))
            observed_all[name] = {k: v for k, v in obs.items() if not k.startswith("_")}
            rows = _compare(name, obs, expected_all.get(name, {}), tol)
            all_rows.extend(rows)
            print(f"[{name}] {title}  ({elapsed:.1f}s)")
            print(f"    gate: {gate_desc} -> {'PASS' if gate_ok else 'FAIL'}")
            if obs.get("_note"):
                print(f"    {obs['_note']}")

    if args.update:
        doc = {
            "_comment": (
                "Expected values for scripts/reproduce.py. Regenerated with "
                "--update; any change here must be explained in a research_log.md "
                "entry, because moving the expectation to match the code is how "
                "a reproducibility check quietly becomes decorative."
            ),
            "recorded_on": time.strftime("%Y-%m-%d"),
            "environment": {
                "python": platform.python_version(),
                "platform": f"{platform.system()} {platform.machine()}",
            },
            "tolerances": DEFAULT_TOL,
            "checks": observed_all,
        }
        try:
            import numpy, pandas, sklearn, scipy
            doc["environment"].update({
                "numpy": numpy.__version__, "pandas": pandas.__version__,
                "scikit-learn": sklearn.__version__, "scipy": scipy.__version__,
            })
        except ImportError:  # pragma: no cover
            pass
        os.makedirs(os.path.dirname(EXPECTED_PATH), exist_ok=True)
        with open(EXPECTED_PATH, "w") as fh:
            json.dump(doc, fh, indent=2)
            fh.write("\n")
        print(f"\nWrote {EXPECTED_PATH} -- now log WHY the expectations moved.")
        return 0

    print("\n" + "-" * 78)
    print(f"{'metric':<42}{'expected':>12}{'observed':>12}{'':>3}{'verdict'}")
    print("-" * 78)
    for metric, exp, t, obs, verdict in all_rows:
        exp_s = "--" if exp is None else f"{exp:.6g}"
        print(f"{metric:<42}{exp_s:>12}{obs:>12.6g}   {verdict}"
              + ("" if verdict == "ok" else f"  (tol {t:g})"))

    mismatches = [r for r in all_rows if r[4] == "MISMATCH"]
    failed_gates = [g for g in gates if not g[2]]
    elapsed = time.time() - t0

    print("-" * 78)
    print(f" {len(all_rows) - len(mismatches)}/{len(all_rows)} metrics within "
          f"tolerance; {len(gates) - len(failed_gates)}/{len(gates)} gates passed "
          f"in {elapsed:.1f}s of compute.")
    for name, desc, ok in failed_gates:
        print(f"   GATE FAILED: {name} -- {desc}")
    for metric, exp, t, obs, _ in mismatches:
        print(f"   MISMATCH: {metric} expected {exp:.6g} +/- {t:g}, got {obs:.6g}")

    ok = not mismatches and not failed_gates
    print()
    print(" RESULT: " + ("REPRODUCED" if ok else "NOT REPRODUCED -- see above"))
    print()
    print(" Scope note: this verifies the SYNTHETIC falsification harness only.")
    print(" The real-data results (trials #1-#13) are historical claims backed by")
    print(" committed artifacts in results/ and rows in research_log.md; they")
    print(" depend on vendor snapshots and are NOT re-derivable from this repo")
    print(" alone. See REPRODUCE.md, 'Reproduced here vs. historical claims'.")

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w") as fh:
        json.dump({
            "reproduced": ok,
            "elapsed_seconds": round(elapsed, 2),
            "environment": {
                "python": platform.python_version(),
                "platform": f"{platform.system()} {platform.machine()}",
            },
            "tolerance_profile": "strict" if args.strict else "default",
            "gates": [{"check": n, "gate": d, "passed": o} for n, d, o in gates],
            "metrics": [
                {"metric": m, "expected": e, "tolerance": t, "observed": o,
                 "verdict": v}
                for m, e, t, o, v in all_rows
            ],
            "observed": observed_all,
        }, fh, indent=2)
        fh.write("\n")
    print(f" Wrote {os.path.relpath(args.report, REPO)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
