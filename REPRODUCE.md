# Reproduce this project in ten minutes

No API keys. No network beyond `pip`. One command produces a verdict.

The claim being defended here is deliberately narrow: **the falsification
harness and the headline numbers that depend on it are reproducible offline,
from a seed, on your machine.** The real-data results are *not* reproducible
from this repo alone, and the last section of this file says exactly which is
which. Overclaiming reproducibility would be the same sin as overclaiming a
Sharpe ratio.

---

## The ten minutes

```bash
git clone https://github.com/Jareedd/qr-alpha-lab && cd qr-alpha-lab
python -m venv .venv && source .venv/bin/activate    # python 3.11 or 3.12
pip install -r requirements-lock.txt                 # ~2-4 min, the slow step
python scripts/reproduce.py                          # ~20 s of compute
```

Expected final lines:

```
 13/13 metrics within tolerance; 4/4 gates passed in ~20s of compute.

 RESULT: REPRODUCED
```

Exit code `0` means reproduced; non-zero prints which metric moved, by how
much, and against which tolerance. A machine-readable copy lands in
`results/repro_report.json`.

`pip install -r requirements.txt` (version *ranges*) also works and is what CI
uses; `requirements-lock.txt` is the exact combination the numbers below were
verified against, so a numeric mismatch can be blamed on the environment or
cleared in one step.

---

## The dataset: a deterministic generator, not a data file

`src/quantlab/synthetic.py` builds the panels from a seed, so the "fixed
dataset" is 3 KB of code rather than a parquet file nobody can audit:

| mode | what it is | what a correct pipeline must do |
|---|---|---|
| `planted` | 60 assets x 3000 days; market + sector + idiosyncratic returns with GARCH-like vol clustering, plus a **known** momentum-like predictable component (`signal_strength=0.03`) | **recover** it |
| `noise` | the same generator with the planted component switched off | **find nothing** |
| `noise` + 1-line leak | `panel["leak_fwd_return"] = panel["label"]` | **be caught** |

Seed `7` throughout, passed explicitly via `--seed`. `scripts/reproduce.py`
verifies not only that the same seed gives the same answer but that a
*different* seed gives a different one — otherwise "deterministic" could just
mean "the number is hardcoded somewhere".

To regenerate a panel yourself:

```python
from quantlab.synthetic import make_panel
prices = make_panel(mode="planted", seed=7)   # (3000 x 60) price panel
```

---

## Exact commands and expected outputs

Every check below is the documented command, run as a subprocess by
`scripts/reproduce.py`, so what you type is what CI runs is what gets
verified. Expected values live in [`repro/expected.json`](repro/expected.json).

### 1. The planted signal must be recovered

```bash
python scripts/run_pipeline.py --data planted --seed 7 --fail-if-dsr-below 0.95
```

| metric | expected | tolerance |
|---|---|---|
| mean rank IC (OOS) | `0.062863` | ±0.001 |
| Newey–West t-stat | `2.0006` | ±0.05 |
| net Sharpe | `0.86458` | ±0.005 |
| annual turnover | `2.17` | ±0.01 |
| **DSR** | `0.991946` | ±0.005 |
| OOS days | `1992` | exact |
| **gate** | DSR ≥ 0.95 → PASS | binary |

### 2. Pure noise must be rejected

```bash
python scripts/run_pipeline.py --data noise --seed 7 --n-trials 20 --fail-if-dsr-above 0.5
```

| metric | expected | tolerance |
|---|---|---|
| mean rank IC (OOS) | `-0.019925` | ±0.001 |
| net Sharpe | `-0.50289` | ±0.005 |
| **DSR** | `0.000435` | ±0.005 |
| OOS days | `1992` | exact |
| **gate** | DSR ≤ 0.5 → PASS | binary |

### 3. A one-line look-ahead leak must be caught

```bash
python scripts/leak_demo.py
```

| metric | expected | tolerance |
|---|---|---|
| planted DSR | `0.9919` | ±0.005 |
| clean-noise DSR | `0.0004` | ±0.005 |
| leaked-noise DSR | `1.0000` | ±0.005 |
| **gate** | leaked noise breaks the DSR ≤ 0.5 bar → leak caught | binary |

### 4. Determinism

Two `--seed 7` runs must agree **bitwise** on every numeric field; a `--seed 99`
run must differ (seed 7 IC `0.062863` vs seed 99 IC `0.097892`).

### And the full suite

```bash
python -m pytest tests/ -q          # 494 passed, 10 skipped, ~6.5 min
```

---

## About the tolerances

A tolerance without a rationale is a fudge factor, so here is the rationale.

The pipeline is deterministic given a seed: rerun on the *same* machine and
the agreement is bitwise (check 4 asserts exactly that). Across a different
CPU, BLAS build, or library patch version, float accumulation order changes.
The drift this project has actually observed machine-to-machine is **~1e-14**
(`research_log.md`, 2026-06-10 laptop→PC migration; re-confirmed 2026-09-08
when the loader rewrite moved the planted metrics by ~1e-14 while leaving IC,
DSR, PSR and turnover byte-identical).

The default tolerances are roughly **1e10 times** that observed drift: loose
enough that a reviewer on different hardware is not chasing ghosts, tight
enough that a real behaviour change cannot hide inside them. `--strict`
tightens everything to 1e-10 for same-machine checks.

**The binary gates are the scientific claim; the point estimates are the
evidence for it.** If a future library version shifts the IC in the fourth
decimal, that is noise. If it flips a gate, that is a finding.

---

## Reproduced here vs. historical claims

This is the distinction that matters most, so it gets its own section rather
than a footnote.

### Reproducible by you, right now, offline

| claim | how to check |
|---|---|
| The pipeline recovers a planted signal (IC 0.063, DSR 0.99) | `scripts/reproduce.py` check 1 |
| The pipeline rejects pure noise (DSR 0.0004) | check 2 |
| A one-line look-ahead leak flips noise to DSR 1.0 and turns the gate red | check 3 |
| Results are seed-deterministic and seed-sensitive | check 4 |
| Walk-forward embargo, cost model, DSR formula, weight construction, baselines behave as documented | `pytest tests/ -q` — 494 known-answer tests, no network |
| Data-loader behaviour under vendor failure (empty downloads, stale caches, partial downloads, duplicate timestamps, invalid input, reruns) | `pytest tests/test_data_loader.py -q` — 32 tests, fake vendor, no network |
| Ingest pipeline: incremental updates, schema validation, safe reruns, backfill, resume-after-failure | `pytest tests/test_ingest.py -q`, and the live demo in `scripts/ingest_demo.py` |

### Historical claims — evidence, not reproductions

Everything derived from **real market data** (trials #1–#13, the live paper
trading, the survivorship-bias case study) depends on vendor snapshots —
yfinance/Tiingo/SEC/Binance pulls taken on specific dates, some behind paid
quotas, all of them silently revised over time by the vendors themselves (the
project measures that revision drift; see `results/live/revisions_*.json`).

You cannot re-derive those numbers from this repo. What you *can* do:

- **Read the committed artifact.** Every real-data number in the README and
  the research note traces to a JSON in `results/` and a dated row in
  `research_log.md` — e.g. the survivorship case study's `0.82 → −0.01` is
  `results/metrics_yfinance_ridge.json` and `results/metrics_sp500_ridge.json`.
- **Re-derive the *presentation* offline.** `python scripts/case_study_chart.py`
  rebuilds the case-study chart from those committed JSONs, so the figure is
  itself reproducible even though the underlying pull is not.
- **Re-run the pull yourself**, with your own keys and today's data, and expect
  *different* numbers — that is the honest expectation, not a failure.

Treat the synthetic checks as verification that the machinery is sound, and
the real-data numbers as reported evidence from a dated experiment. They are
different kinds of claim and this project does not blur them.

### What the synthetic checks do NOT prove

The planted/noise/leak harness catches the leakage it is built to catch:
future information entering through the **feature matrix or the label
alignment** on a panel it generates. It is a strong test of that, and it has
caught real bugs. It is not a proof of "no leakage".

It would not, by construction, catch:

- **Survivorship bias in the universe** — the synthetic universe has no
  delistings, so a biased universe is invisible to it. This is not
  hypothetical: it is exactly the bug the case study documents, and it was
  caught by a point-in-time universe control, not by the gate.
- **Look-ahead in the data source itself** — restated fundamentals, vendor
  back-adjustment of "history", index membership known only in hindsight.
  Synthetic panels are point-in-time by construction, so they cannot exhibit
  the failure.
- **Cross-validation leakage across a real correlation structure** the
  generator does not model (e.g. real sector co-movement during crises).
- **Selection bias in the researcher** — trying 40 variants and reporting the
  best. That is addressed by the trial count and the DSR, not by the gate.
- **Costs, borrow, capacity and impact being wrong** — those are modelling
  assumptions, and being wrong about them is not leakage.

A green gate means "the plumbing did not invent this result". It does not mean
"this result is true".

---

## If it does not reproduce

1. **Check the environment first.** `pip install -r requirements-lock.txt` and
   re-run. `scripts/reproduce.py` prints its interpreter and library versions.
2. **Read the mismatch table.** It names the metric, expected value, observed
   value and tolerance. A gate failure is more serious than a metric drift.
3. **A drifted metric with all gates green** is almost always a library
   version difference. Open an issue with the printed version block.
4. **A failed gate** means the pipeline's behaviour changed. That is a real
   finding either way: report it.

Do not "fix" a mismatch by re-recording expectations. `--update` exists, and
using it without a `research_log.md` entry explaining *why* the expectation
moved is how a reproducibility check quietly becomes decorative.
