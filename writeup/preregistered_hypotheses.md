# Pre-registered hypotheses (Phase 8+ candidates)

The Phase 4 verdict was that the five standard price-only features carry
no defensible edge on the honest universe — and that *new alpha
hypotheses get their own pre-registered trials*, not salvage runs. This
file is where that happens.

**Protocol.** A hypothesis is registered here — with its exact config and
success criteria — BEFORE the run. The run happens once, increments N,
and the outcome is logged whatever it says. Editing a registration after
seeing results is prohibited; a revised idea is a new registration. This
is the difference between testing seven hypotheses and running one
hypothesis seven times until it works.

**Power before spend (MDE).** Every registration must state its minimum
detectable effect — the smallest true net Sharpe it could detect at DSR ≥ 0.95
for its n_obs at the current N — BEFORE running (the fee-first power checks for
the carry tail, trial #10, were the first instance; this makes it universal). A
trial whose plausible edge sits below its MDE is underpowered by construction and
should be redesigned (longer sample, lower turnover) or not run. This spends
thought, not trials.

Template:

```
### H<n>: <one-line hypothesis>
- Status: PROPOSED | RUN (trial #k) | ABANDONED (why)
- Economic prior: why would this be priced? who is on the other side?
- Point-in-time safety: one-line argument per new feature/label
- Exact config: data, features, label, horizon, model, neutralization, costs
- Success criteria (set BEFORE the run): e.g. t_NW ≥ +2 AND DSR ≥ 0.95 at
  the then-current N, net of costs, beats both baselines
- Minimum detectable effect (MDE) — MANDATORY: the smallest TRUE net annual
  Sharpe this trial could clear at DSR ≥ 0.95 given its n_obs and the current N
  (from `scripts/graduation_hurdle.py`). Stated BEFORE the run so an
  underpowered trial is known to be underpowered before it spends N.
- Failure interpretation: what we conclude if it fails
```

---

### H1: Fundamental quality tilts (profitability/accruals) carry cross-sectional signal where price features did not
- Status: **RUN (trial #12, 2026-06-25). Outcome: does NOT graduate** — the CBOP "quality" edge was VALUE IN DISGUISE (RAW net SR +0.58/t_NW +2.30, but the HML-NEUTRAL arm where graduation is judged collapses to -0.18/DSR 0.009; raw-neutral gap +0.76 = the pre-registered value-collinearity signature). N=12. Was PROPOSED (blocked on a data-source decision), resolved free via the SEC name-crosswalk + Tiingo; reproduced leak-fixed on independent data (research_log 2026-06-25).
- Economic prior: quality measures (gross profitability, accrual
  reversal) have survived publication better than price anomalies
  (Novy-Marx 2013); they rebalance slowly, so cost mortality is low —
  exactly the failure mode that killed trials 2–7.
- Point-in-time safety: the hard part. Fundamentals must be lagged by
  filing date, not period end (a ≥3-month report lag as a crude floor).
  Free sources (SEC XBRL frames, FMP free tier) need an explicit
  staleness audit before any run. **This hypothesis cannot run until the
  data question is answered honestly; that audit is its own session.**
- Exact config: PIT sp500, ridge, h=63d, rebalance 63d (quality is slow),
  features = {GP/A, total accruals/A}, sector demean + beta projection,
  10 bps.
- Success criteria: t_NW ≥ +2 and DSR ≥ 0.95 at then-current N, net SR >
  both baselines.
- Failure interpretation: large-cap US quality is also arbitraged away at
  free-data fidelity; the write-up's conclusion extends from price-only
  to price+fundamental features.
- **Amendment, 2026-06-16 — PRE-DATA (status stays PROPOSED; no trial run; still
  blocked on the data-source decision). Refines the spec from the OSAP
  cheapest-kill verification — full record in `research/06_h1_osap_verification.md`.
  Pre-data refinement is legitimate (no H1 result has been seen); the rationale
  is documented here per protocol.**
  1. **Profitability leg = cash-based operating profitability (CBOP), value-weighted
     — NOT the original `z(GP/A) − z(accruals/A)` blend.** Accruals is *subsumed*
     by profitability (Ball, Gerakos, Linnainmaa & Nikolaev 2016), so blending
     double-counts and adds a decayed, microcap/short-leg-concentrated leg. CBOP
     (operating profitability net of accruals, over total assets) captures the
     accruals information cleanly. Any accruals claim is run ONLY as a separate,
     separately-reported, gated hedged claim — never blended.
  2. **Universe: PIT S&P 500, excluding GICS Financials AND Real Estate** (no
     CoGS line). This is **~21% by count** (107/503), not the "~40%" the original
     note implied — ~396 non-financial names → ~40–79 per quintile (stable;
     ≥38/quintile even under coverage haircuts). Large-cap by construction.
  3. **Construction (frozen): value-weighted QUINTILE long-short** (not decile —
     the VW extreme-decile weakens to t=1.88 post-2013). Denominator =
     **current** assets as primary; the **lagged-assets** version reported as a
     robustness leg (it is the cut Hou–Xue–Zhang 2020 find *insignificant*, t=1.04
     — a declared risk). h ≈ 63d, quarterly rebalance (low turnover).
  4. **Data source (the gate): survivorship-safe AND filing-date point-in-time.**
     Sharadar SF1 on an **As-Reported** dimension (ARQ/ART, keyed on `DATEKEY`) —
     NOT MRQ/MRT (restatement look-ahead); OR **WRDS Compustat PIT/Snapshot** —
     NOT vanilla (restated) Compustat. The free SEC source is forbidden
     (survivorship-blocked ~39–75%). Run via `run_fundamentals.py --source
     compustat/sharadar` after the machinery + data gates.
  5. **Annualization fix (confirmed bug):** flow fundamentals (CBOP components)
     must be annualized (TTM / ART, or 10-K-only) BEFORE dividing by point-in-time
     Assets — the free pull mixed 10-Q quarterly flows with stock Assets. Fix
     before the graded run (the machinery gate does not catch it).
  6. **Success criteria (frozen, N=12):** right-signed t_NW ≥ +2 AND net SR > 0
     beating equal-weight and 12-1 momentum baselines net of costs AND **DSR ≥ 0.95
     at N=12** — hurdle **≈ 0.90 net annual SR (daily, ~15-yr) / ≈ 0.86 (weekly)**
     (corrected from the stale 0.83, which was the N=10 value). Value-weighted,
     net of cost. Machinery gate (synthetic `planted_quality` recovered /
     `null_quality` rejected) runs in-env first.
  7. **Honest prior (updated by the verification):** gross profitability is ALIVE
     as a factor (OSAP post-2013 VW NW t≈2.9–3.2, and *strengthened* per
     Novy-Marx & Medhat 2025) — so this is NOT a free kill. BUT the large-cap,
     value-weighted, **net-of-cost** raw cut H1 actually trades is the *marginal*
     version (in-sample large-cap raw t=1.88; the large-cap result is an FF3
     alpha; the lagged-assets denominator is insignificant). **Expected outcome:
     a credible NULL or a marginal pass** — logged whatever it says; either
     extends the survivorship/cost-mortality story to fundamentals.
- **Amendment, 2026-06-24 — PRE-DATA, MACHINERY-ONLY (status stays PROPOSED; ZERO
  trials; N unchanged at 11; H1 remains blocked on the survivorship-safe PIT
  data-source decision). Declares the adjudication protocol BEFORE any real H1 run.**
  (A) RAW-vs-NEUTRAL is now a PRE-REGISTERED, PAIRED success criterion, not a
  post-hoc robustness leg. The H1 quality book is run in TWO arms on the same
  universe / dates / costs: RAW = z(profitability) [CBOP per the 2026-06-16
  amendment]; NEUTRAL = the same signal cross-sectionally residualized against an
  HML/value loading (+ a ones column for dollar-neutrality) via
  `risk_model.cross_sectional_neutralize`, the value loading from a trailing
  past-only `rolling_factor_betas` HML beta (fit_intercept=True — required;
  point-in-time, law #1). H1 GRADUATES on the NEUTRAL arm: a quality claim that is
  merely the value factor re-labeled must NOT count. Criterion: right-signed
  t_NW ≥ +2 AND net SR > 0 beating both baselines AND DSR ≥ 0.95 at the
  then-current N on the NEUTRAL arm; the RAW arm is reported alongside, and a large
  raw-minus-neutral gap is declared IN ADVANCE as evidence the edge was
  value-collinear (interpreted, not hidden).
  (B) PBO/CSCV is now a PRE-REGISTERED selection-overfit gate COMPLEMENTING the
  DSR. Whatever set of H1 configs is considered (raw vs neutral; current- vs
  lagged-assets denominator; quintile vs declared robustness cuts), their OOS
  return paths are assembled into a contiguous, gap-free (T × M) matrix and
  `pbo.cscv_pbo` is reported. Pre-declared threshold:
  PBO ≤ 0.5 required to graduate (PBO > 0.5 means the IS-selection rule does not
  generalize OOS — a hard stop regardless of the chosen config's DSR). DSR guards
  the final track record against luck-of-N; PBO guards the selection PROCESS that
  chose it; BOTH must pass.
  (C) SYNTHETIC TWO-WORLD VALIDATION (run in-env immediately before any real H1
  run, law #4, like the carry/CEF/event gates): `synthetic.make_quality_panel`
  `quality_is_value` (edge IS value → NEUTRAL arm must collapse: static-loading
  neutral SR < 0.3) and `quality_orthogonal` (edge is value-orthogonal → NEUTRAL
  arm must SURVIVE: static-loading neutral SR > 1.0), with the worlds SR-MATCHED on
  the raw arm (discrimination attributable to neutralization, not a Sharpe gap) and
  a placebo-factor control proving the collapse requires the TRUE value factor
  (neutral_true < 0.5 × neutral_placebo). Pinned in `tests/test_quality_value.py`.
  If neutralization cannot tell a value-disguised edge from a genuine one TODAY, no
  real raw-vs-neutral number is trusted → ABORT, no trial spent. Machinery +
  synthetic validation ONLY; spends ZERO trials; N still 11.

### H2: The same pipeline finds (or honestly rejects) carry in crypto perpetuals, where survivorship bias and dead-name gaps do not exist
- Status: **RUN (trial #8, 2026-06-13). Outcome: registered criteria NOT MET
  — but the first NON-NULL the project has produced.** Net SR +0.87,
  IC t_NW −3.54 (correctly signed), funding & price P&L both positive,
  shuffled-funding control flat, survives ex-top-3. The single failed leg
  is the pre-registered DSR ≥ 0.95 (got 0.865), and the return is
  crash-skewed (−1.87, −74% maxDD). Adversarial diagnostics confirmed it
  is a REAL edge, not an artifact (entry-lag decays gracefully → no timing
  leak; only 2/28 dead names held at death → no delisting optimism) — but
  it has DECAYED from SR 2.28 (2020–21) to ~0.4 post-2021 as basis farms
  scaled (McLean–Pontiff in a second asset class). Criteria were NOT
  relaxed post-hoc. Full record: research_log.md trial #8,
  `results/metrics_h2_carry.json`, `results/h2_carry_diagnostics.json`,
  `writeup/h2_carry_design.md`. (Was: PROPOSED.)
- Economic prior: funding-rate carry in perps is a payment for taking the
  crowded side; it is directly observable, not estimated, and the
  CLAUDE.md stretch goal. Exchange data is complete (no delisting-return
  problem — delisted contracts have full terminal histories).
- Point-in-time safety: funding rates and prices are timestamped exchange
  records; features use funding through t only.
- Exact config: top-30 perps by ADV (point-in-time, by listing date),
  feature = trailing funding rate percentile, h=7d, dollar-neutral
  cross-sectional rank portfolio, taker fees + realistic spread, BTC-beta
  projection.
- Success criteria: t_NW ≥ +2 and DSR ≥ 0.95 at then-current N net of
  fees, and the result must survive excluding the top-3 names.
- Failure interpretation: carry is consumed by fees/crowding at our
  fidelity — still a publishable section ("the pipeline generalizes;
  the free lunch does not").
- **Amendment, 2026-06-12 — PRE-DATA (no exchange data has been
  downloaded as of this edit; full design in `writeup/h2_carry_design.md`):**
  1. **Label corrected to funding-INCLUSIVE total return**
     (`mark_return − funding` for a long). The premium lives in the
     funding flows; a price-only label measures the wrong object —
     demonstrated in the synthetic lab, where the same planted-carry book
     scores ≈ +1 funding-inclusive and ≈ −2 price-only
     (`tests/test_synthetic_carry.py`).
  2. **Universe corrected: perps are NOT survivorship-free.** PIT top-30
     by trailing 30d dollar volume among contracts listed at t, with
     delisted contracts (enumerable from the exchange's own public dumps,
     full terminal histories) included through their final day.
  3. **Timestamp convention fixed:** features use only funding SETTLED at
     or before the decision time (8h settlements; trailing 21-settlement
     mean); labels accrue settlements inside the holding window only.
  4. **Machinery gate added:** `synthetic.make_perp_panel` planted_carry
     must be recovered and priced_carry (funding fully offset by drift —
     the true null) rejected, in the same environment, immediately before
     the real run.
  5. **Paired controls registered:** cross-sectionally shuffled-funding
     book must earn ~nothing; funding-income vs price-drag decomposition
     reported; result must not be one regime/period/name.
  6. Costs pinned: taker 5 bps/side + spread 2 bps/side + sqrt impact on
     perp dollar-ADV; weekly rebalance; quartile book on the top-30
     universe.
  Status remains PROPOSED; the run is trial #8 and awaits owner sign-off.

### H3: Momentum works conditionally — only in low-cross-sectional-vol regimes
- Status: PROPOSED — registered partly as a TRAP CHECK
- Economic prior: weak (momentum crashes cluster in high-vol reversals,
  Daniel–Moskowitz 2016) — but regime-conditioning is also the classic
  overfitting move, which is why the criteria are strict.
- Point-in-time safety: regime indicator = trailing 63d cross-sectional
  dispersion, known at t.
- Exact config: trial #2's exact config, weights scaled by the regime
  indicator's trailing percentile (no new fitted parameters — a fixed,
  pre-declared mapping).
- Success criteria: t_NW ≥ +2, DSR ≥ 0.95, AND the unconditional result
  must remain null (if everything improves, suspect leakage first, per
  the prime directive).
- Failure interpretation: regime gating on free daily data is noise; do
  not iterate on the mapping (that is mining with extra steps).

### H4: A causal volatility-regime gate (filtered 2-state HMM) improves the momentum baseline where the fixed dispersion gate (H3) may not
- Status: PROPOSED — machinery built and falsification-validated on
  synthetic data 2026-06-12 (`quantlab/regime.py`, `planted_regime` mode,
  `tests/test_regime.py`); REAL-DATA RUN NOT PERFORMED, awaiting owner
  sign-off. A separate registration from H3 on purpose: H3's fixed
  dispersion mapping stays as registered (editing it post-hoc is
  prohibited); this is a different detector and counts as its own trial.
- Economic prior: same as H3 (Daniel–Moskowitz momentum crashes cluster
  in high-vol regimes) — weak, trap-flagged. The HMM adds persistence
  modeling, not new economics; if H3 and H4 disagree, the disagreement
  itself is evidence of mining.
- Point-in-time safety: the gate is the forward-FILTERED P(calm | market
  returns through t), parameters re-fit on an expanding PAST-only window
  (`causal_regime_probs`). The smoothed (forward-backward) probabilities
  — what off-the-shelf HMM libraries return — condition on the future;
  tests pin both that our filter cannot see the future (perturbing the
  future moves nothing in the past) and that the smoothed version DOES
  (the leak is demonstrated, not asserted). The smoothed output is never
  exposed to strategy code.
- Exact config: trial #2's exact config; rebalance weights scaled by
  filtered P(calm) at each rebalance (fixed mapping, no tuned threshold);
  HMM refit_every=63, min_train=504, on the member-masked equal-weight
  market return.
- Success criteria: t_NW ≥ +2 and DSR ≥ 0.95 at then-current N, net of
  costs; unconditional momentum must remain null; AND the paired
  artifact control must pass — the same gate applied to a label-shuffled
  control must show no lift (the synthetic lab showed residualization ×
  vol-regime interactions can manufacture conditional IC of +0.06 to
  +0.13 from nothing; a real-data result that fails the paired control
  is the artifact, not alpha).
- Failure interpretation: regime gating on free daily data is noise
  regardless of detector sophistication; H3 need not be run separately
  (one trap is enough).

### H5: Vendor data-revision intensity is cross-sectionally informative (the dataset nobody else has)
- Status: PROPOSED — two-stage registration. Stage 1 (now): the idea is
  timestamped BEFORE any data exists to peek at; collection of per-cycle
  revision fingerprints (`results/live/revisions_*.json`, started
  2026-06-11) is fully automated and untouched by hand. Stage 2: when
  ≥ 60 cycles have accumulated (~September 2026), the exact test config
  and success criteria are registered in this file BEFORE the first
  analysis run. No analysis of the revision data is permitted before
  Stage 2 is written down.
- Economic prior: return-cell revisions cluster around dividends,
  splits and corrections — i.e., corporate-action density — and flag
  names whose historical record is least stable. Plausible links: (a)
  realized-vol differences, (b) feature reliability (a name whose past
  keeps changing has noisier features, arguing for down-weighting), (c)
  at minimum a data-quality risk control for the live book. The honest
  prior is that this is a RISK/quality signal, not an alpha signal.
- Point-in-time safety: trivially safe — the fingerprint at t is a
  function of downloads made at t and t−1, both timestamped by CI
  commits; it cannot reference anything later.
- Why this qualifies as an edge at all: the dataset is proprietary by
  construction (daily diffs of a vendor's claimed history exist only if
  someone snapshots daily — we started 2026-06-11 and commit the
  fingerprints publicly, making the record verifiable). Nobody can
  backfill it.
- Failure interpretation: revision intensity is idiosyncratic vendor
  noise with no cross-sectional structure — still a write-up section,
  because measuring it is the only way anyone finds out.

### H6: Closed-end-fund discounts mean-revert from extremes in the small-fund tail, net of costs
- Status: **RUN (trial #11, 2026-06-15). Outcome: registered criteria nominally
  MET at the registered config (net SR 1.11, DSR 0.999, IC t_NW −10.4, beats EW
  0.64, positive skew +0.80, −6% maxDD) — BUT adversarial diagnostics OVERTURN
  it.** The entry-lag sweep collapses the Sharpe **1.11 → 0.10 at a one-week lag**
  (then negative): the entire "edge" is a single-week bounce, not multi-week
  reversion — a microstructure / shared-price (bid-ask-bounce) reversal artifact
  (the discount uses price_w and the next return divides by price_w; you can't
  trade the noisy close you measured). The shuffle control is also seed-fragile
  (breaches 0.3 on 3/6 seeds, max 0.52; the registered seed drew 0.277 by luck).
  The implementable (1-week-lag) edge is **~null (SR 0.10)**. **H6 does NOT
  graduate.** No criteria relaxation. Lesson: the frozen criteria lacked an
  entry-lag / implementability gate (trial #8 ran it as a diagnostic; #11 shows
  it must be a registered criterion for any reversion strategy). Full record:
  research_log.md trial #11, `results/metrics_h6_reversion.json`,
  `results/h6_reversion_diagnostics.json`. N=11. (Was: PROPOSED.) Origin spec:
  `writeup/edge_candidates_2026-06-12.md` §H6.
- Economic prior: a closed-end fund has no creation/redemption mechanism, so
  price can diverge from NAV persistently and structurally — not a picked-over
  statistical factor subject to McLean–Pontiff decay. The sub-$400M tail's
  holder base is retail; tax-season and panic selling widen discounts
  mechanically; Saba-class activists have a scale floor the tail sits beneath.
  The claim is reversion of the discount **z-score vs its own history** (not the
  absolute discount level — that is the value-trap confound). Counterparty: the
  retail panic seller and the absent arbitrageur. Capacity is **$50k–$250k** —
  "too small for Saba" is the why-it-exists, stated plainly.
- **What Stage-1 settled (zero-trial, before this registration):**
  1. A tradable tail exists: **185 funds** (mcap < $400M, dollar-ADV ≥ $250k);
     median discount −7.6%, 101 funds at < −10% (`cef_stage1_census.json`).
  2. NAV staleness is minor: **95% publish daily NAV** (lag ≤ 1d); the
     daily-NAV-only filter retains almost the whole universe.
  3. **Survivorship direction is CONSERVATIVE** (the make-or-break): the
     dead-fund census (SEC EDGAR, 2021–2026, 151 CEF/BDC deaths) found **94%
     are NAV events** (liquidation/merger/open-end/term) and **zero distress
     delistings**. CEF deaths happen at ~NAV, so a current-funds-only backtest
     OMITS winners-at-NAV → it is biased *against* a discount-long, a
     conservative LOWER bound. This is the one idea where missing dead names is
     a tailwind (`cef_dead_fund_census.json`).
  4. Data cadence (constrains this registration): free NAV/discount history is
     **weekly** beyond the trailing year (CEFConnect `All`), back to ~2012;
     distribution-inclusive total-return price is daily via yfinance.
- Exact config (FROZEN):
  - Universe: current CEFConnect funds with mcap < $400M, dollar-ADV ≥ $250k,
    **CEF-only** (CEFConnect lists CEFs; any BDC-category names excluded — BDC
    deaths can distress, unlike CEFs, per the census). Stale-NAV funds are KEPT
    in the main run and isolated by the NAV-staleness control below (refined
    pre-run, before any result is seen, for control coherence — filtering
    daily-NAV upfront would make that control trivially identical to the main
    run). Current-listings-only, accepted as a conservative lower bound per
    Stage-1's survivorship finding (direction measured, and against us).
  - Signal: per fund, `z = (discount − mean_52w) / std_52w` on the **weekly**
    discount series (past-only; ≥ 26w min history), discount = (price − NAV)/NAV
    via `quantlab.cef.discount`.
  - Book: dollar-neutral equal-weight quintiles — **LONG the most-negative-z**
    (widest-discount extreme, expected to revert up), **SHORT the most-positive-z**
    (richest), rebalanced every **4 weeks**, held between.
  - Label / P&L: **weekly distribution-inclusive total return** (yfinance
    adjusted close resampled W-FRI). **Cadence is weekly by design** — the
    signal's information arrives weekly, so weekly is the honest observation
    frequency; this sets n_obs ≈ #weeks and a DSR hurdle of ~1.4 net SR, HIGHER
    than the daily-cadence ~1.1 the graduation doc assumed before Stage-1
    revealed the data is weekly. Choosing the harder, honest hurdle is the point.
  - Costs: **25 bps one-way** on turnover (linear headline; tail spreads are
    wide so 25 bps is the conservative floor — Stage-1 found nothing forcing it
    higher; sqrt-impact is the capacity dimension, reported separately).
  - DSR at **N = 11**.
- Point-in-time safety: discount at week w uses NAV/price published at or before
  w; the z uses a trailing past-only window; weights formed at w earn from w+1.
- Machinery gate (MUST pass in-env immediately before the real run, law #4):
  a synthetic `planted_reversion` CEF world (discounts mean-revert) must be
  recovered and a `random_walk` discount world (no reversion — the true null)
  rejected, as a paired per-seed differential. If the harness cannot tell
  reversion from its absence today, no real number is trusted — ABORT, no trial.
- Registered paired controls (all reported in the single run, not extra trials):
  1. **NAV-staleness control**: re-run on the daily-NAV-only subuniverse; the
     effect must persist (stale NAVs manufacture fake extremes).
  2. **Label-shuffle control**: forward returns permuted across funds within
     each week must earn ~nothing (|SR| < 0.3).
  3. **Seasonality subreport**: SR with and without Dec–Jan (tax-loss window),
     pre-declared, reported not optimized.
- Success criteria (FROZEN — met only if ALL hold): right-signed reversion IC
  with **|t_NW| ≥ 2** AND **net SR > 0** beating an equal-weight-CEF baseline net
  of costs AND **DSR ≥ 0.95 at N=11** AND survives removing the largest-mcap
  decile AND both paired controls pass.
- Kill criteria: machinery gate fails → ABORT (no trial spent); shuffle control
  earns (|SR| ≥ 0.3) or staleness control fails → artifact, logged as such;
  |t_NW| < 2 or wrong-signed → null; DSR < 0.95 → does not graduate (logged like
  trial #8, **NOT relaxed**). NO post-hoc z-window/quantile/holding-period/cost
  scans — each is +1 trial and none is authorized by this registration.
- Failure interpretation: a real-but-DSR-failing or null result is still a
  write-up section ("the one structurally-protected premium, tested honestly on
  free data, and the discipline's verdict") — citable next to the carry trial.

### H7: Daily borrow-fee/short-availability snapshots — collection-only (zero trials)
- Status: REGISTERED 2026-06-12 with owner sign-off, COLLECTION-ONLY —
  the H5 two-stage structure, second instance. This registration
  authorizes data COLLECTION only; it does not authorize any analysis
  and does not increment N. (Originated as candidate H7 in
  `writeup/edge_candidates_2026-06-12.md`, where 17 sibling ideas were
  killed on the record.)
- Economic prior: the borrow fee is the observable price of concentrated
  negative information (the shorting-premium literature is gross of fee;
  net-of-fee tradability is ambiguous — which makes this a dataset, not
  a trade). Honest uses for THIS book: (a) a risk veto for any future
  short leg; (b) a candidate feature for future registrations; (c) an
  unbackfillable public dataset — the moat. Nobody can reconstruct
  yesterday's borrow market tomorrow.
- Source, VERIFIED at registration time: IBKR public short-stock file
  (ftp2.interactivebrokers.com, user 'shortstock', usa.txt; ~20k
  instruments, pipe-delimited, carries its own '#BOF' timestamp).
- Mechanics: non-fatal step in the live cron after each trading cycle
  (`scripts/collect_borrow.py` → `results/live/borrow_{asof}.json`,
  write-once, committed with the prediction logs). Universe = the live
  experiment's own scored cross-section (latest predictions/weights),
  plus whole-file aggregates so format drift is visible. The live
  trading path (`live.py`, deployed config) is untouched.
- Point-in-time safety: trivial — snapshot at t, committed by CI at t;
  the public commit history is the verifiable timestamp.
- Stage 2 condition: ≥ 60 cycles accumulated; the exact analysis config
  and success criteria are registered HERE before the first look. No
  analysis of any kind before Stage 2 is written down.
- Operational success criterion (Stage 1): snapshots present for ≥ 95%
  of trading cycles; schema stable; zero trading-cycle failures
  attributable to the collector.
- Kill criteria: source becomes unreliable or terms-blocked → stop
  collection, log the attempt; the outcome still costs nothing and is a
  write-up sentence about data moats.

### H8: Discretionary S&P 500 deletions earn positive post-effective returns vs a matched control
- Status: **RUN (trial #9, 2026-06-13). Outcome: clean NULL — criteria not
  met.** Daily event-time portfolio net SR −0.04, t_NW −0.10, DSR 0.05
  (75 usable events). Deleted names rebound (~+4.8%/60d) but a
  size+momentum-matched control rebounds +2.6% of it; the +2.2% residual
  is insignificant (t 0.87) and negative pre-2015 — small-loser mean
  reversion, not an index-deletion effect. Greenwood–Sammon's disappeared
  effect reproduced in the post-effective window. Synthetic planted-event
  gate passed first (drift recovered, null rejected), so this is genuine
  absence, not harness impotence. Full record: research_log.md trial #9,
  `results/metrics_h8_events.json`. (Was: PROPOSED.) Power gate had passed
  at zero cost: 124 discretionary deletions since 2010 (`results/h8_event_census.json`).
- Economic prior (weak, declared): index trackers sell at the effective
  close with a tracking-error loss function; if residual overshoot exists
  post-arbitrage it reverts. Counter-prior: Greenwood–Sammon (2025) find
  the announcement→effective deletion effect decayed to ~0.1% for
  2010–2020. This tests the strictly different POST-effective window, with
  an expectation tilted toward a clean, citable null — bought for its
  information value either way.
- Construction: enter the close of effective-date + 1 (avoids the
  depressed forced-flow close), hold 60 trading days, long the deleted
  name vs a SHORT matched-control basket — the hedge IS the control:
  deleted names are mechanically small recent losers, so any rebound must
  beat a basket matched on size and trailing return, not zero. Daily
  event-time portfolio (overlapping events averaged) feeds SR/DSR/t_NW;
  10 bps/side cost per event.
- Point-in-time safety: effective dates are exchange facts; entry t+1 uses
  only data public at t; matching features use data through t only; the
  matched pool is index membership at t.
- **Amendment, 2026-06-13 (PRE-RUN): size matched by log trailing-63d
  dollar volume**, not log-mcap — free point-in-time market cap is
  unavailable; dollar volume is a standard PIT-computable size/liquidity
  proxy. Declared before the run, per law #5 (not a post-hoc change).
- Machinery gate (passed before the real run): synthetic planted post-
  event drift recovered (positive excess + Sharpe), null rejected
  one-sided (no manufactured rebound) — `tests/test_events.py`.
- Success criteria (frozen): right-signed daily-portfolio t_NW ≥ +2 vs the
  matched control AND net SR with DSR ≥ 0.95 at N=9 AND the effect is not
  concentrated in 2010–2014 (pre-declared subperiod report).
- Kill criteria: control rebounds comparably (it was a small-loser effect
  in an index costume); t_NW < 2 → null, logged. NO entry-timing or
  holding-period variants afterward (each is +1 trial, none authorized).
- Failure interpretation: a free replication-with-control of a famous
  disappeared anomaly — citable next to trial #2's survivorship exhibit
  as the project's second in-house reproduction of the published record.

### H9: Long-tail perp funding carry — does the carry premium survive in the liquid tail (ADV ranks 31–150) beneath the majors, where funding is far wider but fills worse?
- Status: **RUN (trial #10, 2026-06-14) — clean NULL, criteria not met** (net SR
  −0.13, gross 0.26, IC t_NW −3.62 correctly signed, DSR 0.024; funding P&L +1.23
  vs price P&L −0.85 → the tail carry is largely PRICED and 20 bps fills finish
  it; shuffled control flat at −0.28; machinery gate passed → genuine economic
  null; `results/metrics_h9_carry_tail.json`). Was PROPOSED as the C1 candidate
  from `writeup/edge_candidates_2026-06-12.md`,
  parked behind H2 with an explicit unparking condition: "if H2 shows
  gross-but-not-net carry in the top-30, the tail version becomes a candidate
  with a fee-first power analysis." H2 (trial #8) ran net-positive-but-DSR-failing
  → the condition is triggered. The **fee-first power analysis ran FIRST** (zero
  trials, `scripts/carry_tail_power.py`): tail = ADV ranks 31–150 has ~2081
  usable days (~8 yr) since 2020-09, median 120 names/day, gross funding spread
  **71.8%/yr** (median 42%) — 10–29× any realistic tail cost. Power exists;
  this is necessary, NOT sufficient (a funding-only spread ignores the
  price-drift offset and the crash skew). **NOT a salvage of #8:** a DISJOINT
  universe (the majors are excluded), pre-declared before the run, judged at the
  SAME bar with NO relaxation.
- Economic prior: carry is proven real in this exact pipeline (trial #8). The
  basis-trade farms that decayed the majors (Ethena-style) concentrate on
  BTC/ETH; the tail is too small and venue-risky for them, so the tail premium
  may be LESS decayed and is structurally wider (the power analysis confirms the
  gross spread). Counter-prior (why it may still fail our bar): in the tail the
  price drift may offset funding more completely (the priced-carry null), fills
  are worse (pre-declared 20 bps/side), and the crash skew is plausibly MORE
  severe than the majors' −1.87 — exactly the failure mode that sank #8 on DSR.
- Point-in-time safety: identical to H2 — funding rates and prices are
  timestamped exchange records; the signal uses funding settled through t only;
  the universe uses trailing ADV through t; delisted contracts fall out when
  their data ends (no survivorship; the cache holds 729 contracts ever, incl.
  delisted).
- Exact config (frozen): PIT tail universe = symbols whose trailing-30d
  dollar-volume rank is in [31, 150] among contracts trading at t (min 20 names
  to form quartiles); signal = trailing-7d mean daily funding; label =
  funding-INCLUSIVE total return (mark_return − funding); book = dollar-neutral
  equal-weight quartiles (SHORT top-funding quartile, LONG bottom), weekly
  rebalance; costs = **20 bps/side** (5 taker + 15 spread, conservative for the
  tail; linear headline, sqrt impact is the capacity dimension); `--n-trials 10`.
  Registered paired control = cross-sectionally shuffled funding (must earn ~0).
  Reported: funding-income vs price-drift decomposition, skew, maxDD,
  ex-top-3-by-weight robustness, and a 2020–21 vs 2022+ subperiod split (the
  decay check). Machinery gate (synthetic planted_carry recovered / priced_carry
  rejected, paired) runs in-env immediately before the real run.
- Success criteria (frozen, SAME as H2 — no relaxation): right-signed IC
  t_NW ≤ −2 AND net SR > 0 AND DSR ≥ 0.95 at N=10 AND survives ex-top-3 AND
  |shuffled-funding control SR| < 0.3. The skew-aware DSR is the judge precisely
  because crash skew is this trade's known weakness ("Sharpe lies on negatively
  -skewed books").
- Kill criteria: machinery gate fails → ABORT, no trial spent; shuffled control
  earns (|SR| ≥ 0.3) → artifact, not carry; t_NW wrong-signed or > −2 → null;
  DSR < 0.95 → does not graduate (logged like #8, NOT relaxed). NO post-hoc
  rank-band scans, cost re-tuning, holding-period or signal-lookback variants —
  each is +1 trial and none is authorized by this registration.
- Failure interpretation: real-but-DSR-failing/crash-skewed → "the tail carry
  premium is real but the discipline refuses it too," McLean–Pontiff in a third
  cut; outright null → "the tail carry is fully priced / consumed by tail fills."
  Both are write-up sections; neither is hidden.

### H10: Opportunistic insider cluster-buying (Form 4) earns positive forward returns — the first EDGAR alternative-data candidate
- Status: **PROPOSED — config FROZEN 2026-06-25 (Stage-2 freeze block below).
  POWER-BLOCKED 2026-06-25: the pre-spend POWER GATE ABORTS on free data, so trial
  #13 was NOT run (N stays 12; see research_log 2026-06-25 row).** Full-universe
  power (computed via `BulkInsiderSource`): median 24 cluster-eligible firms/month →
  top-DECILE long basket ≈2 names, below the ≥5 floor; n_obs 197 passes. The
  registered top-decile book is underpowered on the large-cap survivorship-safe
  universe; clearing the floor by widening the (frozen) quantile would be forbidden
  salvage. A broader-basket design (long ALL cluster names vs EW) would be a NEW
  pre-registration, not a rescue of this one. First word stays PROPOSED for registry
  bookkeeping, but the in-harness POWER GATE is the hard backstop — a naive re-run
  self-aborts without spending N. (Originally: first word PROPOSED so the registry
  gate authorizes exactly one run; it flips to RUN the moment trial #13 is logged.) Running spends a trial and needs explicit owner sign-off, a Stage-1 data
  audit (Form 4 coverage + the price-side survivorship bound below) — now enforced
  in-harness by the POWER GATE, which aborts WITHOUT spending N if the realized
  cross-section is too thin — and the machinery gate. Originated
  from the post-#11 EDGAR alternative-data ideation — the first candidate to clear
  all five free-data screens since the price/crypto funnel went dry
  (`writeup/edge_candidates_2026-06-15.md` screened only price/crypto and missed
  EDGAR alt-data; this fills that gap).
- Economic prior (weak-to-moderate, declared): insiders trade on private
  information, and the INFORMATIVE subset is OPPORTUNISTIC trades — off the
  insider's routine calendar — not routine pre-scheduled ones (Cohen, Malloy &
  Pomorski 2012; Lakonishok & Lee 2001). A CLUSTER (≥k distinct insiders buying in
  a short window) is a stronger, lower-noise signal than a single trade.
  Counterparty: liquidity sellers and a market under-reacting to a
  disclosed-but-noisy signal. **Counter-prior (why it may fail our bar):** the
  effect is PUBLISHED → McLean–Pontiff decay (traded for a decade); it
  concentrates in small-caps (cost/liquidity); and Form 4s are public within 2
  business days, so any edge lives in under-reaction — exactly what arbitrage
  erodes.
- Why it clears the five screens (the reason to test it): (1) **Survivorship,
  signal side — safe:** Form 4 is filed under the issuer CIK and PERSISTS after a
  ticker dies/renames — the exact hole that blocks H1 and killed trials #1–7, on
  the signal side. (2) **Borrow:** the edge is a LONG signal (cluster BUYS) →
  long-tilt dodges the short-leg assassin. (3) **Cost mortality:** event-driven,
  low turnover (hold weeks–quarter). (4) **Implementability (#11 lesson):** the
  signal IS the filing — a discrete public event with its own timestamp, not a
  price-derived quantity — so it cannot be a bid-ask-bounce artifact; the entry-lag
  gate is still required. (5) **Capacity-honest:** small-cap concentration caps
  capacity; declared, not hidden.
- **CRITICAL honest caveat — the price-side survivorship hole is NOT closed, and
  for a LONG signal it is NOT conservative.** "Survivorship-safe by CIK" holds for
  the SIGNAL only. Realized returns still need price history, which free data drops
  for dead/bankrupt names. For a long-insider-BUY signal that gap is OPTIMISTIC
  (insiders who bought names that then went to zero are missing from the priceable
  universe → the long leg looks better than reality) — the OPPOSITE of H6's
  conservative direction. A graded H10 MUST therefore either (a) bound it with a
  synthetic terminal-return scenario on names that go unpriceable mid-window (reuse
  the trial-#2 delisting-return bound), or (b) use delisting-inclusive prices
  (CRSP). Stage-1 must measure how many cluster-buy names go unpriceable before any
  run. (This tempers the ideation's "survivorship-safe by CIK" claim, which is only
  half-true.)
- Point-in-time safety: Form 4 carries a transaction date AND an EDGAR
  acceptance/filed timestamp (filed within 2 business days of the trade); the
  signal at t uses only filings ACCEPTED at or before t. The routine-vs-
  opportunistic classifier uses each insider's trailing-12-month history through t
  only (Cohen–Malloy: "routine" = traded in the same calendar month for ≥3 prior
  years; opportunistic otherwise) — past-only.
- Exact config (to be FROZEN at Stage-2; sketch): universe = PIT S&P (or a broader
  liquid common-stock universe, with the price-survivorship bound stated); signal =
  trailing-W (≈90d) count/dollar of OPPORTUNISTIC open-market BUYS (Form 4 code P)
  by ≥2 distinct insiders, net of opportunistic sells, cross-sectionally ranked;
  label = forward 21–63d return; book = LONG-tilt (long top-decile cluster-buy;
  short leg only if borrow-feasible per H7 data, else long-vs-equal-weight); 10
  bps/side; sector/beta neutral.
- Machinery gate (MUST pass in-env before any run): a synthetic `planted_insider`
  world (planted post-cluster-buy forward drift) recovered, and a null world
  (Form-4 events with no forward drift) rejected, paired per-seed.
- Registered paired controls: (1) **routine-vs-opportunistic differential** —
  routine (calendar-scheduled) cluster buys must be MARKEDLY less informative than
  opportunistic ones (Cohen–Malloy's central result; if routine buys predict
  equally, the signal is generic buying pressure, not information); (2) label-
  shuffle ~0; (3) the price-side survivorship bound, reported as a paired ±scenario.
- Success criteria (frozen at Stage-2): right-signed forward-return IC with t_NW ≥
  +2 AND net SR > 0 beating equal-weight AND 12-1 momentum baselines AND DSR ≥ 0.95
  at then-current N AND survives the one-period entry-lag gate (SR does not collapse
  at +1 period — the #11 requirement, now a criterion not a diagnostic) AND the
  routine-vs-opportunistic differential is right-signed and material.
- Minimum detectable effect (MDE): to be stated at Stage-2 from
  `graduation_hurdle.py` for the realized n_obs. Event-time n_obs on a sparse
  cluster-buy set is modest → a HIGH hurdle; this must be checked FIRST — an
  underpowered event study is the H10-specific risk (the reason the power gate is
  mandatory).
- Kill criteria: machinery gate fails → ABORT; routine control predicts as well as
  opportunistic → generic-buying artifact, logged; entry-lag collapse →
  microstructure, logged like #11; price-survivorship bound flips the sign → the
  long leg was a survivorship mirage, logged. NO post-hoc W/k/horizon scans (each
  is +1 trial).
- Failure interpretation: a null or DSR-failing result is "the most-cited free
  alternative-data anomaly, tested with the same discipline, does not clear the bar
  on free data" — citable next to the carry and CEF trials.

#### H10 — STAGE-2 FROZEN CONFIG (2026-06-25, before any run; edits after this line are a NEW registration)

Every knob below is frozen to a SINGLE value (no ranges, no post-hoc scans — each
scan would be +1 trial). The harness `scripts/run_h10_trial.py` executes exactly
this; `tests/test_run_h10_trial.py` pins it. The run is **trial #13** → the DSR
uses **N = 13**.

- **Universe:** PIT S&P 500 membership at each rebalance date (the survivorship-safe
  `SurvivorshipSafeSECSource.universe()` — same source that ran trial #12). NO
  sector exclusion (insider buying has no CoGS pathology); sector-neutrality is
  handled in the SIGNAL, below. Rationale + honest caveat: the documented edge
  concentrates in SMALL-caps; testing on the large-cap survivorship-safe universe is
  the conservative-on-price-survivorship choice but is the POWER risk — the power
  gate adjudicates whether it is even testable here (it may legitimately ABORT, no N
  spent, logged as a free-data-limitation finding).
- **Rebalance grid:** MONTHLY, month-end as-of dates over the source's
  `start`..`end`. Monthly (not quarterly) maximizes n_obs → power; the 90d signal
  window overlaps across months but the LABEL is non-overlapping 1-month forward
  returns, so NW lags=1 is the correct overlap correction.
- **Signal (per date t × name):** trailing **W = 90 calendar days** count of
  **DISTINCT OPPORTUNISTIC open-market BUYERS** (Form 4 code P / acquired A,
  `filed_date ≤ t` and `> t−90d`), **NET of distinct opportunistic open-market
  SELLERS** (code S / disposed D, same window/PIT): `net = n_opp_buyers −
  n_opp_sellers`. Opportunistic/routine is per-owner-per-ticker, PAST-ONLY (Cohen–
  Malloy–Pomorski: routine = same calendar month ≥3 prior consecutive years; buys
  classified on prior buy history, sells on prior sell history). The signal is then
  **sector-demeaned** (GICS sector via `universe.sector_map`, current map — sectors
  are near-static, same convention H1 uses) and **cross-sectionally z-scored**
  (`_zscore_rows`). The **cluster gate k = 2**: a name is long-eligible only if
  `n_opp_buyers ≥ 2` at t (a single buyer is not a cluster).
- **Label / horizon:** forward **1-month** total return (t → t+1 month, the 21d end
  of the registered 21–63d range — non-overlapping at monthly cadence). Costs **10
  bps/side** on realized turnover. PERIODS_PER_YEAR = 12 (monthly book → sqrt(12)).
- **Book (PRIMARY, "long-vs-EW"):** LONG = equal-weight of names in the **top
  DECILE (quantile = 0.10)** of the sector-neutral signal that also pass the k≥2
  cluster gate; SHORT = equal-weight of the full priceable universe (the benchmark).
  Dollar-neutral by construction (long $1 vs short $1 of EW); beta ≈ 0 (both legs ≈
  market beta) — this is how "beta-neutral" is satisfied without a fitted beta. The
  active return is `mean(long-basket fwd) − mean(EW-universe fwd)`. (The registered
  "short leg only if borrow-feasible" branch is NOT taken — no borrow feed is wired,
  so the registered default long-vs-EW is used.)
- **Machinery gate (law #4, in-env, FIRST after registration):** the existing
  `insider.machinery_gate` — planted-opportunistic world recovered, null rejected,
  paired per seed; min paired differential > 0.5. ABORT if it fails.
- **POWER GATE (mandatory, runs on REAL data BEFORE the verdict, aborts WITHOUT
  spending N):** the H10-specific risk is an underpowered/thin event study. Frozen
  floors — ALL must hold or the run ABORTS (no trial, logged as
  "underpowered/insufficient coverage on free data"): (a) **n_obs ≥ 60** monthly
  periods with a non-empty long basket; (b) **median per-date long-basket size ≥ 5**
  names; (c) the realized **MDE** (net annual Sharpe clearing DSR ≥ 0.95 at N=13 for
  the realized n_obs, from `graduation_hurdle`/`expected_max_sharpe`) is **printed**
  beside the result. A power-gate ABORT is NOT a trial — it is "we cannot test this
  on free data," exactly the trial-#10 fee-first precedent.
- **Registered paired controls (all computed, all reported):**
  1. **Routine-vs-opportunistic differential** (CMP central result): the IDENTICAL
     book built on ROUTINE cluster buys must be MARKEDLY less informative. Frozen
     pass rule: opportunistic net SR > routine net SR **AND** routine |t_NW| < 2
     (routine carries no significant signal). If routine predicts as well →
     generic-buying artifact, FAIL, logged.
  2. **Label-shuffle placebo:** forward returns shuffled cross-sectionally within
     each date → |SR| < 0.3 (else leakage; STOP and hunt it).
  3. **Price-side survivorship bound (paired ±scenario, the load-bearing honest
     caveat):** long-basket names that go UNPRICEABLE mid-hold are assigned the
     trial-#2 delisting terminal return (**−30%** down-scenario) and the book SR
     recomputed. Report base vs down-bounded SR. **Kill rule:** if the −30% bound
     FLIPS the verdict (base passes, bounded fails on SR > 0 / beats-baselines), the
     long leg was a survivorship mirage → does NOT graduate, logged.
- **Entry-lag gate (CRITERION, not a diagnostic — the trial-#11 lesson promoted):**
  re-enter the book ONE month later (signal lagged +1 period). Frozen pass rule:
  lag-1 net SR ≥ **0.5 × lag-0 net SR AND lag-1 net SR > 0**. A collapse (à la H6
  1.11→0.10) means microstructure/under-reaction that vanishes on implementable
  entry → does NOT graduate, logged like #11.
- **PBO (CSCV):** over the honestly-available 4-config family
  {opportunistic, routine} × {lag-0, lag-1} net-return matrix (reuses the controls
  above — NO new knobs). `pbo.cscv_pbo`; require **PBO ≤ 0.5**. If the family cannot
  be assembled (any leg empty) the verdict is BLOCKED (never grade on a degenerate
  matrix — the B5 rule from H1).
- **Success criteria (graduate iff ALL hold on the OPPORTUNISTIC long-vs-EW arm):**
  (1) forward-return IC right-signed with **t_NW ≥ +2**; (2) **net SR > 0** and
  **> both** baselines (EW long-only, 12-1 momentum VW); (3) **DSR ≥ 0.95** at
  N=13; (4) **PBO ≤ 0.5** on the 4-config family; (5) **entry-lag gate** passes;
  (6) **routine-vs-opportunistic differential** right-signed and material;
  (7) **price-survivorship −30% bound does not flip** the sign. Any one failing →
  does NOT graduate; the row is logged whatever it says, N becomes 13.
- **MDE (stated before the run):** computed in-harness from the realized n_obs
  (`expected_max_sharpe` at N=13) and printed beside the result; given a sparse
  large-cap cluster-buy cross-section the hurdle is expected to be HIGH — the power
  gate (n_obs ≥ 60, basket ≥ 5) is the front-line check that the realized design can
  even clear it.
- **Adjudication order (no trial spent until the DATA GATE passes):** registration
  gate (PROPOSED) → machinery gate → DATA GATE (survivorship-safe source required)
  → assemble real panels → POWER GATE (abort-without-N if thin) → opportunistic arm
  + routine arm + lag-1 arms → controls (shuffle, survivorship bound) → PBO/MDE →
  7-gate verdict. Does NOT auto-bump N, does NOT auto-log (the row is added by hand
  after sign-off, same as trial #12).

### H11: 13F institutional-ownership change (crowding) is cross-sectionally informative — secondary EDGAR candidate, WEAK prior
- Status: **PROPOSED (draft 2026-06-16) — DRAFT ONLY, not run.** The weaker sibling
  of H10; registered for completeness of the EDGAR alt-data sweep.
- Economic prior (WEAK, declared): a change in aggregate institutional holdings (Δ
  13F) may carry information OR signal CROWDING that precedes reversal — the two
  point in OPPOSITE directions, which is a reason for skepticism, not confidence.
- **Why it is the weaker bet (counter-prior, up front):** 13F is filed up to **45
  days after quarter-end**, is LONG-ONLY (no shorts, no intra-quarter timing), and
  is window-dressing-prone — so the signal is stale by construction and heavily
  studied. The 45-day lag MUST be hard-enforced (a 13F enters only on its filing
  date, never quarter-end), which alone may kill any edge.
- Survivorship: signal side safe (13F filed under filer CIK, persists); the SAME
  price-side survivorship caveat as H10 applies (and is not conservative for a long
  tilt).
- Point-in-time safety: a quarter-end 13F enters the signal only on its EDGAR
  filing date (≤45d lag), never the period it reports.
- Exact config (to be frozen at Stage-2): Δ institutional ownership
  breadth/concentration, cross-sectionally ranked, long-tilt, slow rebalance, 45-day
  filing lag hard-enforced; SAME machinery gate, paired controls, MDE, and entry-lag
  discipline as H10.
- Success / kill / failure: SAME bar as H10, no relaxation. Given the weak prior,
  the expected and most-citable outcome is a clean null — "free 13F crowding,
  lagged honestly, carries no tradable cross-sectional edge."

### H12: Broader-basket opportunistic insider cluster-buying — long ALL cluster names vs EW (the POWERED redesign of H10 after its top-decile book power-aborted)
- Status: **RUN (trial #13, 2026-06-25) — does NOT graduate (clean NULL).** Opportunistic
  long-all-cluster arm net SR −0.131 (t_NW −0.58, DSR 0.013); ALL 7 gates fail; the
  opportunistic arm is WORSE than routine (no CMP premium). Machinery gate AND power
  gate both PASSED (n_obs 197, median basket 23, MDE 0.42) → genuine economic absence,
  not underpowering. N=13. See research_log 2026-06-25 trial-#13 row. (Was: PROPOSED —
  config FROZEN 2026-06-25, awaiting owner sign-off; first word PROPOSED so the
  registry authorized exactly one run; it flips to RUN now that the trial is logged.)
- **WHY THIS IS NOT SALVAGE (load-bearing integrity statement, read first).** H10's
  frozen TOP-DECILE book POWER-ABORTED pre-spend — it produced NO result and spent
  NO trial (research_log 2026-06-25; `results/h10_power_bulk.json`). The full-universe
  power probe computed ONLY Form 4 buy-COUNTS; **NO forward returns were ever touched.**
  The single design change here (long ALL ~24 cluster names instead of the top decile
  of ~2) is dictated by buy-DENSITY — a POWER requirement — not by any realized P&L.
  This is the trial-#4 discipline applied honestly: a redesign motivated by power is a
  FRESH pre-registration, run once, logged whatever it says — not a relaxation of a
  failed result's criteria (H10 had no result to relax). Criteria below are frozen
  anew at the SAME bar as H10, zero relaxation.
- Economic prior: identical to H10 (Cohen–Malloy–Pomorski 2012 opportunistic
  insiders; off-routine cluster BUYS carry private information), expressed as a broad
  equal-weight tilt toward EVERY firm with an active opportunistic buy-cluster rather
  than the (too-thin, ~2-name) extreme decile. Same weak-to-moderate, published →
  decay counter-prior as H10.
- Data: **SEC BULK insider data set via `BulkInsiderSource`** (cross-checked
  byte-identical to the raw-XML crawl on the recent window, and COMPLETE where the
  crawl is not — `InsiderSource` is recent-only/incomplete for prolific filers,
  verified 2026-06-25, so the crawl must NOT be used). Prices =
  `SurvivorshipSafeSECSource` (Tiingo, delisting-inclusive). Universe = PIT S&P 500.
- Point-in-time safety: identical to H10 — signal keyed on Form 4 FILING date
  (filed_date ≤ t), opportunistic/routine label past-only (same calendar month ≥3
  prior consecutive years = routine), forward label strictly t→t+1.
- **Exact config (FROZEN, single values — no scans):**
  - Universe: PIT S&P 500; monthly month-end rebalance over the source start..end.
  - Cluster gate: a name is LONG-eligible at t iff it has **≥ 2 distinct OPPORTUNISTIC
    open-market buyers** (Form 4 code P / acquired A) with `filed_date ∈ (t−90d, t]`,
    past-only CMP classification. (NO top-decile selection — ALL eligible are longed.)
  - Book (PRIMARY, long-vs-EW): **LONG equal-weight ALL eligible names; SHORT
    equal-weight the full priceable universe.** Dollar-neutral, β≈0. Forward
    **1-month** total return, **10 bps/side**, PERIODS_PER_YEAR=12. (Implementation:
    `insider.long_vs_ew_weights` with quantile = **1.0** — the full eligible set.)
- Machinery gate (law #4, in-env, FIRST): the existing `insider.machinery_gate`
  (planted-opportunistic recovered / null rejected, paired); ABORT if it fails.
- POWER GATE (same frozen floors as H10: n_obs ≥ 60 months, median long basket ≥ 5):
  EXPECTED TO PASS — the probe measured median ~24 cluster-eligible firms/month over
  197 months. Still enforced in-harness (abort-without-N if somehow thin).
- Registered paired controls (all computed, all reported) — SAME as H10:
  1. **Routine-vs-opportunistic differential:** the IDENTICAL long-all-cluster book
     built on ROUTINE clusters must be materially weaker — opp net SR > routine net SR
     AND routine |t_NW| < 2. Else generic-buying artifact → FAIL, logged.
  2. **Label-shuffle placebo:** forward returns shuffled cross-sectionally per date →
     |SR| < 0.3 (else leakage; STOP and hunt it).
  3. **Price-survivorship −30% bound:** long-basket names that go unpriceable mid-hold
     get the trial-#2 −30% terminal return; if the bound FLIPS the verdict, the long
     leg was a survivorship mirage → does NOT graduate, logged.
- Entry-lag gate (CRITERION): re-enter one month later; lag-1 net SR ≥ 0.5 × lag-0
  net SR AND lag-1 net SR > 0. Collapse → microstructure, does NOT graduate (#11).
- PBO (CSCV): 4-config {opportunistic, routine} × {lag-0, lag-1}; require ≤ 0.5;
  BLOCKED (no grade) if the family can't be assembled (B5).
- **Success criteria (graduate iff ALL hold on the OPPORTUNISTIC long-all-cluster
  arm), SAME bar as H10:** (1) IC right-signed, t_NW ≥ +2; (2) net SR > 0 AND > both
  baselines (EW long-only, 12-1 momentum); (3) DSR ≥ 0.95 at **N=13**; (4) PBO ≤ 0.5;
  (5) entry-lag gate passes; (6) routine differential right-signed and material;
  (7) −30% survivorship bound does not flip. Any one failing → does NOT graduate; the
  row is logged whatever it says, N becomes 13.
- **MDE (stated before the run):** from the power probe, n_obs ≈ 197 monthly periods →
  MDE @ N=13 ≈ **0.42** net annual SR. This LOW hurdle (a long 15-yr monthly sample)
  is the redesign's whole advantage over H10's thin decile book — H12 is genuinely
  powered to clear DSR if a real edge exists.
- Kill criteria: machinery gate fails → ABORT; power gate fails → ABORT (no N);
  routine predicts as well → generic-buying artifact, logged; entry-lag collapse →
  microstructure, logged; survivorship bound flips → mirage, logged. **NO post-hoc
  W/k/horizon/basket-width scans — each is a NEW +1 trial.**
- Failure interpretation: a null is "the most-cited free insider-cluster anomaly,
  tested BROADLY (not just the extreme tail) and honestly on free survivorship-safe
  data, carries no tradable cross-sectional edge net of costs" — citable beside the
  carry and CEF trials.

### H13: Post-earnings-announcement drift (PEAD) survives on large-cap US, net of costs — the first INSTITUTIONAL-DATA hypothesis (needs PIT analyst estimates)
- Status: **PROPOSED — config FROZEN 2026-06-26; cleared to run as graded trial
  #14 on owner sign-off.** Graded path = **FMP `epsEstimated`** (free quasi-PIT
  consensus), validated by a fail-closed leakage gate AND an Alpha Vantage
  cross-source PIT check (see PIT-safety below); the bounded Bloomberg/IBES pull
  in `writeup/bloomberg_pead_pull.md` and IBES-via-WRDS remain the gold-standard
  UPGRADE path (re-run on PIT data when access lands). First hypothesis that steps
  past free *price* data onto estimates — the §10 research-note frontier
  ("post-earnings drift with real estimates") made concrete on a credibility-gated
  free feed.
- Economic prior (moderate, declared): markets UNDER-REACT to earnings surprises;
  prices drift in the surprise direction for weeks after the announcement
  (Bernard–Thomas 1989). The load-bearing input is the surprise relative to the
  **pre-announcement consensus** — which free data lacks (free sources carry the
  actual, not the at-the-time consensus); Bloomberg/IBES PIT estimates are the
  unlock. Counterparty: investors slow to fully price the news. **Counter-prior
  (why it may fail our bar):** PEAD is among the most-published anomalies →
  McLean–Pontiff decay; it concentrates in SMALL/illiquid/high-cost names, so on
  large-cap S&P 500 net of costs the honest prior is a DECAYED, possibly-marginal
  effect — exactly the project's recurring story, now on the canonical estimates
  anomaly.
- Point-in-time safety: (1) the surprise uses **FMP `epsEstimated`** as the
  pre-print consensus. It is FREE quasi-PIT, so it is gated, not trusted: a
  fail-closed leakage validator (`validate_pead_events`) PASSES (exact est==actual
  11.6% < 20%, two-signed surprise std 0.99, zero future-dated-with-actual), AND a
  cross-source check vs Alpha Vantage on 15 large caps (~1,533 overlapping events)
  shows the **backfill failure mode is ruled out**: FMP's estimate sits a median
  **0.0003 from AV's estimate vs 0.0400 from AV's actual** (90% closer to the
  estimate on surprise quarters; FMP-vs-AV surprise correlation 0.76), and FMP's
  own internal surprise GROWS with the realized surprise — the opposite of a feed
  backfilled to the actual. **HEADLINE residual risk (declared):** FMP and AV are
  NOT independent — ~49% of estimates and ~70% of actuals are byte-identical, so
  they almost certainly resell a common upstream archive; the cross-check confirms
  CONSISTENCY, not point-in-time-ness. The one PIT failure mode that survives is a
  shared, post-hoc-REVISED (but not realized) consensus, which only a true PIT
  vendor (IBES/WRDS/Bloomberg) can exclude — hence the gold-standard re-run is the
  declared upgrade. (2) The signal enters only AFTER the announcement — **entry at
  T+2** (skip the announcement-day jump so we measure DRIFT, not the immediate
  reaction, and avoid same-bar leakage); verified by a single-bar-jump leakage
  sweep (the book earns zero on the announcement/entry bars, only from the first
  strictly-post-entry return). (3) Forward returns are strictly post-T+2. (4)
  Universe caveat, declared: a CURRENT-S&P-500 first cut carries mild universe
  survivorship — far less corrosive for an EVENT study than for a buy-the-dip (each
  event is dated and local), but a PIT-member pull is the clean upgrade and is
  noted as the residual.
- Exact config (**FROZEN 2026-06-26, before forward returns**): universe = CURRENT
  S&P 500 members (`--universe current`), pulled via FMP subject to the free-tier
  quota — the graded run uses every name successfully fetched, with COVERAGE
  reported (names, events, history span); signal = **SUE** = scale-invariant
  relative surprise **(actual − est)/|est|** (FMP carries no analyst dispersion, so
  the `std_est` branch is unused and the `surprise_pct`/`rel_est` fallback is the
  sole path — confirmed 100% of events; this makes SUE invariant to corporate-action
  split factors, which therefore cancel and cannot leak split-level noise into the
  ranked signal); **DATA-QUALITY FILTER (frozen, declared):** drop announcements
  with **|est_eps| ≤ $0.10** (`pead.drop_low_eps`) — 2-decimal EPS rounding
  quantizes the relative surprise on sub-$0.10-EPS quarters (a 1¢ rounding on a 3¢
  estimate is a 33% swing), which the PIT cross-check found drives ~42% (NVDA) /
  ~35% (WMT) FMP-vs-AV SUE sign-flips on low-price post-split names; dropped count
  is reported. This is a one-time pre-registered cleaning step, **NOT** a tunable
  knob (no post-hoc threshold scan). **event study**: enter T+2, hold 60 trading
  days, LONG top-quintile SUE / SHORT bottom-quintile within each announcement-month
  cohort, plus a cross-sectional monthly variant (rank on the most-recent surprise
  within a trailing 90d window); prices = Tiingo (delisting-inclusive); 10 bps/side;
  size/liquidity reported by tercile (PEAD's small-cap concentration is the headline
  risk). **The leakage gate MUST re-PASS on the actual graded universe** (not just
  the 15-name cross-check sample) before N is spent.
- Machinery gate (MUST pass in-env first): a synthetic `planted_pead` world
  (forward drift injected ONLY after high-|SUE| events) recovered, and a null world
  (same surprise events, no drift) rejected, paired per seed.
- Registered paired controls: (1) **drift-vs-reaction (the PEAD-specific kill):**
  re-enter at T+5 and T+10 — true drift persists for weeks, so the Sharpe must
  retain ≥ 50% from T+2 to T+5; a collapse means we captured the announcement-day
  REACTION (untradable, the trial-#11 entry-lag lesson applied to events), not
  drift. (2) **surprise-shuffle placebo** → ~0. (3) **size-tercile split** — if the
  effect lives ONLY in the smallest/least-liquid tercile, it is not tradable on the
  large-cap universe (logged, not hidden).
- Success criteria (frozen at the data step): right-signed SUE→forward-return IC
  with t_NW ≥ +2 AND net SR > 0 beating equal-weight AND 12-1 momentum AND DSR ≥
  0.95 at then-current N (=14) AND the drift-vs-reaction gate passes (T+5 retains
  ≥50%) AND the effect is not confined to the illiquid tercile.
- Minimum detectable effect (MDE): stated at the data step from `graduation_hurdle`
  for the realized n_obs (event-time obs ≈ # usable announcements; cross-sectional ≈
  months). Large-cap PEAD post-cost is expected near the MDE — checked before the run.
- Kill criteria: machinery gate fails → ABORT; drift collapses by T+5 → it was the
  announcement reaction, logged; effect only in the illiquid tercile → not tradable
  on S&P 500, logged; DSR fail → decayed-PEAD null. NO post-hoc SUE-definition /
  horizon / entry-lag scans (each is +1 trial).
- Failure interpretation: a null is "PEAD, tested with real point-in-time consensus
  estimates on large-cap US net of costs, has decayed below tradability" — extends
  the project's decay story from free-data price/fundamental/alt-data signals to the
  canonical estimates-dependent anomaly, and demonstrates institutional-data fluency.

### H14: Construction variant — the live-vol gap is a construction artifact (cadence + index-beta overlay), not a signal property
- Status: **PROPOSED — frozen 2026-08-06.** Owner sign-off required; a graded
  run spends one trial at then-current N. No new signal anywhere in this
  registration.
- Motivation (diagnosis first, per §5.0 discipline): the committed live record
  shows the INTENDED book (logged weights at public marks) running **20.3%**
  ann vol with realized SPY beta **+0.53**, ex-ante w·β_SPY **positive on 38/38
  cycles** (mean +0.31), and one-way turnover **42×/yr** — while the deployed
  config's own backtest (`results/metrics_sp500_ridge_both_residlabel.json`)
  ran **5.6%** vol at **3.46×/yr**. Execution is ruled out (cum broker-vs-
  intended gap −1.23%; the 2026-07-23 mass short-rejection left a 4.5 bp mark).
  Full decomposition: `results/live_beta_diagnosis.json` +
  `writeup/live_vol_diagnosis_2026-08-06.md`.
- Hypothesis: BOTH live deviations are artifacts of the live layer's
  construction choices, not of the ridge signal: rebuilding the SAME
  walk-forward predictions into (a) the backtest's 21-trading-day rebalance
  cadence and (b) an at-rebalance SPY overlay that zeroes w·β_SPY restores the
  backtest's realized vol and removes the systematic index tilt, without
  degrading the (null) net SR.
- Exact config (FROZEN): identical pipeline to the deployed config
  (PIT S&P 500, 2010→2026 walk-forward, ridge, residualized labels, h=21,
  decile quantile 0.10, sector demean + EW-member-mean beta projection,
  10 bps/side) — three arms through the identical cost-aware backtester on
  identical dates, registered as ONE construction trial:
  - **A0** as-live: decile book refreshed every trading day (the live layer's
    `rebalance_every=1` behavior, reproduced in-backtest);
  - **A1** cadence: identical but rebalanced every 21 trading days (the
    deployed backtest's cadence);
  - **A2** cadence+overlay: A1 plus a SPY position of −(w·β_SPY) set at each
    rebalance, β from 252d rolling (min 126) daily regressions, overlay traded
    at 1 bp/side (index ETF spread, declared), book legs at 10 bps/side.
- Machinery gate (MUST pass in-env before the graded run): a synthetic world
  with a planted cross-sectional alpha PLUS a planted common index loading —
  the overlay must remove the loading (realized book β ≈ 0) while preserving
  the planted alpha; a noise world with the same loading — the overlay must
  not manufacture SR (net SR ~ 0). Paired per seed.
- Success criteria (all three, else the hypothesis FAILS):
  1. A2 realized ann vol ∈ **[0.75, 1.5] ×** the deployed backtest's 0.0559
     (i.e. the vol gap closes to within band);
  2. A2 realized rolling-63d SPY beta: **|mean| ≤ 0.05 and p95|β| ≤ 0.15**
     (graduation criterion 4's bounds, met in-backtest);
  3. A2 net SR ≥ (deployed backtest net SR − 0.15) — the fix must not buy
     neutrality by degrading the return profile beyond noise.
  A0 is the attribution control: A0-vs-A1 isolates cadence, A1-vs-A2 isolates
  the overlay. Turnover and cost drag reported per arm.
- Power/MDE: the registered claims are vol/beta CLOSURE bounds on ~16y of
  daily data (n ≈ 4,000), not an alpha detection — the binding hurdle is the
  pre-stated bands, computed trivially at that n. DSR does not gate a
  null-signal construction check; the trial still increments N (real-data
  evaluation), and the SR floor in criterion 3 is the anti-degradation guard.
- Kill criteria: machinery gate fails → ABORT (no N spent). Criterion 1 fails
  → the vol is a signal/period property, NOT construction — the live equity
  record must then be reinterpreted, logged as such. Criterion 2 fails → the
  EW-vs-SPY beta reference is not the tilt's source; register nothing further
  until re-diagnosed. NO post-hoc cadence/window/hedge-ratio scans (each is
  +1 trial).
- Scope guard (directive: the live config is frozen mid-experiment): this
  trial changes NOTHING deployed. If it passes, whether to deploy A2 to the
  live arm (resetting what the live equity record measures) is a SEPARATE
  owner decision registered at that time. The live-IC arm is unaffected by
  construction either way.

---

Run log: H2 RUN as trial #8 (2026-06-13, criteria not met). H8 RUN as trial #9
(2026-06-13, clean null). H9 RUN as trial #10 (2026-06-14, clean null — tail
carry priced). H6 RUN as trial #11 (2026-06-15, criteria nominally met but
OVERTURNED by entry-lag diagnostic — microstructure reversal artifact, does not
graduate). **N = 11.** Each run requires owner sign-off, increments N by
exactly 1, and is logged in `research_log.md` regardless of outcome. H5/H7 are
collection-only/two-stage and exempt from the run/N language until their Stage-2
analysis is registered. **H10 (Form 4 opportunistic insider clusters) and H11
(13F crowding) are PROPOSED drafts (2026-06-16), DRAFT-ONLY** — the first EDGAR
alternative-data candidates; neither has run, both await a Stage-1 data audit +
owner sign-off, and **MDE (minimum detectable effect) is now a mandatory pre-run
field for every registration** (the protocol note above).
