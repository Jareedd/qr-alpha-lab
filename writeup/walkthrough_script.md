# Five-minute walkthrough — script and evidence pack

**Status: script only. The recording is yours to make** — this file cannot
attest to what you personally wrote, and the section on AI assistance is
deliberately left for you to complete truthfully (see §4). A walkthrough whose
authorship claims were drafted by an assistant would undercut the exact thing
it is meant to demonstrate.

Format: screen recording with voice-over. Five minutes is roughly **750 spoken
words**, so this is tight — the timings below are the budget, not a suggestion.
Rehearse once against a timer; if you run over, cut §5 before cutting §1.

Rule for the whole recording: **every claim on screen must be a file you can
open.** Do not show a slide of numbers. Show the JSON.

---

## 0:00–0:30 — What this is, and the one sentence that matters

> "This is a cross-sectional equity alpha research pipeline. Thirteen logged
> trials, three asset classes, **zero strategies graduated to real capital.**
> That's not the project failing — that's the project working. What I'm
> showing you is a research process that kills its own best-looking numbers,
> and the infrastructure that makes that possible."

*On screen:* `README.md` top, then scroll to the thirteen-trial paragraph.

**Do not oversell.** The strongest opening you have is the zero-graduation
record stated flatly. An interviewer who hears "no edge found, here's why I
believe that" leans in; one who hears "Sharpe 4.6" starts looking for the bug.

## 0:30–1:15 — Why this validation design

*Run live:* `python scripts/reproduce.py` (about 20 seconds — talk over it).

Hit these four points, in this order:

1. **Walk-forward with an embargo, never k-fold.** Labels are 21-day forward
   returns, so consecutive daily observations overlap by 20 of 21 days. K-fold
   puts a test row's future inside a training row's window — the model learns
   the answer. Expanding windows plus a 21-day embargo (≥ the label horizon)
   is the fix. *This is the single most common fatal flaw in student projects,
   and being able to derive why is worth more than any Sharpe.*
2. **Deflated Sharpe with an honest trial count.** The DSR benchmarks your
   Sharpe against the expected maximum of N noise draws. N is 13 here, tracked
   in `research_log.md` and never reset. Tracking N is the discipline; the
   formula is the easy part.
3. **Costs are not a footnote.** 10 bps per side on every unit of turnover,
   and turnover is a headline metric — most published anomalies die here
   (Novy-Marx & Velikov).
4. **Falsification before belief.** The pipeline must recover a planted signal
   *and* reject pure noise before any real-data number is interpretable.

*On screen:* the reproduce output landing on `RESULT: REPRODUCED`.

> "Thirteen numbers, stated tolerances, twenty seconds, no API keys. The
> tolerances are documented with their reasoning in `REPRODUCE.md` — the
> observed cross-machine drift on this project is about 1e-14, and the
> tolerances sit ten orders of magnitude above that."

## 1:15–2:15 — The case study: what assumptions can invalidate the result

*On screen:* `results/case_study_survivorship.png`, then the two metrics JSONs.

> "My first real-data run gave a net Sharpe of 0.82. Two things stopped me
> from believing it. First, equal-weight buy-and-hold on the same universe
> earned a Sharpe of 1.00 — a universe where owning everything beats the
> strategy isn't a test, it's a universe that was selected for going up.
> Second, the t-stat I'd computed was 7.77 using a standard error that assumes
> independent observations, and my daily ICs overlap 20 days out of 21. Under
> Newey–West the same data gives 2.22."
>
> "So I rebuilt the universe point-in-time — 810 members across the window
> instead of today's 60 survivors — and held everything else fixed. Net Sharpe
> 0.82 to −0.01. IC 0.033 to 0.005. t-stat 0.54. **The entire edge was
> hindsight in the universe selection.** That's McLean and Pontiff reproduced
> in-house on my own result."

Then the part that separates you from someone reciting a lesson:

> "And the falsification gate was **green the whole time**. It couldn't have
> caught this — my synthetic panels have no delistings, so survivorship bias
> is invisible to them by construction. A green gate means the plumbing didn't
> invent the result. It does not mean the result is true."

**Assumptions that can still invalidate what's left**, named without prompting:
149 of 810 members are unpriceable on free data (18%, and they're
disproportionately the dead ones); delisting returns are absent and only
*bounded* by a scenario run, never imputed; Wikipedia is not a point-in-time
vendor; sector labels are as-of-today; one market, one sixteen-year regime.

## 2:15–3:15 — What broke, and how I diagnosed it

Pick **one** and tell it properly. The best one is the stale cache, because
the failure mode was silence:

> "My live monitoring report said 'measurable cycles: 0 of 37' when about 17
> should have matured. Nothing crashed. The number was just quietly zero."
>
> "I traced it to the cache key. When you ask the loader for data through
> 'latest', it keys the cache file on the literal string `latest` and then
> returns any existing file unconditionally. CI restores the cache between
> runs — so the report's price panel had been frozen at one June date for six
> weeks, and every maturity-dependent number truncated to zero while looking
> perfectly healthy. Trading was never affected; it uses per-day snapshot
> directories. Only the *measurement* was broken."
>
> "The lesson I took: 'measure your vendor, don't trust it' applies to your own
> cache layer too. The first fix was monitor-side plus a staleness alarm
> written into the committed report. The real fix was at the source — an
> open-ended window now refetches once its cache goes stale, while a closed
> window, which is a frozen question, is cached forever."

*On screen:* `research_log.md` 2026-08-06 entry, then
`tests/test_data_loader.py::test_open_ended_cache_older_than_the_window_is_refetched`.

> "Thirty-two tests now pin that module's behaviour under vendor failure —
> every download empty, partial downloads, duplicate timestamps, corrupt cache
> files. Before this week it had none."

**Backup stories** if asked for another: an equal-weight Sharpe of 3.3 that was
impossible on its face and exposed pad-filled phantom returns for delisted
names; a Binance API that switched kline timestamps to microseconds mid-2025
and overflowed dates to the year 56971; a fundamentals run that returned
market-cap coverage of exactly zero because shares outstanding live under
SEC's `dei` namespace, not `us-gaap`.

## 3:15–4:15 — Operating it, not just running it

*Run live:* `python scripts/ingest_demo.py`, and narrate steps 6 through 9.

> "This is a price store built around one constraint: a run that dies halfway
> has to leave a state the next run can safely continue from. Watch — I kill
> the run at symbol three of six."
>
> "The store is now **partial but consistent**. Partitions are written
> atomically and the manifest is derived from them, so a crash gives you a
> smaller store, never a corrupt one. The resume plan comes off the run report
> that was committed to disk in a `finally` block — not out of my memory."
>
> "Resume the four pending symbols, and the result is byte-identical to a
> clean single-pass run. Same store also does incremental updates from a
> per-symbol watermark, backfills that extend history earlier without
> rewriting what's there, and schema validation that quarantines a poisoned
> batch — a negative price — while leaving the stored history untouched."

*On screen:* the `RESULT` block of the transcript. Mention it runs in CI.

## 4:15–4:45 — What I'd change with ten times the data

Answer in terms of what *breaks first*, not what you'd buy:

1. **The universe reconstruction breaks first.** Wikipedia scraping does not
   survive 10× the names or the history — that becomes a licensed
   point-in-time membership and security-master file (CRSP/Compustat), which
   also fixes the 18% unpriceable names and the as-of-today sector labels.
2. **The store's shape changes, its semantics don't.** One parquet per symbol
   is right at hundreds and wrong at hundreds of thousands: partition by
   symbol *and* date, and the manifest stops being a JSON file you rewrite
   each run. The watermark/atomicity/quarantine/resume design is what carries
   over.
3. **Costs stop being a constant.** 10 bps per side is defensible at this
   scale; at 10× the universe the tail is illiquid and the square-root impact
   model has to move out of the `--capacity` sweep and into every backtest.
4. **More data does not buy more trials.** With 10× the cross-section I'd get
   a genuinely wider universe and shorter effective horizons — but the DSR
   penalty grows with N regardless, so the discipline of pre-registering a
   hypothesis before spending a trial matters *more*, not less.
5. **Statistical power is the real prize** — a wider cross-section raises the
   minimum detectable effect I can rule out. The honest framing is that 10×
   the data would let me state sharper *nulls*, not that it would find alpha.

## 4:45–5:00 — Close

> "The product here is the research process, not the alpha. Everything I've
> shown you reproduces from a clone in about ten minutes, and every number in
> the write-up traces to a committed artifact and a dated row in the research
> log."

---

## §4 — Which parts I built or adapted, including AI assistance

**You must write this section yourself.** It is the one answer nobody can
draft for you, and it is the one an interviewer will probe hardest. What this
file can offer is the shape of a good answer and the questions to answer
honestly:

- Which modules did you write from scratch, and which did you adapt from a
  paper, a textbook (López de Prado ch. 7 for purging/embargo), or a blog?
- Where did you use an AI assistant, and for what — scaffolding, test writing,
  refactoring, documentation, debugging? Be specific and be comfortable with
  the answer.
- **What did you reject or rewrite that the assistant produced, and why?**
  This is the strongest thing you can say. Being able to name a specific
  suggestion you refused — and the reasoning — is direct evidence you were
  driving.
- Which design decisions are *yours*: the pre-registration gate, the
  zero-graduation discipline, the choice to keep the biased universe as a
  comparison, the decision not to relax the DSR bar for trial #8.

The standard to hold yourself to, in one line:

> AI assistance does not invalidate the project. **Being unable to explain or
> repair the output does.**

A useful self-test before recording: open any file in `src/quantlab/` at
random and explain, out loud, what it assumes and how it would fail. If you
cannot do that for a module, you do not yet own it — read it, break it in a
test, and re-record.

---

## Recording notes

- **Terminal, not slides.** Live commands. `reproduce.py` (20 s) and
  `ingest_demo.py` (< 5 s) both fit inside their segments.
- Set the font large enough to read on a phone. Dark-on-light survives
  compression better.
- Do a dry run of the two live commands immediately before recording — a
  failed command on camera is fine if you diagnose it, and a disaster if you
  panic.
- Do not read this script verbatim. The numbers are load-bearing; the phrasing
  is not.
- Anything you cannot defend under follow-up questions, cut. A four-minute
  video you fully own beats a five-minute one you don't.
