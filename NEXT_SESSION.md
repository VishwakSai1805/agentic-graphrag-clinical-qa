# Next session — start here

**Done:** Phases 0–5, all committed to `master`. Phase 3 `b10c385`/`d445423`,
Phase 4 `e75de9c`/`6f9690e`/`1388904`, ADR-025 conflict fix `30700f9`/`ccba531`,
2026-09-07 audit remediation `a714fe0`, Phase 5 evidential calibration
`a0060cf`, Phase 5 temperature-scaling ablation `b9bcda9`. 420 tests pass, ruff clean.

**2026-09-08 small-budget session (code only, no new phase — Phase 6 deliberately
deferred):** four AUDIT items landed as code, plus an overnight script to run
them without spending a session. **Before doing anything else, check whether
`scripts/run_overnight.ps1` has been run** (`results/phase1_nfcorpus/`,
`results/phase1_retrieval_*/`, `results/medqa_B2eq_*/`,
`results/pubmedqa_B*_*/` existing means it has) and read its logs under
`logs/` before trusting anything below that depends on it:

1. **§1.1/§1.2 — Phase 1 traceability, BLOCKER.** `scripts/analysis/
   nfcorpus_significance.py` and `scripts/eval_retrieval.py` now write
   `results/phase1_nfcorpus/metrics.json` and
   `results/phase1_retrieval_<dataset>/metrics.json` per mode instead of
   stdout-only. `docs/04_EVALUATION.md` §5.2 is re-pointed at both paths and
   says plainly its numbers are pre-fix, not yet re-verified — **do that
   re-check first** once the overnight run has produced the artifacts.
2. **§5.4 — PubMedQA wired into `run_baselines.py --dataset`.** Was
   `{medqa, medmcqa}` only. 3-way yes/no/maybe handled in `option_letters()`
   and `gold_option()`; `src/eval/baselines.py`'s prompt/parse/verify logic
   needed no changes (already dict-driven over whatever `options` it's given).
3. **§4.3 — equal-budget B2, `--equal-budget` flag.** Trims B1/B2/B3's bundle
   to `agents.max_evidence` via the exact modality-reserved-slot algorithm
   V4/V5 use (`src.agents.evidence_select.select_evidence`, reused directly).
   Default off, so no committed run's numbers changed. **Once the overnight
   `medqa_B2eq_*` run exists, that's what §5.5's "V4 0.35 vs B2 0.65" gap may
   finally be read against — not before.**
4. `scripts/run_overnight.ps1` — sequential, logs to `logs/`, one PowerShell
   invocation, continues past a failed job rather than aborting the rest,
   prints a pass/fail summary table at the end. Queues: the 4 Phase 1
   regeneration jobs, the equal-budget B2 dev-20 run, and a PubMedQA
   B0–B3 smoke run (`--split dev --limit 20`).

All four items are code-only per the session's explicit scope — none of the
three overnight jobs has been run yet, and none of the numbers they'll
produce exist anywhere in the docs yet. Don't infer results from the code
changes; read the actual `results/*/metrics.json` the overnight run writes.

---

## Phase 5 result (ADR-028) — exit test passes literally; the ablation that matters says C3 needs restating

Fit α,β,γ,δ,T on a 60-question MedQA train slice, measured ECE on a
**disjoint** 60-question held-out slice (both disjoint from Phase 3/4's
dev-20 — `load_question_sample_excluding`). Compared against a REAL
per-token-logprob baseline (`src.llm.client.chat(logprobs=True)`, same
EvidenceBundle as the agent pipeline), with a paired-bootstrap CI on the ECE
difference (ADR-012 discipline, not a point estimate):

- Calibrated ECE 0.0724 vs. raw-logprob-at-T=1 ECE 0.5418 — 95% CI on the
  diff [−0.5426, −0.2826], entirely below 0. Passes the literal exit test.
- **But a temperature-scaled-raw-logprob ablation (no new LLM calls) shows
  this gap is almost entirely a temperature effect: fit a single T on the
  fit-half's saved logprobs (T≈92.03), apply to held-out → ECE 0.0725,
  statistically tied with calibrated's 0.0724 (95% CI [−0.1344, +0.0887]).**
  **C3 is restated: temperature scaling alone matches the evidential
  features' ECE at this sample size; the features' additional value beyond
  temperature scaling is not established.** A real bug surfaced fitting T: an
  initial `t_max=20` search bound silently saturated and reported a
  spuriously bad T-scaled ECE (0.095) that looked like a genuine gap —
  caught by widening the search and confirming the NLL kept improving past
  the old bound. `fit_temperature_only` and `fit_calibration.py` now guard
  against this (boundary-saturation detection + auto-retry at 50× range).
- Ranked output carries per-option citations for all 120 questions × 4
  options (63.1% of option-rows carry a real citation; the rest legitimately
  ungrounded).

**Calibration improved confidence quality while accuracy went down — stated
plainly, not buried.** Calibrated rank-1 accuracy is **0.267** (both fit and
held-out) vs. Phase 4's uncalibrated `rank-1-by-support` **0.35** — different
question sets (a real caveat), but a better-calibrated, less-accurate model
is a genuinely limited contribution and the paper must say so.

**What this does NOT show:** any comparison to B2/B3 (the §4.3 budget
asymmetry — V4/V5 get 12 evidence items, B2/B3 up to ~52 — is still unfixed).

**Gamma (the KG-evidence weight) is unidentifiable — confirmed from a second
direction, exactly as predicted.** Only 12/626 cells (1.9%) on this slice are
KG-modality SUPPORT cells (reproducing §5.5 Finding 2's 1/108 at larger n). A
200-resample bootstrap gives gamma std=0.234 (>3× alpha's 0.072), interval
[−0.40, +0.66] — consistent with zero. The audit layer not citing KG evidence
and a downstream optimizer failing to find a stable weight for it are two
independent measurements pointing at the same fact: the KG modality is nearly
invisible to the agent pipeline as currently wired. Full numbers, feature
definitions, and the fit-vs-held-out ECE gap discussion are in ADR-028.

Artifacts: `results/phase5_calibration/medqa_dev_calibration.json`,
`results/phase5_calibration/medqa_raw_logprob_baseline.json`,
`results/medqa_V4_a714fe0-dirty_20260907T121703Z/`, `configs/v5.yaml`.

**One real environment finding along the way:** `scipy.optimize` segfaults on
a bare `import scipy.optimize` in this project's venv (pre-existing
binary-compatibility issue, confirmed independent of this code). Avoided
rather than risked fixing mid-run — `fit_weights` uses a hand-rolled
analytic-gradient Adam optimizer instead (gradient verified against a
finite-difference check to 1e-8, `tests/test_calibration_softmax_rank.py`).
If a future session wants `scipy.optimize` for something else, that
underlying breakage is still there and unresolved.

---

## 2026-09-07 audit remediation (docs/AUDIT_2026-09-06.md) — done, commit a714fe0

Fixed before Phase 5 could safely start: the Phase 2 KG test-split leakage
(ADR-027), four integrity blockers (surviving-invalid recomputation,
real contamination check, git dirty-stamping, named+superseded B-runs),
the calibration/corpus overlap (ADR-026 — this is *why* Phase 5's fit set
is real dev items, not `pqa_artificial`), the `docs/00_RESEARCH_SPEC.md` §3
rewrite (C1–C5 restated honestly), ADR-021's no-op perturbation floor, the
B3 context-invariance reframing, and five one-line doc corrections. Full
detail in ADR-026/027 and `docs/00_RESEARCH_SPEC.md` §7's disclosure.

**Deferred, tracked in `docs/02_ROADMAP.md`'s "Pre-Phase-7 fixes" section:**
§1.1/§1.2 (real Phase 1 `metrics.json`), §4.1 (B3 prompt-sensitivity), §4.3
(equal-budget B2 comparator — **now also gates reading Phase 5's rank-1
number against anything**), §6.7 (grid-wide multiplicity discipline), all
MINORs.

---

## Where the contributions stand (`docs/00_RESEARCH_SPEC.md` §3 is authoritative)

- **C1** — diagnostic/negative, pending Phase 7 V2-vs-V3. 1/108 (dev-20) and
  12/626 (this 120-slice) audit cells cite KG; gamma unidentifiable (ADR-028).
- **C2** — audit-discrimination supported; conflict *resolution* unverified
  (`conflict_fire_rate` ≈0 both times measured).
- **C3 — restated, not supported as originally framed (ADR-028).** The
  mechanism works (a well-calibrated ranked differential can be produced,
  ECE 0.0724), but a temperature-scaled-raw-logprob ablation ties it
  (0.0725) — temperature scaling alone matches the evidential features at
  this n. The features' additional value is not established, and
  calibrated rank-1 accuracy (0.267) is below the uncalibrated rule's
  (0.35). Needs the Phase 7 grid at n=500 to re-test both the ablation and
  whether gamma stays unidentifiable at scale (if so, drop it).
- **C4** — pending Phase 7. PubMedQA has no baseline code path yet.
- **C5** — SUPPORTED, fully traceable: the KG-hit-rate arithmetic artifact.

---

## Delivery schedule

See `docs/SCHEDULE.md` — hard deadline Saturday 2026-09-12. Mon 7 covered both
the audit-remediation day and Phase 5 (planned for Tue 8) same-session.
**Next: Phase 6** — minimal eval harness + NLI hallucination scoring, per the
schedule's Wed 9 slot — or pull forward the pre-Phase-7 fixes (§4.1, §4.3)
since §4.3 now also blocks reading Phase 5's rank-1 number against B2.

## Standing rules

- Per ADR-012/018/025/027/028: a result that undercuts a contribution is
  reported, not engineered around — including a parameter's own
  unidentifiability (ADR-028's gamma) and a pre-registered-rule deviation
  (ADR-027).
- Every A-vs-B numeric claim gets a paired-bootstrap CI, never a point
  estimate alone — this now explicitly includes ECE comparisons
  (`src.eval.metrics.paired_bootstrap_ece_diff`, ADR-028), not just
  retrieval-metric diffs.
- A fitting set and its evaluation set must be disjoint, always — fitting
  and measuring on the same n makes any positive result circular
  (ADR-028's fit/held-out split; the exact failure this guards against was
  in an earlier draft plan for this phase, caught before implementation).
- n=20/n=60 dev numbers are plumbing/mechanism checks. No "A beats B" claims
  until Phase 7 — and §4.3 (equal-budget B2) must land before Phase 5's own
  rank-1 number can be compared to anything.
- Never select a hyper-parameter, threshold, or ablation decision by reading
  the split Phase 7 reports on (ADR-027).
