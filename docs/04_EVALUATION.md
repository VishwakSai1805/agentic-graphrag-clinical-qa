# 04 — Evaluation Protocol

## 0. Eval-scale design (read before Phase 6/7)

These are **planning estimates** to size the grid, not measured results — every
number here is a design input, distinct from `results/*/metrics.json`. Phase 6
must replace the throughput/latency figures with real benchmark numbers before
Phase 7 is scheduled.

### 0.1 Per-question latency budget

Agent calls (V4/V5) carry an `EvidenceBundle` of ~12 items ≈ 3–4k tokens of
context. At a measured local-Ollama prefill rate of 958 tok/s, that's
`3500 / 958 ≈ 3.7s` prefill *before generation even starts*, per call — the
original 37s/question estimate for V4/V5 did not budget this. Revised budget:

| Variant class | LLM calls/question | Budget/question | Basis |
|---|---|---|---|
| V0–V3 (no agent network) | 1 | ~10s | single call: ~4s prefill + short completion |
| V4, V5, V4−audit, V4−conflict | avg ~5–6, hard cap 12 (`max_llm_calls_per_question`) | **~60s** | ~4s prefill + generation, summed over Hypothesis→Audit→Conflict plus up to 2 re-retrieval rounds |

A question that hits the 12-call budget guard can exceed 60s — that's an
expected `status: budget_exceeded` outcome (see `docs/01_ARCHITECTURE.md`
failure modes), not a bug, and is counted separately rather than padding the
per-question average.

### 0.2 Grid ceiling (re-derived from the 60s/question figure)

Serial (concurrency=1) wall-clock for the full post-cut grid (§0.3), using
10s/question for non-agent variants and 60s/question for agent variants:

| Slice | Question-runs | Est. serial time |
|---|---|---|
| MedQA, 8-variant single-seed (V0,V1,V2,V3 @10s + V4,V5,V4−audit,V4−conflict @60s) × 500 | 4,000 | ~38.9h |
| MedQA, extra 2 seeds × {V3,V5} × 500 | 2,000 | ~19.4h |
| PubMedQA, {V3,V5} × 500 | 1,000 | ~9.7h |
| MedMCQA, {V3,V5} × 500 (subsample, see `docs/03_DATASETS.md`) | 1,000 | ~9.7h |
| **Total, concurrency=1** | 8,000 | **~77.8h** |

This is why item 3 below (concurrency) and item 2 (grid cut) are both
necessary — neither alone brings this into a runnable range on a single 16GB
machine. At `OLLAMA_NUM_PARALLEL=4`, expect *sublinear* speedup (RAM-bound, not
purely compute-bound): budget for **3×**, not 4×, until Phase 6 measures the
real number → **~26h** effective, run as background jobs across sessions, not
in one sitting. The original (pre-cut) 8-variant × 3-dataset × 3-seed grid
would have been roughly 3× this — not runnable on this hardware, which is the
concrete justification for the cut in §0.3.

### 0.3 Grid cut (post-decision, supersedes the implied "everything × everything" reading of Phase 7)

- **Full 8-variant ablation** (V0…V5, V4−audit, V4−conflict) — **MedQA only**,
  500-question fixed seeded subset (seed 42), stated as such in the paper.
- **PubMedQA and MedMCQA** — **V3 and V5 only** (the tri-modal-no-agent and
  full-system endpoints). No intermediate ablation cells on these datasets.
- **Multi-seed (3 seeds: 42, 43, 44)** — **V3 and V5 on MedQA only**, since
  that is the pair the paired-bootstrap significance test (§1) compares. All
  other cells are single-seed.
- MedMCQA uses a fixed 500-question seeded subsample of the official
  validation split (not the full ~4.2k), for parity with the other two
  datasets' scale — see `docs/03_DATASETS.md` and ADR-005.

### 0.4 Concurrency

- Server: `OLLAMA_NUM_PARALLEL=4` (see `.env.example`) so a single Ollama
  process can serve overlapping requests instead of queuing them.
- **The harness concurrency layer and the 1/2/4 benchmark were CUT** (ADR-029,
  `docs/SCHEDULE.md`). `src/eval/harness.py` is a sequential loop — no
  `asyncio.Semaphore`, no `eval.concurrency` wiring. The §0.2 "3×" figure is
  therefore **not** replaced by a measured number; the Phase 7 grid runs
  sequentially, which is acceptable at n=500 within the deadline. The LLM cache
  (`.cache/llm/`) absorbs re-runs of a crashed grid job.
- `configs/base.yaml: eval.concurrency` remains as a knob for a future
  parallel harness but is currently read by nothing.

### 0.5 Keep-alive

Eval runs set `OLLAMA_KEEP_ALIVE=-1` (never idle-unload) — see
`.env.example` and ADR-006. Without it, the default ~5-minute unload timeout
means the first request after any harness pause (a slow question, NLI-model
load between the consensus and scoring phases) pays a full model-load penalty.

### 0.6 Stronger-model sensitivity check (robustness, not a headline result)

V3 vs V5 on **200 MedQA questions** (fixed seeded subset, disjoint from or a
subset of the 500-question dev/test slice — pin exactly which in the run
config) via the **free Gemini Flash tier**, to show the V3→V5 relative gain is
not an artifact of the 7B local model. Reported in the paper as a robustness
check alongside the main grid, not as a headline number, and not subject to
the local-hardware throughput assumptions above (it's API-latency- and
free-tier-rate-limit-bound instead — treat as a background job, not a timed
benchmark). See ADR-008.

**Also record for B3 in this check (added post-Phase-3):** `verifier_decision_rate`
and the `n_supported` distribution, not just accuracy. Phase 3's near-zero
verifier decision rate (§5.4) is evidence for C2's binary bottleneck only if
it is not merely an artifact of the 7B model's SUPPORTED-threshold calibration
— a stronger model must be checked. Tracked under the roadmap's pre-Phase-7
fixes.

## 1. Correctness
- **Accuracy** — exact option match (rank-1 prediction).
- **Macro-F1** — over option positions / classes; the reported headline for
  PubMedQA (class-imbalanced yes/no/maybe).
- Paired bootstrap (10k resamples) vs. the nearest ablation; report p-values.

### 1.0 Pre-registered analysis plan (AUDIT 2026-09-06 §6.7)

**Registered 2026-09-07, before any Phase 7 grid run exists.** Committed at the
Phase 6 checkpoint (commit `ca8f8df`, `docs/05_DECISIONS.md` ADR-030); the grid
launches Wednesday 2026-09-09. Multiple comparison discipline is fixed *here and
now* precisely
because deciding it after seeing grid numbers is worthless: ADR-012 already ran
3 uncorrected comparisons and ADR-018 ran 2, and the grid multiplies that.

**The single confirmatory comparison: V3 vs V5, accuracy, on MedQA (n=500,
seed 42).**
- One test. The metric is **accuracy, singular** — not accuracy *and* Macro-F1
  *and* ECE (that would smuggle a family back into the confirmatory slot).
- **α = 0.05, two-sided.** Two-sided is not negotiable: Phase 4 (`§5.5`
  Finding 3: the uncalibrated agent stack at 0.35 vs equal-budget hybrid RAG at
  0.60) and Phase 5 (ADR-028: temperature scaling alone matches the evidential
  features) both point at V5 possibly *not* beating V3, so the direction may not
  be assumed.
- Method: `src.retrieval.metrics.paired_bootstrap_ci` +
  `paired_bootstrap_p` (10k resamples, seed 42), paired on question id.
- **What V3 = V5 means, written before the number is seen:** if the confirmatory
  test shows no significant accuracy difference, *that is the reported finding* —
  the full tri-modal + multi-agent + evidential-calibration stack does not beat
  tri-modal retrieval alone on MedQA at this scale. Given the fusion tie
  (ADR-012), the Conflict agent firing zero times (ADR-025), and temperature
  scaling matching the evidential features (ADR-028), a null here is a **live
  outcome the paper reports as a result, not a disappointment to re-frame**. The
  contribution in that case is the negative result plus its mechanism (C1/C3 are
  already written as diagnostic, not triumphant — `docs/00` §3).

**Everything else is exploratory** and reported with Holm–Bonferroni correction
across the pre-registered exploratory family below (family-wise α = 0.05,
`src.retrieval.metrics.holm_bonferroni`). **Recomputed 2026-09-08 (ADR-035,
ADR-036), before the grid ran.** The first grid slot runs 5 MedQA cells —
**V1, V3, V5, V4−conflict, B3** (n=500 test, seed 42). **V4−audit is deferred**
to a later slot (it needs the binary-audit implementation, ADR-036) and its
pairs are *not* registered here — they are re-registered before that run. One
zero-GPU derived analysis, **post-hoc-binary** (V5's SupportMatrices with
`SUPPORT`→`SUPPORTED`, `CONTRADICT`∪`NEUTRAL`→`NOT_SUPPORTED`, `s⁻=0`,
re-ranked — ADR-036 Constraint B), joins the family now.

| # | Pair | Metrics tested | count |
|---|---|---|---|
| a | V1 vs V3 | Acc, Macro-F1, halluc-rate | 3 |
| b | V3 vs V5 | Macro-F1, halluc-rate, P@1, P@2, MRR | 5 |
| d | V4−conflict vs V5 | Acc, Macro-F1, halluc-rate, P@1, P@2, MRR | 6 |
| e | V1 vs V5 | Acc, Macro-F1, halluc-rate | 3 |
| g | **V3 vs B3** | Acc, Macro-F1, halluc-rate | 3 |
| h | **V5 vs B3** | Acc, Macro-F1, halluc-rate | 3 |
| i | post-hoc-binary vs V4−conflict | Acc, Macro-F1, halluc-rate, P@1, P@2, MRR | 6 |
| j | post-hoc-binary vs V5 | Acc, Macro-F1, halluc-rate, P@1, P@2, MRR | 6 |

**Family size m = 35.** Holm–Bonferroni over all 35. Fixed contingencies (so
`m` is never chosen after seeing a number):
- **b** has no ECE cell — V3 has no probability vector (this is why b is 5, not
  6); the V5-vs-raw-logprob ECE comparison already lives in §5.6, outside this
  family.
- **g/h** carry no P@K/MRR/ECE — B3 emits one option, not a ranking.
- **i** isolates the graded label cleanly (both sides are *uncalibrated* agent
  rankings — `(s⁺−s⁻)` for V4−conflict, `s⁺`-only for post-hoc-binary). **j**
  is the comparison ADR-036 names but it **confounds** "lost the graded label"
  with "lost calibration" (post-hoc-binary is uncalibrated, V5 is calibrated) —
  reported with that caveat stated.
- If **V4−conflict is dropped for budget** (ADR-035 — the Conflict agent fires
  0×, ADR-025), pairs **d and i leave the family** (−12) → **m = 23**. This one
  substitution is pre-authorised; no other cell may be dropped, and B3 and the
  endpoints (V1/V3/V5) are never dropped.
- P@K/MRR appear only where both sides produce a ranking. V4−conflict and
  post-hoc-binary rankings are uncalibrated — flagged as such wherever reported
  (ADR-029).

**Deferred, to be re-registered *before* the V4−audit slot runs** (ADR-036):
V4−audit vs V5, V4−audit vs V4−conflict, and V4−audit(prompt) vs post-hoc-binary
(the "does the binary *prompt* change the auditor's reasoning" test).

Any comparison **not** in this table that is run after seeing grid numbers is
reported as unplanned, excluded from the Holm family, and flagged as post-hoc.

The halluc-rate tests in the family use the 0.5 entailment threshold; ADR-031
(§5.7) established that threshold is metric-strict, so alongside the family
tests the V3/V5 runs also report halluc-rate at 0.3 / 0.5 / 0.7 as a
robustness display — not a paired test, not in the Holm family, there to show
the V3-vs-V5 *ordering* on halluc-rate does not flip with the threshold.

## 2. Factual fidelity
- **RAGAS faithfulness** — fraction of generated claims entailed by the retrieved
  context. Pin the RAGAS version; run with a fixed judge model recorded in config.
- **Context precision / recall** — retrieval quality independent of the generator.

## 3. Hallucination rate
- Split the answer rationale into atomic claims (`src/eval/nli.py:split_claims`
  — regex, decimals/abbreviations guarded, no LLM).
- For each claim, run NLI (`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`)
  against its **cited** evidence.
- `hallucination_rate = ungrounded_claims / total_claims`, where ungrounded =
  no cited evidence reaches ENTAILMENT ≥ 0.5. Reported under **both**
  denominators (predicted option / all SUPPORT cells; ADR-029).
- **Relative comparison only (ADR-031).** The control in §5.7 shows this
  criterion marks ~70% of clinical rationale claims not-entailed *regardless of
  pipeline* (B2-equal-budget 0.741 vs V5 0.682 on the dev-20, both 0 uncited) —
  it is metric-strict. Use `hallucination_rate` to compare variants at a fixed
  threshold, never as an absolute "X% hallucinated" statement. Phase 7 adds a
  threshold sweep (0.3 / 0.5 / 0.7) on the V3/V5 runs to show the variant
  *ordering* is threshold-stable.
- Also report `invalid_citation_count` (cited id not in the bundle) — a strictly
  worse, unambiguous failure than a weak entailment.

## 4. Diagnostic ranking quality
- **Precision@K** (K = 1, 2) over the ranked option list.
- **MRR** of the gold option.
- **ECE** (15 bins) + reliability diagram over `p_1`.
- **Coverage–risk curve** from the abstention threshold τ.

## 5. Results table (fill only from `results/` — no hand-typed numbers)

Grid per §0.3 and the `docs/SCHEDULE.md` cut: **MedQA only**, cells V1, V3, V5,
V4−audit, V4−conflict, **B3** (n=500 test slice, seed 42; ADR-035 added B3 —
the prior-work comparison the ablation cells cannot substitute for). V0/V2/V4
rows below stay `TBD` unless a spare slot runs them. B3 is scored on
accuracy / Macro-F1 / halluc-rate only — a binary verifier emits one option,
no ranking (P@K / MRR / ECE are `–`).

| Variant | Dataset | Acc | Macro-F1 | Faithfulness | Halluc. rate | P@1 | P@2 | MRR | ECE | run_id |
|---|---|---|---|---|---|---|---|---|---|---|
| V0 zero-shot | MedQA | TBD | TBD | – | TBD | TBD | TBD | TBD | TBD | TBD |
| V1 dense RAG | MedQA | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| V2 hybrid | MedQA | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| V3 +KG | MedQA | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| V4 +agents | MedQA | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| V4−audit | MedQA | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| V4−conflict | MedQA | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| V5 full | MedQA | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| B3 binary verifier | MedQA | TBD | TBD | – | TBD | – | – | – | – | TBD |
| V3 +KG | MedMCQA | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| V5 full | MedMCQA | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| V3 +KG | PubMedQA | TBD | TBD | TBD | TBD | – | – | – | TBD | TBD |
| V5 full | PubMedQA | TBD | TBD | TBD | TBD | – | – | – | TBD | TBD |

Also report, per variant: mean LLM calls/question, tokens/question, wall-clock,
KG-hit rate, and the failure-mode counters from `docs/01_ARCHITECTURE.md`.

### 5.1 Stronger-model sensitivity check (§0.6, robustness only — not a headline row)

| Variant | Model | Dataset (n) | Acc | run_id |
|---|---|---|---|---|
| V3 +KG | Gemini Flash | MedQA (200) | TBD | TBD |
| V5 full | Gemini Flash | MedQA (200) | TBD | TBD |

### 5.2 Phase 1 retrieval-infra validation (real numbers, NOT part of the ablation grid)

These are retrieval-quality checks that validate the tri-modal plumbing
itself (Phase 1 exit test), not QA-accuracy results for any V-variant — they
don't belong in §5's table schema and have no `run_id` (Phase 6's harness
doesn't exist yet). Reproduce via `scripts/eval_retrieval.py` and
`scripts/analysis/nfcorpus_significance.py`. See ADR-012 for full context.

**Traceability fix (2026-09-08, AUDIT 2026-09-06 §1.1/§1.2):** both scripts
previously printed to stdout only, so these numbers existed solely in a
gitignored log — a CLAUDE.md §2 violation. Both now write a committed
artifact: `scripts/analysis/nfcorpus_significance.py` →
`results/phase1_nfcorpus/metrics.json` (means, all three pairwise CIs,
`n_resamples`, `seed`); `scripts/eval_retrieval.py` →
`results/phase1_retrieval_<dataset>/metrics.json` per mode (nfcorpus /
pubmedqa / medqa). **The numbers below are the last values recorded before
this fix and are not yet backed by either artifact** — regeneration is
queued in `scripts/run_overnight.ps1`; once it completes, re-point this
table's citations at the two paths above and treat any number that drifts
from the table below as the current, authoritative one.

| Benchmark | Metric | dense | bm25 | rrf | Verdict |
|---|---|---|---|---|---|
| NFCorpus (BEIR), n=323, gating | nDCG@10 | 0.3362 | 0.2929 | 0.3327 | **PASS** — statistical tie vs. dense (95% CI [-0.0171, +0.0103]); RRF significantly beats bm25 (CI [+0.0242, +0.0564]) |
| PubMedQA (`pqa_artificial`), n=200, infra floor | Recall@10 / MRR | 0.9950 / 0.9925 | 0.9800 / 0.9501 | 0.9900 / 0.9780 | **PASS** — all ≥ 0.9 floor (saturated by design, see ADR-012; not a fusion comparison) |

**Note recorded per ADR-012, not to be quietly revised later:** dense-only
retrieval is a real, evidenced competitor at this point — it saturates
PubMedQA and statistically ties (does not lose to) fused RRF on NFCorpus
while significantly beating BM25 alone. If V1 (dense-only) matches or beats
hybrid/agent variants on MedQA downstream accuracy in the §5 grid, report it
as such.

### 5.3 Phase 2 KG-layer validation (real numbers, NOT part of the ablation grid)

**Split correction (2026-09-07, ADR-027 / AUDIT 2026-09-06 §3.1).** The numbers
in this section were originally measured on the MedQA **test** split — the split
Phase 7 reports on — and the Phase 2 design decisions were made by reading them.
Every figure below has been **re-measured on a 500-question MedQA *train* slice**
and is the train figure. The superseded test-split artifacts
(`medqa_n200.json`, `medqa_n500.json` and the two `kg_rerank` files) are retained
in `results/phase2_kg_probe/` carrying `superseded: true`, and must not be cited.

Reproduce via:

```bash
python -m scripts.kg_probe --dataset medqa --split train --limit 500 \
    --out results/phase2_kg_probe/medqa_train_n500.json
```

reading real PrimeKG data cached under `data/processed/kg_cache/medqa/`
(Neo4j not required — cache-first per ADR-007). Full artifact:
`results/phase2_kg_probe/medqa_train_n500.json` (n=500, 0 cache misses, seed 42).
See ADR-014, ADR-015, ADR-016, ADR-017 and ADR-027 for complete context; this
table is the numeric summary only.

Numbers below use the final, seeded-random triple-selection measurement
(superseding an earlier alphabetical-selection pass whose bias ADR-016
documents and quantifies).

| Metric | Value (train n=500) | Reading |
|---|---|---|
| `aggregate_hit_rate` (question-level) | 0.9540 | Clears the roadmap's ≥0.60 floor, but see the next two rows before citing this as coverage |
| `entity_level_mapping_rate` | 0.1158 | The real per-entity PrimeKG resolution rate — read together with the row above, never alone |
| `naive_expected_hit_rate_under_independence` | 0.9999 | At 75.17 linked entities/question and 11.58% per-entity mapping, a question-level hit is arithmetically near-certain — and this expectation *exceeds* the 0.9540 observed. The question-level rate is that artifact, not evidence PrimeKG covers 95% of MedQA content. This is contribution **C5** (`docs/00_RESEARCH_SPEC.md` §3) |
| `overall_novel_relation_rate` (vs. real 12-item `EvidenceBundle`) | 0.9019 | Fraction of scored KG triples not entailed by top-12 dense passages (corrected from an initial, wrong top-3 measurement — see ADR-016) |
| `overall_relevant_rate` | 0.4744 | Fraction of scored triples mentioning an entity the question/options are about |
| **`overall_novel_and_relevant_rate`** | **0.4237** | **The actual C1 evidence figure** — on-topic *and* not already in dense passages. Middling, not strong; bottlenecked by retrieval precision (only ~47% relevant), not by redundancy with dense retrieval |
| `overall_novel_given_relevant_rate` | 0.8931 | Conditional on relevance, KG evidence is usually novel — the encouraging half of the finding |
| `unmapped_cui.unmapped_rate` | 0.8842 | 33,231/37,584 linked entities have no PrimeKG crosswalk hit, over 6,852 distinct unmapped CUIs (ADR-014: disease/phenotype via real ontology xrefs, drug/gene via name-match only) |
| Per-question triples (post ADR-015 cap) | median 40.0, mean 38.1 (n=500) | 476/500 questions (95.2%) saturate the 40-triple cap — the cap is the normal case, not a backstop. 23 questions yield zero triples |
| PrimeKG raw edges vs. loaded | 8,100,498 raw rows (~4.05M unique undirected edges) → 740,674 loaded (9.1%) | Deliberate 7-item allowlist filter, not an incomplete load — see ADR-014 (corpus/index provenance, traced to `data/MANIFEST.json`) |

Per-category breakdown (`n` shown per this phase's stated discipline;
`MIN_CATEGORY_N_FOR_RATE=15` — at n=500 every category clears it, including
`anatomy`, whose rates were suppressed at n=8 on the old test slice — see
ADR-017):

| Category | n | hit_rate | novel_and_relevant_rate |
|---|---|---|---|
| pharmacology | 127 | 0.976 | 0.483 |
| pathology_diagnosis | 116 | 0.991 | 0.392 |
| treatment_management | 101 | 0.931 | 0.413 |
| other | 92 | 0.913 | 0.365 |
| physiology | 41 | 0.976 | 0.425 |
| anatomy | 23 | 0.870 | 0.530 |

**KG triple reranking ablation (`kg_rerank`, ADR-018 → re-decided in ADR-027) —
the verdict flipped at n=500.** Reusing the Phase 1 MedCPT cross-encoder
(`src/retrieval/rerank.py`) to select which triples survive the ADR-015
40-triple cap (instead of raw traversal order) was tested as an off-by-default
config flag, with a real paired-bootstrap significance check (10k resamples,
seed 42, paired by `question_id`, n_paired=477, n_excluded=23 zero-triple
questions excluded identically in both variants) per CLAUDE.md's "never claim A
beats B from a raw mean alone" rule. Artifact:
`results/phase2_kg_probe/medqa_train_n500_kg_rerank_ablation.json`.

| Metric | naive | kg_rerank | Paired 95% CI | p (two-sided) | Holm survives (m=2) | Verdict (train n=500) | was (test n=200) |
|---|---|---|---|---|---|---|---|
| `relevant_rate_for_question` (secondary) | 0.4744 | 0.5296 | [+0.0203, +0.0889] | 0.0026 | **yes** (rank 1, threshold 0.025) | **significant win** | marginal win |
| `novel_and_relevant_rate_for_question` (**primary**) | 0.4237 | 0.4604 | **[+0.0044, +0.0681]** | 0.0256 | **yes** (rank 2, threshold 0.050) | **significant win** | **tie** |

At n=200 the primary metric was a tie and was reported as one. At n=500 on a
clean split it clears significance, exactly as ADR-018 anticipated when it
pre-committed to re-deciding at larger n. The primary CI's lower bound is
+0.0044 — one-tenth of the point estimate — so this is applied and reported as
a Holm-Bonferroni-corrected result for the 2-metric family
(`src.retrieval.metrics.paired_bootstrap_p` + `holm_bonferroni`, AUDIT §6.8),
not deferred to the grid-wide multiplicity discipline still pending for Phase 7
(§6.7). **Both metrics survive the correction.** This is not a coincidence to
skate past: with exactly 2 comparisons, Holm only tightens the *smaller*
p-value's threshold (secondary, tested at α/2 = 0.025); the *larger* p-value
(primary) is tested at the same threshold as the uncorrected case, α = 0.05,
because it is last in the step-down chain — so the primary's significance here
is genuinely robust to this correction, not narrowly surviving it. It remains a
*marginal* win in effect size (CI lower bound close to zero) even though it is
not marginal in significance. See ADR-027 for the full computation and for the
**pre-registration deviation** this ADR records: ADR-018's rule said adopt on a
cleared verdict, and this decision does not adopt — `kg_rerank` remains **off
by default**, because the metric that moved is an intermediate evidence-quality
proxy and no run yet shows it moving a downstream metric. The Phase 7
`kg_rerank` grid row is what actually settles whether that caution was
warranted.

**The honest position going into Phase 3, per ADR-016, ADR-018 and ADR-027:**
Phase 2's infrastructure is real and correct (verified PrimeKG source, real Neo4j
load, cache-first reproducibility confirmed byte-for-byte). But **the KG
modality's marginal value over dense retrieval is not yet established** — the
widely-visible ~95% hit rate is an arithmetic artifact of entity fan-out, and the
defensible C1 evidence figure (`novel_and_relevant_rate = 0.4237`) is real but
middling, limited primarily by retrieval precision (hub-node noise, ADR-015) and
partial entity crosswalk coverage (ADR-014), not by redundancy with dense
passages. Reranking measurably improves triple *selection* without yet being
shown to improve anything downstream. This is not a result to soften before
Phase 3.

### 5.4 Phase 3 baseline plumbing check (dev slice, n=20 — NOT a grid result)

Real numbers from `scripts/run_baselines.py --dataset medqa --split train
--limit 20` (seed 42). The four runs the table below reports are, exactly
(AUDIT 2026-09-06 §6.2 — no wildcard timestamp, because two committed batches
share the SHA `d5e3af5` and only one is valid):

- `results/medqa_B0_d5e3af5_20260906T101125Z/`
- `results/medqa_B1_d5e3af5_20260906T101242Z/`
- `results/medqa_B2_d5e3af5_20260906T101648Z/`
- `results/medqa_B3_d5e3af5_20260906T102134Z/`

The earlier `~0954–1006Z` batch at the same SHA (`…T095449Z`, `…T095549Z`,
`…T095935Z`, `…T100551Z`) is the pre-`num_ctx`-fix batch. It is **not**
deleted — it is the context-window sensitivity comparator used below and in
ADR-021 — but each of its four `metrics.json` files now carries
`superseded: true` with a `superseded_reason` and a `superseded_by` run id,
stamped by `scripts/mark_superseded.py --preset phase3-num-ctx`, so no reader
and no report script can tabulate it as a headline result by mistake.

Caveat on the recorded SHA: all eight runs were produced from an uncommitted
worktree, so `d5e3af5` does not identify the code that produced them (AUDIT
§6.1 — `src/eval/baselines.py` does not exist at that SHA). `src/eval/results.py`
now stamps `<sha7>-dirty` plus a `git_diff_sha256` for any run made from a dirty
tree; the eight runs above predate that fix and their SHA must be read as
"approximately, the tree around `d5e3af5`", not as a checkout target.

This is the Phase 3 exit test: it certifies that query → tri-modal
`EvidenceBundle` → answer → grounded citations runs end-to-end for all four
baselines. It is **not** a QA-accuracy result — n=20, a dev (train-split)
slice, single-seed, no significance test, no "A beats B" claim. The §5 grid
(Phase 7, 500-question seeded test slice) is the reportable table.

| Baseline | Acc | Macro-F1† | grounded_rate | invalid_citation_count | status |
|---|---|---|---|---|---|
| B0 zero-shot        | 0.55 | 0.52 | 0.00‡ | 0 | 20 ok |
| B1 dense-only RAG   | 0.60 | 0.57 | 0.95 | 1 | 20 ok |
| B2 hybrid (no agents)| 0.65 | 0.61 | 0.95 | 3 | 20 ok |
| B3 binary verifier§  | 0.65 | 0.61 | 0.95 | 3 | 20 ok |

† position-bias check, not a skill metric on 4-option MCQ (ADR-020).
‡ B0 runs no retrieval — ungrounded by construction, not a failure (ADR-020).
§ B3's answer vector is **identical to B2's on 19/20 questions by construction**
(AUDIT §4.6): the verifier decides on only 1/20 (strict prompt), and on the
other 19 it issues the byte-identical B2 generation call — `llm_cache_hits: 19`
in `medqa_B3_d5e3af5_20260906T102134Z/metrics.json`. B3's 0.65 is B2's 0.65
plus one verifier-decided question; the row is not an independent result and
the diagnostic value is `verifier_decision_rate` (below), not accuracy.

**B3 diagnostic — the `verifier_decision_rate` is robust to the context window
but NOT to verifier phrasing (ADR-033, AUDIT §4.1).** The earlier framing
("invariant at 1/20 = 0.05") was premature and is retired.

*Context-window axis (holds).* At the shipped strict verify prompt, the rate is
0.05 under `num_ctx=4096` and 0.05 under `num_ctx=8192`; only the *direction* of
failure changes:

| Run | `num_ctx` | `n_supported` distribution | exactly-one | `verifier_decision_rate` |
|---|---|---|---|---|
| `medqa_B3_d5e3af5_20260906T100551Z` (superseded) | 4096 | `{0: 7, 1: 1, 3: 2, 4: 10}` | 1/20 | **0.05** (over-admits — 10/20 all-four SUPPORTED) |
| `medqa_B3_d5e3af5_20260906T102134Z` (reported) | 8192 | `{0: 15, 1: 1, 2: 4}` | 1/20 | **0.05** (over-rejects — 15/20 zero SUPPORTED) |

*Prompt-wording axis (does not hold).* Re-run on the same dev-20 with two
further verify phrasings (`scripts/analysis/b3_prompt_sensitivity.py`,
`results/phase3_b3_prompt_sensitivity/metrics.json`):

| verify phrasing | `verifier_decision_rate` | `n_supported` distribution |
|---|---|---|
| strict ("SUPPORTED only if the evidence positively supports it" — shipped) | **0.05** | `{0: 15, 1: 1, 2: 4}` |
| neutral ("is option X supported as the correct answer?") | **0.30** | `{0: 8, 1: 6, 2: 3, 3: 1, 4: 2}` |
| permissive ("could option X be correct? SUPPORTED if the evidence is consistent with it") | **0.35** | `{0: 6, 1: 7, 2: 4, 3: 1, 4: 2}` |

Spread 0.30. **The 0.05 is substantially an artifact of the strict instruction**
— it makes the verifier reject nearly everything. Under a fair (neutral or
permissive) instruction the rate is 0.30–0.35. The AUDIT §4.1
"strawman-by-prompt-construction" objection is therefore *confirmed* for the
strong form of the claim.

**What survives.** Even at the *most permissive* phrasing the binary verifier
isolates exactly one option on only 7/20 questions and returns multiple- or
zero-SUPPORTED on the other 13 — a per-option yes/no head cannot rank
mutually-exclusive options, and no phrasing tested gets it above 0.35. That is a
real limitation of binary verification and it motivates C2, but the honest
figure is "isolates a single option on ≤ 35% of questions", **not** "1/20". C2's
audit-discrimination evidence does not depend on this comparison (it rests on
the 108 non-neutral audit cells and the leave-one-agent-out V4−audit grid cell,
`§5.5` / Phase 7); the B3 number is a motivating illustration, quoted with its
0.05–0.35 range, not a headline result. To be re-measured at n=500 in Phase 7.

n=20 single-seed remains a limit on all of the above (0.35 = 7/20 carries a wide
interval).

**Recorded per ADR-021:** the 4096-token batch was Ollama's default context,
which silently truncated the KG-triple block (appended last) and collapsed
B2/B3 to Acc 0.15 with 12/20 parse errors. The main table above is post-fix
(`llm.num_ctx = 8192` + a hard `LLMContextBudgetError` guard). ADR-021 also
records the no-op perturbation floor measured from the same pair of batches:
B0, which does no retrieval and is therefore *unaffected* by the context fix,
still flips one of 20 answers between them — so a 1/20 (0.05) accuracy delta
across these two batches is inside the noise floor and is not read as an effect.

**PubMedQA B3 — the binary verifier fails to isolate an option even on a 3-way
task (ADR-030).** PubMedQA is yes/no/maybe — the exact format MEGA-RAG-class
binary verifiers were built for — and B3 still cannot use it. On the dev-20
(`results/pubmedqa_B3_aa73199-dirty_20260907T165909Z/`, the word-framed prompt
ADR-030 keeps): `verifier_decision_rate = 0.15`, `n_supported = {0: 9, 1: 3,
2: 7, 3: 1}` — it isolates a single SUPPORTED answer on **3/20** three-way
questions and falls back to generation on 17/20. Under the earlier
A/B/C-lettered prompt the rate was **0.00** (`n_supported = {0: 12, 2: 6,
3: 2}`, superseded run `…T163512Z`) — so word framing *helps* the verifier a
little. Both PubMedQA numbers use the shipped *strict* verify instruction; per
ADR-033 that instruction depresses the rate on MedQA (0.05 strict vs 0.30–0.35
neutral/permissive), so the PubMedQA 0.15 is almost certainly a strict-prompt
floor too and should be read as "≤ ~0.35 under a fair instruction", not as a
hard 0.15. The prompt-invariant point stands: a per-option yes/no verifier does
not rank three competing answers any better than four, and no phrasing gets it
near reliable single-option isolation. (The PubMedQA baselines are otherwise a
degenerate one-class predictor at this model scale — see ADR-030; PubMedQA is
not in the Phase 7 grid, `docs/SCHEDULE.md`.)

### 5.5 Phase 4 multi-agent consensus plumbing check (dev slice, n=20 — NOT a grid result)

`python -m src.pipeline --config configs/agents.yaml --limit 20` (MedQA train
dev-20, seed 42), `results/medqa_V4_1388904_20260906T115020Z/`. Certifies that
query → tri-modal bundle → modality-reserved 12-item evidence set →
Hypothesis→Audit→Conflict → `SupportMatrix` with only real citations runs
end-to-end. **Not** a QA-accuracy result. Numbers below are post the ADR-025
claim-level conflict fix.

| Invariant / counter | Value |
|---|---|
| SupportMatrices emitted | 20/20 (+ per-node traces) |
| surviving cells citing a non-bundle id | **0** (exit-test invariant) |
| RAW model citation hallucination (dropped at audit) | 6 cells = 5.3% of produced cells |
| status | `ok` 19 · `insufficient_evidence` 1 |
| `budget_exceeded` rate | 0.00 |
| mean LLM calls / question | 5.75 (cap 12) · mean rounds 1.05 |
| bundle mix | KG slot filled 17/20 · all-passage 3/20 |
| audit cells | 108 (78 SUPPORT, 30 CONTRADICT); 61/80 options get a signal |

**Finding 1 — the Conflict agent never fires under a correct definition.**
With conflict redefined as claim-level (ADR-025: same `claim_idx`, SUPPORT and
CONTRADICT from different evidence ids), `conflict_fire_rate = 0.00` on the
dev-20 — **not one claim** had genuinely opposing evidence. The 11
`unresolved_conflict` statuses in the pre-fix run were entirely artifacts of
the wrong option-level definition (which counted "claim 1 supported, claim 3
contradicted" as a conflict). **Genuine evidence contradictions are rare on
this corpus**, so C2's contribution must be reframed around what the **Audit**
layer does (SUPPORT/CONTRADICT/NEUTRAL discrimination against retrieved
evidence) rather than around conflict *resolution*. This is the honest
position Phase 5 inherits.

**Finding 2 — the KG modality is nearly invisible to the agents.** Of 108
audit cells, **1 cites a KG triple**; 107 cite passages. Even with the KG slot
filled on 17/20 questions, the Audit agent audits claims against passage text
and effectively ignores the verbalised triples. Read with ADR-016
(`novel_and_relevant_rate ≈ 0.41` in retrieval) this puts the KG modality's
actual contribution to the agent pipeline close to nil on this slice — a
Phase 7 `evidence_split` / KG-verbalisation question, recorded not softened.

**Finding 3 — the agent layer underperforms plain hybrid RAG at equal evidence
budget.** `rank-1-by-support` (crude `s⁺ − s⁻` argmax, no calibration,
`resolved` not penalised per ADR-025) = **0.35** on the dev-20. The original
comparison was against B2 at 0.65, but B2 there saw up to ~52 evidence items
(12 reranked passages + all cached KG triples) against the agents' 12 — a
budget confound. That confound is now removed: `scripts/run_baselines.py
--equal-budget` (AUDIT §4.3) trims B2's bundle to the *same* 12-item,
`[8, 4]`-reserved split V4/V5 use, and
`results/medqa_B2eq_aa73199-dirty_20260907T162956Z/` scores **B2 = 0.600**
(macro-F1 0.594) on the identical dev-20. B2's drop from 0.650 → 0.600 under
the trim is **0.05 — exactly ADR-021's measured ±0.05 no-op noise floor** — so
the evidence-budget asymmetry does *not* explain B2's advantage over the
agents.

The unconfounded comparison therefore stands: on the same 20 questions at the
same 12-item budget, **plain hybrid RAG (one LLM call) reaches 0.60 and the
Hypothesis→Audit→Conflict agent stack reaches 0.35**. The gold option carries
the (tied-)max `s⁺` on only 10/20 questions and the Audit agent is
SUPPORT-heavy (78 vs 30). Calibration (Phase 5) fits α,β,γ,δ,T to turn this
count vector into a better logit, but a raw-argmax gap this large against an
equal-budget baseline is a **real finding to report** — the multi-agent layer,
as built, costs ~5.8 LLM calls/question to land below a single-call hybrid RAG
baseline — not a caveat to defer to Phase 7. Phase 7's V2/V3-vs-V4/V5 cells at
n=500 settle whether calibration closes it or whether the agent layer is a net
negative on MedQA.

### 5.6 Phase 5 evidential calibration (disjoint fit/held-out, n=60/60 — NOT a grid result)

`python -m src.pipeline --config configs/agents.yaml --dataset medqa --split
train --limit 120 --exclude-limit 20 --seed 42` → `results/medqa_V4_a714fe0-dirty_20260907T121703Z/`
(a fresh 120-question MedQA train slice, disjoint by construction from Phase
3/4's dev-20 — `load_question_sample_excluding`, joined by `question_id`, not
index arithmetic). Split by sorted `question_id` into fit (first 60) and
held-out (second 60); weights fit on the fit half only, every ECE number below
is measured on the held-out half. See ADR-028 for the full feature
definitions, split rationale, and gamma-identifiability discussion.

**The literal exit test** (`docs/02_ROADMAP.md`: "ECE on dev improves vs. raw
LLM option log-probs; ranked output includes per-option citations for every
question") **passes on its literal reading — and a follow-up ablation shows
that reading is not the one to build a claim on:**

| | value |
|---|---|
| Calibrated ECE, held-out (5 equal-mass bins, n=60) | **0.0724** |
| Raw-logprob ECE (temperature 1), held-out (same 60 questions, same bins) | 0.5418 |
| Paired bootstrap mean diff (calibrated − raw), 10k resamples seed 42 | −0.4204 |
| 95% CI | **[−0.5426, −0.2826]** |
| Verdict | calibrated significantly better than raw-at-T=1 — CI entirely below 0 |

The raw-logprob comparator is a real per-token-logprob readout
(`scripts/raw_logprob_baseline.py`, `src.llm.client.chat(logprobs=True)` —
Ollama's native provider rejects `logprobs`; routed through LiteLLM's
`openai/` provider type at the same server instead, confirmed empirically),
given the **same EvidenceBundle** the agent pipeline saw. Ranked output
(`results/phase5_calibration/medqa_dev_calibration.json`) carries all 120
questions × 4 options with per-option citations structurally present for every
row (63.1% of option-rows carry ≥1 real citation; the rest are legitimately
ungrounded per CLAUDE.md's carve-out, not missing data).

**But this comparison does not isolate what C3 claims, and a direct ablation
shows the gap above is almost entirely a temperature effect, not an
evidential-calibration effect.** A single scalar temperature T was fit on the
fit-60's already-saved raw logprobs (no new LLM calls — `log(raw_probability)`
is a valid stand-in for the true logit here, since softmax is shift-invariant)
and applied, frozen, to the held-out-60:

| comparison (held-out, n=60) | point ECE | diff | 95% CI | verdict |
|---|---|---|---|---|
| Calibrated (0.0724) vs. **T-scaled raw** (0.0725, T=92.03) | −0.0001 | −0.0001 | **[−0.1344, +0.0887]** | **tie** |
| Raw-at-T=1 (0.5418) vs. T-scaled raw (0.0725) | +0.4694 | +0.4694 | [+0.2168, +0.5563] | T-scaling significantly better |

**Temperature scaling alone — zero evidential features, zero agent
reasoning — reaches the same ECE as the full calibrated pipeline.** Nearly all
of the 0.54→0.07 improvement above is a temperature-scaling effect. **C3 is
restated accordingly: at this sample size, temperature scaling of the raw
signal suffices to match the evidential features' ECE; the evidential
features' additional contribution over temperature scaling alone is not
established.** This is not a null result to bury — it is exactly the finding
a careful ablation should surface, and it directly answers the "the gain is
just temperature scaling" objection a reviewer would otherwise have to raise.
It leaves open (Phase 7, larger n, or a different axis such as abstention or
citation quality) whether the features earn their place; it does not, on this
evidence, establish that they already have. Fitting `fit_temperature_only`
also caught and fixed a real methodological bug worth naming: an initial
narrow search bound (`t_max=20`, borrowed from the evidential fit's own
temperature bound) silently saturated, reporting a spuriously low T=20 whose
ECE (0.095) looked like a real gap vs. calibrated — corrected by widening the
search and confirming the true optimum (T≈92) before reporting anything
(see ADR-028 Decision 5; `fit_temperature_only` and `scripts/fit_calibration.py`
now detect and guard against this).

**Calibration improved confidence quality while accuracy went down — stated
plainly, not softened into a caveat.** Phase 4's uncalibrated
`rank-1-by-support` (raw `s+ − s−` argmax, §5.5 Finding 3) was **0.35** on its
own dev-20. This phase's fitted, temperature-scaled model gets **0.267**
rank-1 accuracy on both the fit-60 and held-out-60. These come from different
question sets, not a controlled before/after on the same items — a real
caveat, not a reason to soften the number. A model that is better-calibrated
(ECE 0.07) and less accurate than the cruder rule it replaces is a genuinely
limited contribution, and that is the honest headline alongside the ECE
result, not a footnote under a "not established" note. The §4.3 budget
asymmetry (V4/V5 get 12 evidence items; B2/B3 get up to ~52) separately still
forbids any comparison to B2 (AUDIT 2026-09-06, deferred) — a different
limitation from this one.

**Gamma identifiability, checked from two independent directions and
consistent.** On this 120-question slice, KG-modality SUPPORT cells are 12 of
626 total cells (1.9%) — reproducing §5.5 Finding 2's 1/108 almost exactly at
larger n. A 200-resample bootstrap of the fit gives:

| param | mean | std | range |
|---|---|---|---|
| alpha | −0.0169 | 0.0718 | [−0.2777, +0.1934] |
| beta | +0.1139 | 0.0867 | [−0.2836, +0.3304] |
| **gamma** | **+0.2315** | **0.2340** | **[−0.4031, +0.6642]** |
| delta | +0.0071 | 0.0205 | [−0.0438, +0.1334] |

Gamma's std is >3× alpha's despite a comparable point estimate, and its
interval straddles zero — the signature of an unidentifiable parameter, not a
real moderate effect. This is independent confirmation of the KG-invisibility
finding from a different measurement than the raw cell count: not only does
the audit layer almost never cite KG evidence, but even a downstream optimizer
handed the chance to weight it cannot find a stable weight, because there is
almost no variance in the feature to constrain one. Report gamma's interval
alongside its point estimate; the point estimate alone reads as a real effect
it is not.

### 5.7 Phase 6 harness — hallucination-rate control (dev-20, NOT a grid result)

The Phase 6 harness exit test (`src.eval.harness --config configs/v5.yaml
--dataset medqa --split train --limit 20`,
`results/medqa_V5_aa73199-dirty_20260907T170604Z/`) reported
`hallucination_rate_predicted_option = 0.682` (n=22 SUPPORT-cell claims for the
predicted option; `hallucination_rate_all_support_cells = 0.718`, n=78).
`hallucination_rate` here is: split each SUPPORT cell's rationale into claims
(`src.eval.nli.split_claims` — regex, decimals/abbreviations guarded), NLI each
claim against **its cited evidence text**, ungrounded = no cited evidence
reaches `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` entailment
**≥ 0.5** (`docs/04` §3).

**0.682 needs a control** — a 0.5 DeBERTa-NLI entailment threshold is strict,
and a clinical claim that is genuinely supported *after one step of inference*
scores as not-entailed. So run the identical scorer / threshold / splitter over
a **non-agent** pipeline's rationales on the same 20 questions
(`scripts/analysis/hallucination_control.py`, artifact
`results/phase6_hallucination_control/control.json`):

| pipeline (dev-20, same scorer, threshold 0.5, same `split_claims`) | halluc-rate (predicted option) | n claims | uncited |
|---|---|---|---|
| **B2 equal-budget** (hybrid RAG, one LLM call, 12-item bundle) | **0.741** | 58 | 0 |
| **V5** (agents + calibration) | **0.682** | 22 | 0 |
| V5 − B2 | **−0.060** | — | — |

**The number reflects the NLI criterion's strictness, not the agent layer.**
Both pipelines land near 0.7, **both with zero uncited claims** — so this is
not about missing citations, it is the NLI model declining to score
cited-but-inferential clinical claims as entailed at ≥ 0.5. Example from the B2
run (`train_00409`, cited to the correct passage): *"This protein is crucial
for the activation of B cells by T cells… its deficiency leads to low levels of
IgG, IgM, and IgA"* scores 0.545 (just entailed); the sibling claim *"the
patient has a defect in the activation-induced cytidine deaminase (AID)
protein"* scores 0.331 against the same passage. If anything V5 is marginally
*lower* than plain hybrid RAG (−0.060, n=20 single-seed — no significance
claimed).

**Reading (ADR-031):** at threshold 0.5, DeBERTa NLI marks ~70% of clinical
rationale claims as not-entailed by their cited evidence **regardless of
pipeline**. `hallucination_rate` is therefore usable as a *relative* comparison
across variants at a fixed threshold (its role in the Phase 7 exploratory
family, `§1.0`), **not** as an absolute "70% of claims are hallucinated"
statement — that framing would be measuring the entailment threshold. The
threshold and splitter are stated here and in ADR-031 so the number is
auditable; Phase 7 additionally reports `invalid_citation_count` (a cited id
not in the bundle — a strictly worse and unambiguous failure) alongside it.

## 6. Run discipline
- `results/<run_id>/config.yaml` — full resolved config incl. model ids + seed
- `results/<run_id>/predictions.jsonl` — one row per question: options, p vector,
  citations, status, agent trace ref
- `results/<run_id>/metrics.json` — everything in §1–4
- Only `metrics.json` and `config.yaml` are committed; predictions are large.

## 7. Sanity checks before believing any result
1. Shuffle option order → accuracy should not move more than noise (no position bias).
2. Empty-retrieval control → accuracy should drop toward V0.
3. Gold-evidence oracle → upper bound; if V5 ≈ oracle, retrieval is not the bottleneck.
4. Contamination assertion passed (logged in `metrics.json`).
