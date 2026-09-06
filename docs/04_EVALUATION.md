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
- Harness: `src/eval/harness.py` (Phase 6) issues requests through an
  `asyncio.Semaphore` bound to a `concurrency` config value
  (`configs/base.yaml: eval.concurrency`, default 4).
- Phase 6 exit criteria include benchmarking **1 vs 2 vs 4** concurrent
  requests on a fixed dev slice and recording real throughput
  (questions/hour) into `results/` — replacing the §0.2 3× assumption with a
  measured number before Phase 7 is scheduled.

### 0.5 Keep-alive

Eval runs set `OLLAMA_KEEP_ALIVE=-1` (never idle-unload) — see
`.env.example` and ADR-006. Without it, the default ~5-minute unload timeout
means the first request after any harness pause (checkpoint I/O, a slow
question) pays a full model-load penalty, which would corrupt the throughput
benchmark in §0.4.

### 0.6 Stronger-model sensitivity check (robustness, not a headline result)

V3 vs V5 on **200 MedQA questions** (fixed seeded subset, disjoint from or a
subset of the 500-question dev/test slice — pin exactly which in the run
config) via the **free Gemini Flash tier**, to show the V3→V5 relative gain is
not an artifact of the 7B local model. Reported in the paper as a robustness
check alongside the main grid, not as a headline number, and not subject to
the local-hardware throughput assumptions above (it's API-latency- and
free-tier-rate-limit-bound instead — treat as a background job, not a timed
benchmark). See ADR-008.

## 1. Correctness
- **Accuracy** — exact option match (rank-1 prediction).
- **Macro-F1** — over option positions / classes; the reported headline for
  PubMedQA (class-imbalanced yes/no/maybe).
- Paired bootstrap (10k resamples) vs. the nearest ablation; report p-values.

## 2. Factual fidelity
- **RAGAS faithfulness** — fraction of generated claims entailed by the retrieved
  context. Pin the RAGAS version; run with a fixed judge model recorded in config.
- **Context precision / recall** — retrieval quality independent of the generator.

## 3. Hallucination rate
- Split the answer rationale into atomic claims (`src/eval/nli.py`).
- For each claim, run cross-encoder NLI against its **cited** evidence.
- `hallucination_rate = ungrounded_claims / total_claims`, where ungrounded =
  no cited evidence reaches ENTAILMENT ≥ 0.5.
- Also report `invalid_citation_count` (cited id not in the bundle) — a strictly
  worse failure than a weak entailment.

## 4. Diagnostic ranking quality
- **Precision@K** (K = 1, 2) over the ranked option list.
- **MRR** of the gold option.
- **ECE** (15 bins) + reliability diagram over `p_1`.
- **Coverage–risk curve** from the abstention threshold τ.

## 5. Results table (fill only from `results/` — no hand-typed numbers)

Grid per §0.3: full 8-variant ablation on MedQA only; PubMedQA and MedMCQA get
V3/V5 only; V3/V5 on MedQA are the multi-seed pair (report mean ± std over
seeds 42/43/44 in addition to the single-seed row below).

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

Reproduce via `python -m scripts.kg_probe --dataset medqa --limit 200`,
reading real PrimeKG data cached under `data/processed/kg_cache/medqa/`
(Neo4j not required — cache-first per ADR-007). Full artifact:
`results/phase2_kg_probe/medqa_n200.json`. See ADR-014, ADR-015, ADR-016,
ADR-017 for complete context; this table is the numeric summary only.

Numbers below use the final, seeded-random triple-selection measurement
(superseding an earlier alphabetical-selection pass whose bias ADR-016
documents and quantifies — the effect on the headline metric was small,
0.4078→0.4063, but the fix matters methodologically).

| Metric | Value | Reading |
|---|---|---|
| `aggregate_hit_rate` (question-level) | 0.960 | Clears the roadmap's ≥0.60 floor, but see the next row before citing this as coverage |
| `entity_level_mapping_rate` | 0.1122 | The real per-entity PrimeKG resolution rate — read together with the row above, never alone |
| `naive_expected_hit_rate_under_independence` | 0.9998 | At ~74 linked entities/question and 11.22% per-entity mapping, a question-level hit is arithmetically near-certain — the 0.960 above is this artifact, not evidence PrimeKG covers 96% of MedQA content |
| `overall_novel_relation_rate` (vs. real 12-item `EvidenceBundle`) | 0.9016 | Fraction of scored KG triples not entailed by top-12 dense passages (corrected from an initial, wrong top-3 measurement — see ADR-016) |
| `overall_relevant_rate` | 0.4734 | Fraction of scored triples mentioning an entity the question/options are about |
| **`overall_novel_and_relevant_rate`** | **0.4063** | **The actual C1 evidence figure** — on-topic *and* not already in dense passages. Middling, not strong; bottlenecked by retrieval precision (only ~47% relevant), not by redundancy with dense retrieval |
| `overall_novel_given_relevant_rate` | 0.8581 | Conditional on relevance, KG evidence is usually novel — the encouraging half of the finding |
| `unmapped_cui.unmapped_rate` | 0.8878 | 13,129/14,789 linked entities have no PrimeKG crosswalk hit (ADR-014: disease/phenotype via real ontology xrefs, drug/gene via name-match only) |
| Per-question triples (post ADR-015 cap) | median 40.0, mean 38.4 (n=200) | 192/200 questions (96%) saturate the 40-triple cap — the cap is the normal case, not a backstop |
| PrimeKG raw edges vs. loaded | 8,100,498 raw rows (~4.05M unique undirected edges) → 740,674 loaded (9.1%) | Deliberate 7-item allowlist filter, not an incomplete load — see ADR-014 |

Per-category breakdown (`n` shown per this phase's stated discipline; `anatomy`
n=8 is below `MIN_CATEGORY_N_FOR_RATE=15` and its rates are suppressed —
see ADR-017):

| Category | n | hit_rate | novel_and_relevant_rate |
|---|---|---|---|
| pharmacology | 39 | 1.000 | 0.556 |
| pathology_diagnosis | 43 | 1.000 | 0.416 |
| treatment_management | 48 | 0.979 | 0.381 |
| physiology | 22 | 0.909 | 0.355 |
| other | 40 | 0.875 | 0.277 |
| anatomy | 8 | *suppressed (n<15)* | *suppressed (n<15)* |

**KG triple reranking ablation (`kg_rerank`, ADR-018) — a measured result, not
a shipped default.** Reusing the Phase 1 MedCPT cross-encoder
(`src/retrieval/rerank.py`) to select which triples survive the ADR-015
40-triple cap (instead of raw traversal order) was tested as an off-by-default
config flag on the same 200 MedQA questions, with a real paired-bootstrap
significance check (10k resamples, paired by `question_id`, n=192) per
CLAUDE.md's "never claim A beats B from a raw mean alone" rule:

| Metric | naive | kg_rerank | Paired 95% CI | Verdict |
|---|---|---|---|---|
| `overall_relevant_rate` | 0.4734 | 0.5276 | [+0.0026, +0.1052] | **significant win** (narrow — CI lower bound close to 0) |
| `overall_novel_and_relevant_rate` (primary) | 0.4063 | 0.4443 | [−0.0125, +0.0875] | **tie** (CI includes 0) |

The primary C1 evidence metric does not clear significance at n=200 — this
is reported as a tie, not as "reranking helps," exactly per the ADR-012
precedent. `kg_rerank` ships implemented and tested but **off by default**;
re-testing at a larger n (e.g. Phase 7's ablation grid) is the natural
follow-up, not further tuning of this n=200 result.

**The honest position going into Phase 3, per ADR-016 and ADR-018:** Phase 2's
infrastructure is real and correct (verified PrimeKG source, real Neo4j
load, cache-first reproducibility confirmed byte-for-byte). But **the KG
modality's marginal value over dense retrieval is not yet established** — the
widely-visible 96% hit rate is an arithmetic artifact of entity fan-out, the
defensible C1 evidence figure (`novel_and_relevant_rate = 0.4063`) is
real but middling, limited primarily by retrieval precision (hub-node noise,
ADR-015) and partial entity crosswalk coverage (ADR-014), not by redundancy
with dense passages. This is not a result to soften before Phase 3.

### 5.4 Phase 3 baseline plumbing check (dev slice, n=20 — NOT a grid result)

Real numbers from `scripts/run_baselines.py --dataset medqa --split train
--limit 20` (seed 42), stored under `results/medqa_B{0,1,2,3}_<sha>_<ts>/`.
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
| B3 binary verifier  | 0.65 | 0.61 | 0.95 | 3 | 20 ok |

† position-bias check, not a skill metric on 4-option MCQ (ADR-020).
‡ B0 runs no retrieval — ungrounded by construction, not a failure (ADR-020).

**B3 diagnostic (the useful number):** `verifier_decision_rate = 0.05` — the
binary verifier returned exactly one SUPPORTED option on only 1/20 questions;
`n_supported` distribution `{0: 15, 1: 1, 2: 4}`. So B3 fell back to B2-style
generation 19/20 times, which is why B3 ≡ B2 here. A near-zero verifier
decision rate is **direct empirical evidence for the binary bottleneck C2
targets** (a per-option yes/no verifier cannot rank competing options), to be
re-measured at scale in Phase 7. See ADR-020.

**Recorded per ADR-021:** an earlier run at Ollama's default 4096-token
context silently truncated the KG-triple block (appended last) and collapsed
B2/B3 to Acc 0.15 with 12/20 parse errors. The table above is post-fix
(`llm.num_ctx = 8192` + a hard `LLMContextBudgetError` guard).

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
