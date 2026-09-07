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
| B3 binary verifier  | 0.65 | 0.61 | 0.95 | 3 | 20 ok |

† position-bias check, not a skill metric on 4-option MCQ (ADR-020).
‡ B0 runs no retrieval — ungrounded by construction, not a failure (ADR-020).

**B3 diagnostic (the useful number), reported under both context conditions.**
`verifier_decision_rate = 0.05` in *both* runs — the binary verifier isolated
exactly one SUPPORTED option on 1/20 questions at `num_ctx=4096` and on 1/20
questions at `num_ctx=8192`. What changes between them is only the *direction*
in which it fails:

| Run | `num_ctx` | `n_supported` distribution | exactly-one | `verifier_decision_rate` | failure direction |
|---|---|---|---|---|---|
| `medqa_B3_d5e3af5_20260906T100551Z` (superseded) | 4096 | `{0: 7, 1: 1, 3: 2, 4: 10}` | 1/20 | **0.05** | over-admits — 10/20 questions have *all four* options SUPPORTED |
| `medqa_B3_d5e3af5_20260906T102134Z` (reported) | 8192 | `{0: 15, 1: 1, 2: 4}` | 1/20 | **0.05** | over-rejects — 15/20 questions have *no* option SUPPORTED |

Both distributions are traceable to the `n_supported_distribution` and
`verifier_decision_rate` keys of the two `metrics.json` files named above.

Read correctly, this is a **stronger** result than the single-run version, not a
weaker one. The decision rate is not a fragile 0.05 that happened to fall out of
one context setting: under a perturbation that inverts the verifier's entire
error mode — from admitting everything to admitting nothing — the rate at which
it isolates a single option is *invariant at 1/20*. A per-option yes/no verifier
does not rank competing options; it saturates at one end or the other, and which
end it saturates at is an artifact of how much evidence fits in the window. That
is **direct empirical evidence for the binary bottleneck C2 targets**, and the
invariance across the two conditions is what makes it evidence rather than a
single-point coincidence. To be re-measured at scale in Phase 7. See ADR-020.

Two limits stand regardless: n=20 single-seed (so 1/20 carries a wide interval),
and the prompt-sensitivity check — B3 under a neutral and a permissive verifier
phrasing — has **not** been run (AUDIT §4.1, deferred to the pre-Phase-7 batch;
`docs/02_ROADMAP.md`). Until it has, this result is stated as robust to the
context window, not to prompt wording.

**Recorded per ADR-021:** the 4096-token batch was Ollama's default context,
which silently truncated the KG-triple block (appended last) and collapsed
B2/B3 to Acc 0.15 with 12/20 parse errors. The main table above is post-fix
(`llm.num_ctx = 8192` + a hard `LLMContextBudgetError` guard). ADR-021 also
records the no-op perturbation floor measured from the same pair of batches:
B0, which does no retrieval and is therefore *unaffected* by the context fix,
still flips one of 20 answers between them — so a 1/20 (0.05) accuracy delta
across these two batches is inside the noise floor and is not read as an effect.

### 5.5 Phase 4 multi-agent consensus plumbing check (dev slice, n=20 — NOT a grid result)

`python -m src.pipeline --config configs/agents.yaml --limit 20` (MedQA train
dev-20, seed 42), `results/medqa_V4_1388904_20260906T115020Z/`. Certifies that
query → tri-modal bundle → modality-reserved 12-item evidence set →
Hypothesis→Audit→Conflict → `SupportMatrix` with only real citations runs
end-to-end. **Not** a QA-accuracy result. Numbers below are post the ADR-025
claim-level conflict fix.

| Invariant / counter | Value |
|---|---|
| SupportMatrices emitted | 20/20 (+ per-node traces), 20 distinct |
| surviving cells citing a non-bundle id | **0** (exit-test invariant) |
| RAW model citation hallucination (dropped at audit) | 6 cells = 5.3% of produced cells |
| status | `ok` 19 · `insufficient_evidence` 1 |
| `budget_exceeded` rate | 0.00 |
| mean LLM calls / question | 5.8 (cap 12) · mean rounds 1.05 |
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

**Finding 3 — the uncalibrated agent signal is below B2.** `rank-1-by-support`
(crude `s⁺ − s⁻` argmax, no calibration, `resolved` not penalised per ADR-025)
= **0.35**, vs B2's 0.65 (§5.4) on the same slice. The gold option carries the
(tied-)max `s⁺` on only 10/20 questions, and the Audit agent is SUPPORT-heavy
(78 vs 30). Turning this raw count vector into a calibrated logit that beats
B2 is precisely Phase 5's job (fit α,β,γ,δ,T on dev); a raw argmax
underperforming a direct LLM answer is expected, but the gap is the number
Phase 5 has to close and Phase 6 has to measure at scale.

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
