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
