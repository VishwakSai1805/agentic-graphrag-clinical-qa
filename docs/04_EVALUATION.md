# 04 — Evaluation Protocol

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

| Variant | Dataset | Acc | Macro-F1 | Faithfulness | Halluc. rate | P@1 | P@2 | MRR | ECE | run_id |
|---|---|---|---|---|---|---|---|---|---|---|
| V0 zero-shot | MedQA | TBD | TBD | – | TBD | TBD | TBD | TBD | TBD | TBD |
| V1 dense RAG | MedQA | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| V2 hybrid | MedQA | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| V3 +KG | MedQA | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| V4 +agents | MedQA | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| V5 full | MedQA | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| V5 full | MedMCQA | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| V5 full | PubMedQA | TBD | TBD | TBD | TBD | – | – | – | TBD | TBD |

Also report, per variant: mean LLM calls/question, tokens/question, wall-clock,
KG-hit rate, and the failure-mode counters from `docs/01_ARCHITECTURE.md`.

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
