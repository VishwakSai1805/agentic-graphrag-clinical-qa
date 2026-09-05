# 02 — Roadmap

Execute in order. One phase per Claude Code session where possible. Each phase
has an **exit test** — the phase is not done until that command passes and its
output is real.

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done (add run_id/commit)

---

## Phase 0 — Scaffold & environment  `[x]` (8d457e5, 2026-09-04)
- `pyproject.toml` (uv), `Makefile`, `.gitignore`, `.env.example`
- `src/llm/client.py`: LiteLLM gateway + on-disk cache + token/cost accounting
- `configs/base.yaml`; config loader with hash
- `tests/test_llm_cache.py` (mocked)
- **Exit test:** `pytest -q` green; `python -c "from src.llm.client import chat; print(chat('ping'))"` returns and the second call is a cache hit.
- Deviation from the original phase text, see ADR-004: `client.py` is an
  ordered fallback chain (Ollama primary, Groq/Gemini free-tier overflow)
  rather than a single provider, to satisfy the zero-API-spend constraint.
  Exit test verified live against Ollama alone, no API key present.

## Phase 1 — Corpora & lexical/dense indexes  `[x]` (74457b0, 2026-09-05)
- `scripts/fetch_datasets.py` — PubMedQA (pqa_labeled/pqa_artificial), MedQA-USMLE 4-opt, MedMCQA via HF
  `datasets`; MedQA textbook corpus (18 USMLE textbooks) via the HF mirror `MedRAG/textbooks` — this is
  the primary content source for USMLE-style questions, not a fast-follow (see ADR-009). If no HF
  mirror and the jind11/MedQA GitHub→Google Drive fallback both fail, the script halts rather than
  proceeding on an abstracts-only corpus.
- `scripts/build_corpus.py` — chunk (≈256 tok, 32 overlap), dedupe, `data/processed/corpus.jsonl`,
  `data/MANIFEST.json` (dataset revisions + corpus doc/chunk counts). Real build: 20,000 pubmed-abstract
  docs (fixed-seed subset of `pqa_artificial`) + 18 full textbooks → 113,465 chunks.
- `scripts/build_nfcorpus.py` — separate small NFCorpus (BEIR) index under `indexes/nfcorpus/`, the
  fusion-quality sanity-check corpus (see ADR-012) — not part of the project's actual retrieval corpus.
- `src/retrieval/dense.py` — MedCPT **dual encoder**: `MedCPT-Query-Encoder` for queries,
  `MedCPT-Article-Encoder` for indexed documents (see ADR-010, correcting CLAUDE.md §3) + FAISS;
  `lexical.py` (BM25), `fusion.py` (RRF), `rerank.py` (MedCPT cross-encoder), `metrics.py` (Recall@k,
  MRR, nDCG@k, `paired_bootstrap_ci` — all pure, tested functions).
- `src/eval/contamination.py` — assert no test-item abstract in the index (passed on the real build).
- **Exit test (methodology corrected twice — see ADR-011, then ADR-012 which supersedes ADR-011's
  gating role for PubMedQA):**
  `python -m scripts.eval_retrieval --dataset nfcorpus --limit 323` is the **primary, gating** run —
  NFCorpus (BEIR) is real, human-judged, graded-relevance IR with no single-modality saturation, unlike
  PubMedQA (see ADR-012 for why PubMedQA was retracted as the gate). Judged by a **pre-registered
  statistical-significance rule**, not raw point-estimate inequality: RRF significantly better than the
  best single modality → pass (fusion adds value); statistical tie (95% bootstrap CI includes 0) → pass
  (plumbing correct, dominance not demonstrated — Phase 7's job); RRF significantly worse → fail, real
  bug. **Real result: statistical tie** (rrf−dense mean diff −0.0034, 95% CI [−0.0171, +0.0103]; RRF
  significantly beats bm25, +0.0398 CI [+0.0242,+0.0564]) — gate **PASSED**.
  `python -m scripts.eval_retrieval --dataset pubmedqa --limit 200` is now an **infra-correctness floor**
  (all three modalities Recall@10 ≥ 0.9, a regression guard) plus non-gating diagnostics (Recall@1,
  hard-subset cut) — real result: floor passed comfortably (0.98–0.995 across all three).
  `python -m scripts.eval_retrieval --dataset medqa --limit 200` remains a **secondary diagnostic only**,
  labelled `lexically biased` in its own output, never gating (ADR-011).
  All three modes print `torch.cuda.is_available()` and device capability unconditionally (real run:
  RTX 5070 Ti, capability (12,0), CUDA confirmed working — no silent CPU fallback).

## Phase 2 — Knowledge graph layer  `[ ]`
- `docker/compose.yaml` Neo4j 5 + `scripts/load_primekg.py`
- `src/kg/linker.py` (scispaCy UMLS linking, threshold tuned on 50 hand-checked items)
- `src/kg/traversal.py` — 1–2 hop typed subgraph, edge-type allowlist, degree cap
- `src/kg/queries.py` — every Cypher string lives here
- Triple verbaliser → `Evidence(modality="kg")`
- `scripts/precompute_kg_cache.py` — runs linker + traversal once per question
  across every dataset/split that will be evaluated, writes
  `data/processed/kg_cache/<dataset>/<question_id>.json` (linked CUIs,
  subgraph triples, verbalised evidence). See ADR-007: this is what lets
  Neo4j be shut down for all of Phase 3–7's eval runs on 16GB RAM, and makes
  KG retrieval deterministic/replayable independent of Neo4j uptime.
- `src/kg/traversal.py` gains a cache-first read path: eval-time code reads
  the precomputed file and never opens a live Neo4j connection; only
  `precompute_kg_cache.py` and ad-hoc debugging talk to Neo4j directly.
- **Exit test:** `python -m scripts.kg_probe --dataset medqa --limit 200` reports KG-hit rate ≥ 60% and dumps 10 sample subgraphs for manual inspection; then `docker compose down` (Neo4j stopped) and re-running the same 200 questions from the cache file reproduces identical KG-hit rate and subgraphs byte-for-byte.

## Phase 3 — Tri-modal bundle + baseline RAG  `[ ]`
- `src/pipeline.py` assembling EvidenceBundle
- **Baselines**: B0 zero-shot LLM, B1 dense-only RAG, B2 hybrid RAG (no agents), B3 MEGA-RAG-style binary verifier reimplementation
- **Exit test:** all four baselines score on a 200-item MedQA dev slice, results written to `results/`.

## Phase 4 — Multi-agent consensus network  `[ ]`
- `src/agents/graph.py` LangGraph state machine + checkpointer
- `hypothesis.py`, `audit.py`, `conflict.py` with strict JSON schemas (pydantic) and repair-on-parse-failure
- Budget guards + failure-mode instrumentation from `docs/01_ARCHITECTURE.md`
- `tests/` with a mocked LLM covering: valid path, malformed JSON, invalid citation, conflict, budget trip
- **Exit test:** `python -m src.pipeline --config configs/agents.yaml --limit 20` produces SupportMatrices with zero invalid citations on the dev slice.

## Phase 5 — Evidential calibration & ranking  `[ ]`
- `src/calibration/scoring.py` (logit assembly), `softmax_rank.py` (temperature scaling, ECE), `citations.py`
- Fit α,β,γ,δ,T on a **dev** split only; freeze and record in the config
- Abstention threshold τ + coverage/risk curve
- **Exit test:** ECE on dev improves vs. raw LLM option log-probs; ranked output includes per-option citations for every question.

## Phase 6 — Evaluation harness  `[ ]`
- `src/eval/harness.py` (parallel, resumable, cached), `metrics.py` (Acc, Macro-F1, P@K, MRR, ECE), `ragas_eval.py`, `nli.py`
- Harness issues LLM calls through an `asyncio.Semaphore` sized by
  `configs/base.yaml: eval.concurrency` (default 4, matching
  `OLLAMA_NUM_PARALLEL=4` — see `docs/04_EVALUATION.md` §0.4 and ADR-006).
- Benchmark concurrency = 1, 2, 4 on a fixed dev slice; write the measured
  questions/hour for each into `results/` — this real number replaces the
  planning-estimate 3× speedup assumption in `docs/04_EVALUATION.md` §0.2
  before Phase 7 is scheduled.
- `scripts/report.py` → markdown + LaTeX tables into `paper/tables/`
- **Exit test:** one command reproduces a full dataset run from a config and regenerates identical tables; the concurrency benchmark table (1/2/4) is written to `results/` with no `TBD`.

## Phase 7 — Full runs & ablation grid  `[ ]`

| Variant | Dense | BM25 | KG | Agents | Calibration |
|---|---|---|---|---|---|
| V0 baseline LLM | – | – | – | – | – |
| V1 | ✓ | – | – | – | – |
| V2 | ✓ | ✓ | – | – | – |
| V3 | ✓ | ✓ | ✓ | – | – |
| V4 | ✓ | ✓ | ✓ | ✓ | – |
| V5 (full) | ✓ | ✓ | ✓ | ✓ | ✓ |
| V4−audit, V4−conflict | leave-one-agent-out | | | | |

**Grid scale (ADR-005, `docs/04_EVALUATION.md` §0.3 — the ceiling estimate in
§0.2 is what forced this cut):**
- Full 8-variant ablation (row above, including the two leave-one-out cells) —
  **MedQA only**, 500-question fixed seeded subset (seed 42).
- PubMedQA and MedMCQA — **V3 and V5 only**, 500-question fixed seeded subsets
  each (PubMedQA: official test; MedMCQA: seeded subsample of validation, see
  `docs/03_DATASETS.md`).
- 3 seeds (42/43/44) — **V3 and V5 on MedQA only**, the pair the paired
  bootstrap significance test compares. All other cells single-seed.
- Run with `eval.concurrency=4` and `OLLAMA_KEEP_ALIVE=-1`; expect the grid to
  span multiple background-job sessions (§0.2 estimates ~26h at 3×
  concurrency speedup, to be replaced by the Phase 6 measured number).
- **Sensitivity check (ADR-008):** V3 vs V5, 200 MedQA questions, via free
  Gemini Flash tier — reported alongside the grid as a robustness check, not
  a headline result (`docs/04_EVALUATION.md` §0.6/§5.1).

- **Exit test:** `results/` contains every cell of the cut grid above (not the
  original full cross-product); `docs/04_EVALUATION.md` §5 and §5.1 have no
  `TBD`.

## Phase 8 — Analysis & paper artifacts  `[ ]`
- Error taxonomy from `conflict_log` + failure-mode counters (100 sampled errors, hand-labelled)
- Figures: architecture, reliability diagram, ablation bars, coverage–risk curve
- `paper/` LaTeX with all tables generated by `scripts/report.py`
- **Exit test:** `make paper` builds the PDF from scratch with zero hand-typed numbers.
