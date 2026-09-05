# CLAUDE.md — Agentic Multi-Evidence GraphRAG for Healthcare

This file is auto-loaded into every Claude Code session in this repo. It is the
project constitution: read it before proposing or writing anything.

## 1. What we are building

An **Agentic Multi-Evidence Graph-RAG framework** that extends hallucination
mitigation from binary (yes/no) verification to **multi-choice clinical reasoning
and probabilistically ranked differential diagnoses with explicit citations**.

Research gap: MEGA-RAG-class systems verify answers with binary-constrained
modules (DPAG/DISC). Real clinical workflows need multi-option ranking. We
remove the binary bottleneck with a multi-agent consensus network plus an
evidential calibration layer.

Three-stage pipeline:

1. **Tri-Modal Hybrid Retrieval** — dense (FAISS) + lexical (BM25) + biomedical
   KG traversal (Neo4j / PrimeKG), fused with Reciprocal Rank Fusion.
2. **Multi-Agent Reasoning & Evidence Consensus** — Hypothesis Generation,
   Evidential Audit, Conflict Resolution agents over a deterministic LangGraph.
3. **Multi-Choice Evidential Calibration & Ranking** — per-option evidence
   scores → temperature-scaled softmax → ranked differential + citation set.

Full detail: `docs/00_RESEARCH_SPEC.md`, `docs/01_ARCHITECTURE.md`.
Execution order: `docs/02_ROADMAP.md`. Never invent a phase that isn't there —
amend the roadmap first, then implement.

## 2. Non-negotiable research hygiene

These are correctness rules, not style preferences. Violating them invalidates
the paper.

- **Never fabricate a metric.** Every number that appears in a doc, table, chat
  message or README must be traceable to a JSON file under `results/`. If a run
  hasn't happened, write `TBD`, not a plausible-looking figure.
- **Every eval run is reproducible.** Seeded (`seed: 42` default), config-hashed,
  and written to `results/<run_id>/{config.yaml,predictions.jsonl,metrics.json}`.
  `run_id = <dataset>_<variant>_<git_sha7>_<utc_ts>`.
- **No test-set contamination.** Retrieval corpora are built from training/
  external corpora only. PubMedQA labelled test abstracts must not be indexed
  in the retrieval corpus for the PubMedQA eval. Assert this in code
  (`src/eval/contamination.py`), do not merely comment it.
- **Answers must be grounded.** Every generated claim carries an evidence id
  (`chunk::<doc_id>#<span>` or `kg::<head>|<rel>|<tail>`). An answer with an
  empty citation set is a bug, and is scored as ungrounded.
- **LLM calls are cached.** All model calls go through `src/llm/client.py`,
  which caches on (model, prompt_hash, params) into `.cache/llm/`. Never call a
  provider SDK directly from feature code.
- **Determinism first.** `temperature=0` for audit/calibration agents. Only the
  Hypothesis agent may sample, and only with a recorded seed.
- **Compare systems by significance, not point estimates.** Any claim that
  system/variant A beats B must be backed by paired-bootstrap significance
  (10k resamples, see `docs/04_EVALUATION.md` §1 and
  `src/retrieval/metrics.py: paired_bootstrap_ci`), never by a raw mean
  inequality alone. A small eval set makes point estimates misleading — see
  ADR-012, where a real "RRF loses" point estimate turned out to be a
  statistical tie. A tie is reported as a tie, never as a win.
- **GPU-dependent scripts print CUDA status unconditionally.** Any script
  that uses `torch` on a GPU-relevant path must print
  `torch.cuda.is_available()` (and device name/capability if true) at
  startup — never let a silent CPU fallback hide and quietly cost hours.

## 3. Stack (decided — do not re-litigate without an ADR)

| Layer | Choice | Notes |
|---|---|---|
| Language | Python 3.11 | `uv` for env; fall back to venv+pip |
| Config | Hydra-style YAML in `configs/` | one YAML per experiment variant |
| Dense retrieval | FAISS (IVF-Flat → HNSW) | embeddings: MedCPT dual encoder — `ncbi/MedCPT-Query-Encoder` (queries) + `ncbi/MedCPT-Article-Encoder` (documents), see ADR-010; fallback `BAAI/bge-large-en-v1.5` (symmetric) |
| Lexical | `rank_bm25` (dev) → Pyserini (scale) | |
| KG | Neo4j 5 in Docker, PrimeKG dump | Cypher via `neo4j` driver, all queries in `src/kg/queries.py` |
| Entity linking | scispaCy `en_core_sci_lg` + UMLS linker | QuickUMLS fallback |
| Agent orchestration | **LangGraph** | chosen for deterministic, checkpointable state graphs over CrewAI/AutoGen |
| LLM gateway | LiteLLM | supports Anthropic / OpenAI / local vLLM / Ollama via one config key |
| Reranker | `ncbi/MedCPT-Cross-Encoder` | |
| NLI hallucination check | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli` | |
| Faithfulness | RAGAS | pinned version, offline-capable |
| Metrics | scikit-learn + custom `src/eval/metrics.py` | Acc, Macro-F1, P@K, MRR, ECE |
| Testing | pytest + ruff + mypy (non-strict) | |

## 4. Repo layout

```
configs/            experiment YAMLs (one per ablation variant)
data/raw/           downloaded datasets + corpora (gitignored)
data/processed/     chunked corpora, entity maps (gitignored)
indexes/            FAISS + BM25 artifacts (gitignored)
src/
  llm/              cached LLM gateway, prompt templates
  retrieval/        dense.py, lexical.py, fusion.py (RRF), rerank.py
  kg/               loader.py, linker.py, traversal.py, queries.py
  agents/           graph.py (LangGraph), hypothesis.py, audit.py, conflict.py
  calibration/      scoring.py, softmax_rank.py, citations.py
  eval/             harness.py, metrics.py, ragas_eval.py, nli.py, contamination.py
  pipeline.py       end-to-end entrypoint
tests/              pytest; unit tests mock the LLM gateway
scripts/            one-shot data prep / index build scripts
results/            run outputs (committed: metrics.json only)
docs/               spec, architecture, roadmap, ADRs
paper/              LaTeX, generated tables + figures
```

## 5. Working agreement for Claude Code

- **Plan before code.** For anything beyond a one-file change, enter plan mode
  (`shift+tab` twice / `/phase`), produce the plan, wait for approval.
- **One roadmap phase per session** where possible. At the end of a phase run
  `/checkpoint`.
- **Small vertical slices.** A phase is done when it runs end-to-end on a
  20-question dev slice, not when the code "looks complete".
- **Tests accompany logic.** Any module under `src/` that has branching logic
  gets a `tests/test_<module>.py`. Mock the LLM; never hit a provider in CI.
- **Long jobs go to a script, not the REPL.** Index builds and full evals run as
  `python -m scripts.<name>` with `nohup`/logged output so a crashed session
  doesn't lose work.
- **Record decisions.** Any deviation from this file → append an ADR entry to
  `docs/05_DECISIONS.md` with date, decision, alternatives, rationale.
- **Don't touch `results/` by hand.** Ever.

## 6. Commands

```bash
uv sync                                   # env
make index                                # build FAISS + BM25
make kg-up                                # docker compose up neo4j + load PrimeKG
python -m src.pipeline --config configs/full.yaml --query "..."
python -m src.eval.harness --config configs/medqa_full.yaml --limit 50
pytest -q && ruff check src tests
```

## 7. Definition of done for the project

All rows of `docs/04_EVALUATION.md` §5 filled from real runs: PubMedQA (baseline
parity), MedQA-USMLE and MedMCQA (novelty), full ablation grid, RAGAS
faithfulness, NLI hallucination rate, P@K/MRR, and calibration (ECE + reliability
diagram) — plus `paper/` tables generated by script, never typed by hand.
