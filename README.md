# Agentic Multi-Evidence GraphRAG for Healthcare

Hallucination-mitigated clinical QA that goes beyond binary verification:
tri-modal retrieval (dense + BM25 + biomedical knowledge graph), a multi-agent
evidence-consensus network, and an evidential calibration layer that outputs
**probabilistically ranked differential diagnoses with explicit citations**.

Benchmarks: PubMedQA (parity), MedQA-USMLE (primary novelty), MedMCQA (breadth).

| | |
|---|---|
| How to drive this repo with Claude Code | [`START_HERE.md`](START_HERE.md) |
| Rules every session must follow | [`CLAUDE.md`](CLAUDE.md) |
| Research spec & gap | [`docs/00_RESEARCH_SPEC.md`](docs/00_RESEARCH_SPEC.md) |
| Architecture & agent contracts | [`docs/01_ARCHITECTURE.md`](docs/01_ARCHITECTURE.md) |
| Roadmap (phases 0–8) | [`docs/02_ROADMAP.md`](docs/02_ROADMAP.md) |
| Datasets & corpora | [`docs/03_DATASETS.md`](docs/03_DATASETS.md) |
| Evaluation protocol | [`docs/04_EVALUATION.md`](docs/04_EVALUATION.md) |
| Decision log | [`docs/05_DECISIONS.md`](docs/05_DECISIONS.md) |

**Status:** Phases 0–4 complete (retrieval infra, KG layer, baselines, multi-
agent consensus network). The `docs/04_EVALUATION.md` §5 ablation grid
(Phase 7) is still `TBD` — every number that does exist there is traceable to
a committed `results/*/metrics.json`, per CLAUDE.md §2.

Uses only public, de-identified benchmark data. Research artifact — not a
clinical decision support system and not for patient care.
