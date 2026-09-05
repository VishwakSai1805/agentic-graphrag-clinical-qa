# Next session — start here

**Last completed:** Phase 1 (commits `74457b0`, `976e388`) — tri-modal
retrieval infra (MedCPT dual-encoder dense + FAISS, BM25, RRF, MedCPT
cross-encoder rerank), a 113,465-chunk corpus (20k PubMed abstracts + all 18
USMLE textbooks), contamination assertion, and a three-tier retrieval exit
test. Real result: NFCorpus (BEIR) fusion gate passed as a **statistical
tie** between RRF and dense-only (paired bootstrap, not a raw point
estimate) — see ADR-012 in `docs/05_DECISIONS.md` for the full story,
including two confounded metric attempts (MedQA, then PubMedQA) before
landing on NFCorpus. 47 tests passing, ruff clean, CUDA 12.8 confirmed
working (RTX 5070 Ti).

**Recorded finding to carry forward, not forget:** dense-only retrieval
(V1) is a real, evidenced competitor — it saturates PubMedQA and
significantly beats BM25-only on NFCorpus while statistically tying fused
RRF. If V1 also matches/beats hybrid or agent variants on MedQA downstream
accuracy in Phase 7, report that honestly — don't retune fusion until it
"wins."

**Next:** Phase 2 — Knowledge graph layer (`docs/02_ROADMAP.md`).

---

## 1. Open Claude Code

```powershell
cd "C:\Claude\AI Project"
claude
```

Start a **fresh** session — do not resume the old one. The roadmap and ADRs
are the memory; a clean context works better.

## 2. Paste this as the first message

> Read CLAUDE.md and docs/00 through docs/05 (pay particular attention to
> ADR-007, ADR-012, and ADR-013 in docs/05_DECISIONS.md), then
> `git log --oneline` to see where we are. Phase 1 is done. Run `/phase 2`.

## 3. Expect

Phase 2 builds the KG layer: `docker/compose.yaml` for Neo4j 5, PrimeKG
loading, scispaCy/UMLS entity linking, and Cypher subgraph traversal. Per
ADR-007, design this so KG subgraph retrieval is **precomputed per question
into a cache file** (`scripts/precompute_kg_cache.py`), letting Neo4j be
shut down during eval runs — 16GB RAM is the binding constraint on this
machine, not VRAM, and Neo4j running alongside the LLM + retrieval stack
during eval would compete for it. The Phase 2 exit test explicitly requires
reproducing the same KG-hit rate and sample subgraphs after `docker compose
down`, from the cache alone.

## 4. Open questions parked for later

- Is a 7B local model sufficient for the headline results, or do the final
  runs need a stronger free-tier model? (Decide after the Phase 6
  sensitivity check, ADR-008.)
- Real average LLM calls/question for V4/V5 — unknown until Phase 4 prompts
  exist. Measure on a 20–50 question dev slice before locking the grid scale
  (`docs/04_EVALUATION.md` §0.2's 3× concurrency speedup is also still a
  planning estimate, not measured — Phase 6 replaces it).
- Whether NFCorpus's statistical tie (ADR-012) holds up as more retrieval
  changes land (reranking is implemented but not yet wired into the main
  pipeline/eval — `src/retrieval/rerank.py` exists, unused until Phase 3).
