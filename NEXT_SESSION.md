# Next session — start here

**Last completed:** Phase 2 (commits `82f7824`, `04c7c6f`, follow-up
`36071ef`) — Knowledge graph layer. Real Neo4j 5 + PrimeKG (129,375 nodes /
740,674 edges after the 7-item allowlist, out of 8,100,498 raw directional
rows / ~4.05M unique undirected edges — see ADR-014), scispaCy/UMLS entity
linking, bounded 1–2 hop traversal with a cache-first read path (ADR-007
confirmed byte-for-byte reproducible after a full `docker compose down` +
container removal), triple verbalization, and `scripts/kg_probe.py` as the
KG-hit-rate + marginal-value diagnostic. 115 tests passing, ruff clean.
KG cache precomputed for MedQA test (n=200), PubMedQA (500 test + 500 dev),
and MedMCQA (500-question seeded validation subsample) — matches ADR-005's
evaluation scope, not full splits.

**The literal exit test passed** (`aggregate_hit_rate = 0.960` ≥ 0.60 floor).
**The honest finding underneath it did not, and this is what Phase 3 actually
inherits — read this before assuming KG retrieval is a solved, valuable
modality:**

- The 96% hit rate is a **question-level arithmetic artifact** of high entity
  fan-out (~74 linked entities/question), not evidence of PrimeKG coverage.
  Real per-entity mapping rate is only **11.22%** (`entity_level_mapping_rate`)
  — always cite this alongside any hit rate. See ADR-016.
- The defensible C1 evidence figure is **`novel_and_relevant_rate = 0.4063`**
  (fraction of retrieved KG triples that are both on-topic *and* not already
  in the real 12-item dense `EvidenceBundle`) — **real but middling**,
  bottlenecked by retrieval precision (only ~47% of kept triples are even
  on-topic; hub-node noise, ADR-015) more than by redundancy with dense
  passages. **The KG modality's marginal value over dense retrieval is not
  yet established.**
- An ablatable `kg_rerank` flag (MedCPT cross-encoder triple selection,
  ADR-018) shows a narrow *significant* win on `relevant_rate` (paired
  bootstrap CI [+0.003, +0.105]) but a **tie** on the primary
  `novel_and_relevant_rate` metric (CI [−0.013, +0.088]) at n=200. Ships off
  by default — real signal, not yet strong enough to adopt.

**Next:** Phase 3 — Tri-modal bundle + baseline RAG (`docs/02_ROADMAP.md`).

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
> ADR-007 and ADR-014 through ADR-018 in docs/05_DECISIONS.md), then
> `git log --oneline` to see where we are. Phase 2 is done. Run `/phase 3`.
>
> Strategic context: Phase 2 found that PrimeKG's real value-add over dense
> retrieval is not yet established (`novel_and_relevant_rate ≈ 0.41`,
> middling, bottlenecked by retrieval precision not redundancy). Phase 3
> assembles the tri-modal `EvidenceBundle` and baselines B0–B3 — treat
> whether KG evidence actually moves downstream MedQA accuracy as an open
> empirical question the baselines/ablation must answer, not an assumption
> to build on.

## 3. Expect

Phase 3 assembles `src/pipeline.py`'s `EvidenceBundle` (dense + BM25 + RRF +
rerank, now also KG triples read via `src/kg/traversal.py:
get_subgraph_cached` — cache-only, Neo4j does not need to be running) and
implements four baselines (B0 zero-shot, B1 dense-only RAG, B2 hybrid RAG,
B3 MEGA-RAG-style binary verifier reimplementation), scored on a 200-item
MedQA dev slice. This is the first phase that actually calls the LLM
gateway (`src/llm/client.py`) in a real eval loop — check the Ollama server
is up (`OLLAMA_KEEP_ALIVE`/`OLLAMA_NUM_PARALLEL` per ADR-006) before running
anything at scale.

## 4. Open questions parked for later

- Is a 7B local model sufficient for the headline results, or do the final
  runs need a stronger free-tier model? (Decide after the Phase 6
  sensitivity check, ADR-008.)
- Real average LLM calls/question for V4/V5 — unknown until Phase 4 prompts
  exist. Measure on a 20–50 question dev slice before locking the grid scale
  (`docs/04_EVALUATION.md` §0.2's 3× concurrency speedup is also still a
  planning estimate, not measured — Phase 6 replaces it).
- **BM25 may be under-tuned, which would unfairly weaken B2/V2 in the
  ablation.** Published BEIR baselines put Anserini BM25 on NFCorpus around
  nDCG@10 ≈ 0.32; this build got 0.2929. Likely causes: `rank_bm25` defaults
  (k1=1.5, b=0.75) vs. BEIR's (k1=0.9, b=0.4), no stemming/stopword removal,
  chunking short NFCorpus documents disturbing length normalisation. Not a
  Phase 1 blocker (the gate was about fusion plumbing), but a reviewer will
  notice a lexical baseline below the published number, and every
  BM25-inclusive variant's ablation row depends on it being fairly
  represented. Fix or explicitly justify before Phase 7 runs.
- If V1 (dense-only) matches/beats hybrid or agent variants on MedQA
  downstream accuracy in Phase 7, report that honestly (ADR-012) — don't
  retune fusion or KG retrieval until it "wins."
- `kg_rerank`'s tied primary metric (ADR-018) was measured at n=200 — worth
  re-testing at a larger n (e.g. as part of Phase 7's grid, where it would
  earn its own ablation row) rather than assumed settled either way.
- Full-scope KG novelty scoring (all ~38 triples/question × 12 passages,
  currently a disclosed 10-triple subset per ADR-016) and a paraphrase-aware
  relevance check (currently exact-substring) are legitimate follow-ups if
  the C1 figure needs to be tightened for the paper.
