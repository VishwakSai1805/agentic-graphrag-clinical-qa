# 01 — Architecture

```
                        [Complex Clinical Query + Option Set]
                                       │
                                       ▼
┌───────────────────────────────────────────────────────────────────────┐
│ 1. TRI-MODAL HYBRID RETRIEVAL ENGINE                                  │
│   ┌──────────────┐  ┌──────────────┐  ┌────────────────────────────┐  │
│   │ Dense        │  │ Lexical      │  │ KG Traversal               │  │
│   │ MedCPT+FAISS │  │ BM25         │  │ scispaCy link → Neo4j/     │  │
│   │ top-k=50     │  │ top-k=50     │  │ PrimeKG 2-hop subgraph     │  │
│   └──────┬───────┘  └──────┬───────┘  └──────────────┬─────────────┘  │
│          └────────── RRF fusion (k=60) ──────────────┘                │
│                        │ → MedCPT cross-encoder rerank → top-n=12     │
└────────────────────────┼──────────────────────────────────────────────┘
                         ▼   EvidenceBundle{passages[], triples[]}
┌───────────────────────────────────────────────────────────────────────┐
│ 2. MULTI-AGENT REASONING & EVIDENCE CONSENSUS NETWORK (LangGraph)     │
│                                                                       │
│   HypothesisAgent  ──►  EvidentialAuditAgent  ──►  ConflictAgent      │
│   maps each option      per (option, evidence):     resolves          │
│   to claims + the       SUPPORT/CONTRADICT/         contradictions    │
│   evidence it needs     NEUTRAL + rationale         by source         │
│         │                      │                    precedence &      │
│         │                      │                    recency; may      │
│         └──── targeted re-retrieval loop ◄──────────  request retry   │
│                    (max 2 rounds, budgeted)                           │
└────────────────────────┬──────────────────────────────────────────────┘
                         ▼   SupportMatrix[k options × evidence] + rationales
┌───────────────────────────────────────────────────────────────────────┐
│ 3. MULTI-CHOICE EVIDENTIAL CALIBRATION & RANKING                      │
│   z_i = α·s⁺ − β·s⁻ + γ·kg_i + δ·rerank_i     (weights fit on dev)    │
│   p = softmax(z / T)                          (T fit on dev, ECE)     │
│   → ranked differential + per-option citations + abstain if p1 < τ    │
└───────────────────────────────────────────────────────────────────────┘
```

## Agent contracts

All agents are **pure functions over typed state** — no hidden memory, no free
chat. State is a pydantic model checkpointed by LangGraph so any run can be
replayed from any node.

### HypothesisAgent
- **In:** query, options, EvidenceBundle
- **Out:** `[{option_id, claims:[str], required_evidence_types:[str]}]`
- Decomposes each option into atomic verifiable claims. May sample (seeded).

### EvidentialAuditAgent
- **In:** claims, EvidenceBundle
- **Out:** `SupportMatrix[claim × evidence] ∈ {SUPPORT, CONTRADICT, NEUTRAL}`
  with a required `evidence_id` and a ≤2-sentence rationale per non-neutral cell.
- `temperature=0`. Any cell without a valid `evidence_id` is discarded and logged.

### ConflictResolutionAgent
- **Fires only when** an option has both SUPPORT and CONTRADICT mass above ε.
- Precedence: KG triple from curated source > guideline/textbook passage >
  abstract conclusion > abstract background. Ties → mark `unresolved`, which
  suppresses that option's logit rather than guessing.
- **Out:** resolved SupportMatrix + `conflict_log[]` (goes into the paper's
  qualitative analysis).

### Budget guards
`max_rounds=2`, `max_llm_calls_per_question=12`, `max_evidence=12`. Exceeding a
guard is a hard stop that records `status: budget_exceeded`, never a silent
truncation — the eval harness counts these separately.

## Evidence object

```python
Evidence(
  id="chunk::PMID12345#3" | "kg::hypertension|treated_by|lisinopril",
  modality="dense"|"lexical"|"kg",
  text=...,            # triple verbalised for KG items
  source_meta={...},   # pmid, section, kg source db, edge type
  scores={"dense":.., "bm25":.., "rrf":.., "rerank":..},
)
```

Citations in the final answer are exactly these ids. `src/calibration/citations.py`
resolves them to human-readable references for the output.

## Failure modes to instrument (not just handle)

| Mode | Detection | Recorded as |
|---|---|---|
| Empty KG hit | linker returns no CUI or no edge | `kg_hit=false` per question |
| All-neutral matrix | no SUPPORT/CONTRADICT above ε | `status=insufficient_evidence` |
| Unresolved conflict | ConflictAgent ties | `status=unresolved_conflict` |
| Hallucinated citation | evidence_id not in bundle | `invalid_citation_count` |
| Budget exceeded | guard trip | `status=budget_exceeded` |

Report these rates in the paper — they are a contribution, not an embarrassment.
