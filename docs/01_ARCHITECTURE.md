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
- **Fires per *claim*, not per option** (amended — see ADR-025). A conflict
  exists when the **same claim** carries both SUPPORT and CONTRADICT cells
  from **different evidence ids**. An option with claim 1 supported and claim 3
  contradicted is *partially supported*, not in conflict — that is already
  captured by `(s⁺, s⁻)` and needs no resolution.
- **Semantic check first, precedence as tiebreaker second** (amended — ADR-025).
  On a claim-level conflict the agent (LLM, temperature=0) judges whether the
  two evidence items *genuinely* contradict on the same clinical point. Only
  if genuine does source precedence pick the winner; only if precedence also
  ties is the claim marked `unresolved`. A spurious "conflict" (the two items
  are about different aspects) is recorded as `spurious` and changes nothing.
- Precedence (as implemented, ADR-024): KG triple > textbook passage >
  PubMed abstract. The `abstract conclusion > background` split and "recency"
  from the original design are unimplementable on this corpus (no section
  tags, no dates) and are dropped.
- `unresolved` is a recorded outcome for Phase 5 calibration to act on (it
  *may* suppress that option's logit). It must **not** be used to suppress a
  logit on a degenerate rule — the pre-calibration Phase 4 sanity ranking
  does not penalise it (ADR-025).
- **Out:** resolved SupportMatrix + `conflict_log[]` (one entry per claim-level
  conflict evaluated; goes into the paper's qualitative analysis).

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
