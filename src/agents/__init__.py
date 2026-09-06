"""Phase 4 multi-agent evidence-consensus network.

Hypothesis -> Audit -> Conflict over a deterministic LangGraph
(docs/01_ARCHITECTURE.md). Every agent is a pure function over typed pydantic
state; every LLM output is schema-validated with exactly one repair attempt
(src.agents.schemas.parse_or_repair) before a recorded hard failure. Audit and
Conflict run at temperature=0; Hypothesis is deterministic by default with
sampling available only as an ablation flag (ADR-001, ADR-024).
"""
