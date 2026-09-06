"""Budget guards (docs/01_ARCHITECTURE.md "Budget guards", agents-engineer.md).

`max_rounds=2`, `max_llm_calls_per_question=12`. Tripping a guard is a hard
stop that sets `status="budget_exceeded"` on the state -- never a silent
truncation. The graph routes a non-"running" status straight to END.

These are pure predicates; the graph owns the routing and the call counter
lives on `ConsensusState.llm_calls`.
"""

from __future__ import annotations

from src.config import AgentsConfig


def calls_would_exceed(current_calls: int, planned_calls: int, cfg: AgentsConfig) -> bool:
    """True if issuing `planned_calls` more LLM calls now would break the
    per-question ceiling. A node checks this BEFORE calling and, if True,
    returns `status="budget_exceeded"` instead of proceeding."""
    return current_calls + planned_calls > cfg.max_llm_calls_per_question


def rounds_exhausted(current_round: int, cfg: AgentsConfig) -> bool:
    """True if a re-retrieval round beyond the current one is not allowed."""
    return current_round >= cfg.max_rounds
