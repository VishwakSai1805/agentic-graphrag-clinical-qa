"""Conflict Resolution agent: claim-level, semantic-check-first (ADR-025).

Pure function over typed state (agents-engineer.md). Conflict fires for a
single `(option_id, claim_idx)` when that claim has both SUPPORT and CONTRADICT
cells backed by different evidence ids.

Two stages, in this order:

1. **Semantic judgement (LLM, always, temperature=0).** The model decides only
   whether the two sides *genuinely* contradict on this exact claim, or merely
   appear to (different aspects / populations / conditions). It never picks a
   winner.
2. **Source precedence (code, mechanical).** If the contradiction is genuine,
   the side with the lower `min(source_rank)` wins (lower rank = higher
   precedence). A tie on rank is `unresolved`.

`AgentParseError` from `parse_or_repair` is left to propagate.
"""

from __future__ import annotations

from src.agents.schemas import ConflictOut, parse_or_repair
from src.agents.state import ConflictLogEntry, SupportCell
from src.config import AgentsConfig
from src.llm.client import chat
from src.llm.prompts import load_prompt

# Imported by name so tests can monkeypatch `src.agents.conflict.chat`.
__all__ = ["run_conflict", "chat"]

_SYSTEM = (
    "You judge whether two pieces of clinical evidence genuinely contradict. "
    "Respond with STRICT JSON only."
)

_SPURIOUS_RATIONALE_CHARS = 380
_GENUINE_RATIONALE_CHARS = 340


def _unique(ids: list[str]) -> list[str]:
    """Order-preserving de-duplication."""
    seen: set[str] = set()
    out: list[str] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _side_ids(cells: list[SupportCell]) -> list[str]:
    return _unique([c.evidence_id for c in cells])


def _side_block(cells: list[SupportCell], evidence_by_id: dict) -> str:
    """Render one side as ``[<id>] (rank <n>) <text>`` lines, deduped by id."""
    lines = []
    for eid in _side_ids(cells):
        rank = next(c.source_rank for c in cells if c.evidence_id == eid)
        ev = evidence_by_id.get(eid)
        text = ev.text if ev is not None else "(evidence text unavailable)"
        lines.append(f"[{eid}] (rank {rank}) {text}")
    return "\n".join(lines)


def run_conflict(
    option_id: str,
    claim_idx: int,
    claim_text: str,
    support_cells: list[SupportCell],
    contradict_cells: list[SupportCell],
    evidence_by_id: dict,
    *,
    cfg: AgentsConfig,
    chat_fn=chat,
    seed: int = 42,
) -> tuple[ConflictLogEntry, int]:
    """Evaluate one claim-level SUPPORT/CONTRADICT conflict.

    Returns `(entry, n_llm_calls)`. `n_llm_calls` is 1 for the semantic
    judgement plus 0 or 1 for a repair attempt.
    """
    prompt = load_prompt("conflict", cfg.prompt_versions["conflict"]).format(
        option_id=option_id,
        claim_text=claim_text,
        support_block=_side_block(support_cells, evidence_by_id),
        contradict_block=_side_block(contradict_cells, evidence_by_id),
    )

    resp = chat_fn(prompt, system=_SYSTEM, temperature=0, seed=seed)
    n_llm_calls = 1
    parsed, n_repair = parse_or_repair(
        resp.text,
        ConflictOut,
        chat_fn=chat_fn,
        system=_SYSTEM,
        original_prompt=prompt,
        seed=seed,
    )
    n_llm_calls += n_repair

    model_rationale = (parsed.rationale or "").strip()

    if not parsed.genuine_contradiction:
        return (
            ConflictLogEntry(
                option_id=option_id,
                claim_idx=claim_idx,
                kept=[],
                dropped=[],
                outcome="spurious",
                rationale="spurious: " + model_rationale[:_SPURIOUS_RATIONALE_CHARS],
            ),
            n_llm_calls,
        )

    sup_ids = _side_ids(support_cells)
    con_ids = _side_ids(contradict_cells)
    sup_rank = min(c.source_rank for c in support_cells)
    con_rank = min(c.source_rank for c in contradict_cells)

    if sup_rank < con_rank:
        outcome, kept, dropped = "resolved_support", sup_ids, con_ids
    elif con_rank < sup_rank:
        outcome, kept, dropped = "resolved_contradict", con_ids, sup_ids
    else:
        outcome, kept, dropped = "unresolved", [], _unique(sup_ids + con_ids)

    rationale = (
        model_rationale[:_GENUINE_RATIONALE_CHARS]
        + f" | precedence {sup_rank} vs {con_rank}"
    )
    return (
        ConflictLogEntry(
            option_id=option_id,
            claim_idx=claim_idx,
            kept=list(kept),
            dropped=list(dropped),
            outcome=outcome,
            rationale=rationale,
        ),
        n_llm_calls,
    )
