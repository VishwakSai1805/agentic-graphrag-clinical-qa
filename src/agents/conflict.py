"""Conflict Resolution agent: a source-precedence resolver (ADR-022).

Pure function over typed state (agents-engineer.md). The winner of a
support/contradict conflict is computed DETERMINISTICALLY from `source_rank`
(already stamped on every `SupportCell`; lower rank = higher precedence):

    sup_rank = min(source_rank over the SUPPORT cells)
    con_rank = min(source_rank over the CONTRADICT cells)

The strictly-lower side is the `precedence_winner`; equal ranks are not
resolvable by source alone -> `unresolved`, and no LLM call is made.

When precedence gives a clear winner the LLM is asked ONE narrow question
(temperature=0, always): is the contradiction genuine (same clinical point) or
spurious (different aspects)? It may only return the `precedence_winner` or
`"UNRESOLVED"` -- never the losing side. If it returns the losing side (or an
otherwise unusable winner) we fall back to `precedence_winner` and record a note
on the rationale. `AgentParseError` from `parse_or_repair` is left to propagate.
"""

from __future__ import annotations

from src.agents.schemas import ConflictOut, parse_or_repair
from src.agents.state import _UNRANKED, ConflictLogEntry, SupportCell
from src.config import AgentsConfig
from src.llm.client import chat
from src.llm.prompts import load_prompt

# Imported by name so tests can monkeypatch `src.agents.conflict.chat`.
__all__ = ["run_conflict", "chat"]

_SYSTEM = "You resolve evidence conflicts by source precedence. Respond with STRICT JSON only."

_MAX_RATIONALE_CHARS = 400
_FALLBACK_NOTE = " (model returned an invalid winner; fell back to source precedence)"


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


def _side_block(ids: list[str], cells: list[SupportCell], evidence_by_id: dict) -> str:
    """Render one side as ``[<id>] (rank <n>) <text>`` lines (deduped by id)."""
    rank_by_id: dict[str, int] = {}
    for c in cells:
        rank_by_id.setdefault(c.evidence_id, c.source_rank)
    lines = []
    for eid in ids:
        ev = evidence_by_id.get(eid)
        text = ev.text if ev is not None else "(evidence text unavailable)"
        lines.append(f"[{eid}] (rank {rank_by_id.get(eid, _UNRANKED)}) {text}")
    return "\n".join(lines)


def run_conflict(
    option_id: str,
    support_cells: list[SupportCell],
    contradict_cells: list[SupportCell],
    evidence_by_id: dict,
    *,
    cfg: AgentsConfig,
    chat_fn=chat,
    seed: int = 42,
) -> tuple[ConflictLogEntry, int]:
    """Resolve one option's support/contradict conflict by source precedence.

    Returns `(entry, n_llm_calls)`. `n_llm_calls` is 0 when the source ranks
    tie (no call is made), otherwise 1 for the genuineness check plus 0 or 1
    for a repair attempt.
    """
    sup_ids = _side_ids(support_cells)
    con_ids = _side_ids(contradict_cells)
    all_ids = _unique(sup_ids + con_ids)

    sup_rank = min(c.source_rank for c in support_cells)
    con_rank = min(c.source_rank for c in contradict_cells)

    if sup_rank == con_rank:
        return (
            ConflictLogEntry(
                option_id=option_id,
                kept=[],
                dropped=all_ids,
                outcome="unresolved",
                rationale=(
                    f"tied source precedence (rank {sup_rank}); "
                    "not resolvable by source alone"
                ),
            ),
            0,
        )

    precedence_winner = "SUPPORT" if sup_rank < con_rank else "CONTRADICT"
    if precedence_winner == "SUPPORT":
        winning_ids, losing_ids, win_outcome = sup_ids, con_ids, "resolved_support"
    else:
        winning_ids, losing_ids, win_outcome = con_ids, sup_ids, "resolved_contradict"

    prompt = load_prompt("conflict", cfg.prompt_versions["conflict"]).format(
        option_id=option_id,
        precedence=">".join(cfg.source_precedence),
        support_block=_side_block(sup_ids, support_cells, evidence_by_id),
        contradict_block=_side_block(con_ids, contradict_cells, evidence_by_id),
        precedence_winner=precedence_winner,
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

    rationale = (parsed.rationale or "").strip()[:_MAX_RATIONALE_CHARS]

    if parsed.winner == "UNRESOLVED":
        return (
            ConflictLogEntry(
                option_id=option_id,
                kept=[],
                dropped=all_ids,
                outcome="unresolved",
                rationale=rationale,
            ),
            n_llm_calls,
        )

    if parsed.winner != precedence_winner:
        # The model picked the lower-precedence side (or an otherwise unusable
        # winner). ADR-022: never let the losing side win -- fall back.
        rationale = rationale + _FALLBACK_NOTE

    return (
        ConflictLogEntry(
            option_id=option_id,
            kept=list(winning_ids),
            dropped=list(losing_ids),
            outcome=win_outcome,
            rationale=rationale,
        ),
        n_llm_calls,
    )
