"""Modality-reserved trim of the retrieval bundle to `max_evidence` items
(ADR-023, and the design constraint recorded in docs/02_ROADMAP.md Phase 4).

A single top-`max_evidence` cross-modal ranking would eliminate the KG
modality: MedCPT's cross-encoder is a query-*article* model, so a short
verbalised triple systematically scores below a full passage, and the agent
network would be "tri-modal in name, dense-only in fact" -- the same silent
elimination class as the Phase 3 context-truncation bug (ADR-021).

So: rank WITHIN each modality and reserve slots. `evidence_split = [n_p, n_t]`
is `[passages, triples]`; the triple count is a FLOOR, not a fixed ratio --
unused KG slots backfill with passages so the bundle reaches `max_evidence`
whenever that much evidence exists (a zero-triple question must not hand the
agents fewer items than B2 got). The realised split is returned in a
`MixReport` that callers store as an analysis variable, not just log.
"""

from __future__ import annotations

from collections.abc import Callable

from src.agents.state import MixReport
from src.schema import Evidence

# (query, candidates, top_n) -> reranked top_n. Matches
# src.retrieval.rerank.CrossEncoderReranker.rerank so the real model drops in.
KGSlotRankFn = Callable[[str, list[Evidence], int], list[Evidence]]


def _identity_rank(_query: str, candidates: list[Evidence], top_n: int) -> list[Evidence]:
    """Fallback when KG-slot reranking is disabled: keep cache order."""
    return candidates[:top_n]


def select_evidence(
    *,
    query: str,
    passages: list[Evidence],
    triples: list[Evidence],
    split: tuple[int, int],
    max_evidence: int,
    kg_slot_rank_fn: KGSlotRankFn | None = None,
) -> tuple[list[Evidence], MixReport]:
    """Return `(evidence, mix)`.

    `passages` are assumed already ranked best-first (the Phase 3 pipeline
    cross-encoder-reranks them). `triples` are the bundle's KG evidence,
    ranked into the KG slot by `kg_slot_rank_fn` (default: keep order).
    """
    n_p_want, n_t_want = split
    rank_fn = kg_slot_rank_fn or _identity_rank

    kg_available = len(triples)
    n_t = min(n_t_want, kg_available)
    kg_slot = rank_fn(query, list(triples), n_t) if n_t else []

    # Unused KG slots become passage slots; never exceed max_evidence.
    freed = n_t_want - n_t
    n_p = min(len(passages), n_p_want + freed, max_evidence - n_t)
    passage_slot = passages[:n_p]

    evidence = [*passage_slot, *kg_slot]
    mix = MixReport(
        requested_split=(n_p_want, n_t_want),
        n_passages=len(passage_slot),
        n_triples=len(kg_slot),
        kg_available=kg_available,
        kg_slot_filled=n_t >= n_t_want,
        backfilled_passages=max(0, len(passage_slot) - n_p_want),
    )
    return evidence, mix
