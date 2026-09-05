"""Reciprocal Rank Fusion (RRF) -- the only cross-modality fusion this project
uses (ADR-002). Score normalisation across modalities (dense/lexical/KG) is
forbidden: their raw scores are not comparable, so fusion works on rank
position only.

k=60 is fixed project-wide unless a new ADR changes it (see ADR-002 and
docs/01_ARCHITECTURE.md). Do not make it configurable away from 60 without
recording that decision in docs/05_DECISIONS.md.
"""

from __future__ import annotations

from collections import defaultdict

RRF_K_DEFAULT = 60


def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int = RRF_K_DEFAULT
) -> list[tuple[str, float]]:
    """Fuses one or more ranked lists of ids (e.g. chunk ids) via RRF.

    rankings: one ranked list of ids per modality (best-first, no duplicates
        expected within a single ranking, but duplicates are tolerated -- a
        repeated id within one ranking simply accumulates score from each
        occurrence rather than raising).
    k: RRF constant. score(id) = sum over rankings containing id of
        1 / (k + rank), where rank is 1-indexed position in that ranking.

    Returns (id, score) pairs sorted descending by score. An id missing from
    a given ranking simply contributes 0 from that ranking -- it is not
    penalised beyond "not being there".

    Pure function: no model, no I/O. This is what tests/test_fusion.py
    exercises directly.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
