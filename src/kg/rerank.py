"""KG triple selection for the `kg_rerank` ablation (Phase 2 follow-up,
docs/05_DECISIONS.md -- the ADR this module's addition records). See
CLAUDE.md's owning-agent principle: every Cypher string still lives in
`src/kg/queries.py` only -- this module does no traversal/Cypher, it just
selects *which* already-traversed triples survive `KGConfig.
max_triples_per_question`'s cap.

Context (ADR-015): `src/kg/traversal.py: traverse_live`'s existing behavior
truncates a question's candidate triples to the cap via a deterministic
alphabetical sort on (head, relation, tail). ADR-015's real measurement
(`relevant_rate = 0.4865`) shows broad PrimeKG hub nodes (e.g. "Pain")
frequently crowd out question-specific facts under that ordering. This module
adds an alternative: score each candidate triple's verbalised text against
the question text with the project's existing MedCPT cross-encoder
(`src/retrieval/rerank.py: CrossEncoderReranker`, reused as-is -- it already
scores arbitrary `Evidence` objects against a query string, so a KG triple
only needs verbalising into one first) and keep the top-`cap` by that score.

This is an ablatable flag (`KGConfig.kg_rerank`), not a silent replacement of
the existing behavior -- `select_triples_alphabetical` (the pre-existing
default path, factored out here so both paths are named, comparable
functions) is unchanged and still the `kg_rerank=False` default.
"""

from __future__ import annotations

from collections.abc import Callable

from src.kg.traversal import Triple
from src.kg.verbalize import verbalize_triple
from src.schema import Evidence

# (query, candidates, top_n) -> the top_n candidates from `candidates`, sorted
# descending by relevance to `query`. Matches
# `src.retrieval.rerank.CrossEncoderReranker.rerank`'s real signature exactly,
# so the real reranker's bound method can be passed here directly.
RerankFn = Callable[[str, list[Evidence], int], list[Evidence]]

_default_reranker = None


def _get_default_rerank_fn() -> RerankFn:
    """Lazily loads the real `CrossEncoderReranker` (MedCPT cross-encoder,
    same model/class Phase 1 already uses for dense+BM25 passage reranking --
    not reimplemented here) once, as a module-level singleton, and returns its
    `.rerank` bound method. Tests inject a fake `rerank_fn` instead of calling
    this, so importing this module never forces a torch/transformers load."""
    global _default_reranker
    if _default_reranker is None:
        from src.retrieval.rerank import CrossEncoderReranker

        _default_reranker = CrossEncoderReranker()
    return _default_reranker.rerank


def select_triples_alphabetical(triples: list[Triple], cap: int) -> list[Triple]:
    """The pre-existing `kg_rerank=False` default: deterministic alphabetical
    sort on (head, relation, tail), then first-`cap` truncation. Factored out
    here (mirroring what `traverse_live`'s own final truncation already does
    internally) so the two selection strategies are two named, directly
    comparable functions for the ablation, not one implicit default buried
    inside the traversal function."""
    return sorted(triples, key=lambda t: (t.head, t.relation, t.tail))[:cap]


def select_triples_reranked(
    triples: list[Triple],
    question_text: str,
    cap: int,
    rerank_fn: RerankFn | None = None,
) -> list[Triple]:
    """`kg_rerank=True` path: verbalises every triple in `triples` into an
    `Evidence` (`src.kg.verbalize.verbalize_triple`, unchanged, no
    reimplementation), scores each against `question_text` with `rerank_fn`
    (defaults to the real MedCPT `CrossEncoderReranker.rerank`), and returns
    the top-`cap` triples by that score, mapped back from the scored
    `Evidence` objects via the (head, relation, tail) key preserved in
    `Evidence.source_meta` by `verbalize_triple`.

    If `len(triples) <= cap`, no scoring is needed or performed -- returns
    every triple, alphabetically sorted (matching
    `select_triples_alphabetical`'s tie-breaking convention when there is
    nothing to actually select between; also keeps this function torch-free
    for the common case where a question's candidate pool never exceeded the
    cap in the first place).

    `rerank_fn` is injectable (matches `RerankFn`'s exact signature) so this
    selection logic is unit-testable without loading the real cross-encoder
    model, per this project's existing mocking convention (see
    `tests/test_kg_linker.py`, `src/kg/novelty.py`'s `ScoreFn`).
    """
    if len(triples) <= cap:
        return select_triples_alphabetical(triples, cap)

    key_to_triple: dict[tuple[str, str, str], Triple] = {
        (t.head, t.relation, t.tail): t for t in triples
    }
    evidences = [verbalize_triple(t) for t in triples]
    fn = rerank_fn if rerank_fn is not None else _get_default_rerank_fn()
    reranked_evidence = fn(question_text, evidences, cap)

    selected: list[Triple] = []
    for ev in reranked_evidence:
        meta = ev.source_meta
        key = (meta["head"], meta["relation"], meta["tail"])
        triple = key_to_triple.get(key)
        if triple is not None:
            selected.append(triple)
    return selected
