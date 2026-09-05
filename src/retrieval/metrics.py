"""Pure retrieval-quality metrics: Recall@k, (M)RR, and nDCG@k.

Pure functions over already-computed rankings -- no model, no I/O -- exactly
like fusion.py, and for the same reason: directly unit-testable
(tests/test_metrics.py) without a model/network/GPU.

scripts/eval_retrieval.py imports these rather than computing recall/MRR
inline, and uses ndcg_at_k for the NFCorpus (BEIR) fusion sanity gate, which
needs graded relevance -- PubMedQA/MedQA's binary "is this the gold doc"
notion of relevance is a special case (relevance in {0, 1}).
"""

from __future__ import annotations

import math
import random
from collections.abc import Collection, Mapping, Sequence


def recall_at_k(ranking: Sequence[str], relevant_ids: Collection[str], k: int) -> float:
    """1.0 if any of the top-k ranked ids is in `relevant_ids`, else 0.0.

    Single-query metric -- average with mean_recall_at_k for a corpus-level
    Recall@k. Binary relevance only (an id either is or isn't in
    `relevant_ids`); for graded relevance use ndcg_at_k instead.
    """
    top_k = ranking[:k]
    return 1.0 if any(doc_id in relevant_ids for doc_id in top_k) else 0.0


def reciprocal_rank(ranking: Sequence[str], relevant_ids: Collection[str]) -> float:
    """1 / (1-indexed rank of the first relevant id) over the *full* ranking
    (not cut to any k), or 0.0 if no relevant id appears at all.
    """
    for pos, doc_id in enumerate(ranking, start=1):
        if doc_id in relevant_ids:
            return 1.0 / pos
    return 0.0


def dcg_at_k(ranking: Sequence[str], relevance: Mapping[str, float], k: int) -> float:
    """Discounted Cumulative Gain: sum(rel_i / log2(i + 1)) over the top-k of
    `ranking`, 1-indexed. `relevance` maps id -> graded relevance (0 for ids
    it doesn't contain).
    """
    dcg = 0.0
    for i, doc_id in enumerate(ranking[:k], start=1):
        rel = relevance.get(doc_id, 0.0)
        if rel:
            dcg += rel / math.log2(i + 1)
    return dcg


def ndcg_at_k(ranking: Sequence[str], relevance: Mapping[str, float], k: int) -> float:
    """Normalised DCG@k with graded relevance.

    IDCG is the DCG of the ideal ranking: all ids with a nonzero entry in
    `relevance`, sorted descending by relevance. nDCG = DCG / IDCG, defined
    as 0.0 (not NaN) when IDCG is 0 (the query has no relevant ids at all --
    can't say anything about ranking quality, but a division error is worse
    than a defined zero here).

    Ties in `relevance` are handled correctly: DCG only depends on the
    multiset of relevance values realised at each rank, so any ordering of
    tied-relevance ids yields the same (I)DCG.
    """
    dcg = dcg_at_k(ranking, relevance, k)
    ideal_ranking = sorted(relevance, key=lambda doc_id: relevance[doc_id], reverse=True)
    idcg = dcg_at_k(ideal_ranking, relevance, k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def mean_recall_at_k(
    rankings: Sequence[Sequence[str]], relevant_ids_list: Sequence[Collection[str]], k: int
) -> float:
    if not rankings:
        return 0.0
    return sum(
        recall_at_k(ranking, relevant_ids, k)
        for ranking, relevant_ids in zip(rankings, relevant_ids_list, strict=True)
    ) / len(rankings)


def mean_reciprocal_rank(
    rankings: Sequence[Sequence[str]], relevant_ids_list: Sequence[Collection[str]]
) -> float:
    if not rankings:
        return 0.0
    return sum(
        reciprocal_rank(ranking, relevant_ids)
        for ranking, relevant_ids in zip(rankings, relevant_ids_list, strict=True)
    ) / len(rankings)


def mean_ndcg_at_k(
    rankings: Sequence[Sequence[str]], relevance_maps: Sequence[Mapping[str, float]], k: int
) -> float:
    if not rankings:
        return 0.0
    return sum(
        ndcg_at_k(ranking, relevance, k)
        for ranking, relevance in zip(rankings, relevance_maps, strict=True)
    ) / len(rankings)


def paired_bootstrap_ci(
    diffs: Sequence[float], n_resamples: int = 10_000, seed: int = 42, alpha: float = 0.05
) -> tuple[float, float, float]:
    """95%-style percentile bootstrap CI for the mean of paired per-query
    differences (e.g. per-query nDCG@10 of system A minus system B).

    Returns (mean_diff, ci_lo, ci_hi). `diffs` must be non-empty. Use this
    rather than comparing two raw point-estimate means directly -- see
    docs/04_EVALUATION.md §1 (paired bootstrap, 10k resamples is the
    project's standard significance methodology) and ADR-012 in
    docs/05_DECISIONS.md (the NFCorpus fusion-sanity gate is judged by this,
    not by raw point-estimate inequality, because a small point-estimate gap
    at a few hundred queries can be pure sampling noise).

    A caller checks significance as `ci_lo > 0` (A significantly > B) or
    `ci_hi < 0` (A significantly < B); `ci_lo <= 0 <= ci_hi` is a statistical
    tie -- not evidence either system is better, not "no difference exists."
    """
    if not diffs:
        raise ValueError("diffs must be non-empty")
    n = len(diffs)
    rng = random.Random(seed)
    boot_means = sorted(
        sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_resamples)
    )
    lo_idx = int((alpha / 2) * n_resamples)
    hi_idx = int((1 - alpha / 2) * n_resamples)
    return sum(diffs) / n, boot_means[lo_idx], boot_means[hi_idx]
