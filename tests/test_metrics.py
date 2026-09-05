import math

import pytest

from src.retrieval.metrics import (
    dcg_at_k,
    mean_ndcg_at_k,
    mean_recall_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    paired_bootstrap_ci,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k_hit_and_miss():
    assert recall_at_k(["a", "b", "c"], {"b"}, k=3) == 1.0
    assert recall_at_k(["a", "b", "c"], {"z"}, k=3) == 0.0


def test_recall_at_k_respects_cutoff():
    # relevant id is present but beyond the cutoff
    assert recall_at_k(["a", "b", "c", "d"], {"d"}, k=2) == 0.0
    assert recall_at_k(["a", "b", "c", "d"], {"d"}, k=4) == 1.0


def test_reciprocal_rank_first_relevant_position():
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0
    assert reciprocal_rank(["a", "b", "c"], {"z"}) == 0.0


def test_reciprocal_rank_not_cut_by_any_k():
    ranking = [str(i) for i in range(50)]
    assert reciprocal_rank(ranking, {"49"}) == 1.0 / 50


def test_mean_recall_and_mrr_average_across_queries():
    rankings = [["a", "b"], ["x", "y"]]
    relevant = [{"a"}, {"z"}]
    assert mean_recall_at_k(rankings, relevant, k=2) == 0.5
    assert mean_reciprocal_rank(rankings, relevant) == 0.5


def test_ndcg_perfect_ranking_is_one():
    relevance = {"a": 2.0, "b": 1.0, "c": 0.0}
    perfect_ranking = ["a", "b", "c"]
    assert ndcg_at_k(perfect_ranking, relevance, k=3) == 1.0


def test_ndcg_worse_ranking_scores_lower_than_perfect():
    relevance = {"a": 2.0, "b": 1.0, "c": 0.0}
    perfect = ndcg_at_k(["a", "b", "c"], relevance, k=3)
    worse = ndcg_at_k(["c", "b", "a"], relevance, k=3)
    assert worse < perfect
    assert perfect == 1.0


def test_ndcg_handles_ties_sanely():
    # a and b have equal relevance -- either order is "ideal"
    relevance = {"a": 2.0, "b": 2.0, "c": 0.0}
    assert ndcg_at_k(["a", "b", "c"], relevance, k=3) == 1.0
    assert ndcg_at_k(["b", "a", "c"], relevance, k=3) == 1.0


def test_ndcg_missing_from_ranking_treated_as_zero_relevance():
    relevance = {"a": 3.0}
    # "a" (the only relevant doc) never appears in the ranking at all
    assert ndcg_at_k(["x", "y", "z"], relevance, k=3) == 0.0


def test_ndcg_zero_relevant_docs_returns_zero_not_error():
    assert ndcg_at_k(["a", "b"], {}, k=2) == 0.0
    assert ndcg_at_k(["a", "b"], {"a": 0.0, "b": 0.0}, k=2) == 0.0


def test_dcg_matches_hand_computed_value():
    # rel=[3,2,0] at positions 1,2,3 -> 3/log2(2) + 2/log2(3) + 0/log2(4)
    relevance = {"a": 3.0, "b": 2.0, "c": 0.0}
    expected = 3.0 / math.log2(2) + 2.0 / math.log2(3)
    assert math.isclose(dcg_at_k(["a", "b", "c"], relevance, k=3), expected)


def test_mean_ndcg_averages_across_queries():
    rankings = [["a", "b"], ["a", "b"]]
    relevances = [{"a": 1.0, "b": 0.0}, {"b": 1.0, "a": 0.0}]
    # query 1 is perfect (ndcg=1.0), query 2 is worst-case (ndcg=0 since the
    # only relevant doc "b" is ranked last of 2 -- still > 0 actually, so
    # use an explicit hand check instead of assuming worst-case is exactly 0.
    ndcg_q1 = ndcg_at_k(["a", "b"], relevances[0], k=2)
    ndcg_q2 = ndcg_at_k(["a", "b"], relevances[1], k=2)
    assert math.isclose(mean_ndcg_at_k(rankings, relevances, k=2), (ndcg_q1 + ndcg_q2) / 2)


def test_paired_bootstrap_ci_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        paired_bootstrap_ci([])


def test_paired_bootstrap_ci_detects_significant_positive_difference():
    # A consistently beats B by 0.1 with only tiny noise -- CI should sit
    # entirely above zero (A significantly > B).
    diffs = [0.1 + 0.001 * ((-1) ** i) for i in range(200)]
    mean_diff, lo, hi = paired_bootstrap_ci(diffs, n_resamples=2000, seed=1)
    assert math.isclose(mean_diff, 0.1, abs_tol=1e-6)
    assert lo > 0
    assert hi < 0.2


def test_paired_bootstrap_ci_detects_significant_negative_difference():
    diffs = [-0.1 + 0.001 * ((-1) ** i) for i in range(200)]
    _mean_diff, lo, hi = paired_bootstrap_ci(diffs, n_resamples=2000, seed=1)
    assert hi < 0
    assert lo > -0.2


def test_paired_bootstrap_ci_is_a_tie_when_noise_dominates():
    # A small, noisy positive point estimate with high per-item variance --
    # CI must still include zero (a real statistical tie), not falsely
    # detect a direction just because the point estimate isn't exactly 0.
    diffs = [1.0, -1.0, 1.0, -1.0, 0.5, -0.4]
    mean_diff, lo, hi = paired_bootstrap_ci(diffs, n_resamples=2000, seed=7)
    assert lo <= 0 <= hi
    assert mean_diff != 0  # point estimate isn't exactly zero, but still a tie


def test_paired_bootstrap_ci_deterministic_given_seed():
    diffs = [0.02, -0.01, 0.03, 0.0, -0.02, 0.01]
    result_a = paired_bootstrap_ci(diffs, n_resamples=500, seed=42)
    result_b = paired_bootstrap_ci(diffs, n_resamples=500, seed=42)
    assert result_a == result_b
