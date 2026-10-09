import math

import pytest

from src.retrieval.metrics import (
    dcg_at_k,
    holm_bonferroni,
    marginal_bootstrap_ci,
    mean_ndcg_at_k,
    mean_recall_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    paired_bootstrap_ci,
    paired_bootstrap_p,
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


# --------------------------------------------------------------------------
# marginal_bootstrap_ci (ADR-051 -- descriptive per-variant CIs for Table 1)
# --------------------------------------------------------------------------


def test_marginal_bootstrap_ci_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        marginal_bootstrap_ci([])


def test_marginal_bootstrap_ci_point_estimate_is_exact_mean():
    correct = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    point, _lo, _hi = marginal_bootstrap_ci(correct, n_resamples=500, seed=42)
    assert point == sum(correct) / len(correct)


def test_marginal_bootstrap_ci_high_accuracy_sits_high_away_from_half():
    # 90% correct over 200 questions -- CI should sit entirely above 0.5.
    correct = [1.0] * 180 + [0.0] * 20
    point, lo, hi = marginal_bootstrap_ci(correct, n_resamples=2000, seed=1)
    assert math.isclose(point, 0.9, abs_tol=1e-6)
    assert lo > 0.5
    assert hi < 1.0


def test_marginal_bootstrap_ci_low_accuracy_sits_low_away_from_half():
    correct = [1.0] * 20 + [0.0] * 180
    point, lo, hi = marginal_bootstrap_ci(correct, n_resamples=2000, seed=1)
    assert math.isclose(point, 0.1, abs_tol=1e-6)
    assert hi < 0.5
    assert lo > 0.0


def test_marginal_bootstrap_ci_near_chance_straddles_half():
    # noisy, slightly-above-chance accuracy -- CI should still include 0.5.
    correct = ([1.0, 0.0] * 3) + [1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    point, lo, hi = marginal_bootstrap_ci(correct, n_resamples=2000, seed=7)
    assert lo <= 0.5 <= hi
    assert point != 0.5  # point estimate isn't exactly chance, but still a tie-like spread


def test_marginal_bootstrap_ci_deterministic_given_seed():
    correct = [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0]
    result_a = marginal_bootstrap_ci(correct, n_resamples=500, seed=42)
    result_b = marginal_bootstrap_ci(correct, n_resamples=500, seed=42)
    assert result_a == result_b


# --------------------------------------------------------------------------
# paired_bootstrap_p / holm_bonferroni (AUDIT 2026-09-06 §6.8, ADR-027)
# --------------------------------------------------------------------------


def test_paired_bootstrap_p_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        paired_bootstrap_p([])


def test_paired_bootstrap_p_is_small_for_a_clear_effect():
    diffs = [0.1 + 0.001 * ((-1) ** i) for i in range(200)]
    p = paired_bootstrap_p(diffs, n_resamples=2000, seed=1)
    assert p < 0.01


def test_paired_bootstrap_p_is_large_for_a_tie():
    diffs = [1.0, -1.0, 1.0, -1.0, 0.5, -0.4]
    p = paired_bootstrap_p(diffs, n_resamples=2000, seed=7)
    assert p > 0.05


def test_paired_bootstrap_p_consistent_with_ci_significance():
    """p < alpha must agree with the alpha-level CI excluding zero, for the
    same diffs/resamples/seed -- they share one bootstrap distribution."""
    diffs = [0.05 + 0.02 * ((-1) ** i) for i in range(60)]
    _mean_diff, lo, hi = paired_bootstrap_ci(diffs, n_resamples=4000, seed=3, alpha=0.05)
    p = paired_bootstrap_p(diffs, n_resamples=4000, seed=3)
    ci_significant = lo > 0 or hi < 0
    assert (p < 0.05) == ci_significant


def test_paired_bootstrap_p_deterministic_given_seed():
    diffs = [0.02, -0.01, 0.03, 0.0, -0.02, 0.01]
    assert paired_bootstrap_p(diffs, n_resamples=500, seed=42) == paired_bootstrap_p(
        diffs, n_resamples=500, seed=42
    )


def test_holm_bonferroni_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        holm_bonferroni({})


def test_holm_bonferroni_both_survive_when_both_p_values_are_small():
    out = holm_bonferroni({"secondary": 0.0026, "primary": 0.0256}, alpha=0.05)
    assert out["secondary"]["rank"] == 1
    assert out["secondary"]["threshold"] == pytest.approx(0.025)
    assert out["secondary"]["reject"] is True
    assert out["primary"]["rank"] == 2
    assert out["primary"]["threshold"] == pytest.approx(0.05)
    assert out["primary"]["reject"] is True


def test_holm_bonferroni_step_down_kills_a_later_hypothesis_even_if_its_raw_p_would_pass():
    """The defining property of Holm vs. per-test Bonferroni: once the
    smallest p-value fails its (tightest) threshold, every larger p-value is
    ALSO not rejected, even one smaller than alpha itself."""
    out = holm_bonferroni({"a": 0.03, "b": 0.01}, alpha=0.05)
    # m=2: rank-1 (b, p=0.01) threshold = 0.025 -> rejects.
    # rank-2 (a, p=0.03) threshold = 0.05 -> 0.03 <= 0.05, still rejects here.
    assert out["b"]["reject"] is True
    assert out["a"]["reject"] is True
    # Now force rank-1 to fail: with a large first p, everything after must fail.
    out2 = holm_bonferroni({"a": 0.01, "b": 0.03, "c": 0.02}, alpha=0.03)
    # sorted: a=0.01 (threshold 0.01) borderline, b=0.03 next, c=0.02 next.
    # m=3: rank1 threshold=0.01, rank2 threshold=0.015, rank3 threshold=0.03.
    assert out2["a"]["reject"] is True  # 0.01 <= 0.01
    assert out2["c"]["reject"] is False  # 0.02 > 0.015 -> fails here
    assert out2["b"]["reject"] is False  # later than a failed hypothesis -> also False
    # even though b's own raw threshold (0.03) would have passed p=0.03 <= 0.03
    assert out2["b"]["p"] <= out2["b"]["threshold"]


def test_holm_bonferroni_uncorrected_single_hypothesis_is_plain_alpha():
    out = holm_bonferroni({"only": 0.03}, alpha=0.05)
    assert out["only"]["threshold"] == pytest.approx(0.05)
    assert out["only"]["reject"] is True
