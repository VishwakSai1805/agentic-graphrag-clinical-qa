"""Unit tests for src/calibration/softmax_rank.py: softmax, ranking,
fit_weights (analytic-gradient Adam, no scipy), coverage-risk curve,
abstention threshold, and the bootstrap weight-uncertainty diagnostic."""

from __future__ import annotations

import math

import pytest

from src.calibration.scoring import CalibrationWeights, OptionFeatures
from src.calibration.softmax_rank import (
    bootstrap_weight_uncertainty,
    choose_abstention_threshold,
    coverage_risk_curve,
    fit_weights,
    rank_options,
    softmax,
)


def _f(option_id, s_pos=0, s_neg=0, kg=0, rerank=0.0):
    return OptionFeatures(
        option_id=option_id,
        s_pos=s_pos,
        s_neg=s_neg,
        s_neu=0,
        kg_support_count=kg,
        mean_rerank_score=rerank,
        resolved=True,
    )


# --------------------------------------------------------------------------
# softmax
# --------------------------------------------------------------------------


def test_softmax_sums_to_one():
    probs = softmax([1.0, 2.0, 3.0], temperature=1.0)
    assert sum(probs) == pytest.approx(1.0)


def test_softmax_uniform_logits_gives_uniform_probs():
    probs = softmax([0.0, 0.0, 0.0, 0.0], temperature=1.0)
    for p in probs:
        assert p == pytest.approx(0.25)


def test_softmax_higher_temperature_flattens_distribution():
    sharp = softmax([1.0, 5.0], temperature=1.0)
    flat = softmax([1.0, 5.0], temperature=100.0)
    assert max(sharp) > max(flat)


def test_softmax_rejects_nonpositive_temperature():
    with pytest.raises(ValueError, match="temperature"):
        softmax([1.0, 2.0], temperature=0.0)


def test_softmax_numerically_stable_for_large_logits():
    probs = softmax([1000.0, 1001.0, 999.0], temperature=1.0)
    assert all(math.isfinite(p) for p in probs)
    assert sum(probs) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# rank_options
# --------------------------------------------------------------------------


def test_rank_options_sorts_descending_by_probability():
    feats = [_f("A", s_pos=1), _f("B", s_pos=5), _f("C", s_pos=2)]
    ranked = rank_options(feats, CalibrationWeights())
    assert [oid for oid, _ in ranked] == ["B", "C", "A"]


def test_rank_options_probabilities_sum_to_one():
    feats = [_f("A", s_pos=1), _f("B", s_pos=2)]
    ranked = rank_options(feats, CalibrationWeights())
    assert sum(p for _, p in ranked) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# fit_weights
# --------------------------------------------------------------------------


def test_fit_weights_rejects_empty_examples():
    with pytest.raises(ValueError, match="non-empty"):
        fit_weights([])


def test_fit_weights_recovers_a_clear_separating_signal():
    """Every question: option A has more SUPPORT than B/C/D and is always
    gold. A well-fit alpha should end up positive and rank A first."""
    examples = []
    for _ in range(20):
        feats = [_f("A", s_pos=4), _f("B", s_pos=1), _f("C", s_pos=0), _f("D", s_pos=0)]
        examples.append((feats, "A"))
    weights = fit_weights(examples)
    ranked = rank_options(examples[0][0], weights)
    assert ranked[0][0] == "A"


def test_fit_weights_is_deterministic():
    examples = [([_f("A", s_pos=2), _f("B", s_pos=1)], "A") for _ in range(10)]
    w1 = fit_weights(examples)
    w2 = fit_weights(examples)
    assert w1.model_dump() == w2.model_dump()


def test_fit_weights_temperature_stays_within_bounds():
    examples = [([_f("A", s_pos=100), _f("B", s_pos=0)], "A") for _ in range(5)]
    w = fit_weights(examples)
    assert 0.05 - 1e-6 <= w.temperature <= 20.0 + 1e-6


def test_fit_weights_params_stay_within_bounds_on_separable_data():
    """A perfectly separable small fitting set can drive unregularized
    logistic-style weights to +/-infinity -- the bounds + L2 penalty must
    keep every parameter finite and within [-10, 10]."""
    examples = [([_f("A", s_pos=1), _f("B", s_pos=0)], "A") for _ in range(50)]
    w = fit_weights(examples)
    for name in ("alpha", "beta", "gamma", "delta"):
        assert -10.0 - 1e-6 <= getattr(w, name) <= 10.0 + 1e-6


def test_fit_weights_skips_a_question_whose_gold_is_not_in_features():
    """A malformed example (gold option not present) must not crash the fit."""
    examples = [
        ([_f("A", s_pos=1), _f("B", s_pos=0)], "A"),
        ([_f("A", s_pos=1), _f("B", s_pos=0)], "Z"),  # "Z" not a feature -- skip
    ]
    w = fit_weights(examples)
    assert math.isfinite(w.alpha)


# --------------------------------------------------------------------------
# coverage_risk_curve / choose_abstention_threshold
# --------------------------------------------------------------------------


def test_coverage_risk_curve_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        coverage_risk_curve([], CalibrationWeights())


def test_coverage_risk_curve_at_lowest_threshold_covers_everything():
    examples = [([_f("A", s_pos=3), _f("B", s_pos=1)], "A") for _ in range(4)]
    curve = coverage_risk_curve(examples, CalibrationWeights(), n_thresholds=5)
    assert curve[0]["coverage"] == pytest.approx(1.0)


def test_coverage_risk_curve_all_correct_gives_zero_risk_everywhere():
    examples = [([_f("A", s_pos=5), _f("B", s_pos=0)], "A") for _ in range(6)]
    curve = coverage_risk_curve(examples, CalibrationWeights(), n_thresholds=5)
    for row in curve:
        assert row["risk"] == pytest.approx(0.0)


def test_coverage_risk_curve_single_distinct_p1_value_is_one_point():
    examples = [([_f("A", s_pos=1), _f("B", s_pos=1)], "A") for _ in range(3)]
    curve = coverage_risk_curve(examples, CalibrationWeights(), n_thresholds=5)
    assert len(curve) == 1


def test_choose_abstention_threshold_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        choose_abstention_threshold([])


def test_choose_abstention_threshold_meets_target_prefers_higher_coverage():
    curve = [
        {"tau": 0.1, "coverage": 1.0, "risk": 0.05, "n_covered": 10},
        {"tau": 0.5, "coverage": 0.5, "risk": 0.02, "n_covered": 5},
        {"tau": 0.9, "coverage": 0.1, "risk": 0.0, "n_covered": 1},
    ]
    assert choose_abstention_threshold(curve, target_risk=0.10) == 0.1


def test_choose_abstention_threshold_falls_back_to_lowest_risk_if_none_meet_target():
    curve = [
        {"tau": 0.1, "coverage": 1.0, "risk": 0.5, "n_covered": 10},
        {"tau": 0.9, "coverage": 0.2, "risk": 0.3, "n_covered": 2},
    ]
    assert choose_abstention_threshold(curve, target_risk=0.01) == 0.9


# --------------------------------------------------------------------------
# bootstrap_weight_uncertainty
# --------------------------------------------------------------------------


def test_bootstrap_weight_uncertainty_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        bootstrap_weight_uncertainty([])


def test_bootstrap_weight_uncertainty_reports_all_five_params():
    examples = [([_f("A", s_pos=2), _f("B", s_pos=1)], "A") for _ in range(10)]
    out = bootstrap_weight_uncertainty(examples, n_resamples=10)
    assert set(out) == {"alpha", "beta", "gamma", "delta", "temperature"}
    for stats in out.values():
        assert set(stats) == {"mean", "std", "min", "max"}


def test_bootstrap_weight_uncertainty_gamma_is_near_zero_std_when_feature_is_constant():
    """When kg_support_count is IDENTICAL (here: always 0) across every
    question and option, gamma is unidentifiable -- every resample refits to
    essentially the same (near-zero-signal) gamma, so its bootstrap std should
    be small relative to alpha's, which DOES have a real, varying signal."""
    examples = [([_f("A", s_pos=3, kg=0), _f("B", s_pos=1, kg=0)], "A") for _ in range(20)]
    out = bootstrap_weight_uncertainty(examples, n_resamples=30, seed=1)
    assert out["gamma"]["std"] < 1.0  # no signal to disagree about across resamples


def test_bootstrap_weight_uncertainty_is_deterministic_given_seed():
    examples = [([_f("A", s_pos=2), _f("B", s_pos=1)], "A") for _ in range(8)]
    out1 = bootstrap_weight_uncertainty(examples, n_resamples=15, seed=5)
    out2 = bootstrap_weight_uncertainty(examples, n_resamples=15, seed=5)
    assert out1 == out2


# --------------------------------------------------------------------------
# _loss_and_grad: analytic gradient vs. numerical finite-difference check
# --------------------------------------------------------------------------


def test_loss_and_grad_matches_numerical_gradient():
    """Pins the analytic gradient derivation in `_loss_and_grad` against a
    central finite-difference check -- a wrong analytic gradient would
    silently produce plausible-looking but wrong fits, so this is a hard
    numerical guarantee, not just a smoke test."""
    from src.calibration.softmax_rank import _loss_and_grad

    rng_feats = [
        _f("A", s_pos=3, s_neg=1, kg=1, rerank=0.4),
        _f("B", s_pos=1, s_neg=2, kg=0, rerank=0.7),
        _f("C", s_pos=0, s_neg=0, kg=0, rerank=0.1),
        _f("D", s_pos=2, s_neg=1, kg=1, rerank=0.9),
    ]
    examples = [(rng_feats, "A"), (rng_feats, "B"), (rng_feats, "D")]
    params = [0.3, -0.2, 0.5, 1.1, 0.4]
    _loss, grad = _loss_and_grad(params, examples)

    eps = 1e-5
    for i in range(5):
        p_plus, p_minus = list(params), list(params)
        p_plus[i] += eps
        p_minus[i] -= eps
        l_plus, _ = _loss_and_grad(p_plus, examples)
        l_minus, _ = _loss_and_grad(p_minus, examples)
        numeric = (l_plus - l_minus) / (2 * eps)
        assert grad[i] == pytest.approx(numeric, abs=1e-5)
