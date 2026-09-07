"""Hand-computed toy cases for src/eval/metrics.py (Phase 3).

Per project hygiene: every metric is checked against a case whose expected
value was worked out by hand (see comments) before it scores a real run.

metrics.py duck-types on .pred_option / .gold_option, so these dummies are a
local dataclass rather than the real src.eval.baselines.Prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from src.eval.metrics import (
    accuracy,
    ece,
    macro_f1,
    mean_mrr,
    mean_precision_at_k,
    paired_bootstrap_ece_diff,
    precision_at_k,
    reciprocal_rank_option,
)

LABELS = ["A", "B", "C", "D"]


@dataclass
class DummyPred:
    pred_option: str | None
    gold_option: str
    question_id: str = "q"
    citations: list[str] = field(default_factory=list)


def test_empty_input_accuracy_is_zero_not_nan():
    assert accuracy([]) == 0.0


def test_empty_input_macro_f1_is_zero_not_nan():
    assert macro_f1([], LABELS) == 0.0


def test_perfect_predictions_accuracy_one():
    preds = [DummyPred(x, x) for x in ["A", "B", "C", "D", "A"]]
    assert accuracy(preds) == 1.0


def test_perfect_predictions_macro_f1_one():
    preds = [DummyPred(x, x) for x in ["A", "B", "C", "D"]]
    assert macro_f1(preds, LABELS) == pytest.approx(1.0)


def test_all_A_against_balanced_gold_accuracy_quarter():
    # golds A,B,C,D ; every prediction is "A"
    preds = [DummyPred("A", g) for g in ["A", "B", "C", "D"]]
    assert accuracy(preds) == pytest.approx(0.25)


def test_all_A_against_balanced_gold_macro_f1_collapses():
    # Position-bias signal. Per class:
    #   A: TP=1 FP=3 FN=0 -> P=0.25 R=1.0 F1 = 2*.25*1/1.25 = 0.4
    #   B,C,D: TP=0 -> F1 = 0
    #   macro = 0.4 / 4 = 0.1
    preds = [DummyPred("A", g) for g in ["A", "B", "C", "D"]]
    mf1 = macro_f1(preds, LABELS)
    assert mf1 == pytest.approx(0.1)
    assert mf1 < 0.15  # far below the 0.25 accuracy -> degenerate model


def test_none_prediction_counts_as_wrong_for_accuracy():
    # one correct, one unparseable -> 1/2
    preds = [DummyPred("A", "A"), DummyPred(None, "B")]
    assert accuracy(preds) == pytest.approx(0.5)


def test_none_prediction_never_adds_a_false_positive():
    # All predictions None. Every class has TP=0, FP=0, some FN.
    # Every per-class F1 = 0 -> macro_f1 = 0, and it must not crash.
    preds = [DummyPred(None, g) for g in ["A", "B", "C", "D"]]
    assert macro_f1(preds, LABELS) == 0.0
    assert accuracy(preds) == 0.0


def test_known_three_row_case_exact_f1():
    # rows: (pred A, gold A), (pred A, gold B), (pred None, gold C)
    #   A: TP=1 (row1), FP=1 (row2), FN=0  -> P=1/2=0.5, R=1/1=1.0
    #      F1 = 2*0.5*1.0 / (0.5+1.0) = 1.0 / 1.5 = 0.666666...
    #   B: TP=0, FP=0, FN=1 (row2 gold B) -> P=0, R=0 -> F1=0
    #   C: TP=0, FP=0, FN=1 (row3 gold C) -> F1=0
    #   D: TP=0, FP=0, FN=0               -> F1=0
    #   macro = (2/3 + 0 + 0 + 0) / 4 = 1/6 = 0.166666...
    preds = [DummyPred("A", "A"), DummyPred("A", "B"), DummyPred(None, "C")]
    assert accuracy(preds) == pytest.approx(1 / 3)
    assert macro_f1(preds, LABELS) == pytest.approx(1 / 6)


def test_label_absent_from_preds_and_golds_contributes_zero():
    # Only A and B occur; C and D are dead classes -> each contributes F1=0,
    # dragging the macro mean down over len(labels)=4.
    #   A: TP=1 FP=0 FN=0 -> F1=1.0
    #   B: TP=1 FP=0 FN=0 -> F1=1.0
    #   C,D: F1=0
    #   macro = 2.0 / 4 = 0.5
    preds = [DummyPred("A", "A"), DummyPred("B", "B")]
    assert macro_f1(preds, LABELS) == pytest.approx(0.5)


def test_macro_f1_over_two_class_label_set():
    # labels restricted to {A, B}; same two rows as above -> macro = 1.0
    preds = [DummyPred("A", "A"), DummyPred("B", "B")]
    assert macro_f1(preds, ["A", "B"]) == pytest.approx(1.0)


def test_wrong_but_parseable_predictions_lower_accuracy():
    preds = [DummyPred("B", "A"), DummyPred("C", "A"), DummyPred("A", "A")]
    assert accuracy(preds) == pytest.approx(1 / 3)


# --------------------------------------------------------------------------
# ece() (Phase 5, docs/04_EVALUATION.md §4)
# --------------------------------------------------------------------------


def test_ece_perfectly_calibrated_is_zero():
    # confidence == empirical accuracy in every bin -> ECE == 0.
    # 10 questions at confidence 0.9, 9 correct (acc=0.9==conf).
    probs = [[0.9, 0.1]] * 10
    gold_idx = [0] * 9 + [1]
    result = ece(probs, gold_idx, n_bins=10, binning="equal_width")
    assert result["ece"] == pytest.approx(0.0, abs=1e-9)


def test_ece_maximally_overconfident_and_wrong():
    # Always 100% confident, always wrong -> |0 - 1| = 1.0 for the one bin hit.
    probs = [[1.0, 0.0]] * 5
    gold_idx = [1] * 5  # true answer is index 1, model always picks index 0
    result = ece(probs, gold_idx, n_bins=10, binning="equal_width")
    assert result["ece"] == pytest.approx(1.0)


def test_ece_hand_computed_two_bins():
    # Bin A: confidence 0.6, 2 of 4 correct -> conf=0.6, acc=0.5, |diff|=0.1, weight 4/8
    # Bin B: confidence 0.9, 4 of 4 correct -> conf=0.9, acc=1.0, |diff|=0.1, weight 4/8
    # ECE = 0.5*0.1 + 0.5*0.1 = 0.1
    probs = [[0.6, 0.4]] * 4 + [[0.9, 0.1]] * 4
    gold_idx = [0, 0, 1, 1] + [0, 0, 0, 0]
    result = ece(probs, gold_idx, n_bins=10, binning="equal_width")
    assert result["ece"] == pytest.approx(0.1)


def test_ece_empty_input_is_none_not_zero():
    result = ece([], [], n_bins=10)
    assert result["ece"] is None
    assert result["n"] == 0
    assert result["bins"] == []


def test_ece_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="rows"):
        ece([[0.5, 0.5]], [0, 1])


def test_ece_equal_mass_bins_have_roughly_equal_size():
    # 20 questions, 5 equal-mass bins -> 4 per bin exactly.
    probs = [[0.1 * i, 1 - 0.1 * i] for i in range(1, 21)]
    gold_idx = [0] * 20
    result = ece(probs, gold_idx, n_bins=5, binning="equal_mass")
    assert [b["n"] for b in result["bins"]] == [4, 4, 4, 4, 4]


def test_ece_equal_mass_distributes_remainder_to_early_bins():
    # 7 questions, 3 bins -> sizes 3, 2, 2 (remainder to the front, deterministic).
    probs = [[0.1 * i, 1 - 0.1 * i] for i in range(1, 8)]
    gold_idx = [0] * 7
    result = ece(probs, gold_idx, n_bins=3, binning="equal_mass")
    assert [b["n"] for b in result["bins"]] == [3, 2, 2]


def test_ece_equal_width_omits_empty_bins_not_zero_pads():
    # All confidences in [0.9, 1.0) -> only the top bin of 10 is populated.
    probs = [[0.95, 0.05]] * 3
    gold_idx = [0, 0, 1]
    result = ece(probs, gold_idx, n_bins=10, binning="equal_width")
    assert len(result["bins"]) == 1
    assert result["bins"][0]["n"] == 3


def test_ece_unknown_binning_raises():
    with pytest.raises(ValueError, match="unknown binning"):
        ece([[0.5, 0.5]], [0], binning="bogus")  # type: ignore[arg-type]


def test_ece_works_with_variable_option_counts_per_question():
    # A 3-option and a 4-option question in the same call.
    probs = [[0.7, 0.2, 0.1], [0.4, 0.3, 0.2, 0.1]]
    gold_idx = [0, 0]
    result = ece(probs, gold_idx, n_bins=10, binning="equal_width")
    assert result["n"] == 2
    assert result["ece"] is not None


def test_ece_bins_report_confidence_and_accuracy_means():
    probs = [[0.85, 0.15]] * 4
    gold_idx = [0, 0, 0, 1]  # 3/4 correct at confidence 0.85
    result = ece(probs, gold_idx, n_bins=10, binning="equal_width")
    assert len(result["bins"]) == 1
    b = result["bins"][0]
    assert b["conf_mean"] == pytest.approx(0.85)
    assert b["acc_mean"] == pytest.approx(0.75)


# --------------------------------------------------------------------------
# paired_bootstrap_ece_diff (ADR-012 discipline applied to ECE, Phase 5)
# --------------------------------------------------------------------------


def test_paired_bootstrap_ece_diff_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="probs_a"):
        paired_bootstrap_ece_diff([[0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]], [0])


def test_paired_bootstrap_ece_diff_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        paired_bootstrap_ece_diff([], [], [])


def test_paired_bootstrap_ece_diff_identical_inputs_is_a_tie_at_zero():
    probs = [[0.7, 0.3]] * 20
    gold_idx = [0] * 14 + [1] * 6
    result = paired_bootstrap_ece_diff(probs, probs, gold_idx, n_bins=4, n_resamples=500)
    assert result["point_estimate_diff"] == pytest.approx(0.0)
    assert result["verdict"] == "tie"
    assert result["ci_lo"] <= 0 <= result["ci_hi"]


def test_paired_bootstrap_ece_diff_detects_a_clearly_better_a():
    # a is perfectly calibrated; b is maximally overconfident and wrong.
    probs_a = [[0.5, 0.5]] * 30
    probs_b = [[0.99, 0.01]] * 30
    gold_idx = [0] * 15 + [1] * 15  # a: conf=acc=0.5 exactly; b: conf~1, acc=0.5
    result = paired_bootstrap_ece_diff(probs_a, probs_b, gold_idx, n_bins=4, n_resamples=500)
    assert result["point_ece_a"] < result["point_ece_b"]
    assert result["verdict"] == "a_significantly_better"
    assert result["ci_hi"] < 0


def test_paired_bootstrap_ece_diff_detects_a_clearly_worse_a():
    probs_a = [[0.99, 0.01]] * 30
    probs_b = [[0.5, 0.5]] * 30
    gold_idx = [0] * 15 + [1] * 15
    result = paired_bootstrap_ece_diff(probs_a, probs_b, gold_idx, n_bins=4, n_resamples=500)
    assert result["verdict"] == "b_significantly_better"
    assert result["ci_lo"] > 0


def test_paired_bootstrap_ece_diff_noisy_small_sample_difference_is_a_tie():
    """A noisy, near-random-confidence difference at modest n must resolve
    to a tie despite a nonzero point-estimate gap -- the exact ADR-012
    scenario CLAUDE.md warns about. Fixture and expected CI verified by
    direct computation, not hand-guessed."""
    import random as _random

    rng = _random.Random(3)

    def _noisy_probs(r):
        vals = [max(0.01, min(0.99, 0.5 + 0.1 * r.uniform(-1, 1))) for _ in range(30)]
        return [[p, 1 - p] for p in vals]

    probs_a = _noisy_probs(rng)
    probs_b = _noisy_probs(rng)
    gold_idx = [rng.choice([0, 1]) for _ in range(30)]
    result = paired_bootstrap_ece_diff(
        probs_a, probs_b, gold_idx, n_bins=3, n_resamples=1000, seed=5
    )
    assert result["verdict"] == "tie"
    assert result["ci_lo"] <= 0 <= result["ci_hi"]


def test_paired_bootstrap_ece_diff_deterministic_given_seed():
    probs_a = [[0.7, 0.3]] * 10
    probs_b = [[0.6, 0.4]] * 10
    gold_idx = [0] * 7 + [1] * 3
    r1 = paired_bootstrap_ece_diff(probs_a, probs_b, gold_idx, n_resamples=200, seed=7)
    r2 = paired_bootstrap_ece_diff(probs_a, probs_b, gold_idx, n_resamples=200, seed=7)
    assert r1 == r2


def test_paired_bootstrap_ece_diff_reports_n_and_config():
    probs = [[0.7, 0.3]] * 10
    gold_idx = [0] * 10
    result = paired_bootstrap_ece_diff(probs, probs, gold_idx, n_bins=3, n_resamples=100, seed=1)
    assert result["n"] == 10
    assert result["n_bins"] == 3
    assert result["n_resamples"] == 100
    assert result["seed"] == 1


# --------------------------------------------------------------------------
# precision_at_k / MRR over ranked option lists (Phase 6, docs/04 §4 / H4)
# --------------------------------------------------------------------------


def test_precision_at_k_hit_and_miss():
    # single-gold MCQ: P@k == "gold in top-k"
    assert precision_at_k(["B", "A", "C", "D"], "B", 1) == 1.0
    assert precision_at_k(["B", "A", "C", "D"], "A", 1) == 0.0
    assert precision_at_k(["B", "A", "C", "D"], "A", 2) == 1.0
    assert precision_at_k(["B", "A", "C", "D"], "C", 2) == 0.0
    assert precision_at_k(["B", "A", "C", "D"], "C", 3) == 1.0


def test_precision_at_k_gold_absent_is_zero():
    assert precision_at_k(["A", "B"], "C", 2) == 0.0


def test_precision_at_k_k_below_one_raises():
    with pytest.raises(ValueError, match="k must be"):
        precision_at_k(["A"], "A", 0)


def test_precision_at_1_equals_accuracy_for_a_ranking():
    rankings = [["A", "B"], ["B", "A"], ["C", "A"]]
    golds = ["A", "A", "C"]
    # rank-1 picks: A, B, C  vs golds A, A, C -> 2/3 correct
    assert mean_precision_at_k(rankings, golds, 1) == pytest.approx(2 / 3)


def test_reciprocal_rank_option_positions():
    assert reciprocal_rank_option(["A", "B", "C"], "A") == 1.0
    assert reciprocal_rank_option(["A", "B", "C"], "B") == pytest.approx(0.5)
    assert reciprocal_rank_option(["A", "B", "C"], "C") == pytest.approx(1 / 3)
    assert reciprocal_rank_option(["A", "B", "C"], "D") == 0.0


def test_mean_mrr_hand_computed():
    # gold ranks: 1, 2, 3 -> RR 1, 0.5, 0.333... -> mean = 1.8333/3
    rankings = [["A", "B", "C"], ["B", "A", "C"], ["B", "C", "A"]]
    golds = ["A", "A", "A"]
    assert mean_mrr(rankings, golds) == pytest.approx((1 + 0.5 + 1 / 3) / 3)


def test_mean_precision_and_mrr_empty_is_zero_not_nan():
    assert mean_precision_at_k([], [], 2) == 0.0
    assert mean_mrr([], []) == 0.0


def test_mean_helpers_length_mismatch_raises():
    with pytest.raises(ValueError, match="rankings"):
        mean_precision_at_k([["A"]], ["A", "B"], 1)
    with pytest.raises(ValueError, match="rankings"):
        mean_mrr([["A"]], ["A", "B"])
