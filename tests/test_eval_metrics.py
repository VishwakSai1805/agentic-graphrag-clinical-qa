"""Hand-computed toy cases for src/eval/metrics.py (Phase 3).

Per project hygiene: every metric is checked against a case whose expected
value was worked out by hand (see comments) before it scores a real run.

metrics.py duck-types on .pred_option / .gold_option, so these dummies are a
local dataclass rather than the real src.eval.baselines.Prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from src.eval.metrics import accuracy, macro_f1

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
