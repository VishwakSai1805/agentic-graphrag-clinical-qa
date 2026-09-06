"""Minimal Phase 3 answer-quality metrics: accuracy + macro_f1 only.

This module is deliberately small. Phase 3 only needs a headline accuracy number
and a position-bias sniff test for the multi-choice baselines in
`src/eval/baselines.py`. Phase 6 (`docs/02_ROADMAP.md`) extends this module with
P@K, MRR, ECE (calibration / reliability diagram) and the RAGAS / NLI
faithfulness hooks. When you get there: EXTEND this file, do not rewrite it --
`accuracy` and `macro_f1` are already unit-tested against hand-computed toy
cases (`tests/test_eval_metrics.py`) and other code imports them.

No sklearn: everything here is hand-implemented so the toy cases in the tests can
be checked by hand and there is no hidden averaging convention.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.eval.baselines import Prediction


def accuracy(preds: Sequence[Prediction]) -> float:
    """Fraction of predictions where ``pred_option == gold_option``.

    ``pred_option is None`` (unparseable / no answer) counts as WRONG -- it is
    never a skip and never removed from the denominator. Empty input returns
    0.0 (not NaN, not a crash).

    For 4-option MedQA this is the headline metric.
    """
    if not preds:
        return 0.0
    n_correct = sum(
        1 for p in preds if p.pred_option is not None and p.pred_option == p.gold_option
    )
    return n_correct / len(preds)


def macro_f1(preds: Sequence[Prediction], labels: Sequence[str]) -> float:
    """Macro-averaged F1 over the fixed class set ``labels`` (e.g. A/B/C/D).

    Per label L:
      * precision = TP / (TP + FP), or 0.0 if TP + FP == 0
      * recall    = TP / (TP + FN), or 0.0 if TP + FN == 0
      * F1        = 2 * P * R / (P + R), or 0.0 if P + R == 0
    A label that appears in neither predictions nor golds contributes F1 = 0.
    ``pred_option is None`` is treated as a prediction of "no label": it adds an
    FN to its gold label and never contributes an FP to any label.

    The mean is taken over ``len(labels)`` classes (not over the classes that
    happen to occur). Empty input returns 0.0.

    For 4-option MedQA macro-F1 is a POSITION-BIAS DETECTOR, not a skill metric:
    a model that always answers "A" scores a plausible ~0.25 accuracy on
    balanced data but its macro-F1 collapses (three of four classes get F1 = 0).
    Report accuracy as the headline for MedQA and read macro-F1 as a degeneracy
    check.
    """
    if not preds:
        return 0.0

    f1s: list[float] = []
    for label in labels:
        tp = sum(1 for p in preds if p.pred_option == label and p.gold_option == label)
        fp = sum(1 for p in preds if p.pred_option == label and p.gold_option != label)
        fn = sum(1 for p in preds if p.pred_option != label and p.gold_option == label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)

    return sum(f1s) / len(labels) if labels else 0.0
