"""Phase 3 answer-quality metrics (accuracy, macro_f1) + Phase 5's ECE.

This module started deliberately small (Phase 3: a headline accuracy number
and a position-bias sniff test for the multi-choice baselines in
`src/eval/baselines.py`). `ece()` is Phase 5's addition -- PROJECT_RULES.md's stack
table and `docs/02_ROADMAP.md`'s file layout both place ECE here, not in
`src/calibration/`, so Phase 6's P@K/MRR/faithfulness additions extend this
same file rather than duplicating calibration logic elsewhere.
`accuracy` and `macro_f1` are already unit-tested against hand-computed toy
cases (`tests/test_eval_metrics.py`) and other code imports them -- extend,
don't rewrite.

No sklearn: everything here is hand-implemented so the toy cases in the tests can
be checked by hand and there is no hidden averaging convention.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

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


# --------------------------------------------------------------------------
# Phase 6: ranking quality (docs/04_EVALUATION.md §4, hypothesis H4)
# --------------------------------------------------------------------------


def precision_at_k(ranked_option_ids: Sequence[str], gold_option_id: str, k: int) -> float:
    """`docs/04_EVALUATION.md` §4 "Precision@K (K=1,2) over the ranked option list".

    Multi-choice QA has exactly ONE correct option, so P@K here is the hit
    rate: ``1.0`` iff ``gold_option_id`` appears in the top-``k`` of
    ``ranked_option_ids`` (best-first), else ``0.0``. With a single relevant
    item this coincides with recall@k and with hit@k -- the convention is
    fixed here (and in ADR-029) so "P@2" is read as "did the gold option stay
    in the top-2 differential" (H4), not as ``relevant_in_top_k / k`` which
    would cap a correct top-2 at 0.5. P@1 therefore equals accuracy for a
    ranked prediction.

    ``k`` must be >= 1. A ranking shorter than ``k`` is not padded -- only the
    ids actually present are considered.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    return 1.0 if gold_option_id in list(ranked_option_ids)[:k] else 0.0


def reciprocal_rank_option(ranked_option_ids: Sequence[str], gold_option_id: str) -> float:
    """``1 / rank`` of ``gold_option_id`` in ``ranked_option_ids`` (rank
    1-based, best-first); ``0.0`` if the gold option is absent from the
    ranking. This is the per-question term of MRR (`docs/04` §4)."""
    for pos, oid in enumerate(ranked_option_ids, start=1):
        if oid == gold_option_id:
            return 1.0 / pos
    return 0.0


def mean_precision_at_k(
    rankings: Sequence[Sequence[str]], golds: Sequence[str], k: int
) -> float:
    """Mean of :func:`precision_at_k` over a set of questions. Empty input
    returns ``0.0`` (not NaN), matching :func:`accuracy`."""
    if len(rankings) != len(golds):
        raise ValueError(f"{len(rankings)} rankings but {len(golds)} golds")
    if not rankings:
        return 0.0
    return sum(precision_at_k(r, g, k) for r, g in zip(rankings, golds, strict=True)) / len(
        rankings
    )


def mean_mrr(rankings: Sequence[Sequence[str]], golds: Sequence[str]) -> float:
    """Mean Reciprocal Rank of the gold option over a set of questions
    (`docs/04_EVALUATION.md` §4). Empty input returns ``0.0``."""
    if len(rankings) != len(golds):
        raise ValueError(f"{len(rankings)} rankings but {len(golds)} golds")
    if not rankings:
        return 0.0
    return sum(
        reciprocal_rank_option(r, g) for r, g in zip(rankings, golds, strict=True)
    ) / len(rankings)


# --------------------------------------------------------------------------
# Phase 5: Expected Calibration Error
# --------------------------------------------------------------------------


def ece(
    probs: Sequence[Sequence[float]],
    gold_idx: Sequence[int],
    n_bins: int = 15,
    binning: Literal["equal_width", "equal_mass"] = "equal_width",
) -> dict:
    """Expected Calibration Error over top-1 confidence (Guo et al. 2017).

    `probs[i]` is question i's softmax-normalized probability vector over its
    options (any length >= 2, need not be uniform across questions); `gold_idx[i]`
    is the 0-based index of the correct option in that vector. For each
    question, `confidence = max(probs[i])` and `correct = (argmax(probs[i]) ==
    gold_idx[i])`.

    `binning="equal_width"` (the `docs/04_EVALUATION.md` §4 default, 15 bins):
    fixed-width bins over [0, 1]. `binning="equal_mass"` (ADR-028, dev-scale
    runs): bin EDGES are chosen so each bin holds ~n/n_bins questions
    (quantile split of the sorted confidences) instead of a fixed width --
    avoids the near-empty, noise-dominated bins a small n produces under
    equal-width binning (e.g. 15 bins over 60 questions is ~4 points/bin).
    Ties at a quantile boundary are broken by original list order (stable
    sort), so the split is deterministic.

    ECE = sum_b (n_b / n) * |acc_b - conf_b| over non-empty bins b.

    Returns `{"ece": float, "n_bins": int, "binning": str, "n": int, "bins":
    [{"bin_lo", "bin_hi", "n", "conf_mean", "acc_mean"}, ...]}` (bins listed in
    ascending-confidence order; empty bins are omitted, not zero-padded, since
    an empty bin contributes nothing to the sum and a zero-padded row would
    misread as "0% accuracy" rather than "no data"). `ece: None` and no bins
    when `probs` is empty -- nothing was measured, not a defined zero.
    """
    n = len(probs)
    if n != len(gold_idx):
        raise ValueError(f"probs has {n} rows but gold_idx has {len(gold_idx)}")
    if n == 0:
        return {"ece": None, "n_bins": n_bins, "binning": binning, "n": 0, "bins": []}

    records = []
    for p, g in zip(probs, gold_idx, strict=True):
        conf = max(p)
        pred = max(range(len(p)), key=lambda i: p[i])
        records.append((conf, pred == g))
    records.sort(key=lambda r: r[0])  # ascending confidence, stable

    if binning == "equal_width":
        edges = [i / n_bins for i in range(n_bins + 1)]
        bin_of = []
        for conf, _correct in records:
            idx = min(int(conf * n_bins), n_bins - 1)
            bin_of.append(idx)
        groups: dict[int, list[tuple[float, bool]]] = {}
        for idx, rec in zip(bin_of, records, strict=True):
            groups.setdefault(idx, []).append(rec)
    elif binning == "equal_mass":
        edges = None
        groups = {}
        base, extra = divmod(n, n_bins)
        start = 0
        for b in range(n_bins):
            size = base + (1 if b < extra else 0)
            if size == 0:
                continue
            groups[b] = records[start : start + size]
            start += size
    else:
        raise ValueError(f"unknown binning {binning!r}")

    bins_out = []
    weighted_abs_diff = 0.0
    for b in sorted(groups):
        chunk = groups[b]
        if not chunk:
            continue
        confs = [c for c, _ in chunk]
        corrects = [1.0 if ok else 0.0 for _, ok in chunk]
        conf_mean = sum(confs) / len(confs)
        acc_mean = sum(corrects) / len(corrects)
        weighted_abs_diff += (len(chunk) / n) * abs(acc_mean - conf_mean)
        bins_out.append(
            {
                "bin_lo": edges[b] if edges else min(confs),
                "bin_hi": edges[b + 1] if edges else max(confs),
                "n": len(chunk),
                "conf_mean": conf_mean,
                "acc_mean": acc_mean,
            }
        )

    return {
        "ece": weighted_abs_diff,
        "n_bins": n_bins,
        "binning": binning,
        "n": n,
        "bins": bins_out,
    }


def paired_bootstrap_ece_diff(
    probs_a: Sequence[Sequence[float]],
    probs_b: Sequence[Sequence[float]],
    gold_idx: Sequence[int],
    n_bins: int = 5,
    binning: Literal["equal_width", "equal_mass"] = "equal_mass",
    n_resamples: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """Paired-bootstrap CI for `ece(probs_a) - ece(probs_b)` on the SAME
    question set, applying this project's paired-bootstrap discipline
    (`docs/04_EVALUATION.md` §1, ADR-012: never claim A beats B from a raw
    point-estimate alone) to ECE specifically.

    ECE is not decomposable into independent per-question differences the
    way `src.retrieval.metrics.paired_bootstrap_ci` assumes (it is a
    weighted-bin statistic over the whole set, and the bins themselves are
    re-derived from whichever questions land in a given resample) -- so this
    resamples QUESTION INDICES with replacement (paired: the same resampled
    indices are used for both `probs_a` and `probs_b` and `gold_idx` on each
    draw, exactly as pairing works for `paired_bootstrap_ci`), and
    recomputes the FULL `ece()` metric from scratch for each side on every
    resample, rather than bootstrapping a precomputed per-item diff.

    Returns `{"mean_diff", "ci_lo", "ci_hi", "point_estimate_diff", "verdict",
    "n_resamples", "seed", "n"}`. `verdict` follows the exact ADR-012 rule
    (`ci_hi < 0` -> "a_significantly_better" since a lower ECE is better;
    `ci_lo > 0` -> "b_significantly_better"; otherwise `"tie"` -- CI includes
    0, not evidence of a difference, must be reported as a tie).
    """
    n = len(probs_a)
    if n != len(probs_b) or n != len(gold_idx):
        raise ValueError(
            f"probs_a has {n} rows, probs_b has {len(probs_b)}, gold_idx has {len(gold_idx)}"
        )
    if n == 0:
        raise ValueError("paired_bootstrap_ece_diff: inputs must be non-empty")

    point_ece_a = ece(probs_a, gold_idx, n_bins=n_bins, binning=binning)["ece"]
    point_ece_b = ece(probs_b, gold_idx, n_bins=n_bins, binning=binning)["ece"]
    point_diff = point_ece_a - point_ece_b

    rng = random.Random(seed)
    diffs = []
    for _ in range(n_resamples):
        idxs = [rng.randrange(n) for _ in range(n)]
        r_a = [probs_a[i] for i in idxs]
        r_b = [probs_b[i] for i in idxs]
        r_gold = [gold_idx[i] for i in idxs]
        ece_a = ece(r_a, r_gold, n_bins=n_bins, binning=binning)["ece"]
        ece_b = ece(r_b, r_gold, n_bins=n_bins, binning=binning)["ece"]
        diffs.append(ece_a - ece_b)
    diffs.sort()
    lo_idx = int((alpha / 2) * n_resamples)
    hi_idx = n_resamples - 1 - lo_idx  # symmetric (AUDIT §6.8, matches paired_bootstrap_ci)
    ci_lo, ci_hi = diffs[lo_idx], diffs[hi_idx]

    if ci_hi < 0:
        verdict = "a_significantly_better"
    elif ci_lo > 0:
        verdict = "b_significantly_better"
    else:
        verdict = "tie"

    return {
        "point_ece_a": point_ece_a,
        "point_ece_b": point_ece_b,
        "point_estimate_diff": point_diff,
        "mean_diff": sum(diffs) / len(diffs),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "verdict": verdict,
        "n_resamples": n_resamples,
        "seed": seed,
        "n": n,
        "n_bins": n_bins,
        "binning": binning,
    }
