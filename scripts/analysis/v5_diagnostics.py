"""Three read-only diagnostics on V5's collapse (accuracy 0.316), requested
directly against the already-written `predictions.jsonl` -- no new runs, no
GPU.

(a) "Accuracy of the Hypothesis agent's own stated answer, before the
    support-based ranking is applied." **Not computable as literally asked**:
    per `src/agents/hypothesis.py` and `docs/01_ARCHITECTURE.md`'s own
    `HypothesisAgent` spec, the Hypothesis agent's output is
    `[{option_id, claims, required_evidence_types}]` -- it decomposes every
    option into atomic claims, but never proposes or ranks a candidate
    answer. There is no "Hypothesis agent's stated answer" anywhere in the
    persisted state; this is confirmed from the schema
    (`HypothesisOut(options: list[HypothesisClaimSet])`), not inferred.
    Reported honestly as not computable, rather than substituting a fabricated
    proxy under that name.

    The closest REAL, available diagnostic that answers the underlying
    question ("is the ranking mechanism the defect, precisely") is computed
    instead: V5's own SupportMatrices re-ranked by the RAW uncalibrated
    (s+ - s-) margin (the audit-layer signal, pre-calibration/pre-softmax),
    directly comparable to V4-conflict's own accuracy (0.358, same ranking
    mechanism, an independent run).

(b) Accuracy on the 459 `ok` questions alone vs. all 500.

(c) Why grounded_rate = 0.756 against 0.98-1.00 for V1/V3: which claims lost
    grounding, and why -- broken down by status/abstention/predicted-option
    SUPPORT-cell count.

    uv run python -m scripts.analysis.v5_diagnostics
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.agents.state import SupportMatrix  # noqa: E402
from src.calibration.scoring import extract_option_features  # noqa: E402
from src.eval.metrics import accuracy, macro_f1  # noqa: E402

V5_RUN_DIR = REPO_ROOT / "results" / "medqa_V5_41f7e14-dirty_20260909T103452Z"
OUT_PATH = REPO_ROOT / "results" / "phase7_grid" / "v5_diagnostics.json"
MEDQA_LETTERS = ("A", "B", "C", "D")


def _load_predictions(run_dir: Path) -> list[dict]:
    path = run_dir / "predictions.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


class _AV:
    def __init__(self, p, g):
        self.pred_option = p
        self.gold_option = g


def diagnostic_a(predictions: list[dict]) -> dict:
    """Raw uncalibrated (s+ - s-) margin ranking, computed directly from V5's
    own matrices -- same audit evidence V5's calibrated softmax used, minus
    the calibration/temperature step. Mirrors _agent_record's own
    'uncalibrated_support_margin' path (src/eval/harness.py) exactly."""
    margin_preds = []
    for rec in predictions:
        matrix = SupportMatrix(**rec["agent_matrix"])
        feats = extract_option_features(matrix)
        if feats:
            ranked = sorted(feats, key=lambda f: f.s_pos - f.s_neg, reverse=True)
            pred = ranked[0].option_id
        else:
            pred = None
        margin_preds.append(_AV(pred, rec["gold_option"]))

    return {
        "requested": (
            "Accuracy of the Hypothesis agent's own stated answer, before the "
            "support-based ranking is applied."
        ),
        "computable": False,
        "why_not": (
            "The Hypothesis agent never states a candidate answer. Its schema "
            "(src/agents/schemas.py: HypothesisOut(options: "
            "list[HypothesisClaimSet])) and its documented contract "
            "(docs/01_ARCHITECTURE.md #HypothesisAgent: 'Out: "
            "[{option_id, claims, required_evidence_types}]') both confirm it "
            "only decomposes each option into atomic claims -- there is no "
            "ranking or answer signal from Hypothesis, in the spec or in the "
            "implementation, to measure the accuracy of."
        ),
        "closest_real_substitute": {
            "description": (
                "V5's own SupportMatrices, re-ranked by the raw uncalibrated "
                "(s+ - s-) margin instead of the calibrated softmax -- same "
                "underlying Hypothesis+Audit evidence V5 used, isolating "
                "whether the calibration/softmax step itself is where accuracy "
                "is lost."
            ),
            "n": len(margin_preds),
            "accuracy": accuracy(margin_preds),
            "v5_calibrated_softmax_accuracy": accuracy(
                [_AV(r["pred_option"], r["gold_option"]) for r in predictions]
            ),
            "v4_conflict_accuracy_for_reference": 0.358,
            "reading": (
                "If this raw-margin accuracy is close to V5's calibrated 0.316, "
                "the defect is upstream of calibration (in the audit evidence "
                "itself, or in retrieval). If it is close to V4-conflict's "
                "0.358 (the same uncalibrated-margin mechanism from an "
                "independent run), that corroborates the margin mechanism as "
                "the honest floor and implicates calibration as a further, "
                "separate loss on top of it."
            ),
        },
    }


def diagnostic_b(predictions: list[dict]) -> dict:
    ok = [r for r in predictions if r["status"] == "ok"]
    not_ok = [r for r in predictions if r["status"] != "ok"]
    status_counts = dict(Counter(r["status"] for r in predictions))

    def acc_of(rows: list[dict]) -> float | None:
        if not rows:
            return None
        return accuracy([_AV(r["pred_option"], r["gold_option"]) for r in rows])

    def f1_of(rows: list[dict]) -> float | None:
        if not rows:
            return None
        views = [_AV(r["pred_option"], r["gold_option"]) for r in rows]
        return macro_f1(views, list(MEDQA_LETTERS))

    return {
        "n_total": len(predictions),
        "status_counts": status_counts,
        "accuracy_all_500": acc_of(predictions),
        "macro_f1_all_500": f1_of(predictions),
        "accuracy_ok_only_459": acc_of(ok),
        "macro_f1_ok_only_459": f1_of(ok),
        "accuracy_not_ok_41": acc_of(not_ok),
        "not_ok_breakdown": {
            status: acc_of([r for r in predictions if r["status"] == status])
            for status in status_counts
            if status != "ok"
        },
    }


def diagnostic_c(predictions: list[dict]) -> dict:
    rows_annotated = []
    for rec in predictions:
        matrix = SupportMatrix(**rec["agent_matrix"])
        feats = {f.option_id: f for f in extract_option_features(matrix)}
        pred = rec["pred_option"]
        pred_s_pos = feats[pred].s_pos if pred in feats else None
        grounded = pred is not None and len(rec["citations"].get(pred, [])) > 0
        rows_annotated.append(
            {
                "question_id": rec["question_id"],
                "status": rec["status"],
                "abstained": rec["abstained"],
                "grounded": grounded,
                "pred_option": pred,
                "pred_option_s_pos": pred_s_pos,
                "any_option_has_support": any(f.s_pos > 0 for f in feats.values()),
            }
        )

    n = len(rows_annotated)
    n_grounded = sum(1 for r in rows_annotated if r["grounded"])
    ungrounded_rows = [r for r in rows_annotated if not r["grounded"]]

    by_status = Counter(r["status"] for r in ungrounded_rows)
    n_ungrounded_zero_support = sum(1 for r in ungrounded_rows if r["pred_option_s_pos"] == 0)
    n_ungrounded_had_support_elsewhere = sum(
        1 for r in ungrounded_rows if r["pred_option_s_pos"] == 0 and r["any_option_has_support"]
    )
    n_ungrounded_abstained = sum(1 for r in ungrounded_rows if r["abstained"])

    return {
        "grounded_rate": n_grounded / n,
        "n_grounded": n_grounded,
        "n_ungrounded": len(ungrounded_rows),
        "n_total": n,
        "comparison": {"V1": 1.0, "V3": 0.984, "V5": n_grounded / n},
        "ungrounded_by_status": dict(by_status),
        "ungrounded_where_predicted_option_has_zero_SUPPORT_cells": n_ungrounded_zero_support,
        "of_those_where_some_OTHER_option_did_have_SUPPORT": n_ungrounded_had_support_elsewhere,
        "ungrounded_and_abstained": n_ungrounded_abstained,
        "explanation": (
            "grounded = pred_option has >=1 SUPPORT cell (its citation set is "
            "non-empty). V1/V3 emit a single generation pass and virtually "
            "always cite something (0.984-1.0). V5's pred_option is instead "
            "chosen by softmax(alpha*s+ - beta*s- + gamma*kg + delta*rerank, T) "
            "over PER-OPTION features -- when the calibrated top-ranked option "
            "has zero SUPPORT cells (the audit layer never produced a SUPPORT "
            "verdict for it, e.g. because it only accumulated NEUTRAL/"
            "CONTRADICT cells, or none at all on an insufficient_evidence "
            "question), citations.get(pred_option) is empty by construction "
            "(src/calibration/citations.py: cite() only draws from SUPPORT "
            "cells) and the row scores ungrounded regardless of whether OTHER "
            "options on the same question did have supporting evidence. This "
            "is a structural consequence of ranking on a fitted multi-feature "
            "logit rather than always picking whichever option the audit most "
            "directly supported."
        ),
    }


def main() -> None:
    predictions = _load_predictions(V5_RUN_DIR)
    out = {
        "run_id": "medqa_V5_41f7e14-dirty_20260909T103452Z",
        "n": len(predictions),
        "diagnostic_a_hypothesis_answer_accuracy": diagnostic_a(predictions),
        "diagnostic_b_ok_only_vs_all": diagnostic_b(predictions),
        "diagnostic_c_grounded_rate_explanation": diagnostic_c(predictions),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"[v5_diagnostics] wrote {OUT_PATH}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
