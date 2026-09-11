"""Derive the post-hoc-binary variant from a completed V5 run (ADR-036
Constraint B, zero GPU, zero LLM calls).

Per-question: map `SUPPORT` -> `SUPPORTED`, `CONTRADICT` u `NEUTRAL` ->
`NOT_SUPPORTED`, force s- = 0, and re-rank with configs/v5.yaml's already-fit
weights (alpha*s+ + gamma*kg_support_count + delta*mean_rerank_score,
softmax(z/T) -- beta*s- drops out identically since s- = 0, so the fitted
beta value is irrelevant here, not silently dropped).

This is pure post-processing over the SupportMatrix V5 already logged --
`extract_option_features` already computes s_pos / kg_support_count /
mean_rerank_score from the ORIGINAL 3-way cell labels; the binary collapse
changes nothing about how those three features are computed (SUPPORT cells
are SUPPORT cells either way), so the only real edit is s_neg -> 0. Reuses
`src.calibration.{scoring,softmax_rank,citations}` verbatim -- no scoring code
modified, only imported.

Hallucination-rate is re-aggregated (not re-scored -- zero GPU) from V5's own
already-computed `per_claim` NLI judgements in its metrics.json: claim-level
entailment does not depend on which option is "predicted", only on
(evidence, claim) pairs, which are unchanged by the binary collapse. Only the
`predicted_option` flag is recomputed against post-hoc-binary's own ranking.

    uv run python -m scripts.analysis.posthoc_binary
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.agents.state import SupportMatrix  # noqa: E402
from src.calibration.citations import cite_all  # noqa: E402
from src.calibration.scoring import CalibrationWeights, extract_option_features  # noqa: E402
from src.calibration.softmax_rank import rank_options  # noqa: E402
from src.eval.metrics import accuracy, macro_f1, mean_mrr, mean_precision_at_k  # noqa: E402

V5_RUN_DIR = REPO_ROOT / "results" / "medqa_V5_41f7e14-dirty_20260909T103452Z"
OUT_DIR = REPO_ROOT / "results" / "phase7_posthoc_binary_audit"
MEDQA_LETTERS = ("A", "B", "C", "D")

# From configs/v5.yaml (Phase 5 fit) -- copied, not re-fit (ADR-036: "no
# re-fit, no LLM calls").
V5_WEIGHTS = CalibrationWeights(
    alpha=-0.013896237159311542,
    beta=0.10976837687736825,  # irrelevant here: beta * s_neg(=0) == 0
    gamma=0.2948559455824592,
    delta=0.007722670183812638,
    temperature=0.21447181570951504,
)
ABSTENTION_TAU = 0.5457648774612578


def _load_predictions(run_dir: Path) -> list[dict]:
    path = run_dir / "predictions.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _load_metrics(run_dir: Path) -> dict:
    return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))


def derive_one(rec: dict) -> dict:
    """One post-hoc-binary QuestionRecord-shaped dict from a V5 prediction row."""
    matrix = SupportMatrix(**rec["agent_matrix"])
    feats = extract_option_features(matrix)
    # ADR-036 Constraint B: s- = 0 for every option.
    binary_feats = [f.model_copy(update={"s_neg": 0}) for f in feats]

    if binary_feats:
        ranked_pairs = rank_options(binary_feats, V5_WEIGHTS)
        ranked = [oid for oid, _ in ranked_pairs]
        p_vector = [p for _, p in ranked_pairs]
        pred_option = ranked[0]
        abstained = p_vector[0] < ABSTENTION_TAU
    else:
        ranked, p_vector, pred_option, abstained = [], [], None, False

    citations = cite_all(matrix)
    grounded = pred_option is not None and len(citations.get(pred_option, [])) > 0
    invalid = sorted(
        {
            c.evidence_id
            for c in matrix.cells
            if c.evidence_id and c.evidence_id not in set(matrix.bundle_evidence_ids)
        }
    )

    return {
        "question_id": matrix.question_id,
        "gold_option": matrix.gold_option,
        "pred_option": pred_option,
        "ranked": ranked,
        "p_vector": p_vector,
        "ranking_source": "posthoc_binary_softmax",
        "abstained": abstained,
        "citations": citations,
        "invalid_citations": invalid,
        "grounded": grounded,
        "status": matrix.status,
        "n_llm_calls": matrix.llm_calls,
        "s_neg_forced_zero": True,
        "v5_pred_option": rec["pred_option"],
    }


def _hallucination_rate(
    per_claim_v5: list[dict], posthoc_pred_by_qid: dict[str, str | None]
) -> dict:
    """Re-aggregate V5's per-claim NLI judgements under post-hoc-binary's own
    predicted-option flag (zero GPU -- no new NLI calls)."""
    rows = []
    for row in per_claim_v5:
        is_pred = row["option_id"] == posthoc_pred_by_qid.get(row["question_id"])
        rows.append({**row, "predicted_option": is_pred})

    def agg(scoped: list[dict]) -> dict:
        n = len(scoped)
        n_ungrounded = sum(1 for s in scoped if not s["entailed"])
        n_uncited = sum(1 for s in scoped if not s["cited"])
        return {
            "n_claims": n,
            "n_ungrounded": n_ungrounded,
            "n_uncited": n_uncited,
            "hallucination_rate": (n_ungrounded / n) if n else 0.0,
        }

    predicted = [r for r in rows if r["predicted_option"]]
    agg_pred = agg(predicted)
    agg_all = agg(rows)
    return {
        "primary": "hallucination_rate_predicted_option",
        "hallucination_rate_predicted_option": agg_pred["hallucination_rate"],
        "n_claims_predicted_option": agg_pred["n_claims"],
        "n_ungrounded_predicted_option": agg_pred["n_ungrounded"],
        "n_uncited_predicted_option": agg_pred["n_uncited"],
        "hallucination_rate_all_support_cells": agg_all["hallucination_rate"],
        "n_claims_all_support_cells": agg_all["n_claims"],
        "n_ungrounded_all_support_cells": agg_all["n_ungrounded"],
        "n_uncited_all_support_cells": agg_all["n_uncited"],
        "note": (
            "Re-aggregated from V5's own already-scored per_claim NLI judgements "
            "(same evidence/claim pairs -- the binary label collapse does not "
            "change any SUPPORT cell). Only the predicted_option flag is "
            "recomputed against post-hoc-binary's own ranking. Zero new NLI "
            "calls."
        ),
    }


def main() -> None:
    v5_metrics = _load_metrics(V5_RUN_DIR)
    v5_predictions = _load_predictions(V5_RUN_DIR)

    derived = [derive_one(rec) for rec in v5_predictions]

    class _AV:
        def __init__(self, p, g):
            self.pred_option = p
            self.gold_option = g

    acc_view = [_AV(r["pred_option"], r["gold_option"]) for r in derived]
    n = len(derived)

    metrics: dict = {
        "n_evaluated": n,
        "dataset": "medqa",
        "accuracy": accuracy(acc_view),
        "macro_f1": macro_f1(acc_view, list(MEDQA_LETTERS)),
    }

    rankings = [r["ranked"] for r in derived]
    golds = [r["gold_option"] or "" for r in derived]
    metrics["p_at_1"] = mean_precision_at_k(rankings, golds, 1)
    metrics["p_at_2"] = mean_precision_at_k(rankings, golds, 2)
    metrics["mrr"] = mean_mrr(rankings, golds)
    metrics["ranking_source"] = "posthoc_binary_softmax"

    metrics["invalid_citation_count"] = sum(len(r["invalid_citations"]) for r in derived)
    metrics["grounded_rate"] = sum(1 for r in derived if r["grounded"]) / n
    metrics["abstention_rate"] = sum(1 for r in derived if r["abstained"]) / n
    from collections import Counter

    metrics["status_counts"] = dict(Counter(r["status"] for r in derived))

    posthoc_pred_by_qid = {r["question_id"]: r["pred_option"] for r in derived}
    metrics["hallucination"] = _hallucination_rate(
        v5_metrics["metrics"]["hallucination"]["per_claim"], posthoc_pred_by_qid
    )
    metrics["hallucination_rate"] = metrics["hallucination"]["hallucination_rate_predicted_option"]

    n_pred_changed = sum(1 for r in derived if r["pred_option"] != r["v5_pred_option"])
    metrics["n_pred_option_changed_vs_v5"] = n_pred_changed

    out = {
        "run_id": "phase7_posthoc_binary_audit",
        "dataset": "medqa",
        "variant": "post-hoc-binary",
        "derived_from": v5_metrics["run_id"],
        "derivation": "ADR-036 Constraint B",
        "git_sha7": v5_metrics["git_sha7"],
        "split": v5_metrics["split"],
        "limit": v5_metrics["limit"],
        "seed": v5_metrics["seed"],
        "superseded": False,
        "superseded_by": None,
        "superseded_reason": None,
        "metrics": metrics,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metrics.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    with (OUT_DIR / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in derived:
            f.write(json.dumps(r) + "\n")

    print(f"[posthoc_binary] wrote {OUT_DIR / 'metrics.json'}")
    print(f"[posthoc_binary] n={n} acc={metrics['accuracy']:.3f} macroF1={metrics['macro_f1']:.3f}")
    print(f"  P@1={metrics['p_at_1']:.3f} P@2={metrics['p_at_2']:.3f} MRR={metrics['mrr']:.3f}")
    print(
        f"  grounded_rate={metrics['grounded_rate']:.3f} "
        f"halluc_rate={metrics['hallucination_rate']:.3f}"
    )
    print(f"  status={metrics['status_counts']}  n_pred_changed_vs_v5={n_pred_changed}")


if __name__ == "__main__":
    main()
