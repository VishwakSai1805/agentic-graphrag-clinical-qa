"""The pre-registered exploratory family (docs/04_EVALUATION.md §1.0,
recomputed 2026-09-08, m=35), Holm-Bonferroni corrected. Paired bootstrap,
10k resamples, seed 42, two-sided, on the five real Phase 7 grid runs plus
the post-hoc-binary derived analysis.

Two metric families need different bootstrap procedures (both already this
project's convention -- see `src.retrieval.metrics.paired_bootstrap_ci` vs.
`src.eval.metrics.paired_bootstrap_ece_diff`):
  - Accuracy, P@1, P@2, MRR are additive per question -> pair by
    per-question difference, resample the diffs (`paired_bootstrap_ci/p`).
  - Macro-F1 and hallucination-rate are NOT additive per question (Macro-F1
    is a whole-confusion-matrix statistic; hallucination-rate is a ratio over
    a variable number of claims per question) -> resample QUESTION indices
    jointly for both sides and recompute the full aggregate statistic from
    scratch on every resample (mirrors `paired_bootstrap_ece_diff` exactly),
    implemented locally as `_bootstrap_aggregate_diff` below.

Four of the 35 pre-registered cells are not tested, for two different
reasons -- neither is chosen from an outcome:

**Not computable (discovered data gap, 2 cells).** B3
(`scripts/run_baselines.py`) never computes `claim_records` / NLI
hallucination data -- confirmed empirically: B3's `predictions.jsonl` rows
carry a raw `rationale` string and citation IDs but never resolved evidence
TEXT, and its `metrics.json` has no `hallucination` key at all. Recomputing
it now would mean re-running retrieval to resolve citation IDs back to
passage/triple text, which is out of scope for a read-only, no-rerun
analysis. So pairs **g** (V3 vs B3) and **h** (V5 vs B3) drop their
halluc-rate sub-test: reported as `"computable": false`.

**Not applicable (2 cells, ADR-040).** Pair **b**'s P@2 and MRR. V3 is
`ranking_source: single_prediction` -- its `ranked` list has exactly one
item on every question -- so its P@2 and MRR both collapse to its accuracy
(P@2 = P(gold in a 1-item list) = accuracy; MRR = 1*accuracy + 0). Testing
them against V5's 4-item ranking compares a list that cannot earn rank-2
credit against one that can: V5's apparent advantage is an artifact of
emitting more items, not ranking quality. §1.0's own contingency ("P@K/MRR
appear only where both sides produce a ranking") already excludes them; the
b row of the registration table contradicted its own rule. Reported as
`"applicable": false` with the arithmetic, excluded from Holm. (b:P@1 stays:
top-1 is well-defined for a single prediction -- but note it is numerically
identical to the confirmatory accuracy test, so it is not independent
evidence.)

Family size for the Holm correction is therefore **m=31** (35 pre-registered
- 2 not computable - 2 not applicable). `holm_sensitivity` re-runs Holm at
m=35 (conservative: the two not-applicable cells at their computed p, the
two not-computable cells as p=1.0 placeholders), m=33 (the previous
reading, which tested the not-applicable cells), m=31 (reported) and m=30
(b:P@1 also dropped) and records whether the reject set changes.

    uv run python -m scripts.analysis.exploratory_family
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.eval.metrics import macro_f1  # noqa: E402
from src.retrieval.metrics import (  # noqa: E402
    holm_bonferroni,
    paired_bootstrap_ci,
    paired_bootstrap_p,
)

RUN_DIRS = {
    "V1": REPO_ROOT / "results" / "medqa_V1_41f7e14_20260908T212655Z",
    "V3": REPO_ROOT / "results" / "medqa_V3_41f7e14-dirty_20260908T230633Z",
    "B3": REPO_ROOT / "results" / "medqa_B3_41f7e14-dirty_20260909T023658Z",
    "V5": REPO_ROOT / "results" / "medqa_V5_41f7e14-dirty_20260909T103452Z",
    "V4-conflict": REPO_ROOT / "results" / "medqa_V4-conflict_41f7e14-dirty_20260909T220306Z",
    "post-hoc-binary": REPO_ROOT / "results" / "phase7_posthoc_binary_audit",
}
V5_METRICS_PATH = RUN_DIRS["V5"] / "metrics.json"
OUT_PATH = REPO_ROOT / "results" / "phase7_grid" / "exploratory_family.json"
MEDQA_LETTERS = ("A", "B", "C", "D")

N_RESAMPLES = 10_000
SEED = 42
ALPHA = 0.05
PRE_REGISTERED_M = 35

# (pair_key, side_A, side_B, metrics) -- exactly docs/04_EVALUATION.md §1.0's table.
FAMILY_SPEC: list[tuple[str, str, str, list[str]]] = [
    ("a", "V1", "V3", ["accuracy", "macro_f1", "halluc_rate"]),
    ("b", "V3", "V5", ["macro_f1", "halluc_rate", "p_at_1", "p_at_2", "mrr"]),
    ("d", "V4-conflict", "V5", ["accuracy", "macro_f1", "halluc_rate", "p_at_1", "p_at_2", "mrr"]),
    ("e", "V1", "V5", ["accuracy", "macro_f1", "halluc_rate"]),
    ("g", "V3", "B3", ["accuracy", "macro_f1", "halluc_rate"]),
    ("h", "V5", "B3", ["accuracy", "macro_f1", "halluc_rate"]),
    (
        "i",
        "post-hoc-binary",
        "V4-conflict",
        ["accuracy", "macro_f1", "halluc_rate", "p_at_1", "p_at_2", "mrr"],
    ),
    (
        "j",
        "post-hoc-binary",
        "V5",
        ["accuracy", "macro_f1", "halluc_rate", "p_at_1", "p_at_2", "mrr"],
    ),
]

# B3 has no claim-level NLI data (see module docstring).
NOT_COMPUTABLE = {("g", "halluc_rate"), ("h", "halluc_rate")}
# V3's 1-item list vs V5's 4-item ranking (see module docstring, ADR-040).
NOT_APPLICABLE = {("b", "p_at_2"), ("b", "mrr")}

RANK_METRICS = ("accuracy", "p_at_1", "p_at_2", "mrr")


def _load_predictions(run_dir: Path) -> dict[str, dict]:
    path = run_dir / "predictions.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    return {r["question_id"]: r for r in rows}


def _load_per_claim(run_dir: Path) -> dict[str, list[dict]] | None:
    """question_id -> that question's per_claim rows (predicted_option scope
    filtering happens later), or None if this run has no hallucination data."""
    metrics_path = run_dir / "metrics.json"
    m = json.loads(metrics_path.read_text(encoding="utf-8"))
    h = m["metrics"].get("hallucination")
    if not h:
        return None
    by_q: dict[str, list[dict]] = {}
    for row in h["per_claim"]:
        by_q.setdefault(row["question_id"], []).append(row)
    return by_q


def _posthoc_binary_per_claim(
    v5_per_claim: dict[str, list[dict]], posthoc_preds: dict[str, dict]
) -> dict[str, list[dict]]:
    """Re-flag V5's own per-claim NLI judgements under post-hoc-binary's
    ranking (same procedure as scripts/analysis/posthoc_binary.py -- zero new
    NLI calls, same evidence/claim pairs, only predicted_option changes)."""
    out: dict[str, list[dict]] = {}
    for qid, rows in v5_per_claim.items():
        pred = posthoc_preds[qid]["pred_option"]
        out[qid] = [{**r, "predicted_option": r["option_id"] == pred} for r in rows]
    return out


def _halluc_rate_predicted(per_question_claim_lists: list[list[dict]]) -> float:
    """`per_question_claim_lists`: one list of per_claim rows per (resampled)
    question. Flattens across the resample, then ratio over the
    predicted_option scope -- exactly `src.eval.nli._aggregate`'s formula."""
    all_claims = [c for claims in per_question_claim_lists for c in claims]
    predicted = [r for r in all_claims if r["predicted_option"]]
    if not predicted:
        return 0.0
    n_ungrounded = sum(1 for r in predicted if not r["entailed"])
    return n_ungrounded / len(predicted)


def _bootstrap_aggregate_diff(
    qids: list[str],
    stat_a: dict[str, float | tuple],
    stat_b: dict[str, float | tuple],
    aggregate_fn,
) -> dict:
    """Generic 'resample question indices, recompute the aggregate statistic
    from scratch' bootstrap (mirrors src.eval.metrics.paired_bootstrap_ece_diff).
    `stat_a`/`stat_b`: qid -> whatever per-question payload `aggregate_fn`
    needs (a (pred, gold) tuple for macro-F1, a list of per_claim rows for
    halluc-rate). `aggregate_fn(list_of_payloads) -> float`."""
    n = len(qids)
    point_a = aggregate_fn([stat_a[q] for q in qids])
    point_b = aggregate_fn([stat_b[q] for q in qids])
    point_diff = point_a - point_b

    rng = random.Random(SEED)
    diffs = []
    for _ in range(N_RESAMPLES):
        idxs = [rng.randrange(n) for _ in range(n)]
        sample = [qids[i] for i in idxs]
        a = aggregate_fn([stat_a[q] for q in sample])
        b = aggregate_fn([stat_b[q] for q in sample])
        diffs.append(a - b)
    diffs.sort()
    lo_idx = int((ALPHA / 2) * N_RESAMPLES)
    hi_idx = N_RESAMPLES - 1 - lo_idx
    ci_lo, ci_hi = diffs[lo_idx], diffs[hi_idx]
    n_le0 = sum(1 for d in diffs if d <= 0)
    n_ge0 = sum(1 for d in diffs if d >= 0)
    p = min(1.0, 2 * min(n_le0, n_ge0) / N_RESAMPLES)

    return {
        "point_a": point_a,
        "point_b": point_b,
        "point_diff": point_diff,
        "mean_diff": sum(diffs) / len(diffs),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_value": p,
    }


def _macro_f1_of(preds_gold: list[tuple[str | None, str | None]]) -> float:
    class _AV:
        def __init__(self, p, g):
            self.pred_option = p
            self.gold_option = g

    return macro_f1([_AV(p, g) for p, g in preds_gold], list(MEDQA_LETTERS))


def _rank_metric_test(pa: dict, pb: dict, common: list[str], metric: str) -> dict:
    """Paired bootstrap for a per-question-additive metric."""
    diffs = [_point_val(pa[qid], metric) - _point_val(pb[qid], metric) for qid in common]
    mean_diff, ci_lo, ci_hi = paired_bootstrap_ci(
        diffs, n_resamples=N_RESAMPLES, seed=SEED, alpha=ALPHA
    )
    p = paired_bootstrap_p(diffs, n_resamples=N_RESAMPLES, seed=SEED)
    return {
        "point_a": sum(_point_val(pa[qid], metric) for qid in common) / len(common),
        "point_b": sum(_point_val(pb[qid], metric) for qid in common) / len(common),
        "mean_diff": mean_diff,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_value": p,
    }


def _not_applicable_arithmetic(pa: dict, pb: dict, common: list[str], metric: str) -> dict:
    """The arithmetic that makes a 1-item-list-vs-ranking P@2/MRR test
    meaningless: side A's list lengths, and its metric == its accuracy."""
    return {
        "side_a_ranked_length_counts": dict(
            sorted(Counter(len(pa[q]["ranked"]) for q in common).items())
        ),
        "side_b_ranked_length_counts": dict(
            sorted(Counter(len(pb[q]["ranked"]) for q in common).items())
        ),
        "side_a_accuracy": sum(_point_val(pa[q], "accuracy") for q in common) / len(common),
        "side_a_metric": sum(_point_val(pa[q], metric) for q in common) / len(common),
        "side_b_accuracy": sum(_point_val(pb[q], "accuracy") for q in common) / len(common),
        "side_b_metric": sum(_point_val(pb[q], metric) for q in common) / len(common),
    }


def _holm_scenario(p_values: dict[str, float]) -> dict:
    holm = holm_bonferroni(p_values, alpha=ALPHA)
    rejected = sorted(k for k, h in holm.items() if h["reject"])
    not_rejected_p = sorted(p_values[k] for k in p_values if k not in rejected)
    return {
        "m": len(p_values),
        "n_rejected": len(rejected),
        "rejected": rejected,
        "max_p_among_rejected": max((p_values[k] for k in rejected), default=None),
        "strictest_threshold_alpha_over_m": ALPHA / len(p_values),
        "smallest_p_not_rejected": not_rejected_p[0] if not_rejected_p else None,
    }


def holm_sensitivity(p_values: dict[str, float], not_applicable_p: dict[str, float]) -> dict:
    scenarios = {
        "m35_pre_registered_conservative": {
            **p_values,
            **not_applicable_p,
            **{f"{pair}:{metric}": 1.0 for pair, metric in sorted(NOT_COMPUTABLE)},
        },
        "m33_previous_reading_tested_not_applicable_cells": {**p_values, **not_applicable_p},
        "m31_actually_tested": dict(p_values),
        "m30_if_b_p_at_1_also_dropped": {k: v for k, v in p_values.items() if k != "b:p_at_1"},
    }
    out = {name: _holm_scenario(pv) for name, pv in scenarios.items()}
    common = set.intersection(*(set(pv) for pv in scenarios.values()))
    reject_sets = [set(out[name]["rejected"]) & common for name in scenarios]
    out["reject_set_identical_on_common_cells"] = all(s == reject_sets[0] for s in reject_sets)
    out["placeholder_rule"] = (
        "m35: the two not-computable B3 halluc-rate cells enter as p=1.0 placeholders "
        "(the most conservative placement -- they can only tighten earlier thresholds); "
        "the two not-applicable cells enter at their computed p."
    )
    out["smallest_nonzero_p_attainable"] = 2 / N_RESAMPLES
    return out


def main() -> None:
    preds = {name: _load_predictions(d) for name, d in RUN_DIRS.items()}

    per_claim: dict[str, dict[str, list[dict]] | None] = {}
    for name in ("V1", "V3", "V5", "V4-conflict"):
        per_claim[name] = _load_per_claim(RUN_DIRS[name])
    per_claim["B3"] = None  # confirmed: no hallucination data (module docstring)
    per_claim["post-hoc-binary"] = _posthoc_binary_per_claim(
        per_claim["V5"], preds["post-hoc-binary"]
    )

    results: dict[str, dict] = {}
    p_values: dict[str, float] = {}
    not_applicable_p: dict[str, float] = {}
    not_computable_report: list[dict] = []
    not_applicable_report: list[dict] = []

    for pair_key, side_a, side_b, metrics in FAMILY_SPEC:
        pa, pb = preds[side_a], preds[side_b]
        common = sorted(set(pa) & set(pb))
        if len(common) != len(pa) or len(common) != len(pb):
            raise SystemExit(
                f"[exploratory_family] {pair_key}: question_id mismatch "
                f"{side_a} n={len(pa)} vs {side_b} n={len(pb)}, common={len(common)}"
            )
        for gold_check in common[:1]:
            assert pa[gold_check]["gold_option"] == pb[gold_check]["gold_option"]

        for metric in metrics:
            key = f"{pair_key}:{metric}"

            if (pair_key, metric) in NOT_COMPUTABLE:
                results[key] = {
                    "pair": pair_key,
                    "side_a": side_a,
                    "side_b": side_b,
                    "metric": metric,
                    "computable": False,
                    "why_not": (
                        f"{side_b if side_b == 'B3' else side_a} has no claim-level "
                        "NLI data (scripts/run_baselines.py never computes "
                        "claim_records / resolved evidence text for B3 -- see module "
                        "docstring). Not attempted, not counted toward m."
                    ),
                }
                not_computable_report.append(results[key])
                continue

            if (pair_key, metric) in NOT_APPLICABLE:
                arith = _not_applicable_arithmetic(pa, pb, common, metric)
                if set(arith["side_a_ranked_length_counts"]) != {1} or (
                    arith["side_a_metric"] != arith["side_a_accuracy"]
                ):
                    raise SystemExit(
                        f"[exploratory_family] {key}: registered as not-applicable because "
                        f"{side_a} emits a 1-item list, but the data says otherwise: {arith}"
                    )
                # p computed ONLY for holm_sensitivity's m33/m35 scenarios -- never
                # reported as a test, never in the reported Holm family.
                sens = _rank_metric_test(pa, pb, common, metric)
                not_applicable_p[key] = sens["p_value"]
                results[key] = {
                    "pair": pair_key,
                    "side_a": side_a,
                    "side_b": side_b,
                    "metric": metric,
                    "computable": True,
                    "applicable": False,
                    "why_not": (
                        f"{side_a} is ranking_source=single_prediction: a 1-item list on "
                        f"every question, so its {metric} equals its accuracy. Against "
                        f"{side_b}'s 4-item ranking the test rewards emitting more items, "
                        "not ranking quality. docs/04 §1.0's own rule ('P@K/MRR appear only "
                        "where both sides produce a ranking') excludes it (ADR-040). Not "
                        "counted toward m."
                    ),
                    "arithmetic": arith,
                    "sensitivity_only_p_value": sens["p_value"],
                }
                not_applicable_report.append(results[key])
                continue

            if metric in RANK_METRICS:
                res = _rank_metric_test(pa, pb, common, metric)

            elif metric == "macro_f1":
                stat_a = {qid: (pa[qid]["pred_option"], pa[qid]["gold_option"]) for qid in common}
                stat_b = {qid: (pb[qid]["pred_option"], pb[qid]["gold_option"]) for qid in common}
                res = _bootstrap_aggregate_diff(common, stat_a, stat_b, _macro_f1_of)

            elif metric == "halluc_rate":
                claims_a, claims_b = per_claim[side_a], per_claim[side_b]
                stat_a = {qid: claims_a.get(qid, []) for qid in common}
                stat_b = {qid: claims_b.get(qid, []) for qid in common}
                res = _bootstrap_aggregate_diff(common, stat_a, stat_b, _halluc_rate_predicted)

            else:
                raise ValueError(metric)

            res["pair"] = pair_key
            res["side_a"] = side_a
            res["side_b"] = side_b
            res["metric"] = metric
            res["computable"] = True
            res["applicable"] = True
            res["n"] = len(common)
            if res["ci_lo"] > 0:
                res["verdict"] = f"{side_a} significantly > {side_b}"
            elif res["ci_hi"] < 0:
                res["verdict"] = f"{side_b} significantly > {side_a}"
            else:
                res["verdict"] = "tie (95% CI includes 0)"
            results[key] = res
            p_values[key] = res["p_value"]

    holm = holm_bonferroni(p_values, alpha=ALPHA)
    for key, h in holm.items():
        results[key]["holm_rank"] = h["rank"]
        results[key]["holm_threshold"] = h["threshold"]
        results[key]["holm_reject"] = h["reject"]

    sensitivity = holm_sensitivity(p_values, not_applicable_p)
    m_expected = PRE_REGISTERED_M - len(NOT_COMPUTABLE) - len(NOT_APPLICABLE)
    if len(p_values) != m_expected:
        raise SystemExit(
            f"[exploratory_family] tested {len(p_values)} cells, expected {m_expected}"
        )

    out = {
        "pre_registered_family_size": PRE_REGISTERED_M,
        "actually_tested_family_size": len(p_values),
        "not_computable": not_computable_report,
        "not_applicable": not_applicable_report,
        "n_resamples": N_RESAMPLES,
        "seed": SEED,
        "alpha": ALPHA,
        "multiplicity_correction": (
            "Holm-Bonferroni across actually_tested_family_size "
            "(src.retrieval.metrics.holm_bonferroni)"
        ),
        "holm_sensitivity": sensitivity,
        "results": results,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"[exploratory_family] wrote {OUT_PATH}")
    print(
        f"[exploratory_family] m_pre_registered={PRE_REGISTERED_M} "
        f"m_actually_tested={len(p_values)}"
    )
    for key in sorted(results, key=lambda k: (results[k].get("holm_rank", 999),)):
        r = results[key]
        if not r.get("computable", True):
            print(f"  {key:30s} NOT COMPUTABLE")
            continue
        if not r.get("applicable", True):
            print(f"  {key:30s} NOT APPLICABLE  {r['arithmetic']}")
            continue
        print(
            f"  {key:30s} p={r['p_value']:.4f} rank={r.get('holm_rank')} "
            f"reject={r.get('holm_reject')}  {r['verdict']}"
        )
    for name, sc in sensitivity.items():
        if isinstance(sc, dict):
            print(
                f"  [holm {name}] m={sc['m']} rejected={sc['n_rejected']} "
                f"max_p_rejected={sc['max_p_among_rejected']} "
                f"smallest_p_not_rejected={sc['smallest_p_not_rejected']}"
            )
    print(
        "  [holm] reject set identical on common cells: "
        f"{sensitivity['reject_set_identical_on_common_cells']}"
    )


def _point_val(row: dict, metric: str) -> float:
    gold = row["gold_option"]
    if metric == "accuracy":
        return 1.0 if row["pred_option"] == gold else 0.0
    if metric == "p_at_1":
        return 1.0 if row["ranked"][:1] and gold in row["ranked"][:1] else 0.0
    if metric == "p_at_2":
        return 1.0 if gold in row["ranked"][:2] else 0.0
    if metric == "mrr":
        return (1.0 / (row["ranked"].index(gold) + 1)) if gold in row["ranked"] else 0.0
    raise ValueError(metric)


if __name__ == "__main__":
    main()
