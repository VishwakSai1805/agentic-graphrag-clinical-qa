"""Side-by-side comparison artifact for the `kg_rerank` ablation (Phase 2
follow-up, docs/05_DECISIONS.md -- the ADR this addition records).

Per the project owner's explicit instruction, `kg_rerank` is a measured
on/off ablation, not a silent triple-selection improvement: this script does
NOT duplicate `scripts/kg_probe.py`'s metrics logic (hit-rate, novelty,
relevance -- all of it lives in one place, per CLAUDE.md's "every Cypher
query/every metric lives in one reviewable place" spirit extended here). It
only reads the two JSON files `scripts/kg_probe.py` already produces (one run
against the canonical `data/processed/kg_cache/medqa/` -- kg_rerank=False --
and one run against the separate `data/processed/kg_cache_kg_rerank/medqa/`
-- kg_rerank=True) and writes a side-by-side comparison plus a
human-readable stdout table.

**Significance follow-up (2026-09-06, see docs/05_DECISIONS.md's ADR for this
revision):** the raw-mean overall deltas above (e.g. the first pass's
+0.0380 on `overall_novel_and_relevant_rate`) are a point-estimate inequality
at n=200 questions -- exactly what CLAUDE.md's research-hygiene section
forbids reporting as "reranking helps" without a paired-bootstrap check, per
the ADR-012 precedent (a real point-estimate "RRF loses" reading on NFCorpus
turned out to be a statistical tie once checked). This revision pairs the two
variants' `per_question` arrays (added to `scripts/kg_probe.py`'s output by
the same follow-up) by `question_id` and runs
`src.retrieval.metrics.paired_bootstrap_ci` (10k resamples, the project
standard -- docs/04_EVALUATION.md §1) on the per-question difference
(kg_rerank - naive) for `novel_and_relevant_rate_for_question` (primary) and
`relevant_rate_for_question` (secondary, the more direct "did reranking fix
the selection-precision problem" question). Questions where either variant's
per-question rate is `None` (nothing was scored for that question -- see
`scripts/kg_probe.py: _question_rate`) are excluded from that metric's pairing
entirely, not coerced to 0.0, since a "nothing scored" question says nothing
about reranking's effect on relevance/novelty quality. The verdict follows
the exact ADR-012 rule: CI entirely above 0 -> significant win; CI includes 0
-> statistical tie, must be reported as a tie, not a win; CI entirely below 0
-> significant regression.

Usage (run kg_probe.py TWICE first, then this):
    uv run python -m scripts.kg_probe --dataset medqa --limit 200 \\
        --out results/phase2_kg_probe/medqa_n200.json
    uv run python -m scripts.kg_probe --dataset medqa --limit 200 \\
        --cache-dir data/processed/kg_cache_kg_rerank \\
        --out results/phase2_kg_probe/medqa_n200_kg_rerank_variant.json
    uv run python -m scripts.kg_rerank_ablation \\
        --naive results/phase2_kg_probe/medqa_n200.json \\
        --reranked results/phase2_kg_probe/medqa_n200_kg_rerank_variant.json \\
        --out results/phase2_kg_probe/medqa_n200_kg_rerank_ablation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.retrieval.metrics import paired_bootstrap_ci  # noqa: E402

_OVERALL_METRICS = [
    "aggregate_hit_rate",
    "overall_novel_relation_rate",
    "overall_relevant_rate",
    "overall_novel_and_relevant_rate",
    "overall_novel_given_relevant_rate",
]

_SIGNIFICANCE_METRICS = [
    "novel_and_relevant_rate_for_question",
    "relevant_rate_for_question",
]

N_BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 42


def _category_metrics(result: dict) -> dict:
    out = {}
    for category, stats in result["category_breakdown"].items():
        out[category] = {
            "n": stats["n"],
            "underpowered": stats["underpowered"],
            "hit_rate": stats["hit_rate"],
            "relevant_rate": stats["relevant_rate"],
            "novel_relation_rate": stats["novel_relation_rate"],
            "novel_and_relevant_rate": stats["novel_and_relevant_rate"],
        }
    return out


def pair_per_question_diffs(
    naive: dict, reranked: dict, metric_key: str
) -> tuple[list[float], int]:
    """Pure function (no I/O, no bootstrap) -- pairs the two variants'
    `per_question` arrays by `question_id` and returns
    `([kg_rerank_value - naive_value for each pairable question], n_excluded)`.

    A question is EXCLUDED (not coerced to a 0.0 diff) if:
      - it is missing from either variant's `per_question` array entirely, or
      - either variant's `metric_key` value for that question is `None`
        (nothing was scored for novelty/relevance on that side -- e.g. a
        cache-missing question, or a question whose subgraph had zero
        triples to sample from -- see `scripts/kg_probe.py: _question_rate`).

    `n_excluded` is always reported alongside the diffs so a caller can see
    how much of the nominal n=200 actually contributed to the significance
    test -- CLAUDE.md's `n_evaluated` discipline applies to this pairing step
    too, not just top-level eval runs."""
    naive_by_id = {row["question_id"]: row.get(metric_key) for row in naive.get("per_question", [])}
    reranked_by_id = {
        row["question_id"]: row.get(metric_key) for row in reranked.get("per_question", [])
    }
    all_ids = sorted(set(naive_by_id) | set(reranked_by_id))
    diffs: list[float] = []
    n_excluded = 0
    for qid in all_ids:
        naive_v = naive_by_id.get(qid)
        reranked_v = reranked_by_id.get(qid)
        if naive_v is None or reranked_v is None:
            n_excluded += 1
            continue
        diffs.append(reranked_v - naive_v)
    return diffs, n_excluded


def compute_significance(naive: dict, reranked: dict) -> dict:
    """Runs `paired_bootstrap_ci` (10k resamples, project standard --
    docs/04_EVALUATION.md §1) on the paired per-question difference
    (kg_rerank - naive) for each metric in `_SIGNIFICANCE_METRICS`, and
    applies the exact ADR-012 decision rule (docs/05_DECISIONS.md):
      - ci_lo > 0  -> "significant_win" (kg_rerank significantly better)
      - ci_hi < 0  -> "significant_loss" (kg_rerank significantly worse)
      - otherwise  -> "tie" (CI includes 0 -- not evidence of a difference,
        must be reported as a tie, never as a win)

    Pure aside from calling `paired_bootstrap_ci` (itself pure/deterministic
    given a seed) -- no file I/O, so it's unit-testable directly against
    hand-built `per_question` fixtures (tests/test_kg_rerank_ablation.py)."""
    significance = {
        "n_resamples": N_BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
        "decision_rule": (
            "ADR-012 (docs/05_DECISIONS.md): ci_lo > 0 -> significant_win, "
            "ci_hi < 0 -> significant_loss, otherwise -> tie (CI includes 0, "
            "not evidence of a difference)."
        ),
        "metrics": {},
    }
    for metric_key in _SIGNIFICANCE_METRICS:
        diffs, n_excluded = pair_per_question_diffs(naive, reranked, metric_key)
        if not diffs:
            significance["metrics"][metric_key] = {
                "n_paired": 0,
                "n_excluded": n_excluded,
                "mean_diff": None,
                "ci_lo": None,
                "ci_hi": None,
                "verdict": "no_pairable_questions",
            }
            continue
        mean_diff, ci_lo, ci_hi = paired_bootstrap_ci(
            diffs, n_resamples=N_BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED
        )
        if ci_lo > 0:
            verdict = "significant_win"
        elif ci_hi < 0:
            verdict = "significant_loss"
        else:
            verdict = "tie"
        significance["metrics"][metric_key] = {
            "n_paired": len(diffs),
            "n_excluded": n_excluded,
            "mean_diff": mean_diff,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "verdict": verdict,
        }
    return significance


def build_comparison(naive: dict, reranked: dict) -> dict:
    """Pure function (no file I/O) so it's unit-testable without real
    kg_probe.py output files -- see tests/test_kg_rerank_ablation.py."""
    overall = {}
    for metric in _OVERALL_METRICS:
        naive_v = naive.get(metric)
        reranked_v = reranked.get(metric)
        delta = (
            reranked_v - naive_v if naive_v is not None and reranked_v is not None else None
        )
        overall[metric] = {"naive": naive_v, "kg_rerank": reranked_v, "delta": delta}

    naive_cat = _category_metrics(naive)
    reranked_cat = _category_metrics(reranked)
    category_comparison = {}
    for category in sorted(set(naive_cat) | set(reranked_cat)):
        nc = naive_cat.get(category, {})
        rc = reranked_cat.get(category, {})
        nc_rate = nc.get("novel_and_relevant_rate")
        rc_rate = rc.get("novel_and_relevant_rate")
        delta = (rc_rate - nc_rate) if nc_rate is not None and rc_rate is not None else None
        category_comparison[category] = {
            "n_naive": nc.get("n"),
            "n_kg_rerank": rc.get("n"),
            "naive_novel_and_relevant_rate": nc_rate,
            "kg_rerank_novel_and_relevant_rate": rc_rate,
            "delta_novel_and_relevant_rate": delta,
        }

    return {
        "overall": overall,
        "category_breakdown": category_comparison,
        "significance": compute_significance(naive, reranked),
        "naive_config": naive.get("config"),
        "kg_rerank_config": reranked.get("config"),
        "naive_n_questions": naive.get("n_questions"),
        "kg_rerank_n_questions": reranked.get("n_questions"),
    }


def print_table(comparison: dict) -> None:
    print("=" * 78)
    print("kg_rerank ablation: naive (alphabetical) vs. MedCPT-reranked triple selection")
    print("=" * 78)
    print(f"{'metric':<34}{'naive':>12}{'kg_rerank':>12}{'delta':>12}")
    for metric, vals in comparison["overall"].items():
        naive_s = f"{vals['naive']:.4f}" if vals["naive"] is not None else "n/a"
        rerank_s = f"{vals['kg_rerank']:.4f}" if vals["kg_rerank"] is not None else "n/a"
        delta_s = f"{vals['delta']:+.4f}" if vals["delta"] is not None else "n/a"
        print(f"{metric:<34}{naive_s:>12}{rerank_s:>12}{delta_s:>12}")
    print()
    print(
        f"{'category':<22}{'n(naive/rr)':>14}{'naive nov&rel':>16}"
        f"{'rr nov&rel':>12}{'delta':>10}"
    )
    for category, vals in comparison["category_breakdown"].items():
        n_str = f"{vals['n_naive']}/{vals['n_kg_rerank']}"
        naive_s = (
            f"{vals['naive_novel_and_relevant_rate']:.4f}"
            if vals["naive_novel_and_relevant_rate"] is not None
            else "n/a"
        )
        rr_s = (
            f"{vals['kg_rerank_novel_and_relevant_rate']:.4f}"
            if vals["kg_rerank_novel_and_relevant_rate"] is not None
            else "n/a"
        )
        delta_s = (
            f"{vals['delta_novel_and_relevant_rate']:+.4f}"
            if vals["delta_novel_and_relevant_rate"] is not None
            else "n/a"
        )
        print(f"{category:<22}{n_str:>14}{naive_s:>16}{rr_s:>12}{delta_s:>10}")
    print()
    sig = comparison["significance"]
    print(
        f"paired-bootstrap significance (kg_rerank - naive), "
        f"n_resamples={sig['n_resamples']}, seed={sig['seed']}:"
    )
    for metric_key, vals in sig["metrics"].items():
        if vals["mean_diff"] is None:
            print(f"  {metric_key}: no pairable questions (n_excluded={vals['n_excluded']})")
            continue
        print(
            f"  {metric_key}: mean_diff={vals['mean_diff']:+.4f}  "
            f"95% CI=[{vals['ci_lo']:+.4f}, {vals['ci_hi']:+.4f}]  "
            f"n_paired={vals['n_paired']} n_excluded={vals['n_excluded']}  "
            f"-> {vals['verdict'].upper()}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--naive", required=True, help="kg_probe.py output JSON, kg_rerank=False")
    parser.add_argument(
        "--reranked", required=True, help="kg_probe.py output JSON, kg_rerank=True"
    )
    default_out = REPO_ROOT / "results" / "phase2_kg_probe" / "medqa_n200_kg_rerank_ablation.json"
    parser.add_argument("--out", default=str(default_out))
    args = parser.parse_args()

    naive = json.loads(Path(args.naive).read_text(encoding="utf-8"))
    reranked = json.loads(Path(args.reranked).read_text(encoding="utf-8"))

    comparison = build_comparison(naive, reranked)
    comparison["generated_at_utc"] = datetime.now(UTC).isoformat()
    comparison["naive_source_file"] = str(args.naive)
    comparison["kg_rerank_source_file"] = str(args.reranked)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")

    print_table(comparison)
    print()
    print(f"wrote comparison artifact to {out_path}")


if __name__ == "__main__":
    main()
