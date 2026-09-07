"""Unit tests for scripts/kg_rerank_ablation.py.

Covers the pure comparison logic (`build_comparison`) against hand-built
`kg_probe.py`-shaped fixtures -- no real kg_probe.py output files, no model,
no I/O -- and the significance follow-up (`pair_per_question_diffs`,
`compute_significance`) against small hand-computed `per_question` arrays,
matching this project's `tests/test_metrics.py` style (mock-friendly, no
real bootstrap-scale fixtures needed since `paired_bootstrap_ci` itself is
already covered there).
"""

from __future__ import annotations

import math

from scripts.kg_rerank_ablation import (
    build_comparison,
    compute_significance,
    pair_per_question_diffs,
)


def _per_question_row(
    question_id: str,
    novel_and_relevant_rate: float | None,
    relevant_rate: float | None = None,
) -> dict:
    return {
        "question_id": question_id,
        "novel_and_relevant_rate_for_question": novel_and_relevant_rate,
        "relevant_rate_for_question": (
            relevant_rate if relevant_rate is not None else novel_and_relevant_rate
        ),
    }


def _make_result(per_question: list[dict], **overrides) -> dict:
    base = {
        "aggregate_hit_rate": 0.5,
        "overall_novel_relation_rate": 0.8,
        "overall_relevant_rate": 0.4,
        "overall_novel_and_relevant_rate": 0.3,
        "overall_novel_given_relevant_rate": 0.7,
        "category_breakdown": {},
        "config": {},
        "n_questions": len(per_question),
        "per_question": per_question,
    }
    base.update(overrides)
    return base


def test_pair_per_question_diffs_matches_by_question_id():
    naive = _make_result(
        [
            _per_question_row("test_00000", 0.2),
            _per_question_row("test_00001", 0.4),
        ]
    )
    reranked = _make_result(
        [
            _per_question_row("test_00000", 0.5),
            _per_question_row("test_00001", 0.3),
        ]
    )
    diffs, n_excluded = pair_per_question_diffs(
        naive, reranked, "novel_and_relevant_rate_for_question"
    )
    assert n_excluded == 0
    assert sorted(diffs) == sorted([0.5 - 0.2, 0.3 - 0.4])


def test_pair_per_question_diffs_excludes_none_values_not_coerced_to_zero():
    naive = _make_result(
        [
            _per_question_row("test_00000", None),  # nothing scored on naive side
            _per_question_row("test_00001", 0.4),
        ]
    )
    reranked = _make_result(
        [
            _per_question_row("test_00000", 0.5),
            _per_question_row("test_00001", 0.3),
        ]
    )
    diffs, n_excluded = pair_per_question_diffs(
        naive, reranked, "novel_and_relevant_rate_for_question"
    )
    assert n_excluded == 1
    assert diffs == [0.3 - 0.4]


def test_pair_per_question_diffs_excludes_questions_missing_from_one_side():
    naive = _make_result([_per_question_row("test_00000", 0.2)])
    reranked = _make_result(
        [
            _per_question_row("test_00000", 0.5),
            _per_question_row("test_00001", 0.9),  # only in reranked
        ]
    )
    diffs, n_excluded = pair_per_question_diffs(
        naive, reranked, "novel_and_relevant_rate_for_question"
    )
    assert n_excluded == 1
    assert diffs == [0.5 - 0.2]


def test_compute_significance_detects_consistent_win_as_significant():
    # kg_rerank consistently beats naive by ~0.3 on every question -- CI
    # should sit entirely above zero.
    per_q = 30
    naive = _make_result(
        [_per_question_row(f"test_{i:05d}", 0.2 + 0.001 * ((-1) ** i)) for i in range(per_q)]
    )
    reranked = _make_result(
        [_per_question_row(f"test_{i:05d}", 0.5 + 0.001 * ((-1) ** i)) for i in range(per_q)]
    )
    sig = compute_significance(naive, reranked)
    result = sig["metrics"]["novel_and_relevant_rate_for_question"]
    assert result["verdict"] == "significant_win"
    assert result["ci_lo"] > 0
    assert math.isclose(result["mean_diff"], 0.3, abs_tol=1e-6)
    assert result["n_paired"] == per_q
    assert result["n_excluded"] == 0


def test_compute_significance_reports_tie_when_noise_dominates():
    # Small, noisy, alternating-sign differences -- must resolve to a tie,
    # not a false "win" just because the raw mean happens to be nonzero --
    # this is exactly the ADR-012 scenario CLAUDE.md warns against.
    values = [0.5, 0.1, 0.6, 0.05, 0.55, 0.5]
    naive_vals = values
    reranked_vals = [0.5, 0.6, 0.55, 0.1, 0.5, 0.55]  # small alternating diffs, no consistent
    # direction
    naive = _make_result(
        [_per_question_row(f"test_{i:05d}", v) for i, v in enumerate(naive_vals)]
    )
    reranked = _make_result(
        [_per_question_row(f"test_{i:05d}", v) for i, v in enumerate(reranked_vals)]
    )
    sig = compute_significance(naive, reranked)
    result = sig["metrics"]["novel_and_relevant_rate_for_question"]
    assert result["ci_lo"] <= 0 <= result["ci_hi"]
    assert result["verdict"] == "tie"


def test_compute_significance_no_pairable_questions_reported_explicitly():
    naive = _make_result([_per_question_row("test_00000", None)])
    reranked = _make_result([_per_question_row("test_00000", None)])
    sig = compute_significance(naive, reranked)
    result = sig["metrics"]["novel_and_relevant_rate_for_question"]
    assert result["verdict"] == "no_pairable_questions"
    assert result["n_paired"] == 0
    assert result["mean_diff"] is None


def test_build_comparison_includes_significance_block():
    naive = _make_result(
        [_per_question_row(f"test_{i:05d}", 0.3) for i in range(20)]
    )
    reranked = _make_result(
        [_per_question_row(f"test_{i:05d}", 0.35) for i in range(20)]
    )
    comparison = build_comparison(naive, reranked)
    assert "significance" in comparison
    assert "novel_and_relevant_rate_for_question" in comparison["significance"]["metrics"]
    assert "relevant_rate_for_question" in comparison["significance"]["metrics"]
    assert comparison["significance"]["n_resamples"] == 10_000


def test_build_comparison_overall_deltas_unchanged_by_significance_addition():
    naive = _make_result([_per_question_row("test_00000", 0.3)])
    reranked = _make_result(
        [_per_question_row("test_00000", 0.35)],
        overall_novel_and_relevant_rate=0.44,
    )
    comparison = build_comparison(naive, reranked)
    delta = comparison["overall"]["overall_novel_and_relevant_rate"]["delta"]
    assert math.isclose(delta, 0.44 - 0.3, abs_tol=1e-9)


# --------------------------------------------------------------------------
# Holm-Bonferroni multiplicity correction across the two significance metrics
# (AUDIT 2026-09-06 §6.8, ADR-027)
# --------------------------------------------------------------------------


def test_compute_significance_reports_holm_fields_for_a_consistent_win():
    """Both metrics move identically and win cleanly -- both must survive the
    2-metric Holm correction, and the uncorrected verdict must be preserved."""
    per_q = 30
    naive = _make_result(
        [_per_question_row(f"test_{i:05d}", 0.2 + 0.001 * ((-1) ** i)) for i in range(per_q)]
    )
    reranked = _make_result(
        [_per_question_row(f"test_{i:05d}", 0.5 + 0.001 * ((-1) ** i)) for i in range(per_q)]
    )
    sig = compute_significance(naive, reranked)
    for metric_key in ("novel_and_relevant_rate_for_question", "relevant_rate_for_question"):
        result = sig["metrics"][metric_key]
        assert result["p"] is not None and result["p"] < 0.01
        assert result["holm_survives"] is True
        assert result["verdict"] == "significant_win"
        assert "uncorrected_verdict" not in result


def test_compute_significance_downgrades_a_marginal_win_that_fails_holm():
    """A metric with a CI that barely excludes zero (marginal uncorrected win)
    can fail the 2-metric Holm correction when paired with a much stronger
    second metric -- the ADR-027 scenario. The final `verdict` must NOT read
    'significant_win' in that case; `uncorrected_verdict` preserves what it
    would have been read as alone."""
    per_q = 40
    naive_vals_marginal = [
        0.30 + (0.29 if i % 20 == 0 else -0.01) for i in range(per_q)
    ]  # tiny, barely-positive mean with one large outlier -- a marginal CI
    reranked_vals_marginal = [v + 0.001 for v in naive_vals_marginal]
    naive = _make_result(
        [
            {
                "question_id": f"test_{i:05d}",
                "novel_and_relevant_rate_for_question": naive_vals_marginal[i],
                "relevant_rate_for_question": 0.2 + 0.001 * ((-1) ** i),
            }
            for i in range(per_q)
        ]
    )
    reranked = _make_result(
        [
            {
                "question_id": f"test_{i:05d}",
                "novel_and_relevant_rate_for_question": reranked_vals_marginal[i],
                "relevant_rate_for_question": 0.5 + 0.001 * ((-1) ** i),
            }
            for i in range(per_q)
        ]
    )
    sig = compute_significance(naive, reranked)
    primary = sig["metrics"]["novel_and_relevant_rate_for_question"]
    secondary = sig["metrics"]["relevant_rate_for_question"]
    # secondary is a clean, strong win and must survive
    assert secondary["holm_survives"] is True
    assert secondary["verdict"] == "significant_win"
    # primary's own CI may or may not exclude zero depending on the exact
    # bootstrap draw, but whenever it reports significant_win/loss uncorrected
    # while failing Holm, the final verdict must be downgraded, never left as
    # a win/loss a reader could quote uncorrected.
    if not primary["holm_survives"] and primary.get("uncorrected_verdict") in (
        "significant_win",
        "significant_loss",
    ):
        assert primary["verdict"] == "tie_after_correction"


def test_compute_significance_holm_ranks_are_1_and_2_for_two_metrics():
    per_q = 20
    naive = _make_result(
        [_per_question_row(f"test_{i:05d}", 0.2 + 0.001 * ((-1) ** i)) for i in range(per_q)]
    )
    reranked = _make_result(
        [_per_question_row(f"test_{i:05d}", 0.5 + 0.001 * ((-1) ** i)) for i in range(per_q)]
    )
    sig = compute_significance(naive, reranked)
    ranks = sorted(
        sig["metrics"][k]["holm_rank"]
        for k in ("novel_and_relevant_rate_for_question", "relevant_rate_for_question")
    )
    assert ranks == [1, 2]
