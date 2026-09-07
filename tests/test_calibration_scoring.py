"""Unit tests for src/calibration/scoring.py. No LLM, no I/O -- pure functions
over hand-built SupportMatrix fixtures."""

from __future__ import annotations

import pytest

from src.agents.state import OptionSupport, SupportCell, SupportMatrix
from src.calibration.scoring import CalibrationWeights, compute_logit, extract_option_features


def _matrix(cells, option_support, options=("A", "B", "C", "D")) -> SupportMatrix:
    return SupportMatrix(
        question_id="q1",
        dataset="medqa",
        query="q?",
        options={o: o.lower() for o in options},
        cells=cells,
        option_support=option_support,
    )


def _cell(option_id, claim_idx, evidence_id, label, modality="dense", rerank_score=0.0):
    return SupportCell(
        option_id=option_id,
        claim_idx=claim_idx,
        evidence_id=evidence_id,
        label=label,
        modality=modality,
        rerank_score=rerank_score,
    )


def test_extract_option_features_one_row_per_option():
    m = _matrix([], [OptionSupport(option_id=o) for o in "ABCD"])
    feats = extract_option_features(m)
    assert [f.option_id for f in feats] == ["A", "B", "C", "D"]


def test_extract_option_features_all_zero_for_option_with_no_cells():
    m = _matrix([], [OptionSupport(option_id=o) for o in "ABCD"])
    feats = extract_option_features(m)
    for f in feats:
        assert f.s_pos == 0
        assert f.s_neg == 0
        assert f.kg_support_count == 0
        assert f.mean_rerank_score == 0.0


def test_kg_support_count_counts_only_support_kg_cells():
    cells = [
        _cell("A", 0, "kg::x", "SUPPORT", modality="kg"),
        _cell("A", 1, "kg::y", "CONTRADICT", modality="kg"),  # CONTRADICT, not counted
        _cell("A", 2, "chunk::z#0", "SUPPORT", modality="dense"),  # not kg, not counted
    ]
    m = _matrix(cells, [OptionSupport(option_id="A", s_pos=2, s_neg=1)])
    feats = extract_option_features(m)
    a = next(f for f in feats if f.option_id == "A")
    assert a.kg_support_count == 1


def test_mean_rerank_score_averages_support_and_contradict_not_neutral():
    cells = [
        _cell("A", 0, "chunk::1", "SUPPORT", rerank_score=0.8),
        _cell("A", 1, "chunk::2", "CONTRADICT", rerank_score=0.4),
    ]
    m = _matrix(cells, [OptionSupport(option_id="A", s_pos=1, s_neg=1)])
    feats = extract_option_features(m)
    a = next(f for f in feats if f.option_id == "A")
    assert a.mean_rerank_score == pytest.approx(0.6)


def test_mean_rerank_score_zero_when_no_non_neutral_cells():
    m = _matrix([], [OptionSupport(option_id="A")])
    feats = extract_option_features(m)
    assert feats[0].mean_rerank_score == 0.0


def test_s_pos_s_neg_s_neu_come_from_option_support_not_recomputed():
    """extract_option_features must trust OptionSupport's own counts (the
    single source of truth in src/agents/graph.py), not re-derive them from
    cells -- this pins that contract."""
    m = _matrix([], [OptionSupport(option_id="A", s_pos=7, s_neg=3, s_neu=2, resolved=False)])
    feats = extract_option_features(m)
    a = feats[0]
    assert (a.s_pos, a.s_neg, a.s_neu, a.resolved) == (7, 3, 2, False)


def test_extract_option_features_ignores_cells_for_unknown_options():
    cells = [_cell("Z", 0, "chunk::1", "SUPPORT")]  # "Z" not in matrix.options
    m = _matrix(cells, [OptionSupport(option_id="A")], options=("A",))
    feats = extract_option_features(m)
    assert len(feats) == 1
    assert feats[0].option_id == "A"


def test_compute_logit_matches_formula():
    from src.calibration.scoring import OptionFeatures

    f = OptionFeatures(
        option_id="A",
        s_pos=3,
        s_neg=1,
        s_neu=0,
        kg_support_count=2,
        mean_rerank_score=0.5,
        resolved=True,
    )
    w = CalibrationWeights(alpha=2.0, beta=1.5, gamma=0.5, delta=4.0)
    # 2*3 - 1.5*1 + 0.5*2 + 4*0.5 = 6 - 1.5 + 1 + 2 = 7.5
    assert compute_logit(f, w) == pytest.approx(7.5)


def test_compute_logit_default_weights_is_sanity_ranking():
    """alpha=beta=1, gamma=delta=0 (the module default) reduces to the Phase 4
    pre-calibration s+ - s- argmax rule (docs/01_ARCHITECTURE.md)."""
    from src.calibration.scoring import OptionFeatures

    f = OptionFeatures(
        option_id="A",
        s_pos=5,
        s_neg=2,
        s_neu=0,
        kg_support_count=9,
        mean_rerank_score=9.0,
        resolved=True,
    )
    assert compute_logit(f, CalibrationWeights()) == pytest.approx(3.0)
