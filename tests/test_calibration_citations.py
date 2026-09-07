"""Unit tests for src/calibration/citations.py."""

from __future__ import annotations

from src.agents.state import SupportCell, SupportMatrix
from src.calibration.citations import cite, cite_all


def _matrix(cells, options=("A", "B")) -> SupportMatrix:
    return SupportMatrix(
        question_id="q1",
        dataset="medqa",
        query="q?",
        options={o: o.lower() for o in options},
        cells=cells,
    )


def _cell(option_id, evidence_id, label):
    return SupportCell(option_id=option_id, claim_idx=0, evidence_id=evidence_id, label=label)


def test_cite_returns_only_support_cells_for_the_option():
    m = _matrix(
        [
            _cell("A", "chunk::1", "SUPPORT"),
            _cell("A", "chunk::2", "CONTRADICT"),
            _cell("B", "chunk::3", "SUPPORT"),
        ]
    )
    assert cite("A", m) == ["chunk::1"]


def test_cite_dedupes_and_sorts():
    m = _matrix(
        [
            _cell("A", "chunk::z", "SUPPORT"),
            _cell("A", "chunk::a", "SUPPORT"),
            _cell("A", "chunk::a", "SUPPORT"),  # dup id: two claims cite the same evidence
        ]
    )
    assert cite("A", m) == ["chunk::a", "chunk::z"]


def test_cite_empty_for_option_with_no_support_cells():
    m = _matrix([_cell("A", "chunk::1", "CONTRADICT")])
    assert cite("A", m) == []


def test_cite_empty_for_option_with_no_cells_at_all():
    m = _matrix([])
    assert cite("A", m) == []


def test_cite_all_covers_every_option_including_empty_ones():
    m = _matrix([_cell("A", "chunk::1", "SUPPORT")], options=("A", "B", "C"))
    result = cite_all(m)
    assert result == {"A": ["chunk::1"], "B": [], "C": []}
