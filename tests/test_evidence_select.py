"""Tests for src/agents/evidence_select.py -- modality-reserved trim (ADR-023)."""

from __future__ import annotations

from src.agents.evidence_select import select_evidence
from src.schema import Evidence


def _p(i: int) -> Evidence:
    return Evidence(
        id=f"chunk::D{i}#0", modality="dense", text=f"passage {i}",
        source_meta={"source": "textbook"}, scores={"rerank": 10 - i},
    )


def _t(i: int) -> Evidence:
    return Evidence(
        id=f"kg::h{i}|r|t{i}", modality="kg", text=f"triple {i}", source_meta={}, scores={},
    )


def test_full_split_when_both_modalities_have_enough():
    ev, mix = select_evidence(
        query="q", passages=[_p(i) for i in range(12)], triples=[_t(i) for i in range(6)],
        split=(8, 4), max_evidence=12,
    )
    assert len(ev) == 12
    assert mix.n_passages == 8 and mix.n_triples == 4
    assert mix.kg_slot_filled is True
    assert mix.backfilled_passages == 0
    # KG items come after passages
    assert [e.modality for e in ev] == ["dense"] * 8 + ["kg"] * 4


def test_zero_triples_backfills_passages_to_max_evidence():
    ev, mix = select_evidence(
        query="q", passages=[_p(i) for i in range(15)], triples=[],
        split=(8, 4), max_evidence=12,
    )
    assert len(ev) == 12  # not 8 -- a zero-KG question must not be starved vs B2
    assert mix.n_passages == 12 and mix.n_triples == 0
    assert mix.kg_slot_filled is False
    assert mix.backfilled_passages == 4


def test_partial_triples_partial_backfill():
    ev, mix = select_evidence(
        query="q", passages=[_p(i) for i in range(12)], triples=[_t(0), _t(1)],
        split=(8, 4), max_evidence=12,
    )
    assert len(ev) == 12
    assert mix.n_triples == 2
    assert mix.n_passages == 10  # 8 + 2 freed
    assert mix.backfilled_passages == 2


def test_never_exceeds_max_evidence():
    ev, mix = select_evidence(
        query="q", passages=[_p(i) for i in range(50)], triples=[_t(i) for i in range(50)],
        split=(8, 4), max_evidence=12,
    )
    assert len(ev) == 12


def test_scarce_passages_no_crash():
    ev, mix = select_evidence(
        query="q", passages=[_p(0)], triples=[_t(0), _t(1)],
        split=(8, 4), max_evidence=12,
    )
    assert len(ev) == 3
    assert mix.n_passages == 1 and mix.n_triples == 2


def test_kg_slot_rank_fn_is_used():
    calls = {}

    def fake_rank(query, cands, top_n):
        calls["args"] = (query, len(cands), top_n)
        return list(reversed(cands))[:top_n]

    triples = [_t(i) for i in range(6)]
    ev, mix = select_evidence(
        query="Q", passages=[_p(i) for i in range(8)], triples=triples,
        split=(8, 4), max_evidence=12, kg_slot_rank_fn=fake_rank,
    )
    assert calls["args"] == ("Q", 6, 4)
    assert ev[8].id == "kg::h5|r|t5"  # reversed order, first of top-4
