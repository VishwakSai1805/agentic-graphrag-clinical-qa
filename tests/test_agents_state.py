"""Tests for src/agents/state.py -- source precedence + state<->matrix."""

from __future__ import annotations

from src.agents.state import (
    ConsensusState,
    MixReport,
    OptionSupport,
    SupportCell,
    source_key,
    source_rank,
)
from src.schema import Evidence

_PREC = ["kg", "textbook", "pubmed_abstract"]


def _ev(modality, source=None):
    return Evidence(
        id="x", modality=modality, text="t",
        source_meta={"source": source} if source else {}, scores={},
    )


def test_source_key():
    assert source_key(_ev("kg")) == "kg"
    assert source_key(_ev("dense", "textbook")) == "textbook"
    assert source_key(_ev("lexical", "pubmed_abstract")) == "pubmed_abstract"
    assert source_key(_ev("dense", "weird")) == "other"


def test_source_rank_orders_and_defaults_last():
    assert source_rank(_ev("kg"), _PREC) == 0
    assert source_rank(_ev("dense", "textbook"), _PREC) == 1
    assert source_rank(_ev("dense", "pubmed_abstract"), _PREC) == 2
    assert source_rank(_ev("dense", "weird"), _PREC) == 99


def test_to_matrix_copies_fields_and_maps_running_to_ok():
    st = ConsensusState(
        question_id="train_00001", dataset="medqa", query="q",
        options={"A": "a", "B": "b"}, gold_option="A",
    )
    st.cells = [SupportCell(option_id="A", claim_idx=0, evidence_id="e1", label="SUPPORT")]
    st.option_support = [OptionSupport(option_id="A", s_pos=1)]
    st.mix = MixReport(
        requested_split=(8, 4), n_passages=8, n_triples=4, kg_available=10,
        kg_slot_filled=True, backfilled_passages=0,
    )
    st.raw_invalid_citation_count = 3
    m = st.to_matrix()
    assert m.status == "ok"  # "running" -> "ok"
    assert m.question_id == "train_00001"
    assert m.raw_invalid_citation_count == 3
    assert len(m.cells) == 1
    assert m.mix.n_triples == 4


def test_to_matrix_preserves_terminal_status():
    st = ConsensusState(question_id="q", dataset="medqa", query="q", options={"A": "a"})
    st.status = "budget_exceeded"
    assert st.to_matrix().status == "budget_exceeded"
