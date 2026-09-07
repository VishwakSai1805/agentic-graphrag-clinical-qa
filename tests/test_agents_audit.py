"""Unit tests for the Evidential Audit agent. The LLM gateway is always mocked
via `monkeypatch` on `src.agents.audit.chat` -- no provider is ever hit."""

import json

import pytest

from src.agents import audit as audit_mod
from src.agents.audit import AuditResult, _format_claims, _format_evidence, run_audit
from src.agents.schemas import AgentParseError
from src.config import AgentsConfig
from src.schema import Evidence

CFG = AgentsConfig()  # default source_precedence = ["kg", "textbook", "pubmed_abstract"]


class FakeResp:
    def __init__(self, text: str) -> None:
        self.text = text


class ScriptedChat:
    """Returns canned responses in order; records prompts + kwargs it saw."""

    def __init__(self, *responses: str) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []
        self.kwargs: list[dict] = []

    def __call__(self, prompt, **kwargs):
        self.prompts.append(prompt)
        self.kwargs.append(kwargs)
        if not self._responses:
            raise AssertionError("ScriptedChat ran out of responses")
        return FakeResp(self._responses.pop(0))


@pytest.fixture
def evidence() -> list[Evidence]:
    return [
        Evidence(
            id="kg::asthma|treats|albuterol",
            modality="kg",
            text="Albuterol is indicated for asthma.",
            source_meta={},
            scores={},
        ),
        Evidence(
            id="chunk::tb001#5",
            modality="dense",
            text="First-line therapy for acute asthma exacerbation is a SABA.",
            source_meta={"source": "textbook"},
            scores={},
        ),
        Evidence(
            id="chunk::pma042#0",
            modality="lexical",
            text="A trial found no benefit of antibiotics in asthma exacerbation.",
            source_meta={"source": "pubmed_abstract"},
            scores={},
        ),
    ]


def _cells(*cells: dict) -> str:
    return json.dumps({"cells": list(cells)})


def _patch(monkeypatch, chat_obj) -> None:
    monkeypatch.setattr(audit_mod, "chat", chat_obj)


# --- formatting helpers -------------------------------------------------


def test_format_claims_numbers_lines():
    assert _format_claims(["a", "b"]) == "0: a\n1: b"


def test_format_evidence_has_bracket_id_no_leading_number(evidence):
    out = _format_evidence(evidence)
    assert out.startswith("[kg::asthma|treats|albuterol] Albuterol")
    assert "\n[chunk::tb001#5] First-line" in out


# --- behavior ---------------------------------------------------------


def test_happy_path_two_valid_cells(monkeypatch, evidence):
    chat_obj = ScriptedChat(
        _cells(
            {
                "claim_idx": 0,
                "evidence_id": "chunk::tb001#5",
                "label": "SUPPORT",
                "rationale": "SABA is first-line.",
            },
            {
                "claim_idx": 1,
                "evidence_id": "chunk::pma042#0",
                "label": "CONTRADICT",
                "rationale": "Trial found no antibiotic benefit.",
            },
        )
    )
    _patch(monkeypatch, chat_obj)

    res = run_audit("A", "Albuterol", ["claim zero", "claim one"], evidence, cfg=CFG)

    assert isinstance(res, AuditResult)
    assert len(res.cells) == 2
    assert res.n_llm_calls == 1
    assert res.n_invalid_dropped == 0
    assert res.cells[0].label == "SUPPORT"
    assert res.cells[0].source_rank == 1  # textbook
    assert res.cells[1].label == "CONTRADICT"
    assert res.cells[1].source_rank == 2  # pubmed_abstract
    assert all(c.option_id == "A" for c in res.cells)


def test_temperature_zero_and_seed_passed(monkeypatch, evidence):
    chat_obj = ScriptedChat(_cells())
    _patch(monkeypatch, chat_obj)

    run_audit("A", "opt", ["c0"], evidence, cfg=CFG, seed=99)

    assert chat_obj.kwargs[0]["temperature"] == 0
    assert chat_obj.kwargs[0]["seed"] == 99
    assert chat_obj.kwargs[0]["system"] == audit_mod._SYSTEM


def test_kg_source_rank(monkeypatch, evidence):
    chat_obj = ScriptedChat(
        _cells(
            {
                "claim_idx": 0,
                "evidence_id": "kg::asthma|treats|albuterol",
                "label": "SUPPORT",
                "rationale": "KG indication edge.",
            }
        )
    )
    _patch(monkeypatch, chat_obj)

    res = run_audit("A", "opt", ["c0"], evidence, cfg=CFG)

    assert len(res.cells) == 1
    assert res.cells[0].source_rank == 0  # kg has highest precedence


def test_cell_records_modality_and_rerank_score_from_the_cited_evidence(monkeypatch):
    """Phase 5 (src/calibration/scoring.py) reads modality/rerank_score
    straight off SupportCell -- this pins that run_audit populates them from
    the actual cited Evidence object, not a default."""
    scored_evidence = [
        Evidence(
            id="kg::asthma|treats|albuterol",
            modality="kg",
            text="Albuterol is indicated for asthma.",
            source_meta={},
            scores={"rerank": 0.87},
        ),
    ]
    chat_obj = ScriptedChat(
        _cells(
            {
                "claim_idx": 0,
                "evidence_id": "kg::asthma|treats|albuterol",
                "label": "SUPPORT",
                "rationale": "KG indication edge.",
            }
        )
    )
    _patch(monkeypatch, chat_obj)

    res = run_audit("A", "opt", ["c0"], scored_evidence, cfg=CFG)

    assert len(res.cells) == 1
    assert res.cells[0].modality == "kg"
    assert res.cells[0].rerank_score == pytest.approx(0.87)


def test_cell_rerank_score_defaults_to_zero_when_absent(monkeypatch, evidence):
    """`evidence` fixture's items carry empty `scores` dicts."""
    chat_obj = ScriptedChat(
        _cells(
            {
                "claim_idx": 0,
                "evidence_id": "chunk::tb001#5",
                "label": "SUPPORT",
                "rationale": "Textbook support.",
            }
        )
    )
    _patch(monkeypatch, chat_obj)

    res = run_audit("A", "opt", ["c0"], evidence, cfg=CFG)

    assert res.cells[0].modality == "dense"
    assert res.cells[0].rerank_score == 0.0


def test_bracketed_id_is_normalized_and_kept(monkeypatch, evidence):
    chat_obj = ScriptedChat(
        _cells(
            {
                "claim_idx": 0,
                "evidence_id": "[chunk::tb001#5]",
                "label": "SUPPORT",
                "rationale": "ok",
            }
        )
    )
    _patch(monkeypatch, chat_obj)

    res = run_audit("A", "opt", ["c0"], evidence, cfg=CFG)

    assert len(res.cells) == 1
    assert res.cells[0].evidence_id == "chunk::tb001#5"
    assert res.n_invalid_dropped == 0


def test_hallucinated_id_dropped_and_counted(monkeypatch, evidence):
    chat_obj = ScriptedChat(
        _cells(
            {
                "claim_idx": 0,
                "evidence_id": "chunk::does_not_exist#9",
                "label": "SUPPORT",
                "rationale": "made up",
            },
            {
                "claim_idx": 0,
                "evidence_id": "chunk::tb001#5",
                "label": "SUPPORT",
                "rationale": "real",
            },
        )
    )
    _patch(monkeypatch, chat_obj)

    res = run_audit("A", "opt", ["c0"], evidence, cfg=CFG)

    assert len(res.cells) == 1
    assert res.cells[0].evidence_id == "chunk::tb001#5"
    assert res.n_invalid_dropped == 1


def test_neutral_cell_dropped(monkeypatch, evidence):
    chat_obj = ScriptedChat(
        _cells(
            {
                "claim_idx": 0,
                "evidence_id": "chunk::tb001#5",
                "label": "NEUTRAL",
                "rationale": "merely topical",
            }
        )
    )
    _patch(monkeypatch, chat_obj)

    res = run_audit("A", "opt", ["c0"], evidence, cfg=CFG)

    assert res.cells == []
    assert res.n_invalid_dropped == 0
    assert res.n_llm_calls == 1


def test_claim_idx_out_of_range_dropped(monkeypatch, evidence):
    chat_obj = ScriptedChat(
        _cells(
            {
                "claim_idx": 7,
                "evidence_id": "chunk::tb001#5",
                "label": "SUPPORT",
                "rationale": "wrong index",
            }
        )
    )
    _patch(monkeypatch, chat_obj)

    res = run_audit("A", "opt", ["c0"], evidence, cfg=CFG)

    assert res.cells == []
    assert res.n_invalid_dropped == 0


def test_empty_cells(monkeypatch, evidence):
    chat_obj = ScriptedChat('{"cells": []}')
    _patch(monkeypatch, chat_obj)

    res = run_audit("A", "opt", ["c0"], evidence, cfg=CFG)

    assert res.cells == []
    assert res.n_llm_calls == 1
    assert res.n_invalid_dropped == 0


def test_malformed_json_then_repair(monkeypatch, evidence):
    chat_obj = ScriptedChat(
        "not json at all, sorry",
        _cells(
            {
                "claim_idx": 0,
                "evidence_id": "chunk::tb001#5",
                "label": "SUPPORT",
                "rationale": "recovered",
            }
        ),
    )
    _patch(monkeypatch, chat_obj)

    res = run_audit("A", "opt", ["c0"], evidence, cfg=CFG)

    assert res.n_llm_calls == 2
    assert len(res.cells) == 1


def test_parse_error_propagates_after_failed_repair(monkeypatch, evidence):
    chat_obj = ScriptedChat("garbage", "still garbage")
    _patch(monkeypatch, chat_obj)

    with pytest.raises(AgentParseError):
        run_audit("A", "opt", ["c0"], evidence, cfg=CFG)


def test_rationale_truncated_to_400(monkeypatch, evidence):
    long_rationale = "x" * 900
    chat_obj = ScriptedChat(
        _cells(
            {
                "claim_idx": 0,
                "evidence_id": "chunk::tb001#5",
                "label": "SUPPORT",
                "rationale": long_rationale,
            }
        )
    )
    _patch(monkeypatch, chat_obj)

    res = run_audit("A", "opt", ["c0"], evidence, cfg=CFG)

    assert len(res.cells[0].rationale) == 400


def test_default_chat_fn_is_module_chat(monkeypatch, evidence):
    """`chat_fn` defaults to the module-level `chat`, which tests monkeypatch."""
    calls = []

    def fake_chat(prompt, **kwargs):
        calls.append(prompt)
        return FakeResp('{"cells": []}')

    monkeypatch.setattr(audit_mod, "chat", fake_chat)
    run_audit("A", "opt", ["c0"], evidence, cfg=CFG)
    assert len(calls) == 1
