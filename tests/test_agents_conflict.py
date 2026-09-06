"""Conflict agent -- claim-level, semantic-check-first (ADR-025). LLM mocked."""

from __future__ import annotations

import json

import pytest

from src.agents.conflict import run_conflict
from src.agents.schemas import AgentParseError
from src.agents.state import SupportCell
from src.config import AgentsConfig
from src.schema import Evidence


class FakeResp:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeChat:
    """Returns queued responses in order; records every call's kwargs."""

    def __init__(self, *responses: str) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, prompt, *, system=None, temperature=None, seed=None, **kw):
        self.calls.append(
            {"prompt": prompt, "system": system, "temperature": temperature, "seed": seed}
        )
        return FakeResp(self._responses.pop(0))


def _cfg() -> AgentsConfig:
    return AgentsConfig()


def _cell(evidence_id: str, label: str, rank: int, *, option_id="A", claim_idx=2) -> SupportCell:
    return SupportCell(
        option_id=option_id,
        claim_idx=claim_idx,
        evidence_id=evidence_id,
        label=label,
        rationale="r",
        source_rank=rank,
    )


def _ev(evidence_id: str, text: str = "evidence text") -> Evidence:
    return Evidence(id=evidence_id, modality="kg", text=text, source_meta={}, scores={})


KG_ID = "kg::x|rel|y"
CHUNK_ID = "chunk::doc1#0-9"
EBI = {KG_ID: _ev(KG_ID, "kg triple text"), CHUNK_ID: _ev(CHUNK_ID, "abstract text")}


def _resp(genuine: bool, rationale: str = "same clinical point") -> str:
    return json.dumps({"genuine_contradiction": genuine, "rationale": rationale})


def test_genuine_support_wins_on_rank():
    sup = [_cell(KG_ID, "SUPPORT", 0)]
    con = [_cell(CHUNK_ID, "CONTRADICT", 1)]
    fake = FakeChat(_resp(True))
    entry, n = run_conflict("A", 2, "claim text", sup, con, EBI, cfg=_cfg(), chat_fn=fake)

    assert n == 1
    assert entry.option_id == "A"
    assert entry.claim_idx == 2
    assert entry.outcome == "resolved_support"
    assert entry.kept == [KG_ID]
    assert entry.dropped == [CHUNK_ID]
    assert entry.rationale.startswith("same clinical point")
    assert entry.rationale.endswith("precedence 0 vs 1")


def test_genuine_contradict_wins_on_rank():
    sup = [_cell(CHUNK_ID, "SUPPORT", 1)]
    con = [_cell(KG_ID, "CONTRADICT", 0)]
    fake = FakeChat(_resp(True))
    entry, n = run_conflict("B", 0, "claim", sup, con, EBI, cfg=_cfg(), chat_fn=fake)

    assert n == 1
    assert entry.outcome == "resolved_contradict"
    assert entry.kept == [KG_ID]
    assert entry.dropped == [CHUNK_ID]
    assert entry.rationale.endswith("precedence 1 vs 0")


def test_genuine_tied_ranks_unresolved():
    sup = [_cell(KG_ID, "SUPPORT", 1)]
    con = [_cell(CHUNK_ID, "CONTRADICT", 1)]
    fake = FakeChat(_resp(True))
    entry, n = run_conflict("A", 3, "claim", sup, con, EBI, cfg=_cfg(), chat_fn=fake)

    assert n == 1
    assert entry.outcome == "unresolved"
    assert entry.kept == []
    assert entry.dropped == [KG_ID, CHUNK_ID]
    assert entry.rationale.endswith("precedence 1 vs 1")


def test_spurious():
    sup = [_cell(KG_ID, "SUPPORT", 0)]
    con = [_cell(CHUNK_ID, "CONTRADICT", 1)]
    fake = FakeChat(_resp(False, "different populations"))
    entry, n = run_conflict("A", 2, "claim", sup, con, EBI, cfg=_cfg(), chat_fn=fake)

    assert n == 1
    assert entry.outcome == "spurious"
    assert entry.kept == []
    assert entry.dropped == []
    assert entry.rationale == "spurious: different populations"
    assert entry.claim_idx == 2


def test_malformed_json_then_repair():
    sup = [_cell(KG_ID, "SUPPORT", 0)]
    con = [_cell(CHUNK_ID, "CONTRADICT", 1)]
    fake = FakeChat("not json at all", _resp(True, "fixed"))
    entry, n = run_conflict("A", 1, "claim", sup, con, EBI, cfg=_cfg(), chat_fn=fake)

    assert n == 2
    assert entry.outcome == "resolved_support"
    assert entry.rationale.startswith("fixed")


def test_repair_also_fails_raises():
    sup = [_cell(KG_ID, "SUPPORT", 0)]
    con = [_cell(CHUNK_ID, "CONTRADICT", 1)]
    fake = FakeChat("garbage", "still garbage")
    with pytest.raises(AgentParseError):
        run_conflict("A", 1, "claim", sup, con, EBI, cfg=_cfg(), chat_fn=fake)


def test_blocks_render_and_dedupe():
    sup = [
        _cell(KG_ID, "SUPPORT", 0, claim_idx=2),
        _cell(KG_ID, "SUPPORT", 0, claim_idx=2),  # same evidence id -> one line
    ]
    con = [_cell(CHUNK_ID, "CONTRADICT", 1)]
    fake = FakeChat(_resp(True))
    run_conflict("A", 2, "claim", sup, con, EBI, cfg=_cfg(), chat_fn=fake)

    prompt = fake.calls[0]["prompt"]
    assert prompt.count(f"[{KG_ID}] (rank 0) kg triple text") == 1
    assert f"[{CHUNK_ID}] (rank 1) abstract text" in prompt


def test_blocks_dedupe_reflected_in_kept():
    sup = [
        _cell(KG_ID, "SUPPORT", 0, claim_idx=2),
        _cell(KG_ID, "SUPPORT", 0, claim_idx=2),
    ]
    con = [_cell(CHUNK_ID, "CONTRADICT", 1)]
    fake = FakeChat(_resp(True))
    entry, _ = run_conflict("A", 2, "claim", sup, con, EBI, cfg=_cfg(), chat_fn=fake)

    assert entry.kept == [KG_ID]
    assert entry.dropped == [CHUNK_ID]


def test_spurious_rationale_truncation():
    sup = [_cell(KG_ID, "SUPPORT", 0)]
    con = [_cell(CHUNK_ID, "CONTRADICT", 1)]
    fake = FakeChat(_resp(False, "z" * 900))
    entry, _ = run_conflict("A", 2, "claim", sup, con, EBI, cfg=_cfg(), chat_fn=fake)

    assert entry.rationale == "spurious: " + "z" * 380


def test_genuine_rationale_truncation():
    sup = [_cell(KG_ID, "SUPPORT", 0)]
    con = [_cell(CHUNK_ID, "CONTRADICT", 1)]
    fake = FakeChat(_resp(True, "w" * 900))
    entry, _ = run_conflict("A", 2, "claim", sup, con, EBI, cfg=_cfg(), chat_fn=fake)

    assert entry.rationale == "w" * 340 + " | precedence 0 vs 1"


def test_prompt_carries_claim_and_call_kwargs():
    sup = [_cell(KG_ID, "SUPPORT", 0)]
    con = [_cell(CHUNK_ID, "CONTRADICT", 1)]
    fake = FakeChat(_resp(True))
    run_conflict(
        "A", 2, "diabetes worsens outcome", sup, con, EBI, cfg=_cfg(), chat_fn=fake, seed=7
    )

    call = fake.calls[0]
    assert call["temperature"] == 0
    assert call["seed"] == 7
    assert call["system"] == (
        "You judge whether two pieces of clinical evidence genuinely contradict. "
        "Respond with STRICT JSON only."
    )
    assert "diabetes worsens outcome" in call["prompt"]
    assert KG_ID in call["prompt"] and CHUNK_ID in call["prompt"]
