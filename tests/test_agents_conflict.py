"""Conflict agent (source-precedence resolver, ADR-022) -- LLM mocked."""

from __future__ import annotations

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
    """Returns queued responses in order; records every call."""

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


def _cell(evidence_id: str, label: str, rank: int, *, option_id="A", claim_idx=0) -> SupportCell:
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


def _ok(winner: str, kept=None, rationale="genuine conflict") -> str:
    kept = kept if kept is not None else []
    import json

    return json.dumps({"winner": winner, "kept_evidence_ids": kept, "rationale": rationale})


def test_support_wins_on_rank_model_confirms():
    sup = [_cell(KG_ID, "SUPPORT", 0)]
    con = [_cell(CHUNK_ID, "CONTRADICT", 1)]
    fake = FakeChat(_ok("SUPPORT", [KG_ID]))
    entry, n = run_conflict("A", sup, con, EBI, cfg=_cfg(), chat_fn=fake)

    assert n == 1
    assert entry.option_id == "A"
    assert entry.outcome == "resolved_support"
    assert entry.kept == [KG_ID]
    assert entry.dropped == [CHUNK_ID]
    assert entry.rationale == "genuine conflict"


def test_model_returns_unresolved():
    sup = [_cell(KG_ID, "SUPPORT", 0)]
    con = [_cell(CHUNK_ID, "CONTRADICT", 1)]
    fake = FakeChat(_ok("UNRESOLVED", rationale="different aspects"))
    entry, n = run_conflict("A", sup, con, EBI, cfg=_cfg(), chat_fn=fake)

    assert n == 1
    assert entry.outcome == "unresolved"
    assert entry.kept == []
    assert set(entry.dropped) == {KG_ID, CHUNK_ID}
    assert entry.rationale == "different aspects"


def test_model_returns_losing_side_falls_back():
    sup = [_cell(KG_ID, "SUPPORT", 0)]
    con = [_cell(CHUNK_ID, "CONTRADICT", 1)]
    fake = FakeChat(_ok("CONTRADICT", [CHUNK_ID], rationale="model disagrees"))
    entry, n = run_conflict("A", sup, con, EBI, cfg=_cfg(), chat_fn=fake)

    assert n == 1
    assert entry.outcome == "resolved_support"
    assert entry.kept == [KG_ID]
    assert entry.dropped == [CHUNK_ID]
    assert entry.rationale.startswith("model disagrees")
    assert "fell back to source precedence" in entry.rationale


def test_tied_ranks_skip_llm():
    sup = [_cell(KG_ID, "SUPPORT", 1)]
    con = [_cell(CHUNK_ID, "CONTRADICT", 1)]
    fake = FakeChat()  # nothing queued -- must not be called
    entry, n = run_conflict("A", sup, con, EBI, cfg=_cfg(), chat_fn=fake)

    assert n == 0
    assert fake.calls == []
    assert entry.outcome == "unresolved"
    assert entry.kept == []
    assert set(entry.dropped) == {KG_ID, CHUNK_ID}
    assert "tied source precedence" in entry.rationale


def test_contradict_wins_on_rank():
    sup = [_cell(CHUNK_ID, "SUPPORT", 2)]
    con = [_cell(KG_ID, "CONTRADICT", 0)]
    fake = FakeChat(_ok("CONTRADICT", [KG_ID]))
    entry, n = run_conflict("A", sup, con, EBI, cfg=_cfg(), chat_fn=fake)

    assert n == 1
    assert entry.outcome == "resolved_contradict"
    assert entry.kept == [KG_ID]
    assert entry.dropped == [CHUNK_ID]


def test_malformed_json_then_repair():
    sup = [_cell(KG_ID, "SUPPORT", 0)]
    con = [_cell(CHUNK_ID, "CONTRADICT", 1)]
    fake = FakeChat("not json at all", _ok("SUPPORT", [KG_ID], rationale="fixed"))
    entry, n = run_conflict("A", sup, con, EBI, cfg=_cfg(), chat_fn=fake)

    assert n == 2
    assert entry.outcome == "resolved_support"
    assert entry.rationale == "fixed"


def test_parse_error_propagates():
    sup = [_cell(KG_ID, "SUPPORT", 0)]
    con = [_cell(CHUNK_ID, "CONTRADICT", 1)]
    fake = FakeChat("garbage", "still garbage")
    with pytest.raises(AgentParseError):
        run_conflict("A", sup, con, EBI, cfg=_cfg(), chat_fn=fake)


def test_rationale_truncated_to_400():
    sup = [_cell(KG_ID, "SUPPORT", 0)]
    con = [_cell(CHUNK_ID, "CONTRADICT", 1)]
    long = "x" * 900
    fake = FakeChat(_ok("SUPPORT", [KG_ID], rationale=long))
    entry, _ = run_conflict("A", sup, con, EBI, cfg=_cfg(), chat_fn=fake)

    assert entry.rationale == "x" * 400


def test_rationale_truncated_then_note_appended_on_fallback():
    sup = [_cell(KG_ID, "SUPPORT", 0)]
    con = [_cell(CHUNK_ID, "CONTRADICT", 1)]
    long = "y" * 900
    fake = FakeChat(_ok("CONTRADICT", [CHUNK_ID], rationale=long))
    entry, _ = run_conflict("A", sup, con, EBI, cfg=_cfg(), chat_fn=fake)

    assert entry.rationale.startswith("y" * 400)
    assert entry.rationale.endswith("fell back to source precedence)")


def test_evidence_ids_deduped_across_cells():
    sup = [
        _cell(KG_ID, "SUPPORT", 0, claim_idx=0),
        _cell(KG_ID, "SUPPORT", 0, claim_idx=1),
    ]
    con = [_cell(CHUNK_ID, "CONTRADICT", 1)]
    fake = FakeChat(_ok("SUPPORT", [KG_ID]))
    entry, _ = run_conflict("A", sup, con, EBI, cfg=_cfg(), chat_fn=fake)

    assert entry.kept == [KG_ID]
    assert entry.dropped == [CHUNK_ID]


def test_prompt_is_temperature_zero_and_carries_blocks():
    sup = [_cell(KG_ID, "SUPPORT", 0)]
    con = [_cell(CHUNK_ID, "CONTRADICT", 1)]
    fake = FakeChat(_ok("SUPPORT", [KG_ID]))
    run_conflict("A", sup, con, EBI, cfg=_cfg(), chat_fn=fake)

    call = fake.calls[0]
    assert call["temperature"] == 0
    assert call["system"] == (
        "You resolve evidence conflicts by source precedence. Respond with STRICT JSON only."
    )
    assert KG_ID in call["prompt"]
    assert CHUNK_ID in call["prompt"]
    assert "kg>textbook>pubmed_abstract" in call["prompt"]
    assert "kg triple text" in call["prompt"]
