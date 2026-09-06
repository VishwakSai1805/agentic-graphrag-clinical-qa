"""End-to-end tests for the consensus graph (src/agents/graph.py), LLM mocked.

A `ScriptedChat` routes by which prompt template the call carries, so one fake
serves Hypothesis, Audit and Conflict in one graph run.
"""

from __future__ import annotations

import json

from src.agents.graph import run_consensus
from src.config import AppConfig, LLMConfig, ProviderConfig
from src.schema import Evidence


class FakeResp:
    def __init__(self, text: str):
        self.text = text


class ScriptedChat:
    """Routes on prompt content. `audit_cells_for` maps option_id -> list of
    (claim_idx, evidence_id, label). `conflict_winner` is the verdict string."""

    def __init__(
        self, *, audit_cells_for=None, conflict_winner="SUPPORT", n_claims=2, bad_json_for=None
    ):
        self.audit_cells_for = audit_cells_for or {}
        self.conflict_winner = conflict_winner
        self.n_claims = n_claims
        self.bad_json_for = bad_json_for or set()
        self.calls = []

    def __call__(self, prompt, *, system=None, temperature=None, seed=None, **kw):
        self.calls.append(prompt)
        if "Hypothesis agent" in prompt or "decompose" in prompt:
            if "hypothesis" in self.bad_json_for:
                return FakeResp("not json")
            opts = _options_from_prompt(prompt)
            return FakeResp(json.dumps({"options": [
                {"option_id": o, "claims": [f"claim {i} for {o}" for i in range(self.n_claims)],
                 "required_evidence_types": []}
                for o in opts
            ]}))
        if "Evidential Audit" in prompt or "judge claims against" in prompt.lower():
            oid = _audit_option_from_prompt(prompt)
            cells = self.audit_cells_for.get(oid, [])
            return FakeResp(json.dumps({"cells": [
                {"claim_idx": ci, "evidence_id": eid, "label": lab, "rationale": "r"}
                for ci, eid, lab in cells
            ]}))
        if "Conflict Resolution" in prompt:
            return FakeResp(json.dumps({"winner": self.conflict_winner,
                                        "kept_evidence_ids": [], "rationale": "r"}))
        raise AssertionError(f"unexpected prompt:\n{prompt[:200]}")


def _options_from_prompt(prompt: str) -> list[str]:
    out = []
    for line in prompt.splitlines():
        s = line.strip()
        if len(s) >= 2 and s[0].isalpha() and s[1] == "." and s[0].isupper():
            out.append(s[0])
    return out or ["A", "B"]


def _audit_option_from_prompt(prompt: str) -> str:
    for line in prompt.splitlines():
        if line.startswith("Option "):
            return line.split()[1].rstrip(":")
    return "A"


def _cfg(**agent_overrides) -> AppConfig:
    cfg = AppConfig(
        llm=LLMConfig(fallback_chain=[ProviderConfig(provider="ollama", model="ollama/x")]),
    )
    cfg.agents.enabled = True
    cfg.agents.kg_slot_rerank = False  # don't load the 400MB cross-encoder in unit tests
    for k, v in agent_overrides.items():
        setattr(cfg.agents, k, v)
    return cfg


def _passages(n=8):
    return [Evidence(id=f"chunk::D{i}#0", modality="dense", text=f"p{i}",
                     source_meta={"source": "textbook"}, scores={}) for i in range(n)]


def _triples(n=4):
    return [Evidence(id=f"kg::h{i}|r|t{i}", modality="kg", text=f"t{i}", source_meta={}, scores={})
            for i in range(n)]


def _run(chat, cfg=None, passages=None, triples=None):
    return run_consensus(
        question_id="train_00001", dataset="medqa", query="Q?",
        options={"A": "aa", "B": "bb"},
        bundle_passages=passages if passages is not None else _passages(),
        bundle_triples=triples if triples is not None else _triples(),
        cfg=cfg or _cfg(), gold_option="A",
        chat_fn=chat, retrieve_fn=None,
    )


def test_happy_path_clean_matrix():
    chat = ScriptedChat(audit_cells_for={
        "A": [(0, "chunk::D0#0", "SUPPORT")],
        "B": [(0, "chunk::D1#0", "CONTRADICT")],
    })
    matrix, state = _run(chat)
    assert matrix.status == "ok"
    assert matrix.raw_invalid_citation_count == 0
    assert {c.evidence_id for c in matrix.cells} <= {e.id for e in state.evidence}
    assert all(c.rationale for c in matrix.cells if c.label != "NEUTRAL")
    assert matrix.llm_calls > 0
    assert matrix.mix.n_triples > 0


def test_invalid_citation_is_dropped_and_counted():
    chat = ScriptedChat(audit_cells_for={
        "A": [(0, "chunk::D0#0", "SUPPORT"), (1, "kg::HALLUCINATED|x|y", "SUPPORT")],
        "B": [],
    })
    matrix, _ = _run(chat)
    assert matrix.raw_invalid_citation_count == 1
    assert all(c.evidence_id != "kg::HALLUCINATED|x|y" for c in matrix.cells)


def test_all_neutral_is_insufficient_evidence():
    chat = ScriptedChat(audit_cells_for={"A": [], "B": []})
    matrix, _ = _run(chat)
    assert matrix.status == "insufficient_evidence"
    assert matrix.cells == []


def test_conflict_unresolved_sets_status():
    cells_a = [(0, "kg::h0|r|t0", "SUPPORT"), (1, "chunk::D0#0", "CONTRADICT")]
    chat = ScriptedChat(
        audit_cells_for={"A": cells_a, "B": []},
        conflict_winner="UNRESOLVED",
    )
    matrix, _ = _run(chat)
    assert matrix.conflict_fire_count == 1
    assert matrix.status == "unresolved_conflict"
    assert any(e.outcome == "unresolved" for e in matrix.conflict_log)


def test_conflict_resolved_is_ok():
    cells_a = [(0, "kg::h0|r|t0", "SUPPORT"), (1, "chunk::D0#0", "CONTRADICT")]
    chat = ScriptedChat(
        audit_cells_for={"A": cells_a, "B": []},
        conflict_winner="SUPPORT",
    )
    matrix, _ = _run(chat)
    assert matrix.conflict_fire_count == 1
    assert matrix.status == "ok"


def test_budget_trip_on_low_call_ceiling():
    cfg = _cfg(max_llm_calls_per_question=2)  # hypothesis(1) + audit(2 options) -> would exceed
    chat = ScriptedChat(audit_cells_for={"A": [(0, "chunk::D0#0", "SUPPORT")], "B": []})
    matrix, _ = _run(chat, cfg=cfg)
    assert matrix.status == "budget_exceeded"


def test_hypothesis_parse_failure_is_agent_error():
    chat = ScriptedChat(bad_json_for={"hypothesis"})
    matrix, _ = _run(chat)
    assert matrix.status == "agent_error"


def test_zero_triples_still_runs():
    chat = ScriptedChat(audit_cells_for={"A": [(0, "chunk::D0#0", "SUPPORT")], "B": []})
    matrix, state = _run(chat, triples=[])
    assert matrix.kg_hit is False
    assert matrix.mix.n_triples == 0
    assert matrix.mix.n_passages == len(state.evidence)
    assert matrix.status in ("ok", "insufficient_evidence")


def test_re_retrieval_deepens_kg_then_still_insufficient():
    # 10 triples available, only 4 in slot round 1; still no signal -> round 2
    chat = ScriptedChat(audit_cells_for={"A": [], "B": []})
    matrix, state = _run(chat, cfg=_cfg(max_rounds=2), triples=_triples(10))
    assert state.round == 2
    assert matrix.status == "insufficient_evidence"
    assert matrix.rounds == 2
