"""Tests for src/agents/schemas.py -- first_json_obj + parse_or_repair."""

from __future__ import annotations

import json

import pytest

from src.agents.schemas import (
    AgentParseError,
    HypothesisOut,
    first_json_obj,
    parse_or_repair,
)


class FakeResp:
    def __init__(self, text):
        self.text = text


def test_first_json_obj_plain():
    assert first_json_obj('{"a": 1}') == {"a": 1}


def test_first_json_obj_with_prose_and_fence():
    txt = "Sure!\n```json\n{\"a\": [1, 2], \"b\": {\"c\": 3}}\n```\ndone"
    assert first_json_obj(txt) == {"a": [1, 2], "b": {"c": 3}}


def test_first_json_obj_ignores_braces_in_strings():
    assert first_json_obj('{"a": "has } brace"}') == {"a": "has } brace"}


def test_first_json_obj_none_when_absent():
    assert first_json_obj("no json here") is None


_GOOD = json.dumps(
    {"options": [{"option_id": "A", "claims": ["c1"], "required_evidence_types": []}]}
)


def test_parse_or_repair_happy_no_repair():
    calls = []

    def chat_fn(prompt, **kw):
        calls.append(prompt)
        return FakeResp("unused")

    obj, n = parse_or_repair(
        _GOOD, HypothesisOut, chat_fn=chat_fn, system="s", original_prompt="p", seed=42
    )
    assert n == 0
    assert calls == []
    assert obj.options[0].option_id == "A"


def test_parse_or_repair_one_repair_succeeds():
    def chat_fn(prompt, **kw):
        assert "could not be parsed" in prompt
        return FakeResp(_GOOD)

    obj, n = parse_or_repair(
        "garbage", HypothesisOut, chat_fn=chat_fn, system="s", original_prompt="p", seed=42
    )
    assert n == 1
    assert obj.options[0].claims == ["c1"]


def test_parse_or_repair_double_failure_raises():
    def chat_fn(prompt, **kw):
        return FakeResp("still garbage")

    with pytest.raises(AgentParseError):
        parse_or_repair(
            "garbage", HypothesisOut, chat_fn=chat_fn, system="s", original_prompt="p", seed=42
        )


def test_parse_or_repair_schema_violation_then_fix():
    bad_schema = json.dumps({"options": [{"option_id": "A"}]})  # missing 'claims'

    def chat_fn(prompt, **kw):
        return FakeResp(_GOOD)

    obj, n = parse_or_repair(
        bad_schema, HypothesisOut, chat_fn=chat_fn, system="s", original_prompt="p", seed=42
    )
    assert n == 1
