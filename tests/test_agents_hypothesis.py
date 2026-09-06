"""Unit tests for the Hypothesis agent. The LLM gateway is always mocked by
monkeypatching `src.agents.hypothesis.chat` (or passing `chat_fn=`) -- no
provider is ever hit."""

import json

import pytest

from src.agents.hypothesis import _format_options, run_hypothesis
from src.agents.schemas import AgentParseError
from src.config import AgentsConfig

QUERY = "A 55-year-old with new atrial fibrillation. Best next step?"
OPTIONS = {
    "A": "Start warfarin",
    "B": "Start metoprolol",
    "C": "Immediate cardioversion",
    "D": "Reassurance only",
}


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


def _payload(options: dict[str, list[str]]) -> str:
    return json.dumps(
        {
            "options": [
                {"option_id": oid, "claims": claims, "required_evidence_types": []}
                for oid, claims in options.items()
            ]
        }
    )


def _full_payload() -> str:
    return _payload(
        {
            "A": ["Warfarin is anticoagulation", "AF raises stroke risk"],
            "B": ["Metoprolol controls rate", "Rate control is first-line"],
            "C": ["Cardioversion needs anticoagulation first", "AF <48h lowers risk"],
            "D": ["Untreated AF carries stroke risk", "Reassurance ignores risk"],
        }
    )


def _cfg(**overrides) -> AgentsConfig:
    cfg = AgentsConfig()
    return cfg.model_copy(update=overrides)


def test_format_options_sorted_lines():
    rendered = _format_options({"B": "beta", "A": "alpha"})
    assert rendered == "A. alpha\nB. beta"


def test_happy_path_four_options_one_call():
    chat = ScriptedChat(_full_payload())
    claims, n_calls = run_hypothesis(QUERY, OPTIONS, cfg=_cfg(), chat_fn=chat)

    assert n_calls == 1
    assert [c.option_id for c in claims] == ["A", "B", "C", "D"]
    for c in claims:
        assert 2 <= len(c.claims) <= 4


def test_prompt_contains_question_and_options():
    chat = ScriptedChat(_full_payload())
    run_hypothesis(QUERY, OPTIONS, cfg=_cfg(), chat_fn=chat)
    prompt = chat.prompts[0]
    assert QUERY in prompt
    assert "A. Start warfarin" in prompt
    assert "D. Reassurance only" in prompt


def test_malformed_then_valid_repair_two_calls():
    chat = ScriptedChat("not json at all", _full_payload())
    claims, n_calls = run_hypothesis(QUERY, OPTIONS, cfg=_cfg(), chat_fn=chat)

    assert n_calls == 2
    assert len(claims) == 4


def test_malformed_twice_raises_parse_error():
    chat = ScriptedChat("garbage", "still garbage")
    with pytest.raises(AgentParseError):
        run_hypothesis(QUERY, OPTIONS, cfg=_cfg(), chat_fn=chat)


def test_missing_option_gets_synthesized_fallback():
    chat = ScriptedChat(
        _payload(
            {
                "A": ["Warfarin is anticoagulation", "AF raises stroke risk"],
                "B": ["Metoprolol controls rate", "Rate control is first-line"],
                "C": ["Cardioversion needs anticoagulation", "AF <48h lowers risk"],
            }
        )
    )
    claims, _ = run_hypothesis(QUERY, OPTIONS, cfg=_cfg(), chat_fn=chat)

    assert [c.option_id for c in claims] == ["A", "B", "C", "D"]
    d = next(c for c in claims if c.option_id == "D")
    assert len(d.claims) == 1
    assert "Reassurance only" in d.claims[0]


def test_seven_claims_clamped_to_four():
    chat = ScriptedChat(
        _payload(
            {
                "A": [f"claim {i}" for i in range(7)],
                "B": ["b1", "b2"],
                "C": ["c1", "c2"],
                "D": ["d1", "d2"],
            }
        )
    )
    claims, _ = run_hypothesis(QUERY, OPTIONS, cfg=_cfg(), chat_fn=chat)
    a = next(c for c in claims if c.option_id == "A")
    assert a.claims == ["claim 0", "claim 1", "claim 2", "claim 3"]


def test_unknown_option_id_dropped():
    chat = ScriptedChat(
        _payload(
            {
                "A": ["a1", "a2"],
                "B": ["b1", "b2"],
                "C": ["c1", "c2"],
                "D": ["d1", "d2"],
                "Z": ["should be dropped", "really"],
            }
        )
    )
    claims, _ = run_hypothesis(QUERY, OPTIONS, cfg=_cfg(), chat_fn=chat)
    assert [c.option_id for c in claims] == ["A", "B", "C", "D"]


def test_empty_string_claims_dropped_then_fallback_if_all_empty():
    chat = ScriptedChat(
        _payload(
            {
                "A": ["", "  ", "real claim"],
                "B": ["", ""],
                "C": ["c1", "c2"],
                "D": ["d1", "d2"],
            }
        )
    )
    claims, _ = run_hypothesis(QUERY, OPTIONS, cfg=_cfg(), chat_fn=chat)
    a = next(c for c in claims if c.option_id == "A")
    assert a.claims == ["real claim"]
    b = next(c for c in claims if c.option_id == "B")
    assert len(b.claims) == 1
    assert "Start metoprolol" in b.claims[0]


def test_temperature_zero_when_sampling_disabled():
    chat = ScriptedChat(_full_payload())
    run_hypothesis(QUERY, OPTIONS, cfg=_cfg(hypothesis_sample=False), chat_fn=chat)
    assert chat.kwargs[0]["temperature"] == 0.0
    assert chat.kwargs[0]["seed"] == 42
    assert chat.kwargs[0]["system"].startswith("You decompose clinical answer options")


def test_temperature_used_when_sampling_enabled():
    chat = ScriptedChat(_full_payload())
    run_hypothesis(
        QUERY,
        OPTIONS,
        cfg=_cfg(hypothesis_sample=True, hypothesis_temperature=0.7),
        chat_fn=chat,
        seed=123,
    )
    assert chat.kwargs[0]["temperature"] == 0.7
    assert chat.kwargs[0]["seed"] == 123


def test_monkeypatched_module_chat_is_used(monkeypatch):
    chat = ScriptedChat(_full_payload())
    monkeypatch.setattr("src.agents.hypothesis.chat", chat)
    claims, n_calls = run_hypothesis(QUERY, OPTIONS, cfg=_cfg())
    assert n_calls == 1
    assert len(claims) == 4
