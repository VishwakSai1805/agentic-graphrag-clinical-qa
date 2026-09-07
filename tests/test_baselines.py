"""Unit tests for the Phase 3 baselines. The LLM gateway is always mocked via
`monkeypatch` on `src.eval.baselines.chat` -- no provider is ever hit."""

import pytest

from src.eval.baselines import (
    BASELINES,
    Prediction,
    _format_evidence,
    _normalize_cite,
    _parse_answer,
    run_baseline,
)
from src.kg.dataset_questions import Question
from src.llm.client import LLMContextBudgetError
from src.schema import Evidence

VALID = {"A", "B", "C", "D"}


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


class RaisingChat:
    """Every call raises -- simulates the client's context-budget hard stop."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc
        self.calls = 0

    def __call__(self, prompt, **kwargs):
        self.calls += 1
        raise self._exc


class FakeBundle:
    def __init__(self, evidence):
        self._evidence = list(evidence)

    def evidence(self):
        return list(self._evidence)

    def valid_citation_ids(self):
        return {e.id for e in self._evidence}


class FakePipeline:
    def __init__(self, bundle):
        self._bundle = bundle
        self.modalities_seen: list = []

    def build_bundle(self, query, options, *, dataset, question_id, modalities):
        self.modalities_seen.append(modalities)
        return self._bundle


def _evidence():
    return [
        Evidence(
            id="chunk::doc1#0",
            modality="dense",
            text="Aspirin irreversibly inhibits COX-1.",
            source_meta={},
            scores={},
        ),
        Evidence(
            id="chunk::doc2#0",
            modality="lexical",
            text="Warfarin is a vitamin K antagonist.",
            source_meta={},
            scores={},
        ),
        Evidence(
            id="kg::aspirin|treats|mi",
            modality="kg",
            text="aspirin treats myocardial infarction",
            source_meta={},
            scores={},
        ),
    ]


def _question(qid="test_00001"):
    return Question(
        question_id=qid,
        dataset="medqa",
        split="test",
        question="Which drug irreversibly inhibits COX-1?",
        options={"A": "Aspirin", "B": "Warfarin", "C": "Heparin", "D": "Insulin"},
        answer=None,
    )


def _pipeline():
    return FakePipeline(FakeBundle(_evidence()))


def _patch(monkeypatch, sc):
    monkeypatch.setattr("src.eval.baselines.chat", sc)
    return sc


# --------------------------------------------------------------------------
# constants / dispatch
# --------------------------------------------------------------------------


def test_baselines_constant():
    assert BASELINES == ("B0", "B1", "B2", "B3")


def test_unknown_baseline_raises():
    with pytest.raises(ValueError, match="unknown baseline"):
        run_baseline("B9", _question(), "A", pipeline=None, dataset="medqa")


def test_b1_requires_pipeline():
    with pytest.raises(ValueError, match="requires a pipeline"):
        run_baseline("B1", _question(), "A", pipeline=None, dataset="medqa")


# --------------------------------------------------------------------------
# B0
# --------------------------------------------------------------------------


def test_b0_valid_json(monkeypatch):
    _patch(monkeypatch, ScriptedChat('{"answer": "A", "rationale": "COX-1 inhibitor."}'))
    pred = run_baseline("B0", _question(), "A", pipeline=None, dataset="medqa")
    assert isinstance(pred, Prediction)
    assert pred.pred_option == "A"
    assert pred.correct is True
    assert pred.status == "ok"
    assert pred.citations == []
    assert pred.invalid_citations == []
    assert pred.grounded is False
    assert pred.n_llm_calls == 1
    assert pred.verifier_decided is None
    assert pred.per_option_verdict is None


def test_b0_uses_temperature_zero(monkeypatch):
    sc = _patch(monkeypatch, ScriptedChat('{"answer":"B"}'))
    run_baseline("B0", _question(), "A", pipeline=None, dataset="medqa")
    assert sc.kwargs[0]["temperature"] == 0


def test_b0_incorrect(monkeypatch):
    _patch(monkeypatch, ScriptedChat('{"answer":"C","rationale":"x"}'))
    pred = run_baseline("B0", _question(), "A", pipeline=None, dataset="medqa")
    assert pred.pred_option == "C"
    assert pred.correct is False
    assert pred.grounded is False


def test_b0_grounded_never_true_even_with_citations(monkeypatch):
    # B0 must not surface citations even if the model volunteers some.
    _patch(monkeypatch, ScriptedChat('{"answer":"A","citations":["chunk::doc1#0"]}'))
    pred = run_baseline("B0", _question(), "A", pipeline=None, dataset="medqa")
    assert pred.citations == []
    assert pred.grounded is False


# --------------------------------------------------------------------------
# B1 / B2
# --------------------------------------------------------------------------


def test_b1_mixed_citations(monkeypatch):
    _patch(
        monkeypatch,
        ScriptedChat(
            '{"answer":"A","citations":["chunk::doc1#0","chunk::fake#9"],"rationale":"r"}'
        ),
    )
    pipe = _pipeline()
    pred = run_baseline("B1", _question(), "A", pipeline=pipe, dataset="medqa")
    assert pred.pred_option == "A"
    assert pred.citations == ["chunk::doc1#0", "chunk::fake#9"]
    assert pred.invalid_citations == ["chunk::fake#9"]
    assert pred.grounded is True
    assert pred.n_llm_calls == 1
    assert pipe.modalities_seen == [frozenset({"dense"})]


def test_b2_uses_hybrid_modalities(monkeypatch):
    _patch(monkeypatch, ScriptedChat('{"answer":"A","citations":["kg::aspirin|treats|mi"]}'))
    pipe = _pipeline()
    pred = run_baseline("B2", _question(), "A", pipeline=pipe, dataset="medqa")
    assert pipe.modalities_seen == [frozenset({"dense", "lexical", "kg"})]
    assert pred.grounded is True
    assert pred.invalid_citations == []
    assert pred.n_llm_calls == 1


def test_b1_malformed_json_parse_error(monkeypatch):
    _patch(monkeypatch, ScriptedChat("I think the correct choice is probably Aspirin"))
    pred = run_baseline("B1", _question(), "A", pipeline=_pipeline(), dataset="medqa")
    assert pred.status == "parse_error"
    assert pred.pred_option is None
    assert pred.correct is False
    assert pred.grounded is False


def test_b1_empty_output_no_answer(monkeypatch):
    _patch(monkeypatch, ScriptedChat("   \n  "))
    pred = run_baseline("B1", _question(), "A", pipeline=_pipeline(), dataset="medqa")
    assert pred.status == "no_answer"
    assert pred.pred_option is None
    assert pred.correct is False


def test_b1_letter_out_of_range_parse_error(monkeypatch):
    _patch(monkeypatch, ScriptedChat('{"answer":"E","rationale":"x"}'))
    pred = run_baseline("B1", _question(), "A", pipeline=_pipeline(), dataset="medqa")
    assert pred.status == "parse_error"
    assert pred.pred_option is None


def test_b1_bare_string_citation_tolerated(monkeypatch):
    _patch(monkeypatch, ScriptedChat('{"answer":"A","citations":"chunk::doc2#0"}'))
    pred = run_baseline("B1", _question(), "A", pipeline=_pipeline(), dataset="medqa")
    assert pred.citations == ["chunk::doc2#0"]
    assert pred.grounded is True
    assert pred.invalid_citations == []


def test_b2_json_in_markdown_fence(monkeypatch):
    _patch(
        monkeypatch,
        ScriptedChat('```json\n{"answer": "A", "citations": ["chunk::doc1#0"]}\n```'),
    )
    pred = run_baseline("B2", _question(), "A", pipeline=_pipeline(), dataset="medqa")
    assert pred.pred_option == "A"
    assert pred.grounded is True


# --------------------------------------------------------------------------
# B3
# --------------------------------------------------------------------------


def test_b3_exactly_one_supported(monkeypatch):
    _patch(
        monkeypatch,
        ScriptedChat(
            '{"verdict":"SUPPORTED","evidence_id":"chunk::doc1#0"}',
            '{"verdict":"NOT_SUPPORTED","evidence_id":null}',
            '{"verdict":"NOT_SUPPORTED","evidence_id":null}',
            '{"verdict":"NOT_SUPPORTED","evidence_id":null}',
        ),
    )
    pred = run_baseline("B3", _question(), "A", pipeline=_pipeline(), dataset="medqa")
    assert pred.verifier_decided is True
    assert pred.pred_option == "A"
    assert pred.correct is True
    assert pred.n_supported == 1
    assert pred.n_llm_calls == 4
    assert pred.per_option_verdict == {
        "A": "SUPPORTED",
        "B": "NOT_SUPPORTED",
        "C": "NOT_SUPPORTED",
        "D": "NOT_SUPPORTED",
    }
    assert pred.citations == ["chunk::doc1#0"]
    assert pred.grounded is True


def test_b3_one_supported_but_invalid_evidence_id(monkeypatch):
    _patch(
        monkeypatch,
        ScriptedChat(
            '{"verdict":"NOT_SUPPORTED","evidence_id":null}',
            '{"verdict":"SUPPORTED","evidence_id":"chunk::ghost#1"}',
            '{"verdict":"NOT_SUPPORTED","evidence_id":null}',
            '{"verdict":"NOT_SUPPORTED","evidence_id":null}',
        ),
    )
    pred = run_baseline("B3", _question(), "B", pipeline=_pipeline(), dataset="medqa")
    assert pred.verifier_decided is True
    assert pred.pred_option == "B"
    assert pred.citations == []
    assert pred.invalid_citations == []
    assert pred.grounded is False
    assert pred.n_llm_calls == 4


def test_b3_zero_supported_falls_back(monkeypatch):
    _patch(
        monkeypatch,
        ScriptedChat(
            '{"verdict":"NOT_SUPPORTED"}',
            '{"verdict":"NOT_SUPPORTED"}',
            '{"verdict":"NOT_SUPPORTED"}',
            '{"verdict":"NOT_SUPPORTED"}',
            '{"answer":"A","citations":["chunk::doc1#0"],"rationale":"r"}',
        ),
    )
    pred = run_baseline("B3", _question(), "A", pipeline=_pipeline(), dataset="medqa")
    assert pred.verifier_decided is False
    assert pred.n_supported == 0
    assert pred.pred_option == "A"
    assert pred.correct is True
    assert pred.n_llm_calls == 5
    assert pred.grounded is True
    assert pred.per_option_verdict["A"] == "NOT_SUPPORTED"


def test_b3_two_supported_falls_back(monkeypatch):
    _patch(
        monkeypatch,
        ScriptedChat(
            '{"verdict":"SUPPORTED","evidence_id":"chunk::doc1#0"}',
            '{"verdict":"SUPPORTED","evidence_id":"chunk::doc2#0"}',
            '{"verdict":"NOT_SUPPORTED"}',
            '{"verdict":"NOT_SUPPORTED"}',
            '{"answer":"B","citations":["chunk::doc2#0"],"rationale":"r"}',
        ),
    )
    pred = run_baseline("B3", _question(), "B", pipeline=_pipeline(), dataset="medqa")
    assert pred.verifier_decided is False
    assert pred.n_supported == 2
    assert pred.pred_option == "B"
    assert pred.n_llm_calls == 5
    assert pred.per_option_verdict["A"] == "SUPPORTED"
    assert pred.per_option_verdict["B"] == "SUPPORTED"
    assert pred.grounded is True


def test_b3_fallback_parse_error_still_returns_prediction(monkeypatch):
    _patch(
        monkeypatch,
        ScriptedChat(
            '{"verdict":"NOT_SUPPORTED"}',
            '{"verdict":"NOT_SUPPORTED"}',
            '{"verdict":"NOT_SUPPORTED"}',
            '{"verdict":"NOT_SUPPORTED"}',
            "sorry, no idea",
        ),
    )
    pred = run_baseline("B3", _question(), "A", pipeline=_pipeline(), dataset="medqa")
    assert pred.verifier_decided is False
    assert pred.status == "parse_error"
    assert pred.pred_option is None
    assert pred.correct is False
    assert pred.n_llm_calls == 5


def test_b3_uses_hybrid_modalities(monkeypatch):
    _patch(
        monkeypatch,
        ScriptedChat(
            '{"verdict":"SUPPORTED","evidence_id":"chunk::doc1#0"}',
            '{"verdict":"NOT_SUPPORTED"}',
            '{"verdict":"NOT_SUPPORTED"}',
            '{"verdict":"NOT_SUPPORTED"}',
        ),
    )
    pipe = _pipeline()
    run_baseline("B3", _question(), "A", pipeline=pipe, dataset="medqa")
    assert pipe.modalities_seen == [frozenset({"dense", "lexical", "kg"})]


# --------------------------------------------------------------------------
# _parse_answer direct toy cases
# --------------------------------------------------------------------------


def test_parse_answer_json_ok():
    letter, cites, status = _parse_answer('{"answer":"C","citations":["x"]}', VALID)
    assert (letter, cites, status) == ("C", ["x"], "ok")


def test_parse_answer_regex_fallback():
    letter, cites, status = _parse_answer("answer: B -- best supported choice", VALID)
    assert letter == "B"
    assert status == "ok"
    assert cites == []


def test_parse_answer_blank_is_no_answer():
    assert _parse_answer("", VALID) == (None, [], "no_answer")


def test_parse_answer_unknown_letter_is_parse_error():
    assert _parse_answer('{"answer":"Z"}', VALID) == (None, [], "parse_error")


def test_parse_answer_lowercase_letter_uppercased():
    letter, _cites, status = _parse_answer('{"answer":"a"}', VALID)
    assert (letter, status) == ("A", "ok")


# --------------------------------------------------------------------------
# Bug 1: evidence rendering + citation normalization
# --------------------------------------------------------------------------


def test_format_evidence_has_no_leading_number():
    rendered = _format_evidence(_evidence())
    lines = rendered.splitlines()
    assert lines[0] == "[chunk::doc1#0] Aspirin irreversibly inhibits COX-1."
    assert lines[1].startswith("[chunk::doc2#0] ")
    for line in lines:
        assert not line[:3].strip().rstrip(".").isdigit()


def test_format_evidence_empty():
    assert _format_evidence([]) == "(no evidence retrieved)"


def test_normalize_cite_strips_brackets():
    assert _normalize_cite("[chunk::x#1]") == "chunk::x#1"


def test_normalize_cite_strips_quotes_and_space():
    assert _normalize_cite('  "kg::a|r|b"  ') == "kg::a|r|b"


def test_normalize_cite_leaves_clean_id_untouched():
    assert _normalize_cite("chunk::textbook::InternalMed_Harrison#8538") == (
        "chunk::textbook::InternalMed_Harrison#8538"
    )


def test_normalize_cite_lone_brackets_do_not_crash():
    assert _normalize_cite("[") == ""
    assert _normalize_cite("]") == ""
    assert _normalize_cite("[]") == ""


def test_b1_bracketed_citations_now_match_and_ground(monkeypatch):
    # The Phase 3 exit-run bug: model echoes the bracketed form verbatim.
    _patch(
        monkeypatch,
        ScriptedChat(
            '{"answer":"A","citations":["[chunk::doc1#0]","[kg::aspirin|treats|mi]"],'
            '"rationale":"r"}'
        ),
    )
    pred = run_baseline("B1", _question(), "A", pipeline=_pipeline(), dataset="medqa")
    assert pred.citations == ["chunk::doc1#0", "kg::aspirin|treats|mi"]
    assert pred.invalid_citations == []
    assert pred.grounded is True


def test_b2_citation_via_regex_fallback_normalizes(monkeypatch):
    # No parseable JSON object -> regex path must also normalize each token.
    _patch(
        monkeypatch,
        ScriptedChat('answer = A , "citations": ["chunk::doc2#0", "14"]'),
    )
    pred = run_baseline("B2", _question(), "A", pipeline=_pipeline(), dataset="medqa")
    assert "chunk::doc2#0" in pred.citations
    assert "14" in pred.invalid_citations  # bare position stays unmatched, on purpose
    assert pred.grounded is True


def test_gen_prompt_mentions_exact_id_instruction(monkeypatch):
    sc = _patch(monkeypatch, ScriptedChat('{"answer":"A","citations":["chunk::doc1#0"]}'))
    run_baseline("B1", _question(), "A", pipeline=_pipeline(), dataset="medqa")
    assert "not the position" in sc.prompts[0]


# --------------------------------------------------------------------------
# Bug 2: context-budget hard stop
# --------------------------------------------------------------------------


def test_b0_context_exceeded(monkeypatch):
    rc = _patch(monkeypatch, RaisingChat(LLMContextBudgetError("too big")))
    pred = run_baseline("B0", _question(), "A", pipeline=None, dataset="medqa")
    assert pred.status == "context_exceeded"
    assert pred.pred_option is None
    assert pred.correct is False
    assert pred.citations == []
    assert pred.invalid_citations == []
    assert pred.grounded is False
    assert pred.n_llm_calls == 0
    assert pred.gold_option == "A"
    assert rc.calls == 1


def test_b2_context_exceeded(monkeypatch):
    _patch(monkeypatch, RaisingChat(LLMContextBudgetError("too big")))
    pred = run_baseline("B2", _question(), "C", pipeline=_pipeline(), dataset="medqa")
    assert pred.status == "context_exceeded"
    assert pred.pred_option is None
    assert pred.grounded is False
    assert pred.n_llm_calls == 0
    assert pred.gold_option == "C"
    assert pred.verifier_decided is None


def test_b3_context_exceeded_leaves_verifier_fields_none(monkeypatch):
    _patch(monkeypatch, RaisingChat(LLMContextBudgetError("too big")))
    pred = run_baseline("B3", _question(), "A", pipeline=_pipeline(), dataset="medqa")
    assert pred.status == "context_exceeded"
    assert pred.n_llm_calls == 0
    assert pred.verifier_decided is None
    assert pred.n_supported is None
    assert pred.per_option_verdict is None


# --------------------------------------------------------------------------
# contamination_assertion executes a real check (AUDIT 2026-09-06 §3.2)
# --------------------------------------------------------------------------


class _FakeBundle:
    def __init__(self, texts):
        self.passages = [
            Evidence(id=f"chunk::d{i}#0", modality="dense", text=t, source_meta={}, scores={})
            for i, t in enumerate(texts)
        ]


class _FakePipeline:
    """Stands in for TriModalPipeline: only `cfg.retrieval.top_k_rerank` and
    `build_bundle` are touched by contamination_assertion."""

    def __init__(self, texts_by_qid):
        self._texts = texts_by_qid
        self.cfg = type(
            "C", (), {"retrieval": type("R", (), {"top_k_rerank": 12})()}
        )()

    def build_bundle(self, question, options, *, dataset, question_id):
        return _FakeBundle(self._texts[question_id])


def _q(qid, stem):
    return Question(
        question_id=qid,
        dataset="medqa",
        split="train",
        question=stem,
        options={"A": "a", "B": "b", "C": "c", "D": "d"},
        answer="a",
    )


def test_contamination_assertion_runs_and_reports_clean():
    from scripts.run_baselines import contamination_assertion

    stem = "a 45 year old man presents with crushing substernal chest pain radiating"
    qs = [_q("train_1", stem)]
    out = contamination_assertion("medqa", qs, _FakePipeline({"train_1": ["unrelated filler"]}))
    assert out["checked"] is True
    assert out["applies"] is False
    assert out["n_questions"] == 1
    assert out["retrieved_passages_per_question"] == 12


def test_contamination_assertion_detects_a_stem_reproduced_in_a_retrieved_chunk():
    """The regression test for the literal-dict version: `applies` must flip
    when a real leak is present."""
    from scripts.run_baselines import contamination_assertion

    stem = "a 45 year old man presents with crushing substernal chest pain radiating"
    qs = [_q("train_1", stem)]
    out = contamination_assertion("medqa", qs, _FakePipeline({"train_1": [f"... {stem} ..."]}))
    assert out["checked"] is True
    assert out["applies"] is True
    assert out["n_flagged"] == 1


def test_contamination_assertion_without_a_pipeline_is_not_reported_as_checked():
    """A B0-only run retrieves nothing, so nothing can be measured -- that must
    read `checked: false`, not a free pass."""
    from scripts.run_baselines import contamination_assertion

    out = contamination_assertion("medqa", [_q("train_1", "stem")], None)
    assert out["checked"] is False
    assert out["applies"] is True


def test_contamination_assertion_unknown_dataset_is_not_checked():
    from scripts.run_baselines import contamination_assertion

    out = contamination_assertion("medmcqa", [], None)
    assert out["checked"] is False
