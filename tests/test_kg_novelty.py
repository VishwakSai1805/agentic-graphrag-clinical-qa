"""Unit tests for src/kg/novelty.py.

`decide_novel` (pure) and `is_novel_relation` with an injected fake scorer
are covered here without downloading/loading the real
MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli model or touching torch/CUDA --
`tests/test_kg_novelty.py` per the Phase 2 spec must not require the real
model in CI.
"""

from __future__ import annotations

from src.kg.novelty import decide_novel, is_novel_relation, is_triple_relevant
from src.kg.traversal import Triple


def test_decide_novel_true_when_no_score_reaches_threshold():
    assert decide_novel([0.1, 0.3, 0.49], threshold=0.5) is True


def test_decide_novel_false_when_any_score_reaches_threshold():
    assert decide_novel([0.1, 0.5, 0.2], threshold=0.5) is False


def test_decide_novel_false_when_score_exactly_at_threshold():
    assert decide_novel([0.5], threshold=0.5) is False


def test_decide_novel_true_for_empty_scores():
    """No dense passages at all -- vacuously novel, nothing entails it."""
    assert decide_novel([], threshold=0.5) is True


def test_decide_novel_respects_custom_threshold():
    assert decide_novel([0.6], threshold=0.7) is True
    assert decide_novel([0.75], threshold=0.7) is False


def _make_triple() -> Triple:
    return Triple(
        head="aspirin", relation="indication", tail="myocardial infarction", source_db="primekg"
    )


def test_is_novel_relation_true_when_fake_scorer_never_entails():
    def fake_scorer(premise: str, hypothesis: str) -> float:
        return 0.1  # never entails, regardless of content

    result = is_novel_relation(
        _make_triple(),
        dense_passages=["some unrelated passage about diabetes", "another passage"],
        scorer=fake_scorer,
    )
    assert result is True


def test_is_novel_relation_false_when_a_passage_entails():
    def fake_scorer(premise: str, hypothesis: str) -> float:
        return 0.9 if "aspirin" in premise else 0.05

    result = is_novel_relation(
        _make_triple(),
        dense_passages=["irrelevant passage", "aspirin is indicated for myocardial infarction"],
        scorer=fake_scorer,
    )
    assert result is False


def test_is_novel_relation_passes_verbalised_text_as_hypothesis():
    seen_hypotheses = []

    def recording_scorer(premise: str, hypothesis: str) -> float:
        seen_hypotheses.append(hypothesis)
        return 0.0

    is_novel_relation(_make_triple(), dense_passages=["passage one"], scorer=recording_scorer)

    assert len(seen_hypotheses) == 1
    assert "aspirin" in seen_hypotheses[0]
    assert "myocardial infarction" in seen_hypotheses[0]


def test_is_novel_relation_true_when_no_dense_passages_at_all():
    def fake_scorer(premise: str, hypothesis: str) -> float:
        raise AssertionError("scorer should never be called with zero passages")

    result = is_novel_relation(_make_triple(), dense_passages=[], scorer=fake_scorer)
    assert result is True


def test_is_triple_relevant_true_when_head_in_question_stem():
    triple = _make_triple()  # head=aspirin, tail=myocardial infarction
    assert is_triple_relevant(triple, "A patient is prescribed aspirin daily.") is True


def test_is_triple_relevant_true_when_tail_in_an_option():
    triple = _make_triple()
    options = {"A": "Myocardial infarction", "B": "Stroke"}
    assert is_triple_relevant(triple, "What is the most likely diagnosis?", options) is True


def test_is_triple_relevant_false_when_neither_head_nor_tail_appear():
    triple = _make_triple()
    assert (
        is_triple_relevant(
            triple, "A patient presents with a broken wrist.", {"A": "Splint", "B": "Cast"}
        )
        is False
    )


def test_is_triple_relevant_is_case_insensitive():
    triple = _make_triple()
    assert is_triple_relevant(triple, "ASPIRIN was given.") is True


def test_is_triple_relevant_false_with_no_options_and_no_match():
    triple = _make_triple()
    assert is_triple_relevant(triple, "Unrelated question about the knee.", None) is False
