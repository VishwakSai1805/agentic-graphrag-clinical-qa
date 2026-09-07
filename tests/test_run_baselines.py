"""Unit tests for scripts/run_baselines.py's own logic: gold_option's
per-dataset branching (medqa/medmcqa/pubmedqa) and option_letters. No LLM, no
GPU -- src.eval.baselines's prompt/parse/verify logic is already
dict-driven and covered by tests/test_baselines.py."""

from __future__ import annotations

import pytest

from scripts.run_baselines import gold_option, option_letters
from src.kg.dataset_questions import Question


def _q(dataset, **kwargs):
    defaults = {
        "question_id": "x_00001",
        "dataset": dataset,
        "split": "test",
        "question": "q?",
        "options": None,
        "answer": None,
    }
    defaults.update(kwargs)
    return Question(**defaults)


# --------------------------------------------------------------------------
# option_letters
# --------------------------------------------------------------------------


def test_option_letters_pubmedqa_is_three_way():
    assert option_letters("pubmedqa") == ("A", "B", "C")


def test_option_letters_medqa_is_four_way():
    assert option_letters("medqa") == ("A", "B", "C", "D")


def test_option_letters_medmcqa_is_four_way():
    assert option_letters("medmcqa") == ("A", "B", "C", "D")


# --------------------------------------------------------------------------
# gold_option -- pubmedqa
# --------------------------------------------------------------------------


def test_gold_option_pubmedqa_yes():
    q = _q("pubmedqa", answer="yes")
    assert gold_option(q) == "A"


def test_gold_option_pubmedqa_no():
    q = _q("pubmedqa", answer="no")
    assert gold_option(q) == "B"


def test_gold_option_pubmedqa_maybe():
    q = _q("pubmedqa", answer="maybe")
    assert gold_option(q) == "C"


def test_gold_option_pubmedqa_is_case_insensitive():
    q = _q("pubmedqa", answer="Yes")
    assert gold_option(q) == "A"


def test_gold_option_pubmedqa_missing_answer_raises():
    q = _q("pubmedqa", answer=None)
    with pytest.raises(ValueError, match="missing final_decision"):
        gold_option(q)


def test_gold_option_pubmedqa_unrecognised_answer_raises():
    q = _q("pubmedqa", answer="unsure")
    with pytest.raises(ValueError, match="matched 0"):
        gold_option(q)


# --------------------------------------------------------------------------
# gold_option -- medqa / medmcqa (regression: unchanged by the pubmedqa branch)
# --------------------------------------------------------------------------


def test_gold_option_medqa_matches_text_to_letter():
    q = _q("medqa", options={"A": "aspirin", "B": "ibuprofen"}, answer="ibuprofen")
    assert gold_option(q) == "B"


def test_gold_option_medqa_missing_options_raises():
    q = _q("medqa", options=None, answer="ibuprofen")
    with pytest.raises(ValueError, match="missing options/answer"):
        gold_option(q)


def test_gold_option_medmcqa_is_already_a_letter():
    q = _q("medmcqa", options={"A": "x", "B": "y"}, answer="B")
    assert gold_option(q) == "B"


def test_gold_option_medmcqa_invalid_letter_raises():
    q = _q("medmcqa", options={"A": "x", "B": "y"}, answer="Z")
    with pytest.raises(ValueError, match="not a valid option letter"):
        gold_option(q)


def test_gold_option_unknown_dataset_raises_not_implemented():
    q = _q("unknown_dataset")
    with pytest.raises(NotImplementedError):
        gold_option(q)
