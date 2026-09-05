"""Unit tests for src/kg/dataset_questions.py, against small fixture jsonl
files (not the real multi-hundred-MB raw datasets) written to tmp_path and
monkeypatched into `_SPLIT_PATHS`."""

from __future__ import annotations

import json

from src.kg import dataset_questions as dq


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_medqa_question_id_and_fields(tmp_path, monkeypatch):
    path = tmp_path / "medqa_test.jsonl"
    write_jsonl(
        path,
        [
            {
                "question": "Q1 stem",
                "answer": "B",
                "options": {"A": "opt a", "B": "opt b"},
            },
            {
                "question": "Q2 stem",
                "answer": "A",
                "options": {"A": "opt a2", "B": "opt b2"},
            },
        ],
    )
    monkeypatch.setitem(dq._SPLIT_PATHS, ("medqa", "test"), path)

    questions = dq.load_question_sample("medqa", "test", limit=None)
    assert [q.question_id for q in questions] == ["test_00000", "test_00001"]
    assert questions[0].question == "Q1 stem"
    assert questions[0].options == {"A": "opt a", "B": "opt b"}
    assert questions[0].answer == "B"
    assert questions[0].native_id is None


def test_pubmedqa_native_id_and_no_options(tmp_path, monkeypatch):
    path = tmp_path / "pubmedqa_test.jsonl"
    write_jsonl(
        path,
        [{"pubid": 123456, "question": "Does X cause Y?", "final_decision": "yes"}],
    )
    monkeypatch.setitem(dq._SPLIT_PATHS, ("pubmedqa", "test"), path)

    questions = dq.load_question_sample("pubmedqa", "test", limit=None)
    assert len(questions) == 1
    q = questions[0]
    assert q.question_id == "test_00000"
    assert q.native_id == "123456"
    assert q.options is None
    assert q.answer == "yes"


def test_medmcqa_options_and_answer_letter(tmp_path, monkeypatch):
    path = tmp_path / "medmcqa_validation.jsonl"
    write_jsonl(
        path,
        [
            {
                "id": "abc-123",
                "question": "Which is true?",
                "opa": "A text",
                "opb": "B text",
                "opc": "C text",
                "opd": "D text",
                "cop": 2,
                "subject_name": "Physiology",
            }
        ],
    )
    monkeypatch.setitem(dq._SPLIT_PATHS, ("medmcqa", "validation"), path)

    questions = dq.load_question_sample("medmcqa", "validation", limit=None)
    q = questions[0]
    assert q.native_id == "abc-123"
    assert q.options == {"A": "A text", "B": "B text", "C": "C text", "D": "D text"}
    assert q.answer == "C"
    assert q.subject_name == "Physiology"


def test_link_text_for_question_concatenates_options_for_mcq():
    q = dq.Question(
        question_id="test_00000",
        dataset="medqa",
        split="test",
        question="What drug treats X?",
        options={"A": "aspirin", "B": "metformin"},
    )
    text = dq.link_text_for_question(q)
    assert text == "What drug treats X? aspirin metformin"


def test_link_text_for_question_stem_only_when_no_options():
    q = dq.Question(
        question_id="test_00000",
        dataset="pubmedqa",
        split="test",
        question="Does X cause Y?",
        options=None,
    )
    assert dq.link_text_for_question(q) == "Does X cause Y?"


def test_load_question_sample_limit_is_deterministic(tmp_path, monkeypatch):
    path = tmp_path / "medqa_test.jsonl"
    rows = [{"question": f"Q{i}", "answer": "A", "options": {"A": "x"}} for i in range(20)]
    write_jsonl(path, rows)
    monkeypatch.setitem(dq._SPLIT_PATHS, ("medqa", "test"), path)

    sample1 = dq.load_question_sample("medqa", "test", limit=5, seed=42)
    sample2 = dq.load_question_sample("medqa", "test", limit=5, seed=42)
    assert [q.question_id for q in sample1] == [q.question_id for q in sample2]
    assert len(sample1) == 5


def test_load_question_sample_missing_split_raises_clear_error(tmp_path, monkeypatch):
    import pytest

    with pytest.raises(ValueError, match="no wired raw file"):
        dq.load_question_sample("medqa", "does-not-exist", limit=None)


def test_load_question_sample_missing_file_raises_file_not_found(tmp_path, monkeypatch):
    import pytest

    monkeypatch.setitem(dq._SPLIT_PATHS, ("medqa", "test"), tmp_path / "nope.jsonl")
    with pytest.raises(FileNotFoundError):
        dq.load_question_sample("medqa", "test", limit=None)
