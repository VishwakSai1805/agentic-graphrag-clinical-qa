import json
from pathlib import Path

import pytest

from src.eval.contamination import (
    ContaminationError,
    _load_pqa_labeled_pubids,
    assert_no_contamination,
    blocklisted_pmids,
    pubmedqa_split,
)


def test_pubmedqa_split_deterministic_across_calls():
    test1, dev1 = pubmedqa_split(seed=42)
    test2, dev2 = pubmedqa_split(seed=42)
    assert test1 == test2
    assert dev1 == dev2


def test_pubmedqa_split_test_dev_disjoint():
    test_pmids, dev_pmids = pubmedqa_split()
    assert test_pmids.isdisjoint(dev_pmids)


def test_pubmedqa_split_covers_all_real_pubids():
    real_pubids = set(_load_pqa_labeled_pubids())
    assert len(real_pubids) == 1000

    test_pmids, dev_pmids = pubmedqa_split()
    assert test_pmids | dev_pmids == real_pubids
    assert len(test_pmids) + len(dev_pmids) == 1000


def test_pubmedqa_split_default_sizes_are_500_500():
    test_pmids, dev_pmids = pubmedqa_split()
    assert len(test_pmids) == 500
    assert len(dev_pmids) == 500


def test_pubmedqa_split_different_seed_differs():
    test_a, _ = pubmedqa_split(seed=42)
    test_b, _ = pubmedqa_split(seed=1)
    assert test_a != test_b


def test_blocklisted_pmids_pubmedqa_matches_split_test_set():
    test_pmids, _dev_pmids = pubmedqa_split()
    assert blocklisted_pmids("pubmedqa") == test_pmids


def test_blocklisted_pmids_unknown_dataset_raises():
    with pytest.raises(ValueError):
        blocklisted_pmids("medqa")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_assert_no_contamination_raises_on_blocklisted_doc(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    _write_jsonl(
        corpus_path,
        [
            {"doc_id": "111", "source": "pubmed_abstract"},
            {"doc_id": "222", "source": "pubmed_abstract"},
            {"doc_id": "333", "source": "textbook"},
        ],
    )
    blocklist = {"222", "999"}

    with pytest.raises(ContaminationError) as excinfo:
        assert_no_contamination(corpus_path, blocklist)
    assert "222" in str(excinfo.value)


def test_assert_no_contamination_clean_does_not_raise(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    _write_jsonl(
        corpus_path,
        [
            {"doc_id": "111", "source": "pubmed_abstract"},
            {"doc_id": "333", "source": "textbook"},
        ],
    )
    blocklist = {"222", "999"}

    assert_no_contamination(corpus_path, blocklist)


def test_assert_no_contamination_lists_all_offenders(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    _write_jsonl(
        corpus_path,
        [
            {"doc_id": "111"},
            {"doc_id": "222"},
        ],
    )
    blocklist = {"111", "222"}

    with pytest.raises(ContaminationError) as excinfo:
        assert_no_contamination(corpus_path, blocklist)
    message = str(excinfo.value)
    assert "111" in message
    assert "222" in message
