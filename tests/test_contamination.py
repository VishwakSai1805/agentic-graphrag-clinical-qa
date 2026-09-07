import json
from pathlib import Path

import pytest

from src.eval.contamination import (
    ContaminationError,
    _load_pqa_labeled_pubids,
    assert_calibration_fit_set_uncontaminated,
    assert_no_contamination,
    blocklisted_pmids,
    calibration_fit_pmids,
    max_ngram_overlap,
    medqa_overlap_check,
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


# --------------------------------------------------------------------------
# MedQA n-gram overlap check (AUDIT 2026-09-06 §3.2)
# --------------------------------------------------------------------------


def test_max_ngram_overlap_identical_text_is_one():
    stem = "a 45 year old man presents with crushing substernal chest pain radiating"
    assert max_ngram_overlap(stem, [stem], n=8) == 1.0


def test_max_ngram_overlap_disjoint_text_is_zero():
    stem = "a 45 year old man presents with crushing substernal chest pain radiating"
    other = "the mitochondrion is the powerhouse of the cell indeed"
    assert max_ngram_overlap(stem, [other], n=8) == 0.0


def test_max_ngram_overlap_short_stem_yields_zero_not_error():
    """A stem with fewer than n tokens produces no n-grams; that is 0.0, not a
    crash and not a false 1.0."""
    assert max_ngram_overlap("too short", ["too short"], n=8) == 0.0


def test_max_ngram_overlap_takes_the_max_over_chunks():
    stem = "a 45 year old man presents with crushing substernal chest pain radiating"
    partial = "man presents with crushing substernal chest pain radiating to the jaw"
    assert 0.0 < max_ngram_overlap(stem, ["unrelated filler text here", partial], n=8) < 1.0


def test_max_ngram_overlap_is_case_and_punctuation_insensitive():
    stem = "A 45-year-old man presents with crushing, substernal chest pain radiating."
    chunk = "a 45-year-old MAN PRESENTS with crushing substernal chest pain radiating"
    assert max_ngram_overlap(stem, [chunk], n=5) > 0.0


def test_medqa_overlap_check_reports_checked_and_clean_when_no_overlap():
    res = medqa_overlap_check(
        [
            (
                "a 45 year old man presents with crushing substernal chest pain",
                ["completely unrelated"],
            )
        ]
    )
    assert res["checked"] is True
    assert res["applies"] is False
    assert res["n_flagged"] == 0
    assert res["max_overlap"] == 0.0


def test_medqa_overlap_check_flags_a_verbatim_stem():
    """`applies` must come from the executed outcome -- this is the regression
    test for the literal-dict version the audit found."""
    stem = "a 45 year old man presents with crushing substernal chest pain radiating"
    res = medqa_overlap_check([(stem, [f"prefix {stem} suffix"])])
    assert res["checked"] is True
    assert res["applies"] is True
    assert res["n_flagged"] == 1
    assert res["max_overlap"] == 1.0


def test_medqa_overlap_check_empty_input_is_not_reported_as_checked():
    res = medqa_overlap_check([])
    assert res["checked"] is False
    assert res["applies"] is True


def test_medqa_overlap_check_reports_n_and_distribution_keys():
    stems = [("a 45 year old man presents with crushing substernal chest pain", ["x"])] * 3
    res = medqa_overlap_check(stems)
    assert res["n_questions"] == 3
    for key in ("mean_overlap", "median_overlap", "p95_overlap", "ngram_n", "flag_threshold"):
        assert key in res


# --------------------------------------------------------------------------
# Calibration-fitting split (AUDIT 2026-09-06 §3.3, ADR-026)
# --------------------------------------------------------------------------


def test_calibration_fit_pmids_is_the_dev_half_not_test():
    test_pmids, dev_pmids = pubmedqa_split()
    fit = calibration_fit_pmids()
    assert fit == dev_pmids
    assert not (fit & test_pmids)


def test_calibration_fit_pmids_size_is_500():
    assert len(calibration_fit_pmids()) == 500


def test_assert_calibration_fit_set_uncontaminated_raises_when_fit_doc_indexed(tmp_path):
    """ADR-026's whole point: fitting on an item whose source abstract is
    retrievable must be a hard failure, not a note in a doc."""
    fit_pmid = sorted(calibration_fit_pmids())[0]
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps({"doc_id": fit_pmid, "text": "x"}) + "\n"
        + json.dumps({"doc_id": "not_a_pmid", "text": "y"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ContaminationError):
        assert_calibration_fit_set_uncontaminated(corpus)


def test_assert_calibration_fit_set_uncontaminated_passes_on_clean_corpus(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps({"doc_id": "not_a_pmid", "text": "y"}) + "\n", encoding="utf-8")
    assert_calibration_fit_set_uncontaminated(corpus)
