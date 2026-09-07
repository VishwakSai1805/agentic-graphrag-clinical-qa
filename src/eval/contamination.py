"""Test-set contamination guards for retrieval corpora.

CLAUDE.md non-negotiable: "No test-set contamination. Retrieval corpora are
built from training/external corpora only. PubMedQA labelled test abstracts
must not be indexed in the retrieval corpus for the PubMedQA eval. Assert
this in code, do not merely comment it."

This module is the single source of truth for:

1. The canonical, deterministic PubMedQA test/dev split (``pubmedqa_split``).
   Every phase of this project (index building, eval harness, calibration
   fitting) must call this function rather than re-deriving the split, so the
   500/500 partition is byte-for-byte identical everywhere it is used.
2. The blocklist of PMIDs that must never be indexed as retrievable documents
   for a given dataset's eval (``blocklisted_pmids``).
3. A hard assertion (raises, not logs) that a built corpus file does not
   contain any blocklisted document id (``assert_no_contamination``).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

# Location of the raw PubMedQA labelled split (pqa_labeled), 1000 rows,
# fields include "pubid" (int). This is the HF dataset's only split; we
# derive our own deterministic test/dev partition from it below.
_PQA_LABELED_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "raw" / "pubmedqa" / "pqa_labeled_train.jsonl"
)


class ContaminationError(Exception):
    """Raised when a blocklisted (test-set) document id is found in a corpus file."""


def _load_pqa_labeled_pubids(path: Path = _PQA_LABELED_PATH) -> list[str]:
    """Read pqa_labeled_train.jsonl and return all pubids as strings.

    Order of the returned list follows file order; callers that need
    determinism must sort explicitly (pubmedqa_split does this).
    """
    pubids: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pubids.append(str(row["pubid"]))
    return pubids


def pubmedqa_split(seed: int = 42, n_test: int = 500) -> tuple[set[str], set[str]]:
    """Deterministic (test_pmids, dev_pmids) split of pqa_labeled's 1000 pubids.

    THIS IS THE CANONICAL PubMedQA split for the whole project
    (docs/03_DATASETS.md: "500 test (standard), 500 dev"). Recipe, fixed so
    it can be reproduced byte-for-byte without re-deriving it:

        1. Load all pubids from data/raw/pubmedqa/pqa_labeled_train.jsonl.
        2. Sort them ascending (as ints) for a deterministic starting order,
           independent of file/JSON row order.
        3. Convert to strings, then shuffle with random.Random(seed).shuffle.
        4. The first `n_test` shuffled pubids -> test set.
        5. The remainder -> dev set.

    Returned sets contain pubids as strings (str(pubid)), matching how
    corpus doc_ids are stored as strings for pubmed-abstract-sourced docs.

    Raises:
        ValueError: if n_test is not strictly between 0 and the total number
            of available pubids (a split must be non-trivial on both sides).
    """
    all_pubids_int = sorted(
        int(pid) for pid in _load_pqa_labeled_pubids()
    )
    total = len(all_pubids_int)
    if not (0 < n_test < total):
        raise ValueError(f"n_test={n_test} must be in (0, {total}) for a {total}-row dataset")

    ordered = [str(pid) for pid in all_pubids_int]
    rng = random.Random(seed)
    rng.shuffle(ordered)

    test_pmids = set(ordered[:n_test])
    dev_pmids = set(ordered[n_test:])
    return test_pmids, dev_pmids


def blocklisted_pmids(dataset: str = "pubmedqa") -> set[str]:
    """PMIDs that must never appear as an indexed document in the retrieval
    corpus for `dataset`'s eval.

    For "pubmedqa" this is pubmedqa_split()[0] (the test set only — the dev
    set MAY legitimately be used elsewhere, e.g. for prompt/threshold
    tuning; only the test set is the hard blocklist).

    Raises:
        ValueError: for any dataset name other than "pubmedqa" (MedQA and
            MedMCQA are multiple-choice benchmarks with no source-abstract
            leakage concept, so they have no blocklist to compute yet).
    """
    if dataset != "pubmedqa":
        raise ValueError(
            f"blocklisted_pmids: no blocklist defined for dataset={dataset!r}; "
            "only 'pubmedqa' is supported"
        )
    test_pmids, _dev_pmids = pubmedqa_split()
    return test_pmids


def assert_no_contamination(corpus_path: Path, blocklist: set[str]) -> None:
    """Hard-fail if any document in corpus_path is on the blocklist.

    Reads corpus_path as JSONL, one JSON object per line, each with a
    "doc_id" field. Raises ContaminationError listing every offending
    doc_id if any corpus doc_id is in blocklist. This is an assertion, not
    a warning: per CLAUDE.md it must not merely log.

    Comparison is on the normalized string form of doc_id (str(doc_id)),
    since corpus doc_ids for pubmed-sourced docs are the literal
    str(pubid), not a "chunk::" composite.

    Raises:
        ContaminationError: if one or more corpus doc_ids are blocklisted.
    """
    offending: list[str] = []
    with Path(corpus_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            doc_id = str(row["doc_id"])
            if doc_id in blocklist:
                offending.append(doc_id)

    if offending:
        raise ContaminationError(
            f"{len(offending)} blocklisted document id(s) found in {corpus_path}: "
            f"{sorted(set(offending))}"
        )


# --------------------------------------------------------------------------
# MedQA-side overlap check (AUDIT 2026-09-06 §3.2 / §3.6)
# --------------------------------------------------------------------------
#
# MedQA has no source-abstract blocklist -- its items are not generated from
# indexed documents, so `blocklisted_pmids("medqa")` correctly raises. But the
# corpus DOES contain the 18 MedQA USMLE textbooks (ADR-009), i.e. the source
# material the items were written from, which means "no blocklist applies" was
# never the same statement as "no leakage was measured". Until this function
# existed, `scripts/run_baselines.py: contamination_assertion` returned a
# literal `{"checked": True, ...}` dict without executing anything, and that
# unearned `checked: true` was written into all eight `results/medqa_B*/
# metrics.json`, directly violating CLAUDE.md's "assert this in code, do not
# merely comment it".
#
# What is measured here instead: the fraction of a question stem's word
# n-grams that appear VERBATIM in any passage actually retrieved for that
# question. This is a real executed measurement over the real retrieval path,
# reported as a distribution -- not a pass/fail gate. Some overlap is expected
# and disclosed (ADR-009); the number that would indicate a leak is a stem
# reproduced near-verbatim inside a retrieved chunk.

_OVERLAP_NGRAM_N = 8
_OVERLAP_FLAG_THRESHOLD = 0.5


def _word_ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    """Lowercased, whitespace-tokenised word n-grams, punctuation stripped from
    token edges. Returns an empty set when the text has fewer than n tokens
    (a stem too short to n-gram is not evidence of anything either way)."""
    tokens = [t.strip(".,;:!?()[]{}\"'`") for t in text.lower().split()]
    tokens = [t for t in tokens if t]
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def max_ngram_overlap(stem: str, chunk_texts: list[str], n: int = _OVERLAP_NGRAM_N) -> float:
    """Max over `chunk_texts` of |stem n-grams ∩ chunk n-grams| / |stem n-grams|.

    0.0 when the stem yields no n-grams (too short) or no chunk shares one.
    1.0 means every n-gram of the stem occurs verbatim in a single chunk.
    """
    stem_grams = _word_ngrams(stem, n)
    if not stem_grams:
        return 0.0
    best = 0.0
    for text in chunk_texts:
        shared = len(stem_grams & _word_ngrams(text, n))
        if shared:
            best = max(best, shared / len(stem_grams))
    return best


def medqa_overlap_check(
    stems_and_chunks: list[tuple[str, list[str]]],
    *,
    n: int = _OVERLAP_NGRAM_N,
    flag_threshold: float = _OVERLAP_FLAG_THRESHOLD,
) -> dict:
    """Execute the stem-vs-retrieved-chunk overlap check over a whole eval slice.

    `stems_and_chunks` pairs each question stem with the texts of the passages
    actually retrieved for it. Returns the record written into `metrics.json`:
    `checked` is True because this function ran, and `applies` is set from the
    executed outcome (True iff at least one question's overlap reaches
    `flag_threshold`), never from a hardcoded assumption.
    """
    overlaps = [max_ngram_overlap(stem, chunks, n) for stem, chunks in stems_and_chunks]
    n_q = len(overlaps)
    if not n_q:
        return {
            "checked": False,
            "applies": True,
            "note": "medqa_overlap_check called with zero questions; nothing measured",
        }
    ordered = sorted(overlaps)
    flagged = [o for o in overlaps if o >= flag_threshold]
    return {
        "checked": True,
        "applies": bool(flagged),
        "check": "medqa_stem_vs_retrieved_chunk_ngram_overlap",
        "ngram_n": n,
        "flag_threshold": flag_threshold,
        "n_questions": n_q,
        "n_flagged": len(flagged),
        "max_overlap": max(overlaps),
        "mean_overlap": sum(overlaps) / n_q,
        "median_overlap": ordered[n_q // 2],
        "p95_overlap": ordered[min(n_q - 1, int(0.95 * n_q))],
        "n_questions_with_any_shared_ngram": sum(1 for o in overlaps if o > 0.0),
        "note": (
            f"Executed check: fraction of each question stem's {n}-grams appearing verbatim "
            "in any passage retrieved for that question. MedQA has no source-abstract "
            "blocklist (blocklisted_pmids('medqa') raises by design); the 18 USMLE textbooks "
            "are in the corpus disclosed under ADR-009, so non-zero overlap is expected and "
            f"is reported as a distribution. `applies` is True iff >=1 question reaches "
            f"{flag_threshold} verbatim overlap, which would indicate a stem reproduced "
            "inside a retrieved chunk."
        ),
    }


# --------------------------------------------------------------------------
# Calibration-fitting split (AUDIT 2026-09-06 §3.3, ADR-026)
# --------------------------------------------------------------------------
#
# docs/03_DATASETS.md originally reserved `pqa_artificial` for fitting the
# Phase 5 calibration weights. But `configs/base.yaml` also builds the
# retrieval corpus from a 20,000-document `pqa_artificial` subset
# (`data/MANIFEST.json: n_pubmed_docs = 20000`). Those two uses collide: the
# calibration items' own source abstracts would be indexed and retrievable
# while the temperature is fitted on them, which is the structure behind the
# 0.995 Recall@10 saturation ADR-012 observed. A temperature fitted under
# retrieval that good does not transfer to a test question whose source
# abstract is NOT in the index.
#
# ADR-026 resolves this by fitting calibration on the `pqa_labeled` DEV half
# instead. That half is verified below to be absent from the corpus rather
# than assumed: `pqa_labeled` draws from a pubid pool disjoint from
# `pqa_artificial`, and `assert_calibration_fit_set_uncontaminated` executes
# that check against the built corpus.


def calibration_fit_pmids() -> set[str]:
    """The PMIDs Phase 5 may fit calibration weights on: the `pqa_labeled`
    DEV half (500 items), never the test half and never `pqa_artificial`.

    `pubmedqa_split` is the single source of truth for the partition, so the
    fit set is byte-for-byte the complement of the reported test set.
    """
    _test_pmids, dev_pmids = pubmedqa_split()
    return dev_pmids


def assert_calibration_fit_set_uncontaminated(corpus_path: Path) -> None:
    """Hard-fail if any calibration-fitting document is indexed in the corpus.

    This is the executable form of ADR-026: fitting a calibration temperature
    on items whose source abstracts are retrievable inflates the evidence
    scores the temperature is fitted against, and the resulting temperature
    does not transfer to the reported test set. Phase 5 must call this before
    fitting.

    Raises:
        ContaminationError: if one or more calibration-fit PMIDs are in the corpus.
    """
    assert_no_contamination(corpus_path, calibration_fit_pmids())


# --------------------------------------------------------------------------
# Eval-time contamination record for a PubMedQA run (mid-Phase-6 fix)
# --------------------------------------------------------------------------


def pubmedqa_contamination_report(corpus_path: Path) -> dict:
    """The `contamination_assertion` record for a PubMedQA eval run.

    Earlier both `scripts.run_baselines.contamination_assertion` and the Phase 6
    harness returned ``{"checked": False, "applies": True, ...}`` for PubMedQA
    -- knowing the check applied and skipping it, the same failure class as
    AUDIT 2026-09-06 §3.2 (a `checked` that was not earned). This runs the real
    thing: `assert_no_contamination` scans the built corpus for any PMID on the
    `pqa_labeled` TEST blocklist (`blocklisted_pmids('pubmedqa')`, 500 ids).
    ``checked: True`` here means the corpus was actually read.

    Returns a dict (never raises) so a finished eval still writes its
    metrics.json; ``passed`` carries the outcome.
    """
    try:
        blocklist = blocklisted_pmids("pubmedqa")
    except (FileNotFoundError, ValueError) as e:
        return {
            "checked": False,
            "applies": True,
            "note": f"could not build PubMedQA blocklist: {e}",
        }
    if not Path(corpus_path).exists():
        return {
            "checked": False,
            "applies": True,
            "note": f"corpus {corpus_path} not found -- PubMedQA blocklist check not run",
        }
    base = {
        "applies": True,
        "check": "pubmedqa_test_pmid_blocklist_absent_from_corpus",
        "blocklist_size": len(blocklist),
    }
    try:
        assert_no_contamination(Path(corpus_path), blocklist)
        return {**base, "checked": True, "passed": True, "n_offending": 0}
    except ContaminationError as e:
        return {**base, "checked": True, "passed": False, "error": str(e)}
