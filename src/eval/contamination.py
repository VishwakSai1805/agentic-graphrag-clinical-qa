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
