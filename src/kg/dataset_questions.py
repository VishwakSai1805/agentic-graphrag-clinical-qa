"""Loads question sets for the Phase 2 KG scripts (`scripts/precompute_kg_cache.py`
and `scripts/kg_probe.py`) from the raw dataset files `scripts/fetch_datasets.py`
already produced under `data/raw/<dataset>/`.

No Neo4j / scispaCy / torch imports here -- this module is pure file I/O +
deterministic sampling, so both the Neo4j-touching precompute script and the
cache-only probe script can share exactly one definition of "which questions
make up the seeded n=200 (or whatever `--limit`) slice for a given
(dataset, split, seed)". Sharing this one function is what makes
`precompute_kg_cache.py --limit 200` and `kg_probe.py --limit 200` operate
over the *same* 200 questions.

`question_id` scheme: `f"{split}_{idx:05d}"`, `idx` being the question's
0-based position in the raw split file on disk (a stable, deterministic key
as long as the raw file itself doesn't change -- `scripts/fetch_datasets.py`
pins dataset revisions in `data/MANIFEST.json`). The split name is always
folded into the id (even for pubmedqa/medmcqa, which happen to carry their
own natural unique ids -- `pubid`, `id`) so the same scheme works uniformly
across all three datasets and two different splits of the same dataset never
collide under `data/processed/kg_cache/<dataset>/<question_id>.json` (ADR-007).
The natural id, when one exists, is kept on `Question.native_id` for
cross-referencing, not used as the cache key.

Split files wired (see docs/03_DATASETS.md and `data/raw/<dataset>/`):
  medqa:    "test" -> data/raw/medqa_usmle/test.jsonl
            "train" -> data/raw/medqa_usmle/train.jsonl
  pubmedqa: "test" -> data/raw/pubmedqa/pqa_labeled_train.jsonl (the 1k
                       pqa_labeled release; docs/03_DATASETS.md's 500/500
                       test/dev cut is a Phase 3+ eval-harness concern, not
                       reproduced here -- this loads the whole labeled file)
            "artificial" -> data/raw/pubmedqa/pqa_artificial_train.jsonl
  medmcqa:  "validation" -> data/raw/medmcqa/validation.jsonl
            "train" -> data/raw/medmcqa/train.jsonl
            "test" -> data/raw/medmcqa/test.jsonl (unlabeled, `cop` absent)

Honesty note (see the Phase 2 handoff): this session only actually ran the
precompute + probe pipeline end-to-end for medqa/test. pubmedqa and medmcqa
loaders below are wired against real files and covered by
`tests/test_kg_dataset_questions.py` (against small fixture files, not the
real multi-hundred-MB raw datasets), but the precompute/probe scripts
themselves were not run for real against pubmedqa/medmcqa in this session --
see the report for exactly what was and wasn't run.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = REPO_ROOT / "data" / "raw"

_SPLIT_PATHS: dict[tuple[str, str], Path] = {
    ("medqa", "test"): DATA_RAW / "medqa_usmle" / "test.jsonl",
    ("medqa", "train"): DATA_RAW / "medqa_usmle" / "train.jsonl",
    ("pubmedqa", "test"): DATA_RAW / "pubmedqa" / "pqa_labeled_train.jsonl",
    # "dev" does NOT get its own raw file -- see _load_pubmedqa_dev_sample
    # below. It is listed here only so _split_path's error message lists it
    # as a valid split name; load_question_sample special-cases split=="dev"
    # for dataset=="pubmedqa" before this dict is ever consulted for it.
    ("pubmedqa", "dev"): DATA_RAW / "pubmedqa" / "pqa_labeled_train.jsonl",
    ("pubmedqa", "artificial"): DATA_RAW / "pubmedqa" / "pqa_artificial_train.jsonl",
    ("medmcqa", "validation"): DATA_RAW / "medmcqa" / "validation.jsonl",
    ("medmcqa", "train"): DATA_RAW / "medmcqa" / "train.jsonl",
    ("medmcqa", "test"): DATA_RAW / "medmcqa" / "test.jsonl",
}

# docs/03_DATASETS.md: PubMedQA's `pqa_labeled` release (1,000 questions) is
# cut into a disjoint 500-question "test" (standard) + 500-question "dev"
# split, not two independent seeded samples of the whole file (which, at the
# same seed, would draw the IDENTICAL 500 rows for both and violate
# CLAUDE.md's no-test-contamination principle by making "dev" just a copy of
# "test"). "test" keeps its original behavior/code path unchanged (a plain
# seeded sample of the full file, unconditionally) for backward compatibility
# with existing callers/tests that predate this split. "dev" is defined as
# the complement of whatever "test" would draw at the SAME seed and the same
# fixed partition size below, computed independently of whatever --limit is
# passed to the "dev" call -- so the two splits never overlap regardless of
# call order or limit.
_PUBMEDQA_LABELED_PARTITION_SIZE = 500

_MEDMCQA_OPTION_KEYS = ["opa", "opb", "opc", "opd"]
_MEDMCQA_OPTION_LETTERS = ["A", "B", "C", "D"]


class Question(BaseModel):
    """One question, normalised across datasets for the KG scripts."""

    question_id: str
    dataset: str
    split: str
    native_id: str | None = None
    question: str
    options: dict[str, str] | None = None
    answer: str | None = None
    subject_name: str | None = None  # medmcqa only, real gold category (not used by
    # categorize.py, which is for medqa's *absence* of a gold label -- kept here
    # for reference/future use, e.g. comparing the medqa heuristic's buckets
    # against medmcqa's real subject_name distribution as a sanity check.


def _split_path(dataset: str, split: str) -> Path:
    key = (dataset, split)
    if key not in _SPLIT_PATHS:
        valid = sorted({s for d, s in _SPLIT_PATHS if d == dataset})
        raise ValueError(
            f"no wired raw file for dataset={dataset!r} split={split!r}; "
            f"valid splits for {dataset!r}: {valid}"
        )
    return _SPLIT_PATHS[key]


def _iter_raw_rows(dataset: str, split: str) -> list[dict]:
    path = _split_path(dataset, split)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist -- run scripts/fetch_datasets.py first "
            f"(dataset={dataset!r} split={split!r})."
        )
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _to_question(dataset: str, split: str, idx: int, row: dict) -> Question:
    question_id = f"{split}_{idx:05d}"
    if dataset == "medqa":
        return Question(
            question_id=question_id,
            dataset=dataset,
            split=split,
            native_id=None,
            question=row["question"],
            options=row.get("options"),
            answer=row.get("answer"),
        )
    if dataset == "pubmedqa":
        return Question(
            question_id=question_id,
            dataset=dataset,
            split=split,
            native_id=str(row.get("pubid")),
            question=row["question"],
            options=None,
            answer=row.get("final_decision"),
        )
    if dataset == "medmcqa":
        options = {
            letter: row[key]
            for letter, key in zip(_MEDMCQA_OPTION_LETTERS, _MEDMCQA_OPTION_KEYS, strict=True)
            if row.get(key) is not None
        }
        cop = row.get("cop")
        answer = (
            _MEDMCQA_OPTION_LETTERS[cop]
            if isinstance(cop, int) and 0 <= cop < len(_MEDMCQA_OPTION_LETTERS)
            else None
        )
        return Question(
            question_id=question_id,
            dataset=dataset,
            split=split,
            native_id=str(row.get("id")),
            question=row["question"],
            options=options or None,
            answer=answer,
            subject_name=row.get("subject_name"),
        )
    raise ValueError(f"unknown dataset {dataset!r}")


def link_text_for_question(q: Question) -> str:
    """Text handed to `src.kg.linker.link_entities`. For MCQ datasets
    (medqa, medmcqa) this is the question stem *plus* all answer options --
    concatenating options measurably widens entity coverage for KG linking
    (the correct/distractor option text often names the very
    disease/drug/phenotype the question is testing), at the cost of also
    linking entities that only appear in *wrong* options. Documented, not
    hidden: this is a deliberate precision/coverage tradeoff for a KG-hit-rate
    diagnostic, not a claim that every linked entity is "the" answer entity.
    For pubmedqa (yes/no/maybe over an abstract, no MCQ options), only the
    question stem is used -- the source abstract is not linked over here.
    """
    if q.options:
        return q.question + " " + " ".join(q.options.values())
    return q.question


def gold_letter(q: Question) -> str | None:
    """The gold option letter. medmcqa's `Question.answer` is already a letter;
    medqa's is the option *text*, matched back to its letter (asserting exactly
    one match). pubmedqa has no letter (yes/no/maybe) -> None."""
    if q.answer is None:
        return None
    if q.dataset == "medmcqa":
        return q.answer if q.options and q.answer in q.options else None
    if q.dataset == "medqa" and q.options:
        hits = [k for k, v in q.options.items() if v.strip() == q.answer.strip()]
        if len(hits) != 1:
            raise ValueError(
                f"{q.question_id}: gold text matched {len(hits)} options (expected 1)"
            )
        return hits[0]
    return None


def load_question_sample(
    dataset: str, split: str, limit: int | None, seed: int = 42
) -> list[Question]:
    """Deterministic fixed-seed sample of up to `limit` questions from
    (dataset, split), in the file's own row order for id assignment (see
    module docstring), then subsampled via `random.Random(seed).sample`.
    `limit=None` returns every row in the split, in file order.

    Special case: dataset="pubmedqa", split="dev" does not read a "dev" raw
    file (none exists) -- see `_load_pubmedqa_dev_sample` and the note above
    `_PUBMEDQA_LABELED_PARTITION_SIZE`."""
    if dataset == "pubmedqa" and split == "dev":
        return _load_pubmedqa_dev_sample(limit, seed)
    rows = _iter_raw_rows(dataset, split)
    n = len(rows)
    if limit is None or limit >= n:
        idxs = list(range(n))
    else:
        idxs = sorted(random.Random(seed).sample(range(n), limit))
    return [_to_question(dataset, split, idx, rows[idx]) for idx in idxs]


def load_question_sample_excluding(
    dataset: str, split: str, limit: int, seed: int, exclude_ids: set[str]
) -> list[Question]:
    """Like `load_question_sample`, but draws only from rows whose
    `question_id` is not in `exclude_ids` -- guarantees disjointness from an
    earlier sample (e.g. Phase 5's calibration slice must not overlap Phase
    3/4's dev-20) BY CONSTRUCTION.

    This does NOT reduce to calling `load_question_sample` with a bigger
    `limit` and assuming the smaller sample nests inside it: `random.sample`
    with a fixed seed but a different `k` draws a genuinely different
    sequence internally, so `sample(range(n), 20)` is not a subset of
    `sample(range(n), 120)` at the same seed in general. Excluding by
    resolved `question_id` (not by index arithmetic on the RNG call
    sequence) is the only way to make the guarantee hold regardless of how
    the excluded sample was itself produced -- the same principle
    `_load_pubmedqa_dev_sample` already uses for test/dev disjointness,
    applied here by id instead of by index set."""
    rows = _iter_raw_rows(dataset, split)
    candidate_idxs = [
        i
        for i in range(len(rows))
        if _to_question(dataset, split, i, rows[i]).question_id not in exclude_ids
    ]
    if limit >= len(candidate_idxs):
        idxs = candidate_idxs
    else:
        idxs = sorted(random.Random(seed).sample(candidate_idxs, limit))
    return [_to_question(dataset, split, idx, rows[idx]) for idx in idxs]


def _load_pubmedqa_dev_sample(limit: int | None, seed: int) -> list[Question]:
    """The 500-question complement of pubmedqa's "test" split (see the note
    above `_PUBMEDQA_LABELED_PARTITION_SIZE`), both drawn from the same
    `pqa_labeled_train.jsonl` (1,000 rows). Recomputes exactly what
    `load_question_sample("pubmedqa", "test", limit=_PUBMEDQA_LABELED_PARTITION_SIZE,
    seed=seed)` would have drawn, then returns everything else -- so "test"
    and "dev" are guaranteed disjoint at a given seed without either split
    needing to know about the other's actual runtime `--limit`."""
    rows = _iter_raw_rows("pubmedqa", "test")  # pqa_labeled_train.jsonl
    n = len(rows)
    test_size = min(_PUBMEDQA_LABELED_PARTITION_SIZE, n)
    test_idxs = set(random.Random(seed).sample(range(n), test_size))
    dev_idxs = sorted(set(range(n)) - test_idxs)
    if limit is not None and limit < len(dev_idxs):
        idxs = sorted(random.Random(seed).sample(dev_idxs, limit))
    else:
        idxs = dev_idxs
    return [_to_question("pubmedqa", "dev", idx, rows[idx]) for idx in idxs]
