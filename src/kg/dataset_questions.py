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
    ("pubmedqa", "artificial"): DATA_RAW / "pubmedqa" / "pqa_artificial_train.jsonl",
    ("medmcqa", "validation"): DATA_RAW / "medmcqa" / "validation.jsonl",
    ("medmcqa", "train"): DATA_RAW / "medmcqa" / "train.jsonl",
    ("medmcqa", "test"): DATA_RAW / "medmcqa" / "test.jsonl",
}

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


def load_question_sample(
    dataset: str, split: str, limit: int | None, seed: int = 42
) -> list[Question]:
    """Deterministic fixed-seed sample of up to `limit` questions from
    (dataset, split), in the file's own row order for id assignment (see
    module docstring), then subsampled via `random.Random(seed).sample`.
    `limit=None` returns every row in the split, in file order."""
    rows = _iter_raw_rows(dataset, split)
    n = len(rows)
    if limit is None or limit >= n:
        idxs = list(range(n))
    else:
        idxs = sorted(random.Random(seed).sample(range(n), limit))
    return [_to_question(dataset, split, idx, rows[idx]) for idx in idxs]
