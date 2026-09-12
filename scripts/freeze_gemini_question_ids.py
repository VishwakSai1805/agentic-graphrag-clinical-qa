"""Freeze the §0.6 Gemini sensitivity check's 200-question set (ADR-047).

Writes `results/phase7_gemini/question_ids_200.json`: the FIRST 200 question
ids of the MedQA test-500 seeded order (seed 42) -- i.e. the order the Phase 7
grid cells themselves ran in -- plus a sha256 over that list, so the launcher
can verify at run time that the set has not drifted.

Why an explicit id list instead of `--limit 200`
------------------------------------------------
`load_question_sample("medqa", "test", 200, seed=42)` is NOT the first 200 of
`load_question_sample("medqa", "test", 500, seed=42)`. `random.Random(42).sample`
draws a different internal sequence for a different `k`, so the two sets
overlap on only **190 of 200** questions -- 10 of the 200 lie outside the 500
entirely and were never run by any 7B cell. Running the Gemini check at
`--limit 200` would therefore silently break the pairing that §1.0d's whole
design rests on: the confirmatory test and P2 both compare Gemini against the
7B runs question by question.

This is the same trap `load_question_sample_excluding`'s docstring warns about,
hit from the other direction. The fix is the same one that module already uses:
select by resolved `question_id`, never by re-sampling.

    uv run python -m scripts.freeze_gemini_question_ids          # write it
    uv run python -m scripts.freeze_gemini_question_ids --check  # verify only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.kg.dataset_questions import load_question_sample  # noqa: E402

OUT_PATH = REPO_ROOT / "results" / "phase7_gemini" / "question_ids_200.json"
N_GEMINI = 200
N_GRID = 500
SEED = 42

# The 7B grid runs the Gemini cells are paired against. Every one of them wrote
# predictions in exactly this 500-question order (verified), so the first 200
# ids are present in all of them.
PAIRED_7B_RUNS = {
    "V0": "medqa_V0_b24015a-dirty_20260911T080323Z",
    "V3": "medqa_V3_41f7e14-dirty_20260908T230633Z",
    "B3": "medqa_B3_41f7e14-dirty_20260909T023658Z",
    "V5": "medqa_V5_41f7e14-dirty_20260909T103452Z",
}


def first_200_ids() -> list[str]:
    grid = [q.question_id for q in load_question_sample("medqa", "test", N_GRID, seed=SEED)]
    if len(grid) != N_GRID:
        raise SystemExit(f"expected {N_GRID} grid questions, got {len(grid)}")
    return grid[:N_GEMINI]


def digest(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def build() -> dict:
    ids = first_200_ids()

    # Every pinned id must exist in every 7B run we intend to pair against --
    # checked here, at freeze time, not discovered during analysis.
    coverage = {}
    for cell, run in PAIRED_7B_RUNS.items():
        path = REPO_ROOT / "results" / run / "predictions.jsonl"
        if not path.exists():
            coverage[cell] = "predictions.jsonl absent (gitignored; present on the run machine)"
            continue
        have = {
            json.loads(line)["question_id"]
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        missing = [q for q in ids if q not in have]
        if missing:
            raise SystemExit(f"{cell} ({run}) is missing {len(missing)} pinned ids: {missing[:5]}")
        coverage[cell] = f"all {len(ids)} pinned ids present"

    return {
        "adr": "ADR-047",
        "purpose": (
            "docs/04_EVALUATION.md §1.0d -- the §0.6 stronger-model sensitivity check's "
            "frozen question set: the first 200 of the MedQA test-500 seeded order "
            "(seed 42), which is the order the Phase 7 grid cells ran in."
        ),
        "dataset": "medqa",
        "split": "test",
        "n": len(ids),
        "seed": SEED,
        "selection": (
            f"first {N_GEMINI} of load_question_sample('medqa','test',{N_GRID},seed={SEED})"
        ),
        "why_not_limit_200": (
            f"load_question_sample('medqa','test',{N_GEMINI},seed={SEED}) overlaps that "
            f"{N_GRID}-question set on only 190/{N_GEMINI} ids -- random.sample draws a "
            "different sequence for a different k, so a smaller limit is a different "
            "draw, not a prefix. Pairing requires naming the ids."
        ),
        "paired_7b_runs": PAIRED_7B_RUNS,
        "paired_7b_coverage": coverage,
        "sha256": digest(ids),
        "question_ids": ids,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify the committed file, write nothing")
    args = ap.parse_args()

    built = build()
    if args.check:
        if not OUT_PATH.exists():
            print(f"[freeze_gemini_question_ids] MISSING {OUT_PATH}")
            return 1
        on_disk = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        if on_disk.get("sha256") != built["sha256"]:
            print(
                "[freeze_gemini_question_ids] DRIFT: committed sha256 "
                f"{on_disk.get('sha256')} != recomputed {built['sha256']}"
            )
            return 1
        print(
            f"[freeze_gemini_question_ids] OK -- {on_disk['n']} ids, "
            f"sha256 {built['sha256'][:12]}"
        )
        return 0

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(built, indent=2) + "\n", encoding="utf-8")
    print(f"[freeze_gemini_question_ids] wrote {OUT_PATH}")
    print(f"  n={built['n']}  sha256={built['sha256']}")
    print(f"  first={built['question_ids'][0]}  last={built['question_ids'][-1]}")
    for cell, note in built["paired_7b_coverage"].items():
        print(f"  {cell}: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
