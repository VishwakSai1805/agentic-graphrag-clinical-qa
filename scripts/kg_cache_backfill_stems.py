"""Backfill `question_stem` into pre-existing KG cache files (AUDIT §6.5).

`KGCacheEntry.question_stem` (added 2026-09-08) lets `get_subgraph_cached`
detect a cache that has been mis-associated with the wrong question by a
dataset re-fetch that reordered rows. Files built before that field have
`question_stem == ""` and are read without the check.

This CPU-only script (no Neo4j) walks every cache tree, looks up the question
now sitting at each file's `question_id` in the current dataset fetch, and —
**only if an integrity check passes** — writes the stem in. The integrity
check: at least half of the file's distinct linked-entity surface strings
(captured from the *original* question at build time) must appear verbatim in
the current question text. A file that fails is left untouched and reported —
that failure IS the §6.5 condition (the id↔question binding has drifted).

    uv run python -m scripts.kg_cache_backfill_stems            # dry run
    uv run python -m scripts.kg_cache_backfill_stems --write

After --write, refresh the checksum manifest:
    uv run python -m scripts.kg_cache_manifest --write
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.kg.dataset_questions import (  # noqa: E402
    Question,
    link_text_for_question,
    load_question_sample,
)
from src.kg.traversal import KGCacheEntry, _norm_stem, write_subgraph_cache  # noqa: E402

CACHE_ROOTS = [
    REPO_ROOT / "data" / "processed" / "kg_cache",
    REPO_ROOT / "data" / "processed" / "kg_cache_kg_rerank",
]
# cache filename prefix -> (dataset, split) for load_question_sample
SPLIT_OF_PREFIX = {
    "test": "test",
    "train": "train",
    "dev": "dev",
    "validation": "validation",
}
_MIN_ENTITY_MATCH_FRAC = 0.5


def _question_maps() -> dict[tuple[str, str], dict[str, Question]]:
    """(dataset, split) -> {question_id: Question} for the current fetch."""
    maps: dict[tuple[str, str], dict[str, Question]] = {}
    for root in CACHE_ROOTS:
        if not root.exists():
            continue
        for dataset_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            dataset = dataset_dir.name
            prefixes = {f.name.rsplit("_", 1)[0] for f in dataset_dir.glob("*.json")}
            for prefix in prefixes:
                split = SPLIT_OF_PREFIX.get(prefix, prefix)
                key = (dataset, split)
                if key in maps:
                    continue
                qs = load_question_sample(dataset, split, None, seed=42)
                maps[key] = {q.question_id: q for q in qs}
                print(f"[backfill] loaded {len(maps[key])} {dataset}/{split} questions")
    return maps


def _entity_check(entry: KGCacheEntry, question: Question) -> tuple[bool, float]:
    """Fraction of the file's distinct linked-entity surface strings that
    appear verbatim in the current question's *link text* (stem + options —
    the exact text `precompute_kg_cache` linked against). >= 0.5 -> the
    id<->question binding is intact."""
    surfaces = {e.text.strip().lower() for e in entry.linked_entities if e.text.strip()}
    if not surfaces:
        return True, 1.0  # KG linking found nothing -> nothing to check against
    haystack = link_text_for_question(question).lower()
    matched = sum(1 for s in surfaces if s in haystack)
    frac = matched / len(surfaces)
    return frac >= _MIN_ENTITY_MATCH_FRAC, frac


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="actually write (default: dry run)")
    args = ap.parse_args()

    maps = _question_maps()
    n_files = n_written = n_already = n_missing_q = n_integrity_fail = 0
    failures: list[str] = []

    for root in CACHE_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.json")):
            n_files += 1
            dataset = path.parent.name
            prefix = path.stem.rsplit("_", 1)[0]
            split = SPLIT_OF_PREFIX.get(prefix, prefix)
            entry = KGCacheEntry.model_validate_json(path.read_text(encoding="utf-8"))
            question = maps.get((dataset, split), {}).get(entry.question_id)
            if question is None:
                n_missing_q += 1
                failures.append(f"{path}: no question at {dataset}/{split}/{entry.question_id}")
                continue
            stem = question.question
            if entry.question_stem and _norm_stem(entry.question_stem) == _norm_stem(stem):
                n_already += 1
                continue
            if entry.question_stem:
                n_integrity_fail += 1
                failures.append(
                    f"{path}: EXISTING stem disagrees with current question -- §6.5 drift"
                )
                continue
            ok, frac = _entity_check(entry, question)
            if not ok:
                n_integrity_fail += 1
                failures.append(
                    f"{path}: only {frac:.0%} of linked entities appear in the current "
                    f"question link-text -- binding suspect, NOT backfilled"
                )
                continue
            if args.write:
                write_subgraph_cache(
                    entry.model_copy(update={"question_stem": stem}),
                    dataset,
                    path.parent.parent,
                )
            n_written += 1

    print("\n" + "=" * 70)
    print(f"cache files scanned      : {n_files}")
    print(f"already had a valid stem : {n_already}")
    print(f"{'written' if args.write else 'would write'} question_stem : {n_written}")
    print(f"question_id not in fetch  : {n_missing_q}")
    print(f"integrity-check failures  : {n_integrity_fail}")
    if failures:
        print("\nFAILURES (these are the §6.5 condition -- investigate before Phase 7):")
        for f in failures[:50]:
            print(f"  {f}")
        if len(failures) > 50:
            print(f"  ... and {len(failures) - 50} more")
    if not args.write:
        print("\n(dry run -- pass --write to apply, then scripts.kg_cache_manifest --write)")
    sys.exit(1 if (n_missing_q or n_integrity_fail) else 0)


if __name__ == "__main__":
    main()
