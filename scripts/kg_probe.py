"""Phase 2 exit test (docs/02_ROADMAP.md).

**Revision note (this version corrects four methodological problems the
project owner found in the first real run's report -- see
`docs/05_DECISIONS.md`, the ADR this revision adds. The underlying infra
(Neo4j/PrimeKG load, linker, traversal, cache) is unchanged and was not
re-run; only this script's measurement/reporting logic changed):**

**Second revision note (2026-09-05 Phase 2 follow-up, ADR-016 addendum):**
the novelty/relevance scored-triples-per-question subset
(`NOVELTY_MAX_TRIPLES_SCORED=10`) was previously selected by a deterministic
**alphabetical sort on (head, relation, tail), then first-10** -- ADR-016
already flagged that this over-represents alphabetically-early entity names
(e.g. "Ac-" drug names like Acitretin/Acyclovir/Alfentanil recurring across
unrelated questions' scored subsets is visible directly in the sample
subgraphs). This revision replaces that with a **seeded random sample** of up
to 10 triples per question (`_select_scored_triples`), seeded per-question
from a hash of `(cfg.seed, question_id)` so a full rerun reproduces the exact
same scored subset byte-for-byte (not just "seeded once globally", which
would make the subset order-dependent on how many questions came before it in
the loop -- see `_select_scored_triples`'s docstring). This changes
`overall_novel_relation_rate` / `overall_relevant_rate` /
`overall_novel_and_relevant_rate` from the previous alphabetical-selection
run in an as-measured direction (see the Phase 2 follow-up report for the
before/after numbers) -- not tuned to land anywhere in particular.

1. The old headline `aggregate_hit_rate` (~0.96) used a permissive
   "question-level, >=1 mapped entity" hit definition that is close to an
   arithmetic guarantee given ~74 linked entities/question and an 11.2%
   per-entity mapping rate. This version reports `entity_level_mapping_rate`
   and `naive_expected_hit_rate_under_independence` alongside it, plus an
   explicit `hit_definition` string, so a reader cannot mistake the
   question-level rate for "PrimeKG covers 96% of MedQA content."
2. The old per-category `n_triples` field was actually
   `NOVELTY_MAX_TRIPLES_SCORED` (a scoring-cost cap) echoed back per hit --
   every category's `n_triples/n_hits` ratio was exactly 10.0, not a real
   measurement. This version reports the REAL pre-cap (but post-ADR-015
   40-cap) triples-per-question distribution, read directly from the cache
   files on disk, instead.
3. Novelty was scored against only the top-3 dense passages, not the real
   12-item post-rerank `EvidenceBundle` (`configs/base.yaml:
   retrieval.top_k_rerank`) the agent layer will actually see -- this
   systematically inflates apparent novelty. This version reads
   `retrieval.top_k_rerank` from config (not hardcoded) for the dense side.
   It also adds a relevance-conditioned figure (`novel_and_relevant_rate`)
   since "novel" alone conflates "clinically specific" with "the NLI model
   didn't recognize an irrelevant hub fact."
4. `anatomy` (n=8) and any other category with n < `MIN_CATEGORY_N_FOR_RATE`
   is too small to report a standalone rate. This version suppresses (sets to
   null) the derived *rates* for underpowered categories while keeping their
   raw counts visible.

Reports, for real, on a fixed-seed slice of MedQA test questions:
  - `aggregate_hit_rate` (permissive, question-level definition, disclosed via
    `hit_definition`) alongside `entity_level_mapping_rate` (the real
    per-entity resolution rate) and `naive_expected_hit_rate_under_independence`
    (what the question-level rate would be if it were a pure arithmetic
    consequence of the entity-level rate and mean entities/question).
  - hit rate broken down by `src.kg.categorize.categorize_medqa_question`
    category, always shown with its `n`; categories with n < 15 have their
    rates suppressed (see point 4 above).
  - the real triples-per-question distribution (min/median/mean/max), overall
    and per category, read directly from `KGCacheEntry.triples` on disk --
    the true post-ADR-015-40-cap count, not a novelty-scoring artifact.
  - `novel_relation_rate` (re-scored at the real `retrieval.top_k_rerank`
    dense passages, see point 3) and `novel_and_relevant_rate` /
    `relevant_rate` / `novel_given_relevant_rate`, overall and per category.
  - unmapped-CUI count/rate from the linker (ADR-014: this explains hit-rate
    gaps, not swept under the rug).
  - 10 real sample subgraphs, human-readable.

ADR-007 boundary, made structural rather than just documented: this script
NEVER imports anything from the `neo4j` package, directly or transitively --
it only reads `data/processed/kg_cache/<dataset>/<question_id>.json` via
`src.kg.traversal.get_subgraph_cached`. If a question's cache entry is
missing, that is reported as a cache-miss (counted separately from a real
KG-miss) and the question is treated as a KG-miss for the hit-rate
computation -- this script never falls back to a live Neo4j call to fill the
gap. Concretely: with the full precomputed cache in place, `docker compose
-f docker/compose.yaml down` followed by a rerun of this exact command is
guaranteed to touch nothing Neo4j-related, by construction, not by
assumption. `--from-cache-only` is accepted for CLI-compatibility with the
roadmap's phrasing but is always effectively true and changes nothing --
documented here rather than silently implemented as a dead flag.

**Third revision note (2026-09-06, kg_rerank significance follow-up):** the
output JSON now always carries a `per_question` field -- one record per
question (question_id, category, cache_missing, is_hit,
n_triples_total/scored/novel/relevant/novel_and_relevant, and the three
per-question rates `novel_relation_rate_for_question` /
`relevant_rate_for_question` / `novel_and_relevant_rate_for_question`, `None`
when nothing was scored for that question). This is distinct from
`sample_subgraphs` (only the first `N_SAMPLE_SUBGRAPHS=10` questions with
triples, human-readable, includes full triple lists) -- `per_question` covers
all `n_questions` questions but only the aggregate counts/rates, and exists
specifically so `scripts/kg_rerank_ablation.py` can pair two variants'
per-question rates by `question_id` and run `paired_bootstrap_ci`
(`src/retrieval/metrics.py`) instead of comparing only aggregate means (see
`docs/05_DECISIONS.md` ADR-012's precedent, and the ADR this revision adds).

Usage:
    uv run python -m scripts.kg_probe --dataset medqa --limit 200
    uv run python -m scripts.kg_probe --dataset medqa --limit 200 --from-cache-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config  # noqa: E402
from src.kg.categorize import categorize_medqa_question  # noqa: E402
from src.kg.dataset_questions import load_question_sample  # noqa: E402
from src.kg.novelty import (  # noqa: E402
    NLI_MODEL_NAME,
    get_default_scorer,
    is_novel_relation,
    is_triple_relevant,
)
from src.kg.traversal import get_subgraph_cached  # noqa: E402

CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"
DENSE_DIR = REPO_ROOT / "indexes" / "dense"
CORPUS_PATH = REPO_ROOT / "data" / "processed" / "corpus.jsonl"
RESULTS_DIR = REPO_ROOT / "results" / "phase2_kg_probe"

HIT_RATE_FLOOR = 0.60
HIT_DEFINITION = (
    ">=1 of the question's linked entities resolves to a PrimeKG node (kg_node_id "
    "is not None) AND >=1 triple is retrieved from that node's subgraph. This is a "
    "PERMISSIVE, question-level definition: with ~74 linked entities/question and "
    "only ~11% of individual linked entities resolving (see entity_level_mapping_rate), "
    "a question-level hit is close to an arithmetic guarantee (see "
    "naive_expected_hit_rate_under_independence) rather than evidence that PrimeKG "
    "covers 96% of MedQA content. Read aggregate_hit_rate together with "
    "entity_level_mapping_rate, never alone."
)

NOVELTY_MAX_TRIPLES_SCORED = 10  # per-question cap on how many of that question's
# (already kg.max_triples_per_question=40-capped) triples get scored for novelty.
# Real, disclosed compute-budget cut (ADR-016): scoring all ~38-40 triples/question
# against all top-k dense passages for n=200 is a background-job-scale cost, not
# tractable in-session. This cap is unchanged by this revision -- only the dense
# side (novelty_top_k_dense, now sourced from retrieval.top_k_rerank instead of a
# smaller hardcoded value) was corrected. Triples are selected via a seeded random
# sample (see `_select_scored_triples`), not the earlier alphabetical-first-N
# selection (ADR-016 addendum, 2026-09-05 follow-up) -- reproducible per-question
# via a seed derived from (cfg.seed, question_id), not a fixed sort order.
N_SAMPLE_SUBGRAPHS = 10
MIN_CATEGORY_N_FOR_RATE = 15  # categories with fewer questions than this have their
# derived RATES (not raw counts) suppressed to null in the output -- a rate on a
# thin sample (e.g. anatomy, n=8) reads as more precise than it is.


def print_gpu_status() -> None:
    available = torch.cuda.is_available()
    print(f"torch.cuda.is_available() = {available}")
    if available:
        print(f"torch.cuda.get_device_name(0) = {torch.cuda.get_device_name(0)}")
        print(f"torch.cuda.get_device_capability(0) = {torch.cuda.get_device_capability(0)}")


def load_chunk_texts(corpus_path: Path) -> dict[str, str]:
    chunk_to_text: dict[str, str] = {}
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            chunk_to_text[row["chunk_id"]] = row["text"]
    return chunk_to_text


def _distribution(values: list[int]) -> dict[str, float | int | None]:
    """min/median/mean/max over a list of ints; all None if the list is empty."""
    if not values:
        return {"n": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "n": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "max": max(values),
    }


def _rate_or_none(numerator: int, denominator: int, underpowered: bool) -> float | None:
    if underpowered or denominator == 0:
        return None
    return numerator / denominator


def _question_rate(numerator: int, denominator: int) -> float | None:
    """Same numerator/denominator rate as `_rate_or_none` but for a SINGLE
    question's scored-triple counts (no category-level `underpowered`
    suppression concept applies at n=1 question) -- None (not 0.0) when
    denominator is 0, i.e. this question had no triples scored for novelty,
    so the pairing step (`scripts/kg_rerank_ablation.py`) can distinguish
    "this question contributed zero relevant/novel triples out of some
    scored" from "this question had nothing scored at all" rather than
    silently coercing both to 0.0."""
    if denominator == 0:
        return None
    return numerator / denominator


def _select_scored_triples(triples: list, question_id: str, global_seed: int, max_n: int) -> list:
    """Seeded random sample of up to `max_n` of `triples` (ADR-016 addendum,
    2026-09-05 follow-up) -- replaces the earlier deterministic alphabetical
    first-N selection, which ADR-016 disclosed as over-representing
    alphabetically-early entity names (e.g. "Ac-" drugs) across unrelated
    questions' scored subsets.

    Reproducible byte-for-byte on rerun: each question gets its OWN
    `random.Random` instance seeded from a hash of `(global_seed,
    question_id)` -- not one `random.Random(global_seed)` shared/advanced
    across the whole question loop, which would make a given question's
    scored subset depend on how many other questions (and how many triples
    each had) were processed before it in the loop, i.e. order-dependent and
    not truly per-question-reproducible in isolation. A plain Python `hash()`
    is not used for the per-question seed because `str` hashing is
    randomized per-process by default (`PYTHONHASHSEED`) -- `hashlib.sha256`
    is used instead for a stable, process-independent derived seed.

    If `len(triples) <= max_n`, returns all of them (sorted by
    (head, relation, tail) for readable/deterministic output -- there is
    nothing to actually select between)."""
    if len(triples) <= max_n:
        return sorted(triples, key=lambda t: (t.head, t.relation, t.tail))
    digest = hashlib.sha256(f"{global_seed}:{question_id}".encode()).hexdigest()
    per_question_seed = int(digest[:16], 16)
    rng = random.Random(per_question_seed)
    selected = rng.sample(triples, max_n)
    return sorted(selected, key=lambda t: (t.head, t.relation, t.tail))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["medqa"], default="medqa")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument(
        "--from-cache-only",
        action="store_true",
        help="accepted for CLI-compatibility with the roadmap's phrasing; this script is "
        "ALWAYS cache-only regardless of this flag (see module docstring) -- it never "
        "imports anything neo4j-related.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output JSON path (default: results/phase2_kg_probe/<dataset>_n<limit>.json)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="override kg.cache_dir (default: configs/base.yaml's kg.cache_dir). Used e.g. to "
        "point this script at a separate kg_rerank-ablation cache directory "
        "(data/processed/kg_cache_kg_rerank/<dataset>/) without touching the canonical "
        "data/processed/kg_cache/ default.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 2 KG probe (docs/02_ROADMAP.md exit test) -- corrected reporting")
    print("=" * 70)
    print_gpu_status()

    cfg = load_config(CONFIG_PATH)
    kg_cfg = cfg.kg
    novelty_top_k_dense = cfg.retrieval.top_k_rerank  # real EvidenceBundle size the
    # agent layer will see (docs/01_ARCHITECTURE.md) -- NOT a smaller hardcoded value.

    if not DENSE_DIR.exists():
        print(f"[kg_probe] ERROR: dense index not found at {DENSE_DIR}.", file=sys.stderr)
        sys.exit(2)
    if not CORPUS_PATH.exists():
        print(f"[kg_probe] ERROR: corpus not found at {CORPUS_PATH}.", file=sys.stderr)
        sys.exit(2)

    questions = load_question_sample(args.dataset, args.split, args.limit, seed=cfg.seed)
    print(
        f"[kg_probe] dataset={args.dataset} split={args.split} n_questions={len(questions)} "
        f"seed={cfg.seed} novelty_top_k_dense={novelty_top_k_dense} "
        f"(from retrieval.top_k_rerank)"
    )

    print(f"[kg_probe] loading dense index from {DENSE_DIR} ...")
    from src.retrieval.dense import DenseIndex

    dense = DenseIndex.load(DENSE_DIR)
    chunk_to_text = load_chunk_texts(CORPUS_PATH)

    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(kg_cfg.cache_dir)
    print(f"[kg_probe] reading KG cache from {cache_dir} ...")

    # Per-question bookkeeping.
    n_cache_missing = 0
    n_hit = 0
    total_linked_entities = 0
    total_unmapped = 0
    distinct_unmapped_cuis: set[str] = set()

    category_n: dict[str, int] = defaultdict(int)
    category_hits: dict[str, int] = defaultdict(int)
    category_triples_per_q: dict[str, list[int]] = defaultdict(list)  # REAL, full,
    # post-40-cap triples-per-question counts (not the novelty-scoring subset).
    category_n_scored: dict[str, int] = defaultdict(int)
    category_novel: dict[str, int] = defaultdict(int)
    category_relevant: dict[str, int] = defaultdict(int)
    category_novel_and_relevant: dict[str, int] = defaultdict(int)

    all_triples_per_q: list[int] = []  # REAL distribution, overall.

    n_triples_scored_total = 0
    n_triples_novel = 0
    n_triples_relevant = 0
    n_triples_novel_and_relevant = 0

    samples: list[dict] = []
    per_question: list[dict] = []  # one record per question, ALWAYS populated (not just the
    # N_SAMPLE_SUBGRAPHS=10 human-readable samples above) -- this is what lets a downstream
    # script (scripts/kg_rerank_ablation.py) pair per-question rates across two variants by
    # question_id and run paired_bootstrap_ci, instead of only comparing aggregate means (see
    # docs/05_DECISIONS.md ADR-012's precedent: never compare two systems by raw mean
    # inequality alone).
    scorer = None  # lazy: only load the NLI model once we know we have >=1 triple to score

    t0 = time.time()
    for i, q in enumerate(questions):
        category = categorize_medqa_question(q.question)
        category_n[category] += 1

        entry = get_subgraph_cached(args.dataset, q.question_id, cache_dir)
        if entry is None:
            n_cache_missing += 1
            per_question.append(
                {
                    "question_id": q.question_id,
                    "category": category,
                    "cache_missing": True,
                    "is_hit": False,
                    "n_triples_total": 0,
                    "n_triples_scored": 0,
                    "n_novel": 0,
                    "n_relevant": 0,
                    "n_novel_and_relevant": 0,
                    "novel_relation_rate_for_question": None,
                    "relevant_rate_for_question": None,
                    "novel_and_relevant_rate_for_question": None,
                }
            )
            continue

        total_linked_entities += len(entry.linked_entities)
        unmapped_here = [e.cui for e in entry.linked_entities if e.kg_node_id is None]
        total_unmapped += len(unmapped_here)
        distinct_unmapped_cuis.update(unmapped_here)

        n_resolved = sum(1 for e in entry.linked_entities if e.kg_node_id is not None)
        is_hit = n_resolved > 0 and len(entry.triples) > 0
        if is_hit:
            n_hit += 1
            category_hits[category] += 1

        # REAL pre-cap-by-novelty-subset triples-per-question count -- read
        # straight off the cache entry, no NLI-scoring subset involved. This
        # is the true post-ADR-015-40-cap volume of retrieved evidence.
        all_triples_per_q.append(len(entry.triples))
        category_triples_per_q[category].append(len(entry.triples))

        # Novelty diagnostic: only meaningful when there are triples to judge.
        # Seeded-random subsample (see `_select_scored_triples` and the
        # ADR-016 addendum docstring above) -- reproducible per-question via a
        # seed derived from (cfg.seed, question_id), not the earlier
        # alphabetical-first-N selection. This bounded subset is a disclosed
        # compute-cost cut, unchanged in SIZE by this revision; only the
        # selection method (this revision) and novelty_top_k_dense (the
        # earlier revision) were corrected.
        scored_triples = _select_scored_triples(
            entry.triples, q.question_id, cfg.seed, NOVELTY_MAX_TRIPLES_SCORED
        )
        novel_flags: list[bool] = []
        relevant_flags: list[bool] = []
        n_scored_here = 0
        n_novel_here = 0
        n_relevant_here = 0
        n_novel_and_relevant_here = 0
        if scored_triples:
            if scorer is None:
                print(f"[kg_probe] loading NLI model {NLI_MODEL_NAME} for novelty scoring ...")
                scorer = get_default_scorer()
            dense_hits = dense.search(q.question, novelty_top_k_dense)
            passages = [chunk_to_text[cid] for cid, _score in dense_hits if cid in chunk_to_text]
            for triple in scored_triples:
                novel = is_novel_relation(triple, passages, scorer=scorer)
                relevant = is_triple_relevant(triple, q.question, q.options)
                novel_flags.append(novel)
                relevant_flags.append(relevant)
            n_scored_here = len(scored_triples)
            n_novel_here = sum(novel_flags)
            n_relevant_here = sum(relevant_flags)
            n_novel_and_relevant_here = sum(
                1 for nv, rv in zip(novel_flags, relevant_flags, strict=True) if nv and rv
            )

            n_triples_scored_total += n_scored_here
            n_triples_novel += n_novel_here
            n_triples_relevant += n_relevant_here
            n_triples_novel_and_relevant += n_novel_and_relevant_here

            category_n_scored[category] += n_scored_here
            category_novel[category] += n_novel_here
            category_relevant[category] += n_relevant_here
            category_novel_and_relevant[category] += n_novel_and_relevant_here

        per_question.append(
            {
                "question_id": q.question_id,
                "category": category,
                "cache_missing": False,
                "is_hit": is_hit,
                "n_triples_total": len(entry.triples),
                "n_triples_scored": n_scored_here,
                "n_novel": n_novel_here,
                "n_relevant": n_relevant_here,
                "n_novel_and_relevant": n_novel_and_relevant_here,
                "novel_relation_rate_for_question": _question_rate(n_novel_here, n_scored_here),
                "relevant_rate_for_question": _question_rate(n_relevant_here, n_scored_here),
                "novel_and_relevant_rate_for_question": _question_rate(
                    n_novel_and_relevant_here, n_scored_here
                ),
            }
        )

        if len(samples) < N_SAMPLE_SUBGRAPHS and entry.triples:
            samples.append(
                {
                    "question_id": q.question_id,
                    "category": category,
                    "question": q.question,
                    "linked_entities": [
                        {
                            "text": e.text,
                            "cui": e.cui,
                            "kg_node_id": e.kg_node_id,
                            "kg_node_type": e.kg_node_type,
                            "match_method": e.match_method,
                        }
                        for e in entry.linked_entities
                    ],
                    "n_triples_total": len(entry.triples),
                    "triples": [
                        {
                            "head": t.head,
                            "relation": t.relation,
                            "tail": t.tail,
                            "novel": bool(nv),
                            "relevant": bool(rv),
                        }
                        for t, nv, rv in zip(
                            scored_triples, novel_flags, relevant_flags, strict=True
                        )
                    ],
                    "triples_note": (
                        f"showing the {len(scored_triples)}/{len(entry.triples)} triples "
                        "actually scored for novelty/relevance (see NOVELTY_MAX_TRIPLES_SCORED); "
                        "the rest of entry.triples exist in the cache but were not scored "
                        "in this run."
                        if len(scored_triples) < len(entry.triples)
                        else "all triples for this question were scored."
                    ),
                    "evidence_text": [ev.text for ev in entry.evidence],
                }
            )

        if (i + 1) % 20 == 0 or (i + 1) == len(questions):
            print(
                f"[kg_probe] {i + 1}/{len(questions)} questions processed "
                f"({time.time() - t0:.1f}s elapsed)",
                flush=True,
            )

    elapsed_s = time.time() - t0
    n_scored = len(questions) - n_cache_missing
    aggregate_hit_rate = n_hit / len(questions) if questions else 0.0
    overall_novel_rate = n_triples_novel / n_triples_scored_total if n_triples_scored_total else 0.0
    overall_relevant_rate = (
        n_triples_relevant / n_triples_scored_total if n_triples_scored_total else 0.0
    )
    overall_novel_and_relevant_rate = (
        n_triples_novel_and_relevant / n_triples_scored_total if n_triples_scored_total else 0.0
    )
    overall_novel_given_relevant_rate = (
        n_triples_novel_and_relevant / n_triples_relevant if n_triples_relevant else None
    )

    entity_level_mapping_rate = (
        (total_linked_entities - total_unmapped) / total_linked_entities
        if total_linked_entities
        else 0.0
    )
    mean_linked_entities_per_question = total_linked_entities / n_scored if n_scored else 0.0
    naive_expected_hit_rate_under_independence = (
        1.0 - (1.0 - entity_level_mapping_rate) ** mean_linked_entities_per_question
        if n_scored
        else 0.0
    )

    category_breakdown = {}
    for category in sorted(category_n):
        n = category_n[category]
        underpowered = n < MIN_CATEGORY_N_FOR_RATE
        hits = category_hits[category]
        n_scored_cat = category_n_scored[category]
        n_nov = category_novel[category]
        n_rel = category_relevant[category]
        n_nov_rel = category_novel_and_relevant[category]
        category_breakdown[category] = {
            "n": n,
            "underpowered": underpowered,
            "n_hits": hits,
            "hit_rate": _rate_or_none(hits, n, underpowered),
            "triples_per_question": _distribution(category_triples_per_q[category]),
            "n_triples_scored_for_novelty": n_scored_cat,
            "n_novel": n_nov,
            "novel_relation_rate": _rate_or_none(n_nov, n_scored_cat, underpowered),
            "n_relevant": n_rel,
            "relevant_rate": _rate_or_none(n_rel, n_scored_cat, underpowered),
            "n_novel_and_relevant": n_nov_rel,
            "novel_and_relevant_rate": _rate_or_none(n_nov_rel, n_scored_cat, underpowered),
            "novel_given_relevant_rate": _rate_or_none(n_nov_rel, n_rel, underpowered),
        }

    unmapped_rate = total_unmapped / total_linked_entities if total_linked_entities else 0.0

    gate_passed = aggregate_hit_rate >= HIT_RATE_FLOOR

    result = {
        "dataset": args.dataset,
        "split": args.split,
        "seed": cfg.seed,
        "n_questions": len(questions),
        "n_cache_missing": n_cache_missing,
        "n_scored": n_scored,
        "hit_definition": HIT_DEFINITION,
        "aggregate_hit_rate": aggregate_hit_rate,
        "hit_rate_floor": HIT_RATE_FLOOR,
        "hit_rate_floor_passed": gate_passed,
        "entity_level_mapping_rate": entity_level_mapping_rate,
        "mean_linked_entities_per_question": mean_linked_entities_per_question,
        "naive_expected_hit_rate_under_independence": naive_expected_hit_rate_under_independence,
        "min_category_n_for_rate": MIN_CATEGORY_N_FOR_RATE,
        "category_breakdown": category_breakdown,
        "triples_per_question": _distribution(all_triples_per_q),
        "kg_max_triples_per_question_cap": kg_cfg.max_triples_per_question,
        "n_triples_scored_for_novelty": n_triples_scored_total,
        "n_triples_novel": n_triples_novel,
        "overall_novel_relation_rate": overall_novel_rate,
        "n_triples_relevant": n_triples_relevant,
        "overall_relevant_rate": overall_relevant_rate,
        "n_triples_novel_and_relevant": n_triples_novel_and_relevant,
        "overall_novel_and_relevant_rate": overall_novel_and_relevant_rate,
        "overall_novel_given_relevant_rate": overall_novel_given_relevant_rate,
        "unmapped_cui": {
            "total_linked_entities": total_linked_entities,
            "unmapped_count": total_unmapped,
            "unmapped_rate": unmapped_rate,
            "distinct_unmapped_cuis": len(distinct_unmapped_cuis),
        },
        "sample_subgraphs": samples,
        "per_question": per_question,
        "config": {
            "cache_dir": str(cache_dir),
            "kg_edge_allowlist": kg_cfg.edge_allowlist,
            "kg_max_hops": kg_cfg.max_hops,
            "kg_degree_cap": kg_cfg.degree_cap,
            "kg_max_triples_per_question": kg_cfg.max_triples_per_question,
            "kg_linker_threshold": kg_cfg.linker_threshold,
            "novelty_top_k_dense": novelty_top_k_dense,
            "novelty_top_k_dense_source": "retrieval.top_k_rerank",
            "novelty_max_triples_scored_per_question": NOVELTY_MAX_TRIPLES_SCORED,
            "novelty_triple_selection_method": (
                "seeded random sample per question (seed derived from sha256(seed:question_id)), "
                "not the earlier deterministic alphabetical-first-N selection -- see ADR-016 "
                "addendum, 2026-09-05 follow-up"
            ),
            "novelty_nli_model": NLI_MODEL_NAME,
            "novelty_entailment_threshold": 0.5,
            "relevance_match_method": (
                "case-insensitive substring match of triple head/tail against the "
                "question stem and each answer option's text"
            ),
        },
        "wall_clock_seconds": elapsed_s,
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }

    out_path = Path(args.out) if args.out else RESULTS_DIR / f"{args.dataset}_n{args.limit}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"n_questions={len(questions)}  n_cache_missing={n_cache_missing}")
    print(
        f"aggregate KG-hit rate (permissive, question-level) = {aggregate_hit_rate:.4f} "
        f"({n_hit}/{len(questions)}) -- floor={HIT_RATE_FLOOR} -- "
        f"{'PASSED' if gate_passed else 'FAILED (real result, not adjusted)'}"
    )
    print(f"hit_definition: {HIT_DEFINITION}")
    print(
        f"entity_level_mapping_rate = {entity_level_mapping_rate:.4f} "
        f"({total_linked_entities - total_unmapped}/{total_linked_entities} linked entities "
        "resolved to a PrimeKG node)"
    )
    print(
        f"mean_linked_entities_per_question = {mean_linked_entities_per_question:.2f}  ->  "
        f"naive_expected_hit_rate_under_independence = "
        f"{naive_expected_hit_rate_under_independence:.4f}  "
        f"(compare to the observed {aggregate_hit_rate:.4f} -- if close, the observed rate "
        "is largely an arithmetic artifact of many low-probability entity resolutions per "
        "question, not evidence of broad content coverage)"
    )
    print()
    dist = result["triples_per_question"]
    print(
        f"REAL triples/question distribution (overall, post-40-cap): "
        f"min={dist['min']} median={dist['median']} mean={dist['mean']:.2f} max={dist['max']} "
        f"(cap={kg_cfg.max_triples_per_question})"
    )
    print()
    print(
        f"{'category':<24}{'n':>5}{'hit_rate':>12}{'triples/q(med)':>16}"
        f"{'novel_rate':>12}{'nov&rel_rate':>14}"
    )
    for category, stats in category_breakdown.items():
        if stats["underpowered"]:
            print(
                f"{category:<24}{stats['n']:>5}"
                f"  underpowered (n={stats['n']} < {MIN_CATEGORY_N_FOR_RATE}), rate suppressed"
                f" -- raw counts: n_hits={stats['n_hits']} n_novel={stats['n_novel']} "
                f"n_relevant={stats['n_relevant']} "
                f"n_novel_and_relevant={stats['n_novel_and_relevant']}"
            )
            continue
        med = stats["triples_per_question"]["median"]
        med_str = f"{med:.1f}" if med is not None else "n/a"
        hit_str = f"{stats['hit_rate']:.4f}" if stats["hit_rate"] is not None else "n/a"
        nov_rate = stats["novel_relation_rate"]
        nov_str = f"{nov_rate:.4f}" if nov_rate is not None else "n/a"
        nr_str = (
            f"{stats['novel_and_relevant_rate']:.4f}"
            if stats["novel_and_relevant_rate"] is not None
            else "n/a"
        )
        print(f"{category:<24}{stats['n']:>5}{hit_str:>12}{med_str:>16}{nov_str:>12}{nr_str:>14}")
    print()
    print(
        f"overall novel_relation_rate (re-scored at top-{novelty_top_k_dense} dense passages) = "
        f"{overall_novel_rate:.4f} ({n_triples_novel}/{n_triples_scored_total} triples)"
    )
    print(
        f"overall relevant_rate = {overall_relevant_rate:.4f} "
        f"({n_triples_relevant}/{n_triples_scored_total})"
    )
    print(
        f"overall novel_and_relevant_rate = {overall_novel_and_relevant_rate:.4f} "
        f"({n_triples_novel_and_relevant}/{n_triples_scored_total}) -- this is the figure that "
        "actually speaks to whether KG evidence adds question-specific value beyond dense retrieval"
    )
    if overall_novel_given_relevant_rate is not None:
        print(
            f"overall novel_given_relevant_rate = {overall_novel_given_relevant_rate:.4f} "
            f"({n_triples_novel_and_relevant}/{n_triples_relevant} relevant triples)"
        )
    print()
    print(
        f"unmapped CUIs: {total_unmapped}/{total_linked_entities} linked entities "
        f"({unmapped_rate:.4f}), {len(distinct_unmapped_cuis)} distinct unmapped CUIs"
    )
    print()
    print(f"wall_clock_seconds = {elapsed_s:.1f}")
    print(f"wrote {len(samples)} sample subgraphs + full results to {out_path}")


if __name__ == "__main__":
    main()
