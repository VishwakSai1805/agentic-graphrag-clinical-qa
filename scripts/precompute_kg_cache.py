"""ADR-007 (docs/05_DECISIONS.md): runs entity linking + bounded PrimeKG
traversal once per question and writes the result to
`data/processed/kg_cache/<dataset>/<question_id>.json`. This (plus
`scripts/load_primekg.py` and ad hoc debugging) is the ONLY script allowed to
import anything Neo4j-related -- `src/kg/traversal.py: traverse_live` opens a
real Neo4j session, and every eval-time caller from Phase 3 onward must read
the cache this script writes instead (`get_subgraph_cached`).

Per question:
  1. `src.kg.dataset_questions.load_question_sample` picks the (dataset,
     split, seed)-deterministic question set.
  2. `src.kg.linker.link_entities` runs scispaCy + UMLS linking + the
     PrimeKG crosswalk over the question stem + all answer options (MCQ
     datasets) or just the stem (pubmedqa) -- see
     `dataset_questions.link_text_for_question` for exactly what text is
     linked over, documented there, not here.
  3. The distinct, non-null `kg_node_id`s that resolved become the seed set
     for one `traverse_live` call (bounded by `kg.max_hops`/`kg.degree_cap`/
     `kg.edge_allowlist` from `configs/base.yaml`, per docs/03_DATASETS.md).
     Zero resolved entities -> zero triples, no traversal call is made.
  4. Triples are verbalised (`src.kg.verbalize.verbalize_triple`) into
     `Evidence(modality="kg")` objects.
  5. The whole thing is assembled into a `KGCacheEntry` and written via
     `write_subgraph_cache` -- deterministic, sorted-keys JSON (ADR-007's
     reproducibility requirement).

Idempotent/resumable: a question whose cache file already exists is skipped
unless `--force` is passed, since this is a real, potentially slow,
per-question live-DB job (crosspage entity linking + a Neo4j round trip per
question) that must survive being interrupted and restarted.

Usage:
    uv run python -m scripts.precompute_kg_cache --dataset medqa --split test --limit 200
    uv run python -m scripts.precompute_kg_cache --dataset medqa --split test --limit 200 --force
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

load_dotenv()

from src.config import load_config  # noqa: E402
from src.kg.dataset_questions import (  # noqa: E402
    Question,
    link_text_for_question,
    load_question_sample,
    load_question_sample_excluding,
)
from src.kg.linker import PrimeKGCrosswalk, link_entities  # noqa: E402
from src.kg.rerank import select_triples_reranked  # noqa: E402
from src.kg.traversal import KGCacheEntry, Triple, traverse_live, write_subgraph_cache  # noqa: E402
from src.kg.verbalize import verbalize_triple  # noqa: E402

CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"
PRIMEKG_NODES_PATH = REPO_ROOT / "data" / "raw" / "primekg" / "nodes.csv"
PRIMEKG_CROSSWALK_DIR = REPO_ROOT / "data" / "raw" / "primekg" / "crosswalk"


def build_cache_entry(
    question: Question,
    *,
    crosswalk: PrimeKGCrosswalk,
    linker_threshold: float,
    max_hops: int,
    degree_cap: int,
    edge_allowlist: list[str],
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    max_triples: int | None = None,
    kg_rerank: bool = False,
    rerank_pool_size: int | None = None,
) -> KGCacheEntry:
    """Does the real per-question work: link -> (maybe) traverse_live ->
    (maybe rerank-select) -> verbalise -> assemble. The only function in this
    script that a caller outside this module should need (kept separate from
    `main` so it's independently testable/reusable, e.g. by
    `scripts/kg_probe.py` for an on-demand cache fill -- see that script's
    module docstring for why it does NOT do that by default).

    `kg_rerank` (ADR-018 ablation, see `src/kg/rerank.py`): when False
    (default), behavior is exactly as before this flag existed -- one
    `traverse_live` call truncates directly to `max_triples` via its own
    internal deterministic alphabetical sort. When True, `traverse_live` is
    called with a larger `rerank_pool_size` candidate-pool cap instead (still
    a bounded traversal, just a wider pool for the reranker to choose from --
    not a return to unbounded traversal), and
    `src.kg.rerank.select_triples_reranked` picks the final `max_triples` from
    within that pool by MedCPT cross-encoder relevance to the question text
    instead of alphabetical order.
    """
    link_text = link_text_for_question(question)
    linked = link_entities(link_text, threshold=linker_threshold, crosswalk=crosswalk)

    node_ids = sorted({e.kg_node_id for e in linked if e.kg_node_id is not None})

    triples: list[Triple] = []
    if node_ids:
        if kg_rerank and max_triples is not None:
            pool_cap = max(max_triples, rerank_pool_size or max_triples)
            pool = traverse_live(
                node_ids,
                max_hops=max_hops,
                degree_cap=degree_cap,
                edge_allowlist=edge_allowlist,
                uri=neo4j_uri,
                user=neo4j_user,
                password=neo4j_password,
                max_triples=pool_cap,
            )
            triples = select_triples_reranked(pool, question.question, cap=max_triples)
        else:
            triples = traverse_live(
                node_ids,
                max_hops=max_hops,
                degree_cap=degree_cap,
                edge_allowlist=edge_allowlist,
                uri=neo4j_uri,
                user=neo4j_user,
                password=neo4j_password,
                max_triples=max_triples,
            )

    evidence = [verbalize_triple(t) for t in triples]

    return KGCacheEntry(
        question_id=question.question_id,
        question_stem=question.question,  # AUDIT §6.5: bind the cache to the question text
        linked_entities=linked,
        triples=triples,
        evidence=evidence,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["medqa", "pubmedqa", "medmcqa"], required=True)
    parser.add_argument("--split", required=True, help="e.g. medqa: test|train")
    parser.add_argument("--limit", type=int, default=None, help="cap on number of questions")
    parser.add_argument("--force", action="store_true", help="recompute even if cached")
    parser.add_argument(
        "--kg-rerank",
        action="store_true",
        help="ablation flag (ADR-018, default off): select the final "
        "max_triples_per_question via the MedCPT cross-encoder instead of "
        "naive alphabetical order. Overrides configs/base.yaml: kg.kg_rerank if set. "
        "Always use --cache-dir with this to avoid writing into the canonical cache.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="override kg.cache_dir (default: configs/base.yaml's kg.cache_dir). Use a "
        "separate directory for --kg-rerank runs, e.g. data/processed/kg_cache_kg_rerank, "
        "so the canonical data/processed/kg_cache/ files are never touched.",
    )
    parser.add_argument(
        "--exclude-limit",
        type=int,
        default=None,
        help="exclude the question_ids that load_question_sample(dataset, split, "
        "exclude_limit, seed=exclude_seed) would draw before sampling --limit "
        "questions -- matches src/pipeline.py's --exclude-limit so a precompute "
        "run and an eval run over 'the same disjoint slice' actually agree.",
    )
    parser.add_argument(
        "--exclude-seed",
        type=int,
        default=None,
        help="seed for --exclude-limit's sample (default: cfg.seed)",
    )
    args = parser.parse_args()

    cfg = load_config(CONFIG_PATH)
    kg_cfg = cfg.kg
    kg_rerank = args.kg_rerank or kg_cfg.kg_rerank

    password = os.environ.get(kg_cfg.neo4j_password_env)
    if not password:
        print(
            f"[precompute_kg_cache] ERROR: env var {kg_cfg.neo4j_password_env} is not set.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[precompute_kg_cache] loading PrimeKG crosswalk from {PRIMEKG_NODES_PATH} ...")
    crosswalk = PrimeKGCrosswalk(
        PRIMEKG_NODES_PATH, PRIMEKG_CROSSWALK_DIR, download_if_missing=True
    )

    if args.exclude_limit is not None:
        excl_seed = args.exclude_seed if args.exclude_seed is not None else cfg.seed
        excluded = load_question_sample(
            args.dataset, args.split, args.exclude_limit, seed=excl_seed
        )
        exclude_ids = {q.question_id for q in excluded}
        questions = load_question_sample_excluding(
            args.dataset, args.split, args.limit, cfg.seed, exclude_ids
        )
        print(
            f"[precompute_kg_cache] excluding {len(exclude_ids)} questions "
            f"(load_question_sample({args.dataset!r}, {args.split!r}, {args.exclude_limit}, "
            f"seed={excl_seed})) before sampling"
        )
    else:
        questions = load_question_sample(args.dataset, args.split, args.limit, seed=cfg.seed)
    print(
        f"[precompute_kg_cache] dataset={args.dataset} split={args.split} "
        f"n_questions={len(questions)} seed={cfg.seed}"
    )

    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(kg_cfg.cache_dir)
    print(
        f"[precompute_kg_cache] kg_rerank={kg_rerank} "
        f"(pool_size={kg_cfg.kg_rerank_pool_size} if kg_rerank else n/a) cache_dir={cache_dir}"
    )
    n_skipped = 0
    n_computed = 0
    n_zero_entities = 0
    n_zero_triples = 0
    t0 = time.time()

    for i, q in enumerate(questions):
        cache_path = cache_dir / args.dataset / f"{q.question_id}.json"
        if cache_path.exists() and not args.force:
            n_skipped += 1
            continue

        entry = build_cache_entry(
            q,
            crosswalk=crosswalk,
            linker_threshold=kg_cfg.linker_threshold,
            max_hops=kg_cfg.max_hops,
            degree_cap=kg_cfg.degree_cap,
            edge_allowlist=kg_cfg.edge_allowlist,
            neo4j_uri=kg_cfg.neo4j_uri,
            neo4j_user=kg_cfg.neo4j_user,
            neo4j_password=password,
            max_triples=kg_cfg.max_triples_per_question,
            kg_rerank=kg_rerank,
            rerank_pool_size=kg_cfg.kg_rerank_pool_size,
        )
        write_subgraph_cache(entry, args.dataset, cache_dir)
        n_computed += 1
        if not any(e.kg_node_id for e in entry.linked_entities):
            n_zero_entities += 1
        if not entry.triples:
            n_zero_triples += 1

        if (i + 1) % 20 == 0 or (i + 1) == len(questions):
            elapsed = time.time() - t0
            print(
                f"[precompute_kg_cache] {i + 1}/{len(questions)} processed "
                f"({n_computed} computed, {n_skipped} skipped-cached, "
                f"{elapsed:.1f}s elapsed)",
                flush=True,
            )

    print(
        f"[precompute_kg_cache] done: {n_computed} computed, {n_skipped} skipped "
        f"(already cached), {n_zero_entities} computed questions with zero resolved "
        f"KG entities, {n_zero_triples} computed questions with zero triples, "
        f"cache_dir={cache_dir / args.dataset}"
    )


if __name__ == "__main__":
    main()
