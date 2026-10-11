"""KG subgraph retrieval: a live-Neo4j path (`traverse_live`) and a
cache-only path (`get_subgraph_cached`).

ADR-007 (docs/05_DECISIONS.md) is binding: `traverse_live` is the *only*
function in this module that opens a Neo4j session, and it is only ever
called from `scripts/precompute_kg_cache.py` (Phase 2, written in a follow-up
session) and ad hoc debugging. `get_subgraph_cached` is what every eval-time
caller (Phase 3 onward) must use instead -- it reads a precomputed JSON file
from disk and does not import anything Neo4j-related on its call path, so
Neo4j can be `docker compose down` for the rest of a session once the cache
has been built.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from src.kg import queries
from src.schema import Evidence, LinkedEntity


class Triple(BaseModel):
    head: str
    relation: str
    tail: str
    source_db: str


# PrimeKG's kg.csv stores every relation in this allowlist as TWO literal
# rows per fact -- (x=A,y=B) and (x=B,y=A) -- as its way of representing an
# "undirected" edge (verified directly against the real downloaded dump: e.g.
# every `drug_effect` row has a matching reverse-direction row with x/y
# swapped). For a relation whose two node types are semantically
# asymmetric (indication: always drug -> disease, never disease -> drug),
# naively trusting Neo4j's `startNode`/`endNode` per-row produces a
# genuinely backwards sentence half the time (e.g. "Hearing impairment has
# the effect Acetylsalicylic acid" from the reverse-direction row) -- this
# is a real bug caught by testing against the live loaded graph, not a
# hypothetical. `_EXPECTED_SUBJECT_TYPE` fixes the semantic subject side per
# relation so `_canonicalize` can reorient (and dedupe) both directions into
# one consistently-phrased triple regardless of which row Neo4j happened to
# traverse. `disease_disease` is not in this map -- both sides are type
# "disease" so there is no wrong-direction phrasing risk (either order reads
# correctly under its symmetric template), but its reverse-direction row is
# NOT deduped here (deliberately out of scope for this fix; a caller wanting
# exactly-once-per-undirected-fact disease_disease triples needs a separate
# name-based canonical ordering, not implemented).
_EXPECTED_SUBJECT_TYPE: dict[str, str] = {
    "indication": "drug",
    "contraindication": "drug",
    "off-label use": "drug",
    "disease_phenotype_positive": "disease",
    "disease_protein": "disease",
    "drug_effect": "drug",
}


def _canonicalize(
    head_id: str,
    head_name: str,
    head_type: str,
    relation: str,
    tail_id: str,
    tail_name: str,
    tail_type: str,
) -> tuple[str, str, str, str]:
    """Returns (head_id, head_name, tail_id, tail_name) reoriented so head is
    the semantic subject for `relation` (see _EXPECTED_SUBJECT_TYPE), or the
    input order unchanged if `relation` has no fixed subject type or the
    types don't match what's expected (logged, not guessed further)."""
    expected = _EXPECTED_SUBJECT_TYPE.get(relation)
    if expected is None:
        return head_id, head_name, tail_id, tail_name
    if head_type == expected:
        return head_id, head_name, tail_id, tail_name
    if tail_type == expected:
        return tail_id, tail_name, head_id, head_name
    return head_id, head_name, tail_id, tail_name


class KGCacheStemMismatch(Exception):
    """Raised when a cache file's stored question stem does not match the
    question now sitting at that `question_id` (AUDIT 2026-09-06 §6.5)."""


def _norm_stem(text: str) -> str:
    """Whitespace- and case-insensitive normalisation for stem comparison."""
    return " ".join((text or "").split()).strip().lower()


class KGCacheEntry(BaseModel):
    question_id: str
    linked_entities: list[LinkedEntity]
    triples: list[Triple]
    evidence: list[Evidence]
    # AUDIT 2026-09-06 §6.5: `question_id` is a positional index into a raw HF
    # file (`f"{split}_{idx:05d}"`), and the KG cache is keyed on it. A re-fetch
    # that reorders rows silently re-associates every cached subgraph with the
    # wrong question. Storing the stem here and asserting it on read
    # (`get_subgraph_cached(expected_stem=...)`) makes that detectable. "" on
    # a file that predates this field -> the check is skipped for that file
    # (a warning, not a hard failure) until it is re-precomputed or backfilled.
    question_stem: str = ""


def traverse_live(
    node_ids: list[str],
    max_hops: int,
    degree_cap: int,
    edge_allowlist: list[str],
    *,
    uri: str,
    user: str,
    password: str,
    max_triples: int | None = None,
) -> list[Triple]:
    """Opens a real Neo4j session (the `neo4j` driver) and returns a bounded
    1-2 hop typed subgraph starting from `node_ids` (PrimeKG `Entity.id`
    values, i.e. node_index strings -- see src/kg/queries.py and
    src/kg/linker.py's crosswalk).

    Bounded by construction, not just by convention:
      - `max_hops` (<=2 per docs/03_DATASETS.md) hops are expanded via
        repeated single-hop queries, not one recursive/variable-length Cypher
        query -- this keeps the degree cap enforceable and auditable per hop.
      - `edge_allowlist`: only these relation types are ever followed
        (src/kg/queries.py: NEIGHBORS_ONE_HOP filters on `r.relation IN
        $edge_allowlist`).
      - `degree_cap`: at most this many edges are pulled per expanded node
        per hop (enforced by the `CALL { ... } LIMIT` subquery in
        NEIGHBORS_ONE_HOP -- a plain top-level `LIMIT` would not scope
        per-node and would not actually prevent a hub-node explosion).
      - `max_triples` (new, real-data-driven addition -- see
        docs/05_DECISIONS.md, the ADR this phase adds): a per-node
        `degree_cap` alone does NOT bound the total size of a multi-seed,
        2-hop traversal. Verified live against the real loaded PrimeKG graph:
        a single MedQA question with 8 resolved seed entities (several broad
        disease/anatomy nodes -- "transitional cell carcinoma", "bladder")
        produced 15,212 triples at max_hops=2/degree_cap=200, because hop 1
        alone can produce a frontier of up to `len(node_ids) * degree_cap`
        distinct new nodes, and hop 2 then expands *each of those* by up to
        `degree_cap` again. This is PROJECT_RULES.md's "unbounded traversal on a hub
        node is a bug, not a slow query" surfacing for real, just via
        multi-seed compounding rather than one single famous hub (e.g.
        aspirin) -- both are the same underlying failure mode. When set,
        traversal stops expanding to the next hop as soon as the
        accumulated triple count reaches `max_triples`, and the final result
        is deterministically truncated to at most `max_triples` entries
        (sorted by (head, relation, tail) so truncation is reproducible, not
        insertion-order-dependent). `None` (default) preserves the original,
        uncapped behavior for callers/tests that don't need it.

    Callers other than scripts/precompute_kg_cache.py and ad hoc debugging
    must not call this function -- see get_subgraph_cached below (ADR-007).
    """
    if max_hops < 1 or max_hops > 2:
        raise ValueError(f"max_hops must be 1 or 2 (bounded traversal), got {max_hops}")

    from neo4j import GraphDatabase  # imported here, not at module top level,
    # so that importing src.kg.traversal for get_subgraph_cached alone never
    # requires the neo4j package to be installed/importable at eval time.

    seen_node_ids: set[str] = set(node_ids)
    triples_by_key: dict[tuple[str, str, str], Triple] = {}
    frontier = list(node_ids)

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            for _hop in range(max_hops):
                if not frontier:
                    break
                query = queries.NEIGHBORS_ONE_HOP
                params = queries.neighbors_one_hop_params(frontier, edge_allowlist, degree_cap)
                result = session.run(query, params)
                next_frontier: set[str] = set()
                for record in result:
                    relation = record["relation"]
                    neighbor_id = record["neighbor_id"]

                    head_id, head_name, tail_id, tail_name = _canonicalize(
                        record["head_id"],
                        record["head_name"],
                        record["head_type"],
                        relation,
                        record["tail_id"],
                        record["tail_name"],
                        record["tail_type"],
                    )

                    key = (head_id, relation, tail_id)
                    if key not in triples_by_key:
                        triples_by_key[key] = Triple(
                            head=head_name,
                            relation=relation,
                            tail=tail_name,
                            source_db="primekg",
                        )
                    if neighbor_id not in seen_node_ids:
                        next_frontier.add(neighbor_id)
                seen_node_ids.update(next_frontier)
                frontier = list(next_frontier)

                if max_triples is not None and len(triples_by_key) >= max_triples:
                    # Stop expanding to the next hop -- see max_triples's
                    # docstring above. The current hop's own query has
                    # already run (bounded per-node by degree_cap, so this
                    # doesn't itself explode), only the *next* hop is
                    # skipped.
                    break
    finally:
        driver.close()

    triples = list(triples_by_key.values())
    if max_triples is not None and len(triples) > max_triples:
        # Deterministic truncation (sorted by the natural triple key, not
        # insertion/dict order) so repeated runs over an unchanged graph
        # truncate to the exact same subset -- required for ADR-007's
        # byte-for-byte cache reproducibility.
        triples = sorted(triples, key=lambda t: (t.head, t.relation, t.tail))[:max_triples]
    return triples


def _cache_path(dataset: str, question_id: str, cache_dir: Path) -> Path:
    return Path(cache_dir) / dataset / f"{question_id}.json"


def get_subgraph_cached(
    dataset: str,
    question_id: str,
    cache_dir: Path,
    *,
    expected_stem: str | None = None,
) -> KGCacheEntry | None:
    """Cache-only read path (ADR-007): reads
    `<cache_dir>/<dataset>/<question_id>.json` and returns a `KGCacheEntry`,
    or None if no cache entry exists for this question yet (a cache miss --
    the question hasn't been through scripts/precompute_kg_cache.py, or KG
    linking found nothing to traverse from).

    Never opens a Neo4j connection and never imports the `neo4j` package --
    the only import in this function's call path is `json` + `pathlib`, so
    Neo4j can be stopped entirely once the cache exists. This is the function
    all eval-time (Phase 3+) code must call; do not call `traverse_live` from
    eval code.

    `expected_stem` (AUDIT 2026-09-06 §6.5): the text of the question that
    *should* be at `question_id`. When given and the cache file carries a
    non-empty `question_stem`, a mismatch raises `KGCacheStemMismatch` -- the
    raw dataset file was re-fetched in a different row order and every cached
    subgraph is now bound to the wrong question. A file with an empty
    `question_stem` (predates the field) is read without the check.
    """
    path = _cache_path(dataset, question_id, cache_dir)
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    entry = KGCacheEntry.model_validate(raw)
    if expected_stem is not None and entry.question_stem:
        if _norm_stem(entry.question_stem) != _norm_stem(expected_stem):
            raise KGCacheStemMismatch(
                f"{path}: cached question_stem does not match the question now at "
                f"{dataset}/{question_id}. The raw dataset file's row order changed "
                f"since the KG cache was built, so every cached subgraph is "
                f"mis-associated (AUDIT §6.5). Re-run "
                f"`scripts.precompute_kg_cache --dataset {dataset} --split "
                f"{question_id.rsplit('_', 1)[0]} --force`.\n"
                f"  cached: {entry.question_stem[:120]!r}\n"
                f"  now   : {expected_stem[:120]!r}"
            )
    return entry


def write_subgraph_cache(entry: KGCacheEntry, dataset: str, cache_dir: Path) -> Path:
    """Writes a `KGCacheEntry` to the same path `get_subgraph_cached` reads.
    Used by scripts/precompute_kg_cache.py. Deterministic byte-for-byte
    output (sorted keys, fixed separators) so repeated precompute runs over
    an unchanged graph are diffable/reproducible per ADR-007's exit test.
    """
    path = _cache_path(dataset, entry.question_id, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(entry.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path
