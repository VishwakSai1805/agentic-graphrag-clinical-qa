"""Every Cypher string used anywhere in Phase 2 lives in this module (CLAUDE.md:
"Every Cypher string lives in src/kg/queries.py. No inline query strings
anywhere else, ever"). Functions return `(query, params)`-shaped pieces --
callers (scripts/load_primekg.py, src/kg/traversal.py) never write their own
Cypher, they only import from here.

Graph schema (see scripts/load_primekg.py for the load path):
  (:Entity {id, node_type, name, source})
    -[:KG_EDGE {relation, display_relation}]->
  (:Entity {...})

`id` is PrimeKG's own `node_index` (a globally unique integer across all
10 node types, stringified) -- not `node_id` (which is a per-type external
vocabulary id, e.g. a DrugBank id, and for grouped MONDO disease nodes can be
an underscore-joined compound of several MONDO ids). `node_index` is the
stable, always-unique primary key the raw dump itself uses; see
docs/03_DATASETS.md and ADR entries this phase adds for the entity-linking
crosswalk that maps a UMLS CUI to this same `id`.

Edges are loaded exactly as directed in the source dump (`x_index -> y_index`)
but traversal queries match them in either direction (`-[r]-`, not `-[r]->`)
and recover the true stored direction via `startNode(r)`/`endNode(r)` --
this matters because relations like `indication` are not symmetric (head is
always the drug, tail the disease) and a caller must not silently guess which
side is which.
"""

from __future__ import annotations

CONSTRAINT_ENTITY_ID = """
CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
FOR (n:Entity) REQUIRE n.id IS UNIQUE
"""


UPSERT_NODES = """
UNWIND $rows AS row
MERGE (n:Entity {id: row.id})
SET n.node_type = row.node_type,
    n.name = row.name,
    n.source = row.source
"""


def upsert_nodes_params(rows: list[dict]) -> dict:
    """rows: [{"id": <node_index str>, "node_type": ..., "name": ..., "source": ...}, ...]"""
    return {"rows": rows}


UPSERT_EDGES = """
UNWIND $rows AS row
MATCH (x:Entity {id: row.x_id})
MATCH (y:Entity {id: row.y_id})
MERGE (x)-[r:KG_EDGE {relation: row.relation}]->(y)
SET r.display_relation = row.display_relation
"""


def upsert_edges_params(rows: list[dict]) -> dict:
    """rows: [{"x_id": <node_index str>, "y_id": <node_index str>,
    "relation": ..., "display_relation": ...}, ...]. x_id/y_id are `Entity.id`
    values (node_index), matching UPSERT_NODES -- callers must load nodes
    before edges referencing them or the MATCH silently drops the edge."""
    return {"rows": rows}


COUNT_NODES = "MATCH (n:Entity) RETURN count(n) AS n"

COUNT_EDGES = "MATCH ()-[r:KG_EDGE]->() RETURN count(r) AS n"

COUNT_EDGES_BY_RELATION = """
MATCH ()-[r:KG_EDGE]->()
RETURN r.relation AS relation, count(r) AS n
ORDER BY n DESC
"""

COUNT_NODES_BY_TYPE = """
MATCH (n:Entity)
RETURN n.node_type AS node_type, count(n) AS n
ORDER BY n DESC
"""


# --- Traversal (src/kg/traversal.py: traverse_live only -- never called from
# eval-time code, see ADR-007 and the module docstring in traversal.py). ---

NEIGHBORS_ONE_HOP = """
UNWIND $node_ids AS start_id
MATCH (n:Entity {id: start_id})
CALL (n) {
  MATCH (n)-[r:KG_EDGE]-(m:Entity)
  WHERE r.relation IN $edge_allowlist
  RETURN r AS r, m AS m
  ORDER BY m.id
  LIMIT $degree_cap
}
RETURN
  n.id AS from_id,
  startNode(r).id AS head_id, startNode(r).name AS head_name,
  startNode(r).node_type AS head_type,
  r.relation AS relation, r.display_relation AS display_relation,
  endNode(r).id AS tail_id, endNode(r).name AS tail_name,
  endNode(r).node_type AS tail_type,
  m.id AS neighbor_id
"""


def neighbors_one_hop_params(
    node_ids: list[str], edge_allowlist: list[str], degree_cap: int
) -> dict:
    """One bounded hop out of every id in node_ids: only edges whose relation
    is in edge_allowlist are followed, and at most degree_cap edges are
    returned per start node (the `CALL { ... } LIMIT` subquery scopes the
    LIMIT per `n`, not over the whole result set -- this is what actually
    enforces the per-node degree cap, see docs/03_DATASETS.md's aspirin
    hub-explosion example)."""
    return {"node_ids": node_ids, "edge_allowlist": edge_allowlist, "degree_cap": degree_cap}
