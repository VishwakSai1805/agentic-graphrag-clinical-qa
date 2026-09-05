"""Unit tests for src/kg/traversal.py.

ADR-007 (docs/05_DECISIONS.md) makes a hard boundary claim: eval-time code
uses `get_subgraph_cached` and that function must never open a Neo4j
connection. `test_get_subgraph_cached_never_imports_neo4j` makes that claim
falsifiable rather than just documented, by blocking the `neo4j` import at
the interpreter level and proving the cache-read path still works.

`traverse_live` is tested with the `neo4j` driver entirely faked (no live
Neo4j needed) to check the bounded multi-hop BFS / dedup logic.
"""

from __future__ import annotations

import sys

import pytest

from src.kg.traversal import (
    KGCacheEntry,
    Triple,
    get_subgraph_cached,
    write_subgraph_cache,
)
from src.schema import Evidence, LinkedEntity


def make_entry(question_id="q1"):
    return KGCacheEntry(
        question_id=question_id,
        linked_entities=[
            LinkedEntity(
                text="aspirin",
                cui="C0004057",
                score=0.95,
                kg_node_id="14012",
                kg_node_type="drug",
                match_method="name_match",
            )
        ],
        triples=[
            Triple(
                head="aspirin",
                relation="indication",
                tail="myocardial infarction",
                source_db="primekg",
            )
        ],
        evidence=[
            Evidence(
                id="kg::aspirin|indication|myocardial infarction",
                modality="kg",
                text="aspirin is indicated for myocardial infarction.",
                source_meta={"source_db": "primekg", "relation": "indication"},
                scores={},
            )
        ],
    )


def test_get_subgraph_cached_returns_none_on_miss(tmp_path):
    assert get_subgraph_cached("medqa", "does-not-exist", tmp_path) is None


def test_write_then_get_subgraph_cached_roundtrip(tmp_path):
    entry = make_entry()
    path = write_subgraph_cache(entry, "medqa", tmp_path)
    assert path.exists()

    loaded = get_subgraph_cached("medqa", "q1", tmp_path)
    assert loaded is not None
    assert loaded == entry


def test_write_subgraph_cache_is_byte_for_byte_reproducible(tmp_path):
    entry = make_entry()
    p1 = write_subgraph_cache(entry, "medqa", tmp_path / "run1")
    p2 = write_subgraph_cache(entry, "medqa", tmp_path / "run2")
    assert p1.read_text(encoding="utf-8") == p2.read_text(encoding="utf-8")


def test_get_subgraph_cached_never_imports_neo4j(tmp_path, monkeypatch):
    """Blocks `import neo4j` at the module-resolution level (a documented
    technique: a `None` entry in sys.modules makes any import of that name
    raise ImportError) and proves get_subgraph_cached's call path -- reading
    a real cache file from disk -- still succeeds without needing it."""
    entry = make_entry()
    write_subgraph_cache(entry, "medqa", tmp_path)

    monkeypatch.setitem(sys.modules, "neo4j", None)
    try:
        loaded = get_subgraph_cached("medqa", "q1", tmp_path)
    finally:
        monkeypatch.delitem(sys.modules, "neo4j", raising=False)

    assert loaded == entry


# --- traverse_live: neo4j driver faked entirely, no live Neo4j required ---


class _FakeResult:
    def __init__(self, records):
        self._records = records

    def __iter__(self):
        return iter(self._records)


class _FakeSession:
    def __init__(self, responses):
        self._responses = responses
        self.calls = []

    def run(self, query, params):
        self.calls.append(params)
        return _FakeResult(self._responses.pop(0))

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeDriver:
    def __init__(self, session):
        self._session = session

    def session(self):
        return self._session

    def close(self):
        pass


def test_traverse_live_bounds_hops_and_dedupes(monkeypatch):
    hop1 = [
        {
            "from_id": "1",
            "head_id": "1",
            "head_name": "aspirin",
            "head_type": "drug",
            "relation": "indication",
            "display_relation": "indication",
            "tail_id": "2",
            "tail_name": "headache",
            "tail_type": "disease",
            "neighbor_id": "2",
        }
    ]
    hop2 = [
        {
            "from_id": "2",
            "head_id": "2",
            "head_name": "headache",
            "head_type": "disease",
            "relation": "disease_phenotype_positive",
            "display_relation": "phenotype",
            "tail_id": "3",
            "tail_name": "pain",
            "tail_type": "effect/phenotype",
            "neighbor_id": "3",
        }
    ]
    fake_session = _FakeSession([hop1, hop2])
    fake_driver = _FakeDriver(fake_session)

    fake_neo4j = type(sys)("neo4j")
    fake_neo4j.GraphDatabase = type(
        "GD", (), {"driver": staticmethod(lambda uri, auth: fake_driver)}
    )
    monkeypatch.setitem(sys.modules, "neo4j", fake_neo4j)

    from src.kg.traversal import traverse_live

    triples = traverse_live(
        ["1"],
        max_hops=2,
        degree_cap=200,
        edge_allowlist=["indication", "disease_phenotype_positive"],
        uri="bolt://fake",
        user="neo4j",
        password="x",
    )

    assert len(fake_session.calls) == 2
    assert fake_session.calls[0]["node_ids"] == ["1"]
    assert fake_session.calls[1]["node_ids"] == ["2"]  # hop 2 starts from hop 1's new neighbor only

    assert {(t.head, t.relation, t.tail) for t in triples} == {
        ("aspirin", "indication", "headache"),
        ("headache", "disease_phenotype_positive", "pain"),
    }


def test_traverse_live_rejects_out_of_bound_hops():
    with pytest.raises(ValueError):
        from src.kg.traversal import traverse_live

        traverse_live(
            [], max_hops=3, degree_cap=200, edge_allowlist=[], uri="x", user="y", password="z"
        )


def test_traverse_live_canonicalizes_reverse_direction_rows(monkeypatch):
    """Real bug caught against the live loaded PrimeKG graph: kg.csv stores
    every allowlisted relation as two literal rows per fact (both
    directions), so Neo4j's raw startNode/endNode alternates which side is
    "head" depending on which row got traversed. Without reorienting by
    node type, half of all `drug_effect`/`indication`/etc. triples render
    backwards (e.g. "Hearing impairment has the effect Acetylsalicylic
    acid"). Both directions of the same fact must collapse to one
    consistently-oriented triple."""
    forward_row = {
        "from_id": "1",
        "head_id": "1",
        "head_name": "aspirin",
        "head_type": "drug",
        "relation": "drug_effect",
        "display_relation": "side effect",
        "tail_id": "2",
        "tail_name": "hearing impairment",
        "tail_type": "effect/phenotype",
        "neighbor_id": "2",
    }
    reverse_row = {
        "from_id": "1",
        "head_id": "2",
        "head_name": "hearing impairment",
        "head_type": "effect/phenotype",
        "relation": "drug_effect",
        "display_relation": "side effect",
        "tail_id": "1",
        "tail_name": "aspirin",
        "tail_type": "drug",
        "neighbor_id": "1",
    }
    fake_session = _FakeSession([[forward_row, reverse_row]])
    fake_driver = _FakeDriver(fake_session)

    fake_neo4j = type(sys)("neo4j")
    fake_neo4j.GraphDatabase = type(
        "GD", (), {"driver": staticmethod(lambda uri, auth: fake_driver)}
    )
    monkeypatch.setitem(sys.modules, "neo4j", fake_neo4j)

    from src.kg.traversal import traverse_live

    triples = traverse_live(
        ["1"],
        max_hops=1,
        degree_cap=200,
        edge_allowlist=["drug_effect"],
        uri="bolt://fake",
        user="neo4j",
        password="x",
    )

    assert len(triples) == 1
    assert triples[0].head == "aspirin"
    assert triples[0].tail == "hearing impairment"


def test_traverse_live_max_triples_stops_early_and_truncates_deterministically(monkeypatch):
    """Real hub-explosion finding (see traverse_live's max_triples docstring
    and the ADR this phase adds): a per-node degree_cap does not bound the
    total size of a multi-seed 2-hop traversal. `max_triples` must (a) stop
    expanding to a further hop once the cap is already reached, and (b)
    truncate the final result deterministically (sorted by (head, relation,
    tail), not insertion order) so repeated runs over an unchanged graph
    truncate to the exact same subset."""
    hop1 = [
        {
            "from_id": "1",
            "head_id": "1",
            "head_name": "seed",
            "head_type": "drug",
            "relation": "indication",
            "display_relation": "indication",
            "tail_id": str(n),
            "tail_name": f"disease{n:02d}",
            "tail_type": "disease",
            "neighbor_id": str(n),
        }
        for n in range(5)
    ]
    fake_session = _FakeSession([hop1])  # only ONE call expected -- hop 2 must never run
    fake_driver = _FakeDriver(fake_session)

    fake_neo4j = type(sys)("neo4j")
    fake_neo4j.GraphDatabase = type(
        "GD", (), {"driver": staticmethod(lambda uri, auth: fake_driver)}
    )
    monkeypatch.setitem(sys.modules, "neo4j", fake_neo4j)

    from src.kg.traversal import traverse_live

    triples = traverse_live(
        ["1"],
        max_hops=2,
        degree_cap=200,
        edge_allowlist=["indication"],
        uri="bolt://fake",
        user="neo4j",
        password="x",
        max_triples=3,
    )

    assert len(fake_session.calls) == 1  # hop 2 skipped -- cap already reached after hop 1
    assert len(triples) == 3
    assert [t.tail for t in triples] == ["disease00", "disease01", "disease02"]


def test_traverse_live_max_triples_none_preserves_uncapped_behavior(monkeypatch):
    hop1 = [
        {
            "from_id": "1",
            "head_id": "1",
            "head_name": "aspirin",
            "head_type": "drug",
            "relation": "indication",
            "display_relation": "indication",
            "tail_id": "2",
            "tail_name": "headache",
            "tail_type": "disease",
            "neighbor_id": "2",
        }
    ]
    fake_session = _FakeSession([hop1])
    fake_driver = _FakeDriver(fake_session)

    fake_neo4j = type(sys)("neo4j")
    fake_neo4j.GraphDatabase = type(
        "GD", (), {"driver": staticmethod(lambda uri, auth: fake_driver)}
    )
    monkeypatch.setitem(sys.modules, "neo4j", fake_neo4j)

    from src.kg.traversal import traverse_live

    triples = traverse_live(
        ["1"],
        max_hops=1,
        degree_cap=200,
        edge_allowlist=["indication"],
        uri="bolt://fake",
        user="neo4j",
        password="x",
    )
    assert len(triples) == 1
