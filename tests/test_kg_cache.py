"""KGCacheEntry round-trip + byte-for-byte determinism, per ADR-007's
reproducibility requirement (docs/05_DECISIONS.md) and the Phase 2 exit test
(docs/02_ROADMAP.md: "re-running the same 200 questions from the cache file
reproduces identical KG-hit rate and subgraphs byte-for-byte").

`tests/test_kg_traversal.py` already covers `write_subgraph_cache`/
`get_subgraph_cached`'s basic contract (round trip, the neo4j-import
boundary). This file is the dedicated cache-determinism suite the Phase 2
task calls for: multiple entries, multiple write passes, and an explicit
diff (not just an equality assert) so a future regression shows exactly what
changed.
"""

from __future__ import annotations

from src.kg.traversal import KGCacheEntry, Triple, get_subgraph_cached, write_subgraph_cache
from src.schema import Evidence, LinkedEntity


def make_entry(question_id: str) -> KGCacheEntry:
    return KGCacheEntry(
        question_id=question_id,
        linked_entities=[
            LinkedEntity(
                text="metformin",
                cui="C0025598",
                score=0.91,
                kg_node_id="9001",
                kg_node_type="drug",
                match_method="name_match",
            ),
            LinkedEntity(
                text="type 2 diabetes",
                cui="C0011860",
                score=0.97,
                kg_node_id="4021",
                kg_node_type="disease",
                match_method="mondo_xref",
            ),
        ],
        triples=[
            Triple(
                head="metformin",
                relation="indication",
                tail="type 2 diabetes",
                source_db="primekg",
            ),
            Triple(
                head="type 2 diabetes",
                relation="disease_phenotype_positive",
                tail="polyuria",
                source_db="primekg",
            ),
        ],
        evidence=[
            Evidence(
                id="kg::metformin|indication|type 2 diabetes",
                modality="kg",
                text="metformin is indicated for type 2 diabetes.",
                source_meta={"source_db": "primekg", "relation": "indication"},
                scores={},
            ),
            Evidence(
                id="kg::type 2 diabetes|disease_phenotype_positive|polyuria",
                modality="kg",
                text="polyuria is a phenotype associated with type 2 diabetes.",
                source_meta={"source_db": "primekg", "relation": "disease_phenotype_positive"},
                scores={},
            ),
        ],
    )


def test_roundtrip_preserves_all_fields(tmp_path):
    entry = make_entry("test_00001")
    write_subgraph_cache(entry, "medqa", tmp_path)
    loaded = get_subgraph_cached("medqa", "test_00001", tmp_path)

    assert loaded is not None
    assert loaded.question_id == entry.question_id
    assert loaded.linked_entities == entry.linked_entities
    assert loaded.triples == entry.triples
    assert loaded.evidence == entry.evidence
    assert loaded == entry


def test_multiple_entries_do_not_collide(tmp_path):
    e1 = make_entry("test_00001")
    e2 = make_entry("test_00002")
    write_subgraph_cache(e1, "medqa", tmp_path)
    write_subgraph_cache(e2, "medqa", tmp_path)

    assert get_subgraph_cached("medqa", "test_00001", tmp_path) == e1
    assert get_subgraph_cached("medqa", "test_00002", tmp_path) == e2


def test_datasets_do_not_collide_even_with_same_question_id(tmp_path):
    entry = make_entry("test_00001")
    write_subgraph_cache(entry, "medqa", tmp_path)
    write_subgraph_cache(entry, "medmcqa", tmp_path)

    assert (tmp_path / "medqa" / "test_00001.json").exists()
    assert (tmp_path / "medmcqa" / "test_00001.json").exists()
    assert get_subgraph_cached("medqa", "test_00001", tmp_path) == entry
    assert get_subgraph_cached("medmcqa", "test_00001", tmp_path) == entry


def test_rewrite_is_byte_for_byte_identical_and_diffable(tmp_path):
    """Writes the same entry twice (simulating two full precompute runs over
    an unchanged graph) and explicitly diffs the two files line-by-line --
    an empty diff, not just `==`, is what the Phase 2 exit test actually
    checks in practice (`diff run1.json run2.json`)."""
    import difflib

    entry = make_entry("test_00042")
    p1 = write_subgraph_cache(entry, "medqa", tmp_path / "run1")
    p2 = write_subgraph_cache(entry, "medqa", tmp_path / "run2")

    lines1 = p1.read_text(encoding="utf-8").splitlines(keepends=True)
    lines2 = p2.read_text(encoding="utf-8").splitlines(keepends=True)
    diff = list(difflib.unified_diff(lines1, lines2))

    assert diff == []
    assert lines1 == lines2


def test_cache_miss_returns_none_and_does_not_create_a_file(tmp_path):
    assert get_subgraph_cached("medqa", "never_computed", tmp_path) is None
    assert not (tmp_path / "medqa" / "never_computed.json").exists()
