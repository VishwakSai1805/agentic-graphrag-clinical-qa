"""Unit tests for src/kg/linker.py.

Per CLAUDE.md's "mock the LLM" spirit extended to heavy models generally:
these tests never load the real en_core_sci_lg model or the real UMLS KB --
`link_entities`'s scispaCy pipeline is monkeypatched with fakes, and
`PrimeKGCrosswalk` is exercised against small on-disk fixture files instead
of the real ~1GB PrimeKG download / MONDO / HPO ontology files.
"""

from __future__ import annotations

import logging

import pytest

from src.kg import linker as linker_mod
from src.kg.linker import PrimeKGCrosswalk, link_entities

# --- fixtures: tiny stand-ins for nodes.csv / mondo xref / hp.obo ---


@pytest.fixture
def nodes_csv(tmp_path):
    path = tmp_path / "nodes.csv"
    path.write_text(
        "node_index,node_id,node_type,node_name,node_source\n"
        "27165,8019,disease,mullerian aplasia and hyperandrogenism,MONDO\n"
        "22117,1507,effect/phenotype,Growth abnormality,HPO\n"
        "14012,DB09130,drug,Copper,DrugBank\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def mondo_xref_tsv(tmp_path):
    crosswalk_dir = tmp_path / "crosswalk"
    crosswalk_dir.mkdir()
    path = crosswalk_dir / "mondo_exactmatch_umls.sssom.tsv"
    path.write_text(
        "# comment line, must be skipped\n"
        "subject_id\tsubject_label\tpredicate_id\tobject_id\tmapping_justification\n"
        "MONDO:0008019\tmullerian aplasia and hyperandrogenism\tskos:exactMatch\t"
        "UMLS:C2675014\tsemapv:ManualMappingCuration\n",
        encoding="utf-8",
    )
    return crosswalk_dir


@pytest.fixture
def crosswalk_dir_with_hpo(mondo_xref_tsv):
    path = mondo_xref_tsv / "hp.obo"
    path.write_text(
        "[Term]\n"
        "id: HP:0001507\n"
        "name: Growth abnormality\n"
        "xref: UMLS:C0221755 UMLS\n"
        "\n"
        "[Term]\n"
        "id: HP:0000107\n"
        "name: Renal cyst\n",
        encoding="utf-8",
    )
    return mondo_xref_tsv


def make_crosswalk(nodes_csv, crosswalk_dir_with_hpo):
    return PrimeKGCrosswalk(nodes_csv, crosswalk_dir_with_hpo, download_if_missing=False)


# --- PrimeKGCrosswalk ---


def test_crosswalk_resolves_disease_via_mondo_xref(nodes_csv, crosswalk_dir_with_hpo):
    cw = make_crosswalk(nodes_csv, crosswalk_dir_with_hpo)
    node_id, node_type, method = cw.lookup(
        "C2675014", "mullerian aplasia and hyperandrogenism", ["T047"]
    )
    assert node_id == "27165"
    assert node_type == "disease"
    assert method == "mondo_xref"


def test_crosswalk_resolves_phenotype_via_hpo_xref(nodes_csv, crosswalk_dir_with_hpo):
    cw = make_crosswalk(nodes_csv, crosswalk_dir_with_hpo)
    node_id, node_type, method = cw.lookup("C0221755", "Growth abnormality", ["T184"])
    assert node_id == "22117"
    assert node_type == "effect/phenotype"
    assert method == "hpo_xref"


def test_crosswalk_falls_back_to_name_match_for_drug(nodes_csv, crosswalk_dir_with_hpo):
    cw = make_crosswalk(nodes_csv, crosswalk_dir_with_hpo)
    # No UMLS-CUI crosswalk exists for DrugBank in this project (see linker.py
    # docstring) -- a drug CUI must resolve via normalized-name match only.
    node_id, node_type, method = cw.lookup("C0009244", "Copper", ["T121"])
    assert node_id == "14012"
    assert node_type == "drug"
    assert method == "name_match"


def test_crosswalk_returns_none_for_truly_unmapped_cui(nodes_csv, crosswalk_dir_with_hpo):
    cw = make_crosswalk(nodes_csv, crosswalk_dir_with_hpo)
    node_id, node_type, method = cw.lookup("C9999999", "totally unknown concept", ["T047"])
    assert (node_id, node_type, method) == (None, None, None)


def test_crosswalk_missing_xref_files_does_not_download_or_raise(tmp_path, nodes_csv):
    empty_crosswalk_dir = tmp_path / "no_such_dir"
    cw = PrimeKGCrosswalk(nodes_csv, empty_crosswalk_dir, download_if_missing=False)
    node_id, node_type, method = cw.lookup(
        "C2675014", "mullerian aplasia and hyperandrogenism", ["T047"]
    )
    # id-xref maps are simply empty; a name-match fallback can still succeed.
    assert node_id == "27165"
    assert method == "name_match"


# --- link_entities: threshold filtering + crosswalk wiring, scispaCy mocked ---


class _FakeUnderscore:
    def __init__(self, kb_ents):
        self.kb_ents = kb_ents


class _FakeSpan:
    def __init__(self, text, kb_ents):
        self.text = text
        self._ = _FakeUnderscore(kb_ents)


class _FakeDoc:
    def __init__(self, ents):
        self.ents = ents


class _FakeUmlsEntity:
    def __init__(self, canonical_name, types):
        self.canonical_name = canonical_name
        self.types = types


class _FakeKB:
    def __init__(self, entities):
        self.cui_to_entity = entities


class _FakeLinkerPipe:
    def __init__(self, entities):
        self.kb = _FakeKB(entities)


class _FakeNlp:
    def __init__(self, doc, linker_pipe):
        self._doc = doc
        self._linker_pipe = linker_pipe

    def __call__(self, text):
        return self._doc

    def get_pipe(self, name):
        assert name == "scispacy_linker"
        return self._linker_pipe


def test_link_entities_filters_below_threshold(monkeypatch):
    doc = _FakeDoc(
        [
            _FakeSpan("aspirin", [("C0004057", 0.95), ("C0000001", 0.5)]),
        ]
    )
    entities = {
        "C0004057": _FakeUmlsEntity("aspirin", ["T121"]),
        "C0000001": _FakeUmlsEntity("something else", ["T121"]),
    }
    fake_nlp = _FakeNlp(doc, _FakeLinkerPipe(entities))
    monkeypatch.setattr(linker_mod, "_get_nlp", lambda: fake_nlp)

    results = link_entities("patient takes aspirin daily", threshold=0.85)
    assert len(results) == 1
    assert results[0].cui == "C0004057"
    assert results[0].score == pytest.approx(0.95)
    assert results[0].kg_node_id is None  # no crosswalk passed


def test_link_entities_wires_crosswalk_hits_and_logs_misses(
    monkeypatch, caplog, nodes_csv, crosswalk_dir_with_hpo
):
    cw = make_crosswalk(nodes_csv, crosswalk_dir_with_hpo)
    doc = _FakeDoc(
        [
            _FakeSpan("mullerian aplasia", [("C2675014", 0.9)]),
            _FakeSpan("mystery disease", [("C9999999", 0.9)]),
        ]
    )
    entities = {
        "C2675014": _FakeUmlsEntity("mullerian aplasia and hyperandrogenism", ["T047"]),
        "C9999999": _FakeUmlsEntity("totally unknown concept", ["T047"]),
    }
    fake_nlp = _FakeNlp(doc, _FakeLinkerPipe(entities))
    monkeypatch.setattr(linker_mod, "_get_nlp", lambda: fake_nlp)

    with caplog.at_level(logging.INFO):
        results = link_entities("...", threshold=0.85, crosswalk=cw)

    assert len(results) == 2
    hit = next(r for r in results if r.cui == "C2675014")
    assert hit.kg_node_id == "27165"
    assert hit.match_method == "mondo_xref"

    miss = next(r for r in results if r.cui == "C9999999")
    assert miss.kg_node_id is None
    assert "kg_unmapped_cui" in caplog.text
    assert "C9999999" in caplog.text
