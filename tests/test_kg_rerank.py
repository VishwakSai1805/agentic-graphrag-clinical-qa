"""Unit tests for src/kg/rerank.py.

Per this project's existing mocking convention (see tests/test_kg_linker.py,
src/kg/novelty.py's injectable `ScoreFn`): these tests never load the real
MedCPT cross-encoder -- `select_triples_reranked`'s `rerank_fn` is a fake
matching the real `CrossEncoderReranker.rerank`'s (query, candidates, top_n)
-> list[Evidence] signature exactly, so no torch/transformers import happens
here.
"""

from __future__ import annotations

from src.kg.rerank import select_triples_alphabetical, select_triples_reranked
from src.kg.traversal import Triple
from src.kg.verbalize import verbalize_triple


def make_triple(head: str, relation: str, tail: str) -> Triple:
    return Triple(head=head, relation=relation, tail=tail, source_db="primekg")


# --- select_triples_alphabetical (unchanged pre-existing default) ---


def test_select_triples_alphabetical_sorts_and_truncates():
    triples = [
        make_triple("Zebra drug", "drug_effect", "nausea"),
        make_triple("Aspirin", "drug_effect", "bleeding"),
        make_triple("Metformin", "drug_effect", "lactic acidosis"),
    ]
    result = select_triples_alphabetical(triples, cap=2)
    assert [t.head for t in result] == ["Aspirin", "Metformin"]


def test_select_triples_alphabetical_returns_all_when_under_cap():
    triples = [make_triple("Aspirin", "drug_effect", "bleeding")]
    result = select_triples_alphabetical(triples, cap=40)
    assert result == triples


# --- select_triples_reranked ---


def _fake_rerank_fn_factory(score_by_tail: dict[str, float]):
    """A fake matching CrossEncoderReranker.rerank's real signature: scores
    each candidate Evidence by looking up its verbalized triple's `tail` in
    `score_by_tail`, then returns the top_n sorted descending by that score."""

    def fake_rerank(query, candidates, top_n):
        scored = [(c, score_by_tail[c.source_meta["tail"]]) for c in candidates]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [c for c, _score in scored[:top_n]]

    return fake_rerank


def test_select_triples_reranked_picks_top_n_by_fake_score():
    triples = [
        make_triple("Aspirin", "drug_effect", "bleeding"),
        make_triple("Ibuprofen", "drug_effect", "renal impairment"),
        make_triple("Pain", "drug_effect", "hub noise 1"),
        make_triple("Pain", "drug_effect", "hub noise 2"),
    ]
    # "bleeding" and "renal impairment" are the question-relevant triples;
    # the two "Pain" hub-noise triples score lowest.
    fake_fn = _fake_rerank_fn_factory(
        {
            "bleeding": 0.9,
            "renal impairment": 0.8,
            "hub noise 1": 0.1,
            "hub noise 2": 0.05,
        }
    )
    result = select_triples_reranked(
        triples, "What drug causes bleeding?", cap=2, rerank_fn=fake_fn
    )
    assert {(t.head, t.tail) for t in result} == {
        ("Aspirin", "bleeding"),
        ("Ibuprofen", "renal impairment"),
    }


def test_select_triples_reranked_no_scoring_needed_under_cap():
    """If there are already <= cap triples, no rerank_fn call is needed --
    passing rerank_fn=None must not raise (it would if the real lazy-loaded
    cross-encoder were touched)."""
    triples = [make_triple("Aspirin", "drug_effect", "bleeding")]
    result = select_triples_reranked(triples, "irrelevant query", cap=40, rerank_fn=None)
    assert result == triples


def test_select_triples_reranked_maps_back_to_original_triple_objects():
    """The returned triples must be the actual input Triple objects (matched
    via the (head, relation, tail) key preserved in Evidence.source_meta by
    verbalize_triple), not reconstructed/duplicated copies."""
    t1 = make_triple("Aspirin", "drug_effect", "bleeding")
    t2 = make_triple("Ibuprofen", "drug_effect", "renal impairment")
    t3 = make_triple("Pain", "drug_effect", "hub noise")
    triples = [t1, t2, t3]

    fake_fn = _fake_rerank_fn_factory({"bleeding": 1.0, "renal impairment": 0.5, "hub noise": 0.0})
    result = select_triples_reranked(triples, "q", cap=1, rerank_fn=fake_fn)
    assert result == [t1]
    assert result[0] is t1


def test_select_triples_reranked_calls_fn_with_verbalized_evidence():
    """Sanity check that the candidates handed to rerank_fn are Evidence
    objects carrying the real verbalized text, not raw Triple objects."""
    triples = [make_triple("Aspirin", "drug_effect", "bleeding")] * 2  # duplicate key ok
    captured = {}

    def capturing_fn(query, candidates, top_n):
        captured["query"] = query
        captured["candidates"] = candidates
        captured["top_n"] = top_n
        return candidates[:top_n]

    # Need >cap triples to force scoring; make a distinct third triple too.
    extra = make_triple("Ibuprofen", "drug_effect", "renal impairment")
    result = select_triples_reranked(
        [triples[0], extra], "does aspirin cause bleeding?", cap=1, rerank_fn=capturing_fn
    )
    assert captured["query"] == "does aspirin cause bleeding?"
    assert captured["top_n"] == 1
    assert all(isinstance(c.text, str) and c.text for c in captured["candidates"])
    assert captured["candidates"][0].text == verbalize_triple(triples[0]).text
    assert len(result) == 1
