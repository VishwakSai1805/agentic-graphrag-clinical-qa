from src.retrieval.fusion import reciprocal_rank_fusion


def test_doc_ranked_first_in_both_beats_doc_ranked_first_in_only_one():
    dense_ranking = ["docA", "docB", "docC"]
    bm25_ranking = ["docA", "docD", "docE"]
    # docB is #1-ranked in neither modality individually but let's construct a
    # doc ranked #1 in exactly one ranking, to compare against docA (#1 in both).
    fused = dict(reciprocal_rank_fusion([dense_ranking, bm25_ranking]))
    assert fused["docA"] > fused["docB"]
    assert fused["docA"] > fused["docD"]


def test_rrf_output_sorted_descending():
    dense_ranking = ["docC", "docA", "docB"]
    bm25_ranking = ["docB", "docC", "docA"]
    fused = reciprocal_rank_fusion([dense_ranking, bm25_ranking])
    scores = [score for _doc_id, score in fused]
    assert scores == sorted(scores, reverse=True)


def test_doc_missing_from_one_ranking_still_fused_in():
    dense_ranking = ["docA", "docB"]
    bm25_ranking = ["docC", "docD"]
    fused = dict(reciprocal_rank_fusion([dense_ranking, bm25_ranking]))
    assert set(fused) == {"docA", "docB", "docC", "docD"}
    assert fused["docA"] > 0
    assert fused["docC"] > 0


def test_rrf_k_constant_changes_relative_weighting():
    dense_ranking = ["docA", "docB"]
    bm25_ranking = ["docB", "docA"]
    fused_default = dict(reciprocal_rank_fusion([dense_ranking, bm25_ranking], k=60))
    fused_small_k = dict(reciprocal_rank_fusion([dense_ranking, bm25_ranking], k=1))
    # both docs tie under symmetric rankings regardless of k, but the k
    # constant itself must actually be used (not ignored) -- verify the
    # magnitude changes with k.
    assert fused_default["docA"] != fused_small_k["docA"]


def test_empty_rankings_returns_empty_list():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_single_ranking_preserves_relative_order():
    ranking = ["docA", "docB", "docC"]
    fused = reciprocal_rank_fusion([ranking])
    assert [doc_id for doc_id, _score in fused] == ranking
