import types
from typing import Any, Dict, Sequence

from helpers import _search


def test_search_prefers_semantic_hits(monkeypatch):
    """
    Ensure hybrid fusion surfaces a semantic hit even when lexical is lower.
    """

    query_obj = _search._keywords.Query(boosted_terms=["alpha"], normal_terms=["beta"], section_filter=None)

    def _fake_extract(question: str):
        return [query_obj], {"stub": True, "question": question}

    chunks = {
        "doc1#0": {"id": "doc1#0", "doc_id": "doc1", "text": "first chunk about alpha beta"},
        "doc1#1": {"id": "doc1#1", "doc_id": "doc1", "text": "second chunk about beta only"},
    }

    def _stub_build_lexical_dsl(query_input: Any) -> Dict[str, Any]:
        return {"_raw_query": query_input}

    def _stub_search_index(
        es: Any,
        index_name: str,
        dsl: Dict[str, Any],
        *,
        size: int,
        source_includes: Sequence[str] | None,
        track_total_hits: object | None,
    ) -> Dict[str, Any]:
        del es, index_name, track_total_hits
        # Lexical favors chunk0
        hits = [
            {"_id": "doc1#0", "_score": 10.0, "_source": chunks["doc1#0"]},
            {"_id": "doc1#1", "_score": 5.0, "_source": chunks["doc1#1"]},
        ]
        return {"hits": {"hits": hits[:size]}, "took": 1}

    def _stub_knn_search(
        es: Any,
        index_name: str,
        vector: Sequence[float],
        *,
        k: int,
        num_candidates: int | None = None,
        filter_query: Dict[str, Any] | None = None,
        source_includes: Sequence[str] | None = None,
    ) -> Dict[str, Any]:
        del es, index_name, num_candidates, filter_query, source_includes, vector
        # Semantic favors chunk1
        hits = [{"_id": "doc1#1", "_score": 20.0, "_source": chunks["doc1#1"]}]
        return {"hits": {"hits": hits}, "took": 1}

    monkeypatch.setattr(_search._keywords, "extract_queries", _fake_extract)
    monkeypatch.setattr(_search._es_index, "build_lexical_dsl", _stub_build_lexical_dsl)
    monkeypatch.setattr(_search._es_index, "search_index", _stub_search_index)
    monkeypatch.setattr(_search._es_index, "knn_search", _stub_knn_search)

    dummy_embedder = types.SimpleNamespace(encode=lambda text: [1.0])
    result = _search.search(
        es=object(),
        question="alpha beta",
        index_name="idx",
        embedder=dummy_embedder,  # bypass Specter2 dependency
    )

    assert result.contexts, "Expected contexts from hybrid search"
    top_chunk = result.contexts[0]["chunk_ids"][0]
    assert top_chunk == "doc1#1", "Semantic-preferred chunk should rank first"
    fused_scores = [ctx["provenance"]["fused_score"] for ctx in result.contexts]
    assert all(score > 0 for score in fused_scores)
