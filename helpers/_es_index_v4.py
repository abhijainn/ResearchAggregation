"""
Lightweight Elasticsearch helpers for v4 search that use simulated answer text.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

from elasticsearch import Elasticsearch

DEFAULT_SOURCE_FIELDS: Sequence[str] = (
    "id",
    "doc_id",
    "chunk_id",
    "text",
    "section_title",
    "headings",
    "page_num",
    "char_start",
    "char_end",
)


def build_lexical_dsl(answer_text: str) -> Dict[str, Any]:
    """Construct a simple BM25 match query over the simulated answer text."""
    return {
        "_raw_query": answer_text,
        "query": {
            "match": {
                "text": {
                    "query": answer_text,
                    "operator": "and",
                }
            }
        },
    }


def search_index(
    es: Elasticsearch,
    index_name: str,
    dsl: Mapping[str, Any],
    *,
    size: int = 10,
    source_includes: Optional[Sequence[str]] = None,
    track_total_hits: Optional[object] = None,
) -> Dict[str, Any]:
    body = dict(dsl)
    body.setdefault("size", size)
    if source_includes is not None:
        body["_source"] = {"includes": list(source_includes)}
    if track_total_hits not in (None, False):
        body["track_total_hits"] = track_total_hits
    return es.search(index=index_name, body=body)


def knn_search(
    es: Elasticsearch,
    index_name: str,
    vector: Sequence[float],
    *,
    k: int = 50,
    source_includes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "knn": {
            "field": "specter2_vec",
            "query_vector": list(vector),
            "k": k,
            "num_candidates": max(k * 2, 100),
        }
    }
    if source_includes is not None:
        body["_source"] = {"includes": list(source_includes)}
    return es.search(index=index_name, body=body)
