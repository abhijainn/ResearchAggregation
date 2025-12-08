"""
Elasticsearch index + retrieval utilities for chunk-level quote finding.

This module centralises the responsibilities described in the hybrid-search
blueprint:
    * index lifecycle management (settings, analyzers, mappings)
    * document chunking + metadata enrichment
    * ingestion helpers (single doc + bulk)
    * lexical + semantic search entrypoints
    * doc-anchor discovery utilities for downstream query builders
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import ssl
from urllib.parse import urlparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

try:
    from elasticsearch import Elasticsearch, exceptions as es_exceptions, helpers as es_helpers
except ModuleNotFoundError:  # pragma: no cover - allows smoke tests without dependency
    class _UnavailableElasticsearch:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
            raise RuntimeError(
                "Elasticsearch client is unavailable; install the 'elasticsearch' package to proceed."
            )

    class _UnavailableHelpers:
        def __getattr__(self, name: str) -> Any:
            raise RuntimeError(
                f"Elasticsearch helpers are unavailable (missing dependency); attempted to access '{name}'."
            )

    class _UnavailableExceptions:
        ConnectionError = RuntimeError

    Elasticsearch = _UnavailableElasticsearch  # type: ignore
    es_helpers = _UnavailableHelpers()  # type: ignore
    es_exceptions = _UnavailableExceptions  # type: ignore

try:
    from ._log import log  # type: ignore
except ImportError:  # pragma: no cover
    from _log import log  # type: ignore

try:  # Prefer v3 keyword utilities when available
    from ._keywords_v3 import Query, normalize_keyword_terms, extract_queries  # type: ignore
except ImportError:  # pragma: no cover
    try:
        from _keywords_v3 import Query, normalize_keyword_terms, extract_queries  # type: ignore
    except ImportError:  # pragma: no cover
        try:
            from ._keywords_v2 import Query, normalize_keyword_terms, extract_queries  # type: ignore
        except ImportError:  # pragma: no cover
            try:
                from _keywords_v2 import Query, normalize_keyword_terms, extract_queries  # type: ignore
            except ImportError:  # pragma: no cover
                try:
                    from ._keywords import Query, normalize_keyword_terms, extract_queries  # type: ignore
                except ImportError:  # pragma: no cover
                    from _keywords import Query, normalize_keyword_terms, extract_queries  # type: ignore

try:
    from ._load_data import Specter2Embeddings, EMBED_MODEL_NAME as _DEFAULT_EMBED_MODEL  # type: ignore
except ImportError:  # pragma: no cover
    try:
        from _load_data import Specter2Embeddings, EMBED_MODEL_NAME as _DEFAULT_EMBED_MODEL  # type: ignore
    except ImportError:  # pragma: no cover - embeddings optional
        Specter2Embeddings = None  # type: ignore
        _DEFAULT_EMBED_MODEL = None  # type: ignore

DEFAULT_EMBED_MODEL = _DEFAULT_EMBED_MODEL if _DEFAULT_EMBED_MODEL else None

__all__ = [
    "Chunk",
    "ChunkingConfig",
    "create_index",
    "delete_index",
    "upsert_mappings",
    "chunk_document",
    "index_document",
    "bulk_index_chunks",
    "build_lexical_dsl",
    "search_index",
    "knn_search",
    "fetch_chunk_by_id",
    "get_doc_anchors",
    "DEFAULT_FIELD_BOOSTS",
]


DEFAULT_FIELD_BOOSTS: Mapping[str, float] = {
    "text": 1.0,
    "text_shingle": 1.2,
    "section_title": 3.0,
    "headings": 2.5,
    "model_names": 2.0,
}

TOKEN_PATTERN = re.compile(r"\S+")
NUMERIC_HEADING_RE = re.compile(r"^(?:[0-9]+(?:\.[0-9]+)*)\s+[^\d]+")
MODEL_NAME_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9]*(?:Net|Former|BERT|GPT|GAN|LM|Transformer|Model|Network|System))\b"
)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

HEADER_SCORE_FLOOR = 1e-6  # rank_feature fields must be positive; treat ~0 as tiny positive

SMOKE_SOURCE_FIELDS: Sequence[str] = (
    "text",
    "section_title",
    "headings",
    "doc_id",
    "chunk_id",
    "page_num",
    "id",
)

STRUCTURAL_METADATA_KEYS = {
    "sections",
    "headings",
    "section_spans",
    "page_spans",
    "pages",
    "doc_metadata",
}


@dataclass
class ChunkingConfig:
    """Configuration for doc->chunk conversion."""

    tokens_per_chunk: int = 512
    overlap_tokens: int = 128
    min_chunk_tokens: int = 40
    header_window_chars: int = 400
    detect_model_names: bool = True
    model_name_regex: str = MODEL_NAME_RE.pattern
    default_section_title: str = "Body"

    def __post_init__(self) -> None:
        if self.tokens_per_chunk <= 0:
            raise ValueError("tokens_per_chunk must be positive.")
        if self.overlap_tokens < 0:
            raise ValueError("overlap_tokens must be >= 0.")
        if self.overlap_tokens >= self.tokens_per_chunk:
            raise ValueError("overlap_tokens must be smaller than tokens_per_chunk.")
        if self.min_chunk_tokens <= 0:
            raise ValueError("min_chunk_tokens must be positive.")

    @property
    def model_name_pattern(self) -> re.Pattern[str]:
        return re.compile(self.model_name_regex)


@dataclass
class Chunk:
    """Chunk-as-document representation."""

    id: str
    doc_id: str
    chunk_id: int
    text: str
    char_start: int
    char_end: int
    token_count: int
    section_title: str
    section_level: int
    headings: List[str]
    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    page_num: Optional[int] = None
    header_score: float = 0.0
    model_names: List[str] = field(default_factory=list)
    specter2_vec: Optional[Sequence[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    keyword_key: List[str] = field(default_factory=list)

    def to_document(self) -> Dict[str, Any]:
        document = {
            "id": self.id,
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "text_shingle": self.text,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "token_count": self.token_count,
            "section_title": self.section_title,
            "section_level": self.section_level,
            "headings": " > ".join(head for head in self.headings if head),
            "prev_chunk_id": self.prev_chunk_id,
            "next_chunk_id": self.next_chunk_id,
            "page_num": self.page_num,
            "header_score": self.header_score,
            "model_names": list(dict.fromkeys(self.model_names)),
        }
        if self.specter2_vec is not None:
            document["specter2_vec"] = list(self.specter2_vec)
        if self.keyword_key:
            document["keyword_key"] = list(self.keyword_key)
        document.update(self.metadata)
        return document


def create_index(
    es: Elasticsearch,
    index_name: str,
    *,
    recreate: bool = False,
    vector_dims: int = 768,
    extra_settings: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Create an Elasticsearch index with analyzers/mappings suitable for hybrid retrieval.
    """

    try:
        exists = es.indices.exists(index=index_name)
    except es_exceptions.ConnectionError:
        raise

    if exists and recreate:
        es.indices.delete(index=index_name)
        exists = False

    if exists:
        log(f"Index '{index_name}' already exists; skipping creation.")
        return

    settings = _build_index_settings(vector_dims=vector_dims)
    if extra_settings:
        settings = _merge_dict(settings, extra_settings)

    es.indices.create(index=index_name, **settings)
    log(f"Created index '{index_name}'.")


def delete_index(es: Elasticsearch, index_name: str) -> None:
    """Delete the given index if it exists."""
    try:
        if es.indices.exists(index=index_name):
            es.indices.delete(index=index_name)
            log(f"Deleted index '{index_name}'.")
    except es_exceptions.ConnectionError:
        raise


def upsert_mappings(
    es: Elasticsearch, index_name: str, mapping_patch: Mapping[str, Any]
) -> None:
    """Apply mapping updates to an existing index."""
    es.indices.put_mapping(index=index_name, body=mapping_patch)


def chunk_document(
    doc_id: str,
    text: str,
    metadata: Optional[Mapping[str, Any]] = None,
    chunk_cfg: Optional[ChunkingConfig] = None,
) -> List[Chunk]:
    """Split the provided document text into overlapping, metadata-rich chunks."""
    chunk_cfg = chunk_cfg or ChunkingConfig()
    tokens = list(TOKEN_PATTERN.finditer(text))
    if not tokens:
        return []

    meta = dict(metadata or {})
    doc_metadata = dict(meta.get("doc_metadata") or {})
    if not doc_metadata:
        doc_metadata = {
            k: v
            for k, v in meta.items()
            if k not in STRUCTURAL_METADATA_KEYS and not k.startswith("_")
        }

    sections = _build_sections(
        meta.get("sections")
        or meta.get("headings")
        or meta.get("section_spans")
        or [],
        text=text,
        fallback_title=chunk_cfg.default_section_title,
    )
    page_spans = meta.get("page_spans") or meta.get("pages") or []
    model_name_pattern = chunk_cfg.model_name_pattern if chunk_cfg.detect_model_names else None

    chunks: List[Chunk] = []
    start_token = 0
    chunk_index = 0

    while start_token < len(tokens):
        end_token = min(len(tokens), start_token + chunk_cfg.tokens_per_chunk)
        token_slice = tokens[start_token:end_token]
        if len(token_slice) < chunk_cfg.min_chunk_tokens and chunks:
            break

        char_start = token_slice[0].start()
        char_end = token_slice[-1].end()
        chunk_text = text[char_start:char_end].strip()
        if not chunk_text:
            start_token = end_token
            continue

        section_title, section_level, breadcrumb = _resolve_section(sections, char_start)
        header_score = 1.0 if _near_section_header(sections, char_start, chunk_cfg.header_window_chars) else HEADER_SCORE_FLOOR
        page_num = _resolve_page(page_spans, char_start)
        chunk_id = f"{doc_id}#{chunk_index:04d}"

        model_names = list(dict.fromkeys(meta.get("model_names", [])))
        if chunk_cfg.detect_model_names and model_name_pattern:
            regex_hits = model_name_pattern.findall(chunk_text)
            model_names.extend(regex_hits)

        keyword_terms = normalize_keyword_terms(chunk_text)

        chunk = Chunk(
            id=chunk_id,
            doc_id=doc_id,
            chunk_id=chunk_index,
            text=chunk_text,
            char_start=char_start,
            char_end=char_end,
            token_count=len(token_slice),
            section_title=section_title,
            section_level=section_level,
            headings=breadcrumb,
            page_num=page_num,
            header_score=header_score,
            model_names=list(dict.fromkeys(model_names)),
            metadata=copy.deepcopy(doc_metadata),
            keyword_key=keyword_terms,
        )
        if chunks:
            prev = chunks[-1]
            prev.next_chunk_id = chunk.id
            chunk.prev_chunk_id = prev.id
        chunks.append(chunk)

        if end_token >= len(tokens):
            break
        start_token = max(0, end_token - chunk_cfg.overlap_tokens)
        chunk_index += 1

    return chunks


def index_document(
    es: Elasticsearch,
    index_name: str,
    doc_id: str,
    text: str,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
    chunk_cfg: Optional[ChunkingConfig] = None,
    embedder: Optional[Callable[[List[str]], List[Sequence[float]]]] = None,
    refresh: bool = False,
) -> List[str]:
    """
    Chunk a document (if raw text is provided), attach embeddings when possible,
    and index the resulting records. Returns the chunk identifiers written.
    """

    chunks = chunk_document(doc_id=doc_id, text=text, metadata=metadata, chunk_cfg=chunk_cfg)
    if not chunks:
        return []

    if embedder is not None:
        texts = [chunk.text for chunk in chunks]
        vectors = embedder(texts)
        if len(vectors) != len(chunks):
            raise ValueError("Embedding function returned unexpected vector count.")
        for chunk, vector in zip(chunks, vectors):
            chunk.specter2_vec = vector

    actions = (
        {
            "_op_type": "index",
            "_index": index_name,
            "_id": chunk.id,
            "_source": chunk.to_document(),
        }
        for chunk in chunks
    )
    es_helpers.bulk(es, actions, refresh=refresh)
    log(f"Indexed {len(chunks)} chunks for doc '{doc_id}' into '{index_name}'.")
    return [chunk.id for chunk in chunks]


def bulk_index_chunks(
    es: Elasticsearch,
    index_name: str,
    chunks: Iterable[Mapping[str, Any] | Chunk],
    *,
    refresh: bool = False,
) -> None:
    """Index already-prepared chunk dictionaries."""

    def _iter_actions() -> Iterable[Mapping[str, Any]]:
        for chunk in chunks:
            if isinstance(chunk, Chunk):
                document = chunk.to_document()
            else:
                document = dict(chunk)
            chunk_id = document.get("id")
            if not chunk_id:
                raise ValueError("Each chunk must include a stable 'id'.")
            yield {
                "_op_type": "index",
                "_index": index_name,
                "_id": chunk_id,
                "_source": document,
            }

    es_helpers.bulk(es, _iter_actions(), refresh=refresh)


def build_lexical_dsl(
    query_input: Query,
    *,
    field_boosts: Optional[Mapping[str, float]] = None,
    use_rescore: bool = True,
    rescore_window: int = 200,
) -> Dict[str, Any]:
    """Construct a dis_max + boosts DSL for the provided keyword query."""

    parsed, section_filter = _normalize_query_input(query_input)
    boosts = field_boosts or DEFAULT_FIELD_BOOSTS

    boosted_fields = [
        f"{field}^{boost}"
        for field, boost in boosts.items()
        if field in {"text", "text_shingle", "section_title", "headings", "model_names"}
    ]
    if not boosted_fields:
        boosted_fields = ["text^1.0"]

    must_clauses: List[Dict[str, Any]] = []
    phrase_shoulds: List[Dict[str, Any]] = []

    for clause in parsed.clauses:
        clause_query = clause.text
        if not clause_query:
            continue
        if clause.is_phrase:
            phrase_shoulds.append(
                {
                    "multi_match": {
                        "query": clause_query,
                        "type": "phrase",
                        "slop": 1,
                        "fields": [
                            f"section_title^{boosts.get('section_title', 3.0)}",
                            f"headings^{boosts.get('headings', 2.5)}",
                        ],
                    }
                }
            )
        match_body: Dict[str, Any] = {
            "query": clause_query,
            "fields": boosted_fields,
        }
        if clause.must:
            match_body["operator"] = "and"
            must_clauses.append({"multi_match": match_body})

    dis_max_query = {
        "dis_max": {
            "queries": [
                {
                    "multi_match": {
                        "query": parsed.combined,
                        "fields": boosted_fields,
                        "type": "best_fields",
                        "tie_breaker": 0.2,
                    }
                },
                {
                    "multi_match": {
                        "query": parsed.combined,
                        "fields": ["text^1.0"],
                        "type": "phrase",
                        "slop": 2,
                    }
                },
            ],
            "tie_breaker": 0.1,
        }
    }

    keyword_terms = normalize_keyword_terms(parsed.combined) if parsed.combined else []

    keyword_should: Optional[Dict[str, Any]] = None
    if keyword_terms:
        keyword_should = {
            "terms_set": {
                "keyword_key": {
                    "terms": keyword_terms,
                    "minimum_should_match_script": {
                        "source": (
                            "int required = (int)Math.ceil(params.num_terms * params.ratio); "
                            "return Math.max(1, required);"
                        ),
                        "params": {"num_terms": len(keyword_terms), "ratio": 0.6},
                    },
                }
            }
        }

    shingle_phrase = {
        "match_phrase": {
            "text_shingle": {
                "query": parsed.combined,
                "slop": 1,
                "boost": 1.5,
            }
        }
    }

    should_clauses: List[Dict[str, Any]] = [dis_max_query, shingle_phrase, *phrase_shoulds]
    if keyword_should:
        should_clauses.append(keyword_should)
    base_query: Dict[str, Any] = {
        "bool": {
            "must": must_clauses,
            "should": should_clauses,
            "minimum_should_match": 1 if not must_clauses else 0,
        }
    }
    if section_filter:
        base_query["bool"].setdefault("filter", []).append(
            {"term": {"section_title.keyword": section_filter}}
        )

    dsl: Dict[str, Any] = {
        "query": {
            "function_score": {
                "query": base_query,
                "functions": [
                    {
                        "script_score": {
                            "script": {
                                "source": (
                                    "doc['header_score'].size()==0 ? 0 : doc['header_score'].value"
                                )
                            }
                        }
                    }
                ],
                "boost_mode": "sum",
                "score_mode": "sum",
            }
        }
    }

    if use_rescore:
        dsl["rescore"] = {
            "window_size": rescore_window,
            "query": {
                "rescore_query": {
                    "bool": {
                        "should": [
                            {
                                "match_phrase": {
                                    "text": {
                                    "query": parsed.combined,
                                        "slop": 1,
                                    }
                                }
                            },
                            {
                                "match_phrase": {
                                    "text_shingle": {
                                    "query": parsed.combined,
                                        "slop": 1,
                                    }
                                }
                            },
                        ]
                    }
                },
                "query_weight": 1.0,
                "rescore_query_weight": 2.0,
            },
        }

    return dsl


def search_index(
    es: Elasticsearch,
    index_name: str,
    dsl: Mapping[str, Any],
    *,
    size: int = 10,
    source_includes: Optional[Sequence[str]] = None,
    track_total_hits: Optional[object] = None,
) -> Dict[str, Any]:
    """Execute a BM25 lexical search."""
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
    k: int = 200,
    num_candidates: Optional[int] = None,
    filter_query: Optional[Mapping[str, Any]] = None,
    source_includes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Execute an ANN cosine search over `specter2_vec`."""
    num_candidates = num_candidates or max(k * 2, 100)
    body: Dict[str, Any] = {
        "knn": {
            "field": "specter2_vec",
            "query_vector": list(vector),
            "k": k,
            "num_candidates": num_candidates,
        }
    }
    if filter_query:
        body["knn"]["filter"] = filter_query
    if source_includes is not None:
        body["_source"] = {"includes": list(source_includes)}
    return es.search(index=index_name, body=body)


def _hybrid_search(
    es: Elasticsearch,
    index_name: str,
    query: Query,
    *,
    embedder: Optional["Specter2Embeddings"],
    top_n: int,
    source_fields: Sequence[str],
) -> List[Dict[str, Any]]:
    query_terms = query.normal_terms or query.boosted_terms
    query_text = " ".join(term.strip() for term in query_terms if term.strip()) or ""
    normalized_query_terms = normalize_keyword_terms(query_text)
    lexical_size = max(top_n * 4, 20)
    dsl = build_lexical_dsl(query)
    lexical_response = search_index(
        es,
        index_name,
        dsl,
        size=lexical_size,
        source_includes=source_fields,
        track_total_hits=False,
    )
    lexical_hits = lexical_response.get("hits", {}).get("hits", [])

    combined: Dict[str, Dict[str, Any]] = {}
    lexical_scores: List[float] = []
    for hit in lexical_hits:
        chunk_id = hit.get("_id")
        if not chunk_id:
            continue
        entry = combined.setdefault(
            chunk_id,
            {
                "chunk_id": chunk_id,
                "source": hit.get("_source", {}),
                "lexical_score": 0.0,
                "vector_score": 0.0,
            },
        )
        entry["source"] = entry.get("source") or hit.get("_source", {})
        score = float(hit.get("_score") or 0.0)
        entry["lexical_score"] = max(entry.get("lexical_score", 0.0), score)
        lexical_scores.append(score)

    vector_scores: List[float] = []
    if embedder is not None:
        try:
            query_vector = embedder.embed_query(" ".join(query.normal_terms or query.boosted_terms))
        except Exception as exc:
            raise RuntimeError(f"Failed to embed query for hybrid search: {exc}") from exc
        knn_response = knn_search(
            es,
            index_name,
            query_vector,
            k=max(top_n * 5, 50),
            num_candidates=max(top_n * 20, 200),
            source_includes=source_fields,
        )
        vector_hits = knn_response.get("hits", {}).get("hits", [])
        for hit in vector_hits:
            chunk_id = hit.get("_id")
            if not chunk_id:
                continue
            entry = combined.setdefault(
                chunk_id,
                {
                    "chunk_id": chunk_id,
                    "source": hit.get("_source", {}),
                    "lexical_score": 0.0,
                    "vector_score": 0.0,
                },
            )
            entry["source"] = entry.get("source") or hit.get("_source", {})
            score = float(hit.get("_score") or 0.0)
            entry["vector_score"] = max(entry.get("vector_score", 0.0), score)
            vector_scores.append(score)

    if not combined:
        return []

    lex_max = max(lexical_scores) if lexical_scores else 0.0
    vec_max = max(vector_scores) if vector_scores else 0.0
    for entry in combined.values():
        lex_component = (entry.get("lexical_score", 0.0) / lex_max) if lex_max else 0.0
        vec_component = (entry.get("vector_score", 0.0) / vec_max) if vec_max else 0.0
        entry["hybrid_score"] = lex_component + vec_component
        best_sentence = _best_sentence_match(
            entry["source"].get("text", ""),
            query_text,
            normalized_query_terms,
        )
        entry["best_sentence"] = best_sentence
        overlap = _term_overlap_score(best_sentence, normalized_query_terms)
        entry["keyword_overlap"] = overlap
        entry["final_score"] = entry["hybrid_score"] + (overlap * 0.5)
        entry["query"] = query_text

    ranked = sorted(
        combined.values(),
        key=lambda item: (
            item.get("final_score", item.get("hybrid_score", 0.0)),
            item.get("lexical_score", 0.0),
            item.get("vector_score", 0.0),
        ),
        reverse=True,
    )
    return ranked[:top_n]


def _interactive_query_loop(
    es: Elasticsearch,
    index_name: str,
    *,
    embedder: Optional["Specter2Embeddings"],
    top_n: int,
    source_fields: Sequence[str],
) -> None:
    print("\n=== Interactive Hybrid Search ===")
    print("Enter a question to retrieve matching chunks (press Enter or Ctrl-D to exit).")
    if embedder is None:
        print("[INFO] Embeddings unavailable; falling back to lexical scoring only.")
    while True:
        try:
            user_query = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting interactive search.")
            break
        if not user_query:
            print("Exiting interactive search.")
            break
        try:
            query_obj = Query(
                boosted_terms=[],
                normal_terms=_tokenize_question_for_smoke(user_query),
                section_filter=None,
            )
            hits = _hybrid_search(
                es,
                index_name,
                query=query_obj,
                embedder=embedder,
                top_n=top_n,
                source_fields=source_fields,
            )
        except Exception as exc:
            print(f"[FAIL] Hybrid search failed: {exc}")
            continue
        if not hits:
            print("No hits returned.")
            continue
        display_count = min(top_n, len(hits))
        print(f"\nTop {display_count} chunks (requested {top_n}):")
        for rank, entry in enumerate(hits[:top_n], start=1):
            source = entry.get("source", {})
            section = source.get("section_title") or "Unknown section"
            headings = source.get("headings")
            doc_id = source.get("doc_id")
            page_num = source.get("page_num")
            lex_score = entry.get("lexical_score", 0.0)
            vec_score = entry.get("vector_score", 0.0)
            hybrid_score = entry.get("hybrid_score", 0.0)
            print(
                f"#{rank} chunk_id={entry.get('chunk_id')} doc_id={doc_id} "
                f"section='{section}' page={page_num} hybrid={hybrid_score:.3f} "
                f"(lex={lex_score:.3f}, vec={vec_score:.3f})"
            )
            if headings:
                print(f"   headings: {headings}")
            best_sentence = entry.get("best_sentence") or ""
            if best_sentence:
                print(f"   sentence: {best_sentence}")
            chunk_text = (source.get("text") or "").strip()
            if chunk_text:
                formatted = "\n      ".join(chunk_text.splitlines())
                print("   text:")
                print(f"      {formatted}")
        print("")


def _run_demo_query(
    es: Elasticsearch,
    index_name: str,
    question: str,
    *,
    embedder: Optional["Specter2Embeddings"],
    top_n: int,
    source_fields: Sequence[str],
) -> None:
    print("\n=== Demo Keyword Query ===")
    print(f"Question: {question.strip()}")
    keyword_queries, _debug = extract_queries(question)
    if not keyword_queries:
        print("[WARN] Keyword extractor returned no queries for the demo question.")
        return
    query_obj = keyword_queries[0]
    try:
        hits = _hybrid_search(
            es,
            index_name,
            query=query_obj,
            embedder=embedder,
            top_n=top_n,
            source_fields=source_fields,
        )
    except Exception as exc:
        print(f"[FAIL] Demo query failed: {exc}")
        return
    if not hits:
        print("[INFO] Demo query returned no matching chunks.")
        return
    print(f"Top {min(top_n, len(hits))} chunks (quote shown first):")
    for rank, entry in enumerate(hits[:top_n], start=1):
        source = entry.get("source", {})
        quote = entry.get("best_sentence") or ""
        chunk_text = (source.get("text") or "").strip()
        doc_id = source.get("doc_id")
        section = source.get("section_title") or "Unknown section"
        print(
            f"#{rank} chunk_id={entry.get('chunk_id')} doc_id={doc_id} section='{section}' "
            f"hybrid={entry.get('hybrid_score', 0.0):.3f}"
        )
        if quote:
            print(f"   quote: {quote}")
        if chunk_text:
            formatted = "\n      ".join(chunk_text.splitlines())
            print("   chunk:")
            print(f"      {formatted}")


def fetch_chunk_by_id(es: Elasticsearch, index_name: str, chunk_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a single chunk document by its id."""
    try:
        record = es.get(index=index_name, id=chunk_id)
    except es_exceptions.NotFoundError:
        return None
    return record.get("_source")


def get_doc_anchors(
    es: Elasticsearch,
    index_name: str,
    doc_ids: Sequence[str],
    *,
    top_n_headers: int = 5,
    top_n_models: Optional[int] = None,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Return representative section headers + model names per document for downstream keyword building.
    """
    if not doc_ids:
        return {}
    top_n_models = top_n_models or top_n_headers
    aggs_body = {
        "size": 0,
        "query": {"terms": {"doc_id": list(dict.fromkeys(doc_ids))}},
        "aggs": {
            "docs": {
                "terms": {"field": "doc_id", "size": len(doc_ids)},
                "aggs": {
                    "sections": {
                        "terms": {
                            "field": "section_title.keyword",
                            "size": top_n_headers,
                            "order": {"_count": "desc"},
                        }
                    },
                    "models": {
                        "terms": {
                            "field": "model_names",
                            "size": top_n_models,
                            "order": {"_count": "desc"},
                        }
                    },
                },
            }
        },
    }
    response = es.search(index=index_name, body=aggs_body)
    aggregations = response.get("aggregations", {})
    doc_buckets = aggregations.get("docs", {}).get("buckets", [])
    anchors: Dict[str, Dict[str, List[str]]] = {}
    for bucket in doc_buckets:
        doc_id = bucket.get("key")
        section_titles = [sec["key"] for sec in bucket.get("sections", {}).get("buckets", []) if sec.get("key")]
        model_names = [mod["key"] for mod in bucket.get("models", {}).get("buckets", []) if mod.get("key")]
        anchors[doc_id] = {
            "section_titles": section_titles,
            "model_names": model_names,
        }
    return anchors


def _tokenize_question_for_smoke(question: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?", question or "")
    return tokens or (question.split() if question else [])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _merge_dict(base: Mapping[str, Any], patch: Mapping[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _build_index_settings(*, vector_dims: int) -> Dict[str, Any]:
    settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {
                "analyzer": {
                    "english_extended": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "english_stop", "kstem"],
                    },
                    "text_shingle": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": [
                            "lowercase",
                            "english_stop",
                            "kstem",
                            "shingle_filter",
                        ],
                    },
                },
                "filter": {
                    "english_stop": {"type": "stop", "stopwords": "_english_"},
                    "shingle_filter": {
                        "type": "shingle",
                        "min_shingle_size": 2,
                        "max_shingle_size": 3,
                        "output_unigrams": False,
                    },
                },
                "normalizer": {
                    "lowercase_keyword": {
                        "type": "custom",
                        "filter": ["lowercase"],
                    }
                },
            },
            "similarity": {
                "bm25": {"type": "BM25", "k1": 1.4, "b": 0.75},
            },
        },
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "id": {"type": "keyword"},
                "doc_id": {"type": "keyword"},
                "chunk_id": {"type": "integer"},
                "prev_chunk_id": {"type": "keyword"},
                "next_chunk_id": {"type": "keyword"},
                "section_level": {"type": "integer"},
                "section_title": {
                    "type": "text",
                    "analyzer": "english_extended",
                    "fields": {"keyword": {"type": "keyword"}},
                },
                "headings": {
                    "type": "text",
                    "analyzer": "english_extended",
                    "fields": {"keyword": {"type": "keyword"}},
                },
                "model_names": {
                    "type": "keyword",
                    "normalizer": "lowercase_keyword",
                },
                "text": {
                    "type": "text",
                    "analyzer": "english_extended",
                },
                "text_shingle": {
                    "type": "text",
                    "analyzer": "text_shingle",
                },
                "keyword_key": {
                    "type": "keyword",
                    "normalizer": "lowercase_keyword",
                },
                "header_score": {"type": "float"},
                "page_num": {"type": "integer"},
                "char_start": {"type": "integer"},
                "char_end": {"type": "integer"},
                "token_count": {"type": "integer"},
                "published_at": {"type": "date", "format": "strict_date_optional_time"},
            },
        },
    }

    if vector_dims:
        settings["mappings"]["properties"]["specter2_vec"] = {
            "type": "dense_vector",
            "dims": vector_dims,
            "index": True,
            "similarity": "cosine",
            "index_options": {"type": "hnsw", "m": 16, "ef_construction": 128},
        }
    return settings


def _build_sections(
    raw_sections: Sequence[Mapping[str, Any]],
    *,
    text: str,
    fallback_title: str,
) -> List[Tuple[int, str, int]]:
    sections: List[Tuple[int, str, int]] = []
    for entry in raw_sections:
        title = (entry.get("title") or entry.get("section_title") or "").strip()
        if not title:
            continue
        start = int(entry.get("char_start") or entry.get("start") or 0)
        level = int(entry.get("level") or entry.get("section_level") or 1)
        sections.append((start, title, level))

    if not sections:
        sections = _heuristic_sections(text)

    if not sections:
        sections = [(0, fallback_title, 1)]

    sections.sort(key=lambda item: item[0])
    return sections


def _heuristic_sections(text: str) -> List[Tuple[int, str, int]]:
    sections: List[Tuple[int, str, int]] = []
    offset = 0
    for line in text.splitlines(True):
        stripped = line.strip()
        if _looks_like_heading(stripped):
            level = 1
            if NUMERIC_HEADING_RE.match(stripped):
                # Use number of dots as proxy for depth.
                level = stripped.count(".") + 1
            sections.append((offset, stripped.rstrip(":").strip(), level))
        offset += len(line)
    return sections


def _looks_like_heading(line: str) -> bool:
    if not line:
        return False
    if len(line) > 120:
        return False
    if line.endswith("."):
        return False
    if NUMERIC_HEADING_RE.match(line):
        return True
    alpha_chars = sum(1 for ch in line if ch.isalpha())
    if alpha_chars == 0:
        return False
    upper_ratio = sum(1 for ch in line if ch.isupper()) / alpha_chars
    title_case = line == line.title()
    return upper_ratio > 0.6 or title_case


def _resolve_section(
    sections: Sequence[Tuple[int, str, int]], char_start: int
) -> Tuple[str, int, List[str]]:
    chosen_title = sections[0][1]
    chosen_level = sections[0][2]
    breadcrumb: List[str] = []

    chosen_index = 0
    for index, (start, _title, _level) in enumerate(sections):
        if start <= char_start:
            chosen_index = index
        else:
            break
    _chosen_start, chosen_title, chosen_level = sections[chosen_index]
    breadcrumb = [
        title for start, title, _level in sections[: chosen_index + 1] if start <= char_start
    ]
    if not breadcrumb:
        breadcrumb = [chosen_title]
    return chosen_title, chosen_level, breadcrumb


def _near_section_header(
    sections: Sequence[Tuple[int, str, int]],
    char_start: int,
    window: int,
) -> bool:
    for start, _title, _level in sections:
        if 0 <= char_start - start <= window:
            return True
    return False


def _resolve_page(page_spans: Sequence[Mapping[str, Any]], char_start: int) -> Optional[int]:
    for span in page_spans:
        page = span.get("page") or span.get("page_num")
        start = span.get("start") or span.get("char_start") or 0
        end = span.get("end") or span.get("char_end") or math.inf
        if start <= char_start < end:
            return int(page)
    return None


def _best_sentence_match(text: str, query: str, normalized_terms: Sequence[str]) -> str:
    if not text:
        return ""
    sentences = [segment.strip() for segment in SENTENCE_SPLIT_RE.split(text) if segment.strip()]
    if not sentences:
        sentences = [text.strip()]
    query_vocab = {term for term in normalized_terms if term}
    best_sentence = sentences[0]
    best_score = -1.0
    for sentence in sentences:
        if not _is_informative_sentence(sentence):
            continue
        sentence_terms = normalize_keyword_terms(sentence)
        if query_vocab:
            overlap = len(query_vocab & set(sentence_terms))
            if overlap == 0:
                coverage = 0.0
            else:
                coverage = overlap / max(1, len(query_vocab))
        else:
            coverage = 0.0
        alpha_ratio = _alpha_ratio(sentence)
        score = coverage + alpha_ratio * 0.1
        if score > best_score:
            best_sentence = sentence
            best_score = score
    return best_sentence


def _is_informative_sentence(sentence: str) -> bool:
    stripped = sentence.strip()
    if not stripped:
        return False
    if len(stripped) < 12:
        return False
    if stripped.isupper() and len(stripped) > 15:
        return False
    if _alpha_ratio(stripped) < 0.4:
        return False
    unique_tokens = set(re.findall(r"\w+", stripped.lower()))
    return len(unique_tokens) >= 4


def _term_overlap_score(sentence: str, query_terms: Sequence[str]) -> float:
    if not sentence or not query_terms:
        return 0.0
    sentence_terms = set(normalize_keyword_terms(sentence))
    if not sentence_terms:
        return 0.0
    query_vocab = {term for term in query_terms if term}
    if not query_vocab:
        return 0.0
    overlap = len(sentence_terms & query_vocab)
    if overlap == 0:
        return 0.0
    return overlap / max(1, len(query_vocab))


def _alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    alpha = sum(1 for ch in text if ch.isalpha())
    return alpha / max(1, len(text))


@dataclass
class ParsedClause:
    text: str
    is_phrase: bool
    must: bool


@dataclass
class ParsedQuery:
    clauses: List[ParsedClause]
    combined: str


def _parse_query_string(raw_query: str) -> ParsedQuery:
    clauses: List[ParsedClause] = []
    text_terms: List[str] = []
    i = 0
    length = len(raw_query)
    while i < length:
        while i < length and raw_query[i].isspace():
            i += 1
        if i >= length:
            break
        must_flag = False
        if raw_query[i] in {"+", "-"}:
            must_flag = raw_query[i] == "+"
            i += 1
            while i < length and raw_query[i].isspace():
                i += 1
        if i >= length:
            break
        if raw_query[i] == '"':
            i += 1
            start = i
            while i < length and raw_query[i] != '"':
                i += 1
            phrase = raw_query[start:i].strip()
            if i < length and raw_query[i] == '"':
                i += 1
            if phrase:
                clauses.append(ParsedClause(text=phrase, is_phrase=True, must=must_flag))
                text_terms.append(phrase)
        else:
            start = i
            while i < length and not raw_query[i].isspace():
                i += 1
            term = raw_query[start:i].strip()
            if term:
                clauses.append(ParsedClause(text=term, is_phrase=False, must=must_flag))
                text_terms.append(term)
    joined = " ".join(text_terms)
    return ParsedQuery(clauses=clauses, combined=joined)


def _normalize_query_input(query_input: Query) -> Tuple[ParsedQuery, Optional[str]]:
    parsed = _parsed_from_query_object(query_input)
    combined = parsed.combined or " ".join(query_input.normal_terms or query_input.boosted_terms)
    combined = combined.strip() or "keyword"
    parsed = ParsedQuery(clauses=parsed.clauses, combined=combined)
    return parsed, query_input.section_filter


def _parsed_from_query_object(query: Query) -> ParsedQuery:
    clauses: List[ParsedClause] = []
    seen: Set[Tuple[str, bool]] = set()

    def _append(term: str, must: bool) -> None:
        normalized = term.strip()
        if not normalized:
            return
        key = (normalized.lower(), must)
        if key in seen:
            return
        seen.add(key)
        clauses.append(
            ParsedClause(
                text=normalized,
                is_phrase=_is_phrase(normalized),
                must=must,
            )
        )

    for term in query.boosted_terms:
        _append(term, True)
    for term in query.normal_terms:
        _append(term, False)

    combined_terms = query.normal_terms or query.boosted_terms
    combined = " ".join(term.strip() for term in combined_terms if term.strip())
    return ParsedQuery(clauses=clauses, combined=combined)


def _is_phrase(term: str) -> bool:
    return bool(re.search(r"\s", term))


def _run_arxiv_smoke_test(
    es: Elasticsearch,
    *,
    index_name: str,
    payload_path: Path,
    arxiv_id: str = "2504.04915",
    question: Optional[str] = None,
    recreate_index: bool = False,
    embedder: Optional["Specter2Embeddings"] = None,
    hybrid_top_k: int = 3,
    demo_question: Optional[str] = None,
    demo_top_k: int = 5,
    run_demo: bool = False,
) -> None:
    """Ingest a cached arXiv payload into Elasticsearch and launch an interactive hybrid search."""

    log(
        f"Running Elasticsearch helper smoke test for {arxiv_id} "
        f"using payload '{payload_path.name}' -> index '{index_name}'"
    )

    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:  # pragma: no cover - failing fast is sufficient
        raise RuntimeError(f"Cached payload is missing: {payload_path}") from exc

    text = payload.get("processed_text") or payload.get("full_text") or ""
    if not text:
        raise RuntimeError("Cached payload did not include any text.")

    doc_id = payload.get("arxiv_id") or arxiv_id
    metadata = {
        "doc_metadata": {
            "title": payload.get("title"),
            "authors": payload.get("authors"),
            "categories": payload.get("categories"),
            "published": payload.get("published"),
            "doi": payload.get("doi"),
        }
    }
    structural_metadata = {
        key: payload[key]
        for key in ("sections", "headings", "section_spans", "page_spans", "pages")
        if key in payload
    }

    print("\n=== Document Metadata ===")
    print(json.dumps(metadata["doc_metadata"], indent=2))

    chunks = chunk_document(doc_id=doc_id, text=text, metadata=structural_metadata)
    if not chunks:
        raise RuntimeError("Chunking produced zero records.")

    vector_dims = 0
    if embedder is not None:
        print("[INFO] Computing chunk embeddings for hybrid search...")
        try:
            chunk_texts = [chunk.text for chunk in chunks]
            vectors = embedder.embed_documents(chunk_texts)
        except Exception as exc:
            raise RuntimeError(f"Failed to compute chunk embeddings: {exc}") from exc
        if vectors and len(vectors) != len(chunks):
            raise RuntimeError(
                f"Embedding count ({len(vectors)}) did not match chunk count ({len(chunks)})."
            )
        if vectors:
            vector_dims = len(vectors[0])
            for chunk, vector in zip(chunks, vectors):
                chunk.specter2_vec = vector
        print("[INFO] Chunk embeddings ready.")
    else:
        print("[INFO] Embeddings disabled; hybrid search will be lexical-only.")

    create_index(es, index_name=index_name, recreate=recreate_index, vector_dims=vector_dims or 0)
    try:
        bulk_index_chunks(es, index_name=index_name, chunks=chunks, refresh=True)
    except es_helpers.BulkIndexError as exc:
        errors = getattr(exc, "errors", [])
        sample_errors = errors[:3] if errors else []
        raise RuntimeError(
            f"Bulk indexing failed for {len(errors)} document(s). Sample: {sample_errors}"
        ) from exc
    log(f"Indexed {len(chunks)} chunks for doc '{doc_id}' into '{index_name}'.")

    sample_chunk = chunks[0].to_document()
    print("\n=== First Chunk Preview ===")
    print(
        json.dumps(
            {
                "id": sample_chunk["id"],
                "section_title": sample_chunk["section_title"],
                "token_count": sample_chunk["token_count"],
                "model_names": sample_chunk.get("model_names", []),
                "headings": sample_chunk.get("headings"),
            },
            indent=2,
        )
    )

    title = metadata["doc_metadata"].get("title") or f"arXiv:{doc_id}"
    query_text = question or f"What problem does '{title}' aim to solve?"
    query_obj = Query(
        boosted_terms=[],
        normal_terms=_tokenize_question_for_smoke(query_text),
        section_filter=None,
    )
    dsl = build_lexical_dsl(query_obj)
    print("\n=== Sample Lexical DSL ===")
    print(json.dumps(dsl, indent=2))

    _interactive_query_loop(
        es,
        index_name,
        embedder=embedder,
        top_n=hybrid_top_k,
        source_fields=SMOKE_SOURCE_FIELDS,
    )

    if run_demo:
        demo_text = demo_question or "What problem does MoC Mixtures of Text Chunking Learners for Retrieval-Augmented Generation System aim to solve"
        _run_demo_query(
            es,
            index_name,
            demo_text,
            embedder=embedder,
            top_n=demo_top_k,
            source_fields=SMOKE_SOURCE_FIELDS,
        )


def _build_smoke_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chunk a cached arXiv payload, index it into Elasticsearch, and run a sample query."
    )
    default_payload = Path(__file__).resolve().parent.parent / "2402.09642.json"
    parser.add_argument("--payload", default=str(default_payload), help="Path to cached get_full_text JSON payload.")
    parser.add_argument("--index", default="arxiv-chunks-smoke", help="Elasticsearch index name to use.")
    parser.add_argument("--host", default="https://localhost:9200", help="Elasticsearch host URL.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to display per query.")
    parser.add_argument("--arxiv-id", default="2504.04915", help="Fallback arXiv identifier for metadata.")
    parser.add_argument("--question", help="Custom question to convert into a lexical DSL query.")
    parser.add_argument(
        "--recreate",
        action="store_true",
        default=True,
        help="Drop and recreate the Elasticsearch index before indexing (default: enabled).",
    )
    parser.add_argument(
        "--no-recreate",
        dest="recreate",
        action="store_false",
        help="Do not drop/recreate the index; reuse the existing mappings.",
    )
    parser.add_argument("--user", default="elastic", help="Elasticsearch basic-auth username.")
    parser.add_argument(
        "--password",
        help="Elasticsearch basic-auth password. Defaults to ELASTIC_PASSWORD if omitted.",
    )
    parser.add_argument(
        "--ca-cert",
        dest="ca_cert",
        default="~/http_ca.crt",
        help="Path to a CA certificate for HTTPS connections (default: ~/http_ca.crt).",
    )
    parser.add_argument(
        "--allow-insecure",
        action="store_true",
        help="Disable TLS verification. Use only for local testing.",
    )
    parser.add_argument(
        "--run-demo-query",
        action="store_true",
        help="Run a keyword-search demo using helpers._keywords after the interactive loop.",
    )
    parser.add_argument(
        "--demo-question",
        default="What problem does MoC Mixtures of Text Chunking Learners for Retrieval-Augmented Generation System aim to solve",
        help="Question to feed into helpers._keywords for the demo query.",
    )
    parser.add_argument(
        "--demo-top-k",
        type=int,
        default=5,
        help="Number of demo results to display when --run-demo-query is set.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBED_MODEL,
        help="Embedding model to use for hybrid search. Defaults to config vector model when available.",
    )
    parser.add_argument(
        "--embedding-device",
        help="Device identifier for the embedding model (e.g., cpu, cuda).",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=16,
        help="Batch size for embedding inference.",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Disable embeddings and hybrid vector search.",
    )
    return parser


def _main() -> None:
    parser = _build_smoke_arg_parser()
    args = parser.parse_args()

    payload_path = Path(args.payload).expanduser()
    if not payload_path.exists():
        raise SystemExit(f"Payload file not found: {payload_path}")
    payload_path = payload_path.resolve()

    password = os.getenv("ELASTIC_PASSWORD")
    if args.user and not password:
        raise SystemExit("Password not provided. Supply --password or set ELASTIC_PASSWORD.")

    es_kwargs: Dict[str, Any] = {}
    parsed_host = urlparse(args.host)
    use_https = parsed_host.scheme == "https"
    if use_https:
        if args.ca_cert:
            ca_cert_path = Path(args.ca_cert).expanduser()
            if not ca_cert_path.exists():
                raise SystemExit(f"CA certificate not found: {ca_cert_path}")
            try:
                ssl_context = ssl.create_default_context(cafile=str(ca_cert_path))
            except Exception as exc:
                raise SystemExit(f"Failed to load CA certificate '{ca_cert_path}': {exc}") from exc
            es_kwargs["ssl_context"] = ssl_context
        elif not args.allow_insecure:
            print("[WARN] No CA certificate supplied; relying on system trust store.")
        if args.allow_insecure:
            es_kwargs["verify_certs"] = False
            print("[WARN] TLS verification disabled (allow-insecure).")
    else:
        if args.ca_cert:
            print("[WARN] --ca-cert specified but host scheme is HTTP; ignoring certificate.")
        if args.allow_insecure:
            print("[WARN] --allow-insecure has no effect for HTTP connections.")
    if args.user and password:
        es_kwargs["basic_auth"] = (args.user, password)

    try:
        es_client = Elasticsearch(args.host, **es_kwargs)
    except es_exceptions.ConnectionError as exc:  # pragma: no cover - network dependent
        raise SystemExit(
            "Failed to connect to Elasticsearch during client creation; ensure it is running and accessible."
        ) from exc

    embedder = None
    if args.no_embeddings:
        print("[INFO] Embeddings disabled via --no-embeddings.")
    elif Specter2Embeddings is None:
        print("[WARN] Specter2Embeddings unavailable; install embedding dependencies for hybrid search.")
    else:
        model_name = args.embedding_model or DEFAULT_EMBED_MODEL
        if not model_name:
            print("[WARN] No embedding model specified; hybrid search will use lexical scoring only.")
        else:
            print(f"[INFO] Loading embedding model: {model_name}")
            try:
                embedder = Specter2Embeddings(
                    model_name=model_name,
                    device=args.embedding_device,
                    batch_size=args.embedding_batch_size,
                )
            except Exception as exc:
                raise SystemExit(f"Failed to load embedding model '{model_name}': {exc}") from exc

    try:
        _run_arxiv_smoke_test(
            es_client,
            index_name=args.index,
            payload_path=payload_path,
            arxiv_id=args.arxiv_id,
            question=args.question,
            recreate_index=args.recreate,
            embedder=embedder,
            hybrid_top_k=args.top_k,
            demo_question=args.demo_question,
            demo_top_k=args.demo_top_k,
            run_demo=args.run_demo_query,
        )
    except Exception as exc:  # pragma: no cover - smoke-test surface
        log(f"Smoke test failed: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    _main()
