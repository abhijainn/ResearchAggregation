"""
Simplified hybrid search pipeline.

This version:
  * pulls keyword Query objects from `_keywords_v3`
  * builds Elasticsearch lexical DSLs via `_es_index_v3`
  * rewrites the user question with OpenAI for semantic embedding
  * fuses lexical + semantic scores with a lightweight reciprocal-rank scheme
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set

# Ensure project root is importable when running as a script (`python helpers/_search.py`)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from elasticsearch import Elasticsearch

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore

try:
    from . import _es_index_v3 as _es_index  # type: ignore
    from . import _keywords_v3 as _keywords  # type: ignore
    from ._query_rewrite import rewrite_for_semantic_query  # type: ignore
except ImportError:  # pragma: no cover - script mode fallback
    import _es_index_v3 as _es_index  # type: ignore
    import _keywords_v3 as _keywords  # type: ignore
    from _query_rewrite import rewrite_for_semantic_query  # type: ignore

DEFAULT_EMBED_MODEL = "allenai/specter2_base"

DEFAULT_SOURCE_FIELDS = [
    "id",
    "doc_id",
    "chunk_id",
    "text",
    "section_title",
    "section_level",
    "headings",
    "page_num",
    "char_start",
    "char_end",
    "header_score",
    "model_names",
]

__all__ = ["SearchConfig", "SearchResult", "search"]


@dataclass
class SearchConfig:
    per_query_size_lex: int = 40
    semantic_k: int = 80
    return_k: int = 8
    rrf_k: int = 60
    embed_model_name: str = DEFAULT_EMBED_MODEL
    cache_size: int = 128
    semantic_model: str = "gpt-4o-mini"


@dataclass
class Candidate:
    chunk_id: str
    doc_id: str
    source: Dict[str, Any]
    lexical_scores: Dict[str, float] = field(default_factory=dict)
    lexical_ranks: Dict[str, int] = field(default_factory=dict)
    semantic_score: Optional[float] = None
    semantic_rank: Optional[int] = None
    fused_score: float = 0.0

    @property
    def text(self) -> str:
        return self.source.get("text", "")


@dataclass
class SearchResult:
    contexts: List[Dict[str, Any]]
    queries_used: List[Dict[str, Any]]
    debug: Dict[str, Any]


def _load_specter2_embedder(model_name: str):
    """
    Dynamically load Specter2Embeddings from helpers._load_data (or _load_data).
    Raises immediately if unavailable so semantic search fails fast.
    """
    for module_name in ("helpers._load_data", "_load_data"):
        try:
            module = import_module(module_name)
            embedder_cls = getattr(module, "Specter2Embeddings", None)
            if embedder_cls is not None:
                return embedder_cls(model_name)
        except Exception:
            continue
    raise ImportError(
        "Specter2Embeddings could not be imported. Ensure helpers/_load_data.py is on PYTHONPATH "
        "and dependencies (torch, transformers, adapters) are installed."
    )


class QuestionEmbedder:
    """Small wrapper around a Specter2-compatible embedder with an LRU cache."""

    def __init__(self, model_name: str, cache_size: int = 128, loader: Callable[[str], Any] = _load_specter2_embedder):
        self.model_name = model_name
        self.cache_size = cache_size
        self.loader = loader
        self._embedder: Optional[Any] = None
        self._cache: Dict[str, List[float]] = {}
        self._lru: List[str] = []

    def encode(self, text: str) -> List[float]:
        key = hashlib.md5(text.encode("utf-8")).hexdigest()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if self._embedder is None:
            self._embedder = self.loader(self.model_name)
        vector = self._embedder.embed_query(text)
        self._cache[key] = vector
        self._lru.append(key)
        if len(self._lru) > self.cache_size:
            oldest = self._lru.pop(0)
            self._cache.pop(oldest, None)
        return vector


def _rewrite_with_openai(question: str, keyword_queries: Sequence["_keywords.Query"], model: str) -> str:
    """
    Use OpenAI to produce a semantic-friendly rewrite.
    Falls back to the deterministic helper if the API is unavailable.
    """

    def _term_string() -> str:
        terms: List[str] = []
        seen: Set[str] = set()
        for query in keyword_queries:
            for term in list(query.normal_terms):
                norm = term.strip()
                if not norm:
                    continue
                lower = norm.lower()
                if lower in seen:
                    continue
                seen.add(lower)
                terms.append(norm)
        return ", ".join(terms[:12])

    terms = _term_string()
    if OpenAI is None:
        return rewrite_for_semantic_query(question, keyword_queries)

    prompt = (
        "Rewrite the user question to guide semantic search AND draft a 1-2 sentence mock answer. "
        "Keep the rewrite concise, include the key concepts, and explicitly avoid references/bibliographies. "
        "Mock answer should be a plausible, self-contained guess to tighten semantic similarity.\n"
        f"Question: {question.strip()}\n"
        f"Key terms: {terms or '(none)'}\n"
        "Return as: Rewritten: <text> || Mock answer: <text>"
    )
    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        content = resp.choices[0].message.content if resp.choices else None
        rewritten = content.strip() if content else ""
        if rewritten:
            return rewritten
    except Exception:
        pass
    base = rewrite_for_semantic_query(question, keyword_queries)
    term_snippet = " ".join((terms.split(",") if terms else [])[:3]).strip()
    mock_answer = f"Mock answer: This likely discusses {term_snippet or question}."
    return f"{base} {mock_answer}".strip()


def _record_hits(
    candidate_map: Dict[str, Candidate],
    hits: Sequence[Mapping[str, Any]],
    *,
    query_id: str,
) -> None:
    for rank, hit in enumerate(hits, start=1):
        source = hit.get("_source", {}) or {}
        chunk_id = source.get("id") or hit.get("_id")
        doc_id = source.get("doc_id")
        if not chunk_id or not doc_id:
            continue
        candidate = candidate_map.get(chunk_id)
        if candidate is None:
            candidate = Candidate(chunk_id=chunk_id, doc_id=doc_id, source=source)
            candidate_map[chunk_id] = candidate
        score = float(hit.get("_score") or 0.0)
        if query_id == "semantic":
            candidate.semantic_score = score
            candidate.semantic_rank = rank
        else:
            candidate.lexical_scores[query_id] = score
            candidate.lexical_ranks[query_id] = rank


def _apply_rrf(candidates: List[Candidate], rrf_k: int) -> None:
    for cand in candidates:
        score = 0.0
        for rank in cand.lexical_ranks.values():
            score += 1.0 / (rrf_k + rank)
        if cand.semantic_rank is not None:
            score += 1.0 / (rrf_k + cand.semantic_rank)
        cand.fused_score = score


def search(
    es: Elasticsearch,
    question: str,
    index_name: str,
    *,
    config: Optional[SearchConfig] = None,
    embedder: Optional[QuestionEmbedder] = None,
) -> SearchResult:
    """Run simplified hybrid search using keyword queries + semantic rewrite."""
    config = config or SearchConfig()
    normalized_question = question.strip()

    keyword_queries, keyword_debug = _keywords.extract_queries(normalized_question)
    rewritten_question = _rewrite_with_openai(normalized_question, keyword_queries, config.semantic_model)

    candidate_map: Dict[str, Candidate] = {}
    lexical_debug: List[Dict[str, Any]] = []
    lexical_hits_all: List[Dict[str, Any]] = []

    for idx, query_obj in enumerate(keyword_queries):
        dsl = _es_index.build_lexical_dsl(query_obj)
        response = _es_index.search_index(
            es,
            index_name,
            dsl,
            size=config.per_query_size_lex,
            source_includes=DEFAULT_SOURCE_FIELDS,
            track_total_hits=False,
        )
        hits = response.get("hits", {}).get("hits", [])
        query_id = f"lex_{idx}"
        _record_hits(candidate_map, hits, query_id=query_id)
        lexical_hits_all.append(
            {
                "query_id": query_id,
                "query": query_obj.to_dict(),
                "hits": hits,
            }
        )
        lexical_debug.append(
            {
                "query": query_obj.to_dict(),
                "query_id": query_id,
                "hit_count": len(hits),
                "took_ms": response.get("took"),
            }
        )

    semantic_debug: Dict[str, Any] = {"enabled": False}
    semantic_hits_all: List[Dict[str, Any]] = []
    if config.semantic_k > 0:
        if embedder is None:
            embedder = QuestionEmbedder(config.embed_model_name, cache_size=config.cache_size)
        vector = embedder.encode(rewritten_question)
        response = _es_index.knn_search(
            es,
            index_name,
            vector,
            k=config.semantic_k,
            source_includes=DEFAULT_SOURCE_FIELDS,
        )
        semantic_hits = response.get("hits", {}).get("hits", [])
        _record_hits(candidate_map, semantic_hits, query_id="semantic")
        semantic_debug = {
            "enabled": True,
            "hit_count": len(semantic_hits),
            "took_ms": response.get("took"),
            "rewritten_question": rewritten_question,
        }
        semantic_hits_all.append(
            {
                "query_id": "semantic",
                "rewritten_question": rewritten_question,
                "hits": semantic_hits,
            }
        )

    candidates = list(candidate_map.values())
    _apply_rrf(candidates, config.rrf_k)
    ranked = sorted(candidates, key=lambda cand: cand.fused_score, reverse=True)
    top_candidates = ranked[: config.return_k]

    contexts: List[Dict[str, Any]] = []
    for candidate in top_candidates:
        contexts.append(
            {
                "doc_id": candidate.doc_id,
                "chunk_ids": [candidate.chunk_id],
                "text": candidate.text,
                "section_title": candidate.source.get("section_title"),
                "headings": candidate.source.get("headings"),
                "page_num": candidate.source.get("page_num"),
                "char_span": (candidate.source.get("char_start"), candidate.source.get("char_end")),
                "provenance": {
                    "lexical_ranks": candidate.lexical_ranks,
                    "lexical_scores": candidate.lexical_scores,
                    "semantic_rank": candidate.semantic_rank,
                    "semantic_score": candidate.semantic_score,
                    "fused_score": candidate.fused_score,
                },
            }
        )

    debug_payload = {
        "keyword_debug": keyword_debug,
        "lexical": lexical_debug,
        "semantic": semantic_debug,
        "candidate_count": len(candidates),
        "rewritten_question": rewritten_question,
        "lexical_hits": lexical_hits_all,
        "semantic_hits": semantic_hits_all,
    }

    return SearchResult(
        contexts=contexts,
        queries_used=[q.to_dict() for q in keyword_queries],
        debug=debug_payload,
    )


if __name__ == "__main__":
    # Simple CLI harness to exercise the pipeline without Streamlit.
    default_payload = Path(__file__).resolve().parent.parent / "helpers" / "test.xml"
    default_question = "What problem does MoC Mixtures of Text Chunking Learners aim to solve?"

    payload_path = Path(input(f"Payload path [{default_payload}]: ").strip() or default_payload).expanduser()
    if not payload_path.exists():
        raise SystemExit(f"Payload file not found: {payload_path}")
    if payload_path.suffix.lower() == ".xml":
        raw_xml = payload_path.read_text(encoding="utf-8")
        # naive tag strip for harness use
        text_only = re.sub(r"<[^>]+>", " ", raw_xml)
        payload = {"full_text": text_only}
    else:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    question = input(f"Question [{default_question}]: ").strip() or default_question

    document_text = payload.get("processed_text") or payload.get("full_text") or ""
    doc_id = payload.get("arxiv_id") or "sample-doc"
    metadata = {k: payload.get(k) for k in ("sections", "headings", "section_spans", "page_spans", "pages") if payload.get(k)}

    # Build in-memory chunks of 3 sentences each and stub ES helpers for quick testing.
    def _chunk_three_sentences(text: str, doc_id: str) -> List[Dict[str, Any]]:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        chunks: List[Dict[str, Any]] = []
        for idx in range(0, len(sentences), 3):
            window = sentences[idx : idx + 3]
            if not window:
                continue
            chunk_text = " ".join(window)
            chunk_id = f"{doc_id}#{idx//3:04d}"
            chunks.append(
                {
                    "id": chunk_id,
                    "doc_id": doc_id,
                    "chunk_id": idx // 3,
                    "text": chunk_text,
                    "section_title": "",
                    "section_level": 1,
                    "headings": "",
                    "prev_chunk_id": chunks[-1]["id"] if chunks else None,
                    "next_chunk_id": None,
                    "page_num": None,
                    "char_start": None,
                    "char_end": None,
                    "header_score": 0.0,
                    "model_names": [],
                }
            )
            if len(chunks) > 1:
                chunks[-2]["next_chunk_id"] = chunk_id
        return chunks

    chunk_docs = _chunk_three_sentences(document_text, doc_id)
    chunk_by_id = {doc["id"]: doc for doc in chunk_docs}
    harness_embedder = _load_specter2_embedder(DEFAULT_EMBED_MODEL)
    chunk_vectors = harness_embedder.embed_documents([doc.get("text") or "" for doc in chunk_docs])
    chunk_vector_map: Dict[str, List[float]] = {doc["id"]: vec for doc, vec in zip(chunk_docs, chunk_vectors)}

    def _stub_search_index(es: Any, index_name: str, dsl: Mapping[str, Any], *, size: int, source_includes: Optional[Sequence[str]], track_total_hits: Optional[object]) -> Dict[str, Any]:  # noqa: D401
        del es, index_name, track_total_hits
        query_obj = dsl.get("_raw_query") or getattr(dsl.get("query", {}), "_raw_query", None)
        if hasattr(query_obj, "normal_terms"):
            terms = query_obj.normal_terms
        else:
            terms = _keywords.normalize_keyword_terms(question)
        terms = [t.lower() for t in terms if t]
        hits = []
        for doc in chunk_docs:
            text_lower = (doc.get("text") or "").lower()
            # enforce 3-sentence chunking: already chunked that way above
            score = sum(text_lower.count(t) for t in terms) or 0.0
            if score > 0:
                source = {field: doc.get(field) for field in source_includes} if source_includes else dict(doc)
                hits.append({"_id": doc["id"], "_score": float(score), "_source": source})
        hits.sort(key=lambda h: h["_score"], reverse=True)
        return {"hits": {"hits": hits[:size]}, "took": 0}

    def _stub_knn_search(es: Any, index_name: str, vector: Sequence[float], *, k: int, num_candidates: Optional[int] = None, filter_query: Optional[Mapping[str, Any]] = None, source_includes: Optional[Sequence[str]] = None) -> Dict[str, Any]:  # noqa: D401
        del es, index_name, num_candidates, filter_query
        hits = []
        for chunk_id, vec in chunk_vector_map.items():
            score = float(sum(a * b for a, b in zip(vector, vec)))
            source = {field: chunk_by_id[chunk_id].get(field) for field in source_includes} if source_includes else dict(chunk_by_id[chunk_id])
            hits.append({"_id": chunk_id, "_score": score, "_source": source})
        hits.sort(key=lambda h: h["_score"], reverse=True)
        return {"hits": {"hits": hits[:k]}, "took": 0}

    _es_index.search_index = _stub_search_index  # type: ignore
    _es_index.knn_search = _stub_knn_search  # type: ignore

    def _first_sentence(text: str) -> str:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return parts[0].strip() if parts else text.strip()

    embedder_for_search = QuestionEmbedder(DEFAULT_EMBED_MODEL)
    result = search(es=object(), question=question, index_name="in-memory", embedder=embedder_for_search)
    print("\n[Keyword Retrieved]")
    for block in result.debug.get("lexical_hits", []):
        qid = block.get("query_id")
        qdict = block.get("query", {})
        hits = block.get("hits", []) or []
        print(f"Query {qid}: {qdict}")
        for rank, hit in enumerate(hits, start=1):
            src = hit.get("_source", {}) or {}
            sentence = _first_sentence(src.get("text") or "")
            print(f"  #{rank} chunk={hit.get('_id')} score={hit.get('_score')} sentence={sentence}")

    print("\n[Semantic Retrieved]")
    for block in result.debug.get("semantic_hits", []):
        hits = block.get("hits", []) or []
        print(f"Rewritten: {block.get('rewritten_question')}")
        for rank, hit in enumerate(hits, start=1):
            src = hit.get("_source", {}) or {}
            sentence = _first_sentence(src.get("text") or "")
            print(f"  #{rank} chunk={hit.get('_id')} score={hit.get('_score')} sentence={sentence}")

    print("\n[Final Reranked Contexts]")
    for idx, ctx in enumerate(result.contexts, start=1):
        print(f"\n#{idx} doc={ctx['doc_id']} chunks={ctx['chunk_ids']} fused={ctx['provenance']['fused_score']:.3f}")
        sentence = _first_sentence(ctx["text"])
        print(f"Sentence match: {sentence}")
