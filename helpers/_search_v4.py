"""
Search v4: use a simulated answer string for both lexical and semantic retrieval.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

from elasticsearch import Elasticsearch

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from . import _keywords_v4 as _keywords  # type: ignore
from . import _es_index_v4 as _es_index  # type: ignore
from ._query_rewrite import rewrite_for_semantic_query  # reuse deterministic version
from importlib import import_module

DEFAULT_EMBED_MODEL = "allenai/specter2_base"
DEFAULT_SOURCE_FIELDS = list(_es_index.DEFAULT_SOURCE_FIELDS)


def _load_specter2_embedder(model_name: str):
    for module_name in ("helpers._load_data", "_load_data"):
        try:
            module = import_module(module_name)
            cls = getattr(module, "Specter2Embeddings", None)
            if cls:
                return cls(model_name)
        except Exception:
            continue
    raise ImportError("Specter2Embeddings unavailable; install deps and ensure _load_data.py is importable.")


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
    lexical_score: float = 0.0
    lexical_rank: Optional[int] = None
    semantic_score: Optional[float] = None
    semantic_rank: Optional[int] = None
    fused_score: float = 0.0

    @property
    def text(self) -> str:
        return self.source.get("text", "")


@dataclass
class SearchResult:
    contexts: List[Dict[str, Any]]
    simulated_answer: str
    debug: Dict[str, Any]


class QuestionEmbedder:
    """Cache wrapper around a Specter2-style embedder."""

    def __init__(self, model_name: str, cache_size: int = 128):
        self.model_name = model_name
        self.cache_size = cache_size
        self._embedder = None
        self._cache: Dict[str, List[float]] = {}
        self._lru: List[str] = []

    def encode(self, text: str) -> List[float]:
        key = hashlib.md5(text.encode("utf-8")).hexdigest()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if self._embedder is None:
            self._embedder = _load_specter2_embedder(self.model_name)
        vector = self._embedder.embed_query(text)
        self._cache[key] = vector
        self._lru.append(key)
        if len(self._lru) > self.cache_size:
            oldest = self._lru.pop(0)
            self._cache.pop(oldest, None)
        return vector


def _rewrite_with_openai(simulated_answer: str, question: str, model: str) -> str:
    """Optionally refine the simulated answer with OpenAI, otherwise reuse deterministic rewrite."""
    if OpenAI is None:
        return rewrite_for_semantic_query(simulated_answer or question)
    prompt = (
        "Polish the following simulated answer so it tightly matches the user's question for semantic retrieval.\n"
        f"Question: {question.strip()}\n"
        f"Simulated answer: {simulated_answer.strip()}\n"
        "Return only the refined answer."
    )
    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        content = resp.choices[0].message.content if resp.choices else None
        refined = content.strip() if content else ""
        if refined:
            return refined
    except Exception:
        pass
    return rewrite_for_semantic_query(simulated_answer or question)


def _record_hits(candidate_map: Dict[str, Candidate], hits: Sequence[Mapping[str, Any]], *, kind: str) -> None:
    for rank, hit in enumerate(hits, start=1):
        source = hit.get("_source", {}) or {}
        chunk_id = source.get("id") or hit.get("_id")
        doc_id = source.get("doc_id")
        if not chunk_id or not doc_id:
            continue
        cand = candidate_map.get(chunk_id)
        if cand is None:
            cand = Candidate(chunk_id=chunk_id, doc_id=doc_id, source=source)
            candidate_map[chunk_id] = cand
        score = float(hit.get("_score") or 0.0)
        if kind == "semantic":
            cand.semantic_score = score
            cand.semantic_rank = rank
        else:
            cand.lexical_score = score
            cand.lexical_rank = rank


def _apply_rrf(candidates: List[Candidate], rrf_k: int) -> None:
    for cand in candidates:
        score = 0.0
        if cand.lexical_rank:
            score += 1.0 / (rrf_k + cand.lexical_rank)
        if cand.semantic_rank:
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
    config = config or SearchConfig()
    queries, debug_kw = _keywords.extract_queries(question)
    simulated_answer = queries[0].text if queries else question.strip()
    refined_answer = _rewrite_with_openai(simulated_answer, question, config.semantic_model)

    candidate_map: Dict[str, Candidate] = {}
    lexical_hits_all: List[Dict[str, Any]] = []

    dsl = _es_index.build_lexical_dsl(refined_answer)
    response = _es_index.search_index(
        es,
        index_name,
        dsl,
        size=config.per_query_size_lex,
        source_includes=DEFAULT_SOURCE_FIELDS,
        track_total_hits=False,
    )
    lex_hits = response.get("hits", {}).get("hits", [])
    _record_hits(candidate_map, lex_hits, kind="lexical")
    lexical_hits_all.append({"query": refined_answer, "hits": lex_hits})

    semantic_hits_all: List[Dict[str, Any]] = []
    if embedder is None:
        embedder = QuestionEmbedder(config.embed_model_name, cache_size=config.cache_size)
    vector = embedder.encode(refined_answer)
    response = _es_index.knn_search(
        es,
        index_name,
        vector,
        k=config.semantic_k,
        source_includes=DEFAULT_SOURCE_FIELDS,
    )
    sem_hits = response.get("hits", {}).get("hits", [])
    _record_hits(candidate_map, sem_hits, kind="semantic")
    semantic_hits_all.append({"query": refined_answer, "hits": sem_hits})

    candidates = list(candidate_map.values())
    _apply_rrf(candidates, config.rrf_k)
    ranked = sorted(candidates, key=lambda c: c.fused_score, reverse=True)[: config.return_k]

    contexts: List[Dict[str, Any]] = []
    for cand in ranked:
        contexts.append(
            {
                "doc_id": cand.doc_id,
                "chunk_ids": [cand.chunk_id],
                "text": cand.text,
                "provenance": {
                    "lexical_rank": cand.lexical_rank,
                    "lexical_score": cand.lexical_score,
                    "semantic_rank": cand.semantic_rank,
                    "semantic_score": cand.semantic_score,
                    "fused_score": cand.fused_score,
                },
            }
        )

    debug_payload = {
        "keyword_debug": debug_kw,
        "lexical_hits": lexical_hits_all,
        "semantic_hits": semantic_hits_all,
        "refined_answer": refined_answer,
    }

    return SearchResult(contexts=contexts, simulated_answer=refined_answer, debug=debug_payload)


if __name__ == "__main__":
    default_payload = Path(__file__).resolve().parent.parent / "helpers" / "test.xml"
    default_question = "What problem does MoC Mixtures of Text Chunking Learners aim to solve?"

    payload_path = Path(input(f"Payload path [{default_payload}]: ").strip() or default_payload).expanduser()
    if not payload_path.exists():
        raise SystemExit(f"Payload file not found: {payload_path}")
    if payload_path.suffix.lower() == ".xml":
        raw_xml = payload_path.read_text(encoding="utf-8")
        import re
        document_text = re.sub(r"<[^>]+>", " ", raw_xml)
    else:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        document_text = payload.get("processed_text") or payload.get("full_text") or ""
    question = input(f"Question [{default_question}]: ").strip() or default_question
    doc_id = "doc"

    sentences = [s.strip() for s in __import__("re").split(r"(?<=[.!?])\s+", document_text) if s.strip()]
    chunk_docs = []
    for idx in range(0, len(sentences), 3):
        window = sentences[idx : idx + 3]
        if not window:
            continue
        chunk_text = " ".join(window)
        chunk_id = f"{doc_id}#{idx//3:04d}"
        chunk_docs.append({"id": chunk_id, "doc_id": doc_id, "text": chunk_text})
    chunk_by_id = {doc["id"]: doc for doc in chunk_docs}

    embedder_backend = _load_specter2_embedder(DEFAULT_EMBED_MODEL)
    chunk_vectors = embedder_backend.embed_documents([doc.get("text") or "" for doc in chunk_docs])
    chunk_vector_map: Dict[str, List[float]] = {doc["id"]: vec for doc, vec in zip(chunk_docs, chunk_vectors)}

    def _stub_search_index(es: Any, index_name: str, dsl: Mapping[str, Any], *, size: int, source_includes: Optional[Sequence[str]], track_total_hits: Optional[object]) -> Dict[str, Any]:
        del es, index_name, track_total_hits
        answer_text = dsl.get("_raw_query") or ""
        terms = [t.strip().lower() for t in answer_text.split() if t.strip()]
        hits = []
        for doc in chunk_docs:
            text_lower = (doc.get("text") or "").lower()
            score = sum(text_lower.count(t) for t in terms) or 0.0
            if score > 0:
                source = {field: doc.get(field) for field in source_includes} if source_includes else dict(doc)
                hits.append({"_id": doc["id"], "_score": float(score), "_source": source})
        hits.sort(key=lambda h: h["_score"], reverse=True)
        return {"hits": {"hits": hits[:size]}, "took": 0}

    def _stub_knn_search(es: Any, index_name: str, vector: Sequence[float], *, k: int, source_includes: Optional[Sequence[str]] = None, num_candidates: Optional[int] = None, filter_query: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
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

    search_embedder = QuestionEmbedder(DEFAULT_EMBED_MODEL)
    result = search(es=object(), question=question, index_name="in-memory", embedder=search_embedder)
    print(json.dumps(result.debug, indent=2))
    print("\n[Final Contexts]")
    for idx, ctx in enumerate(result.contexts, start=1):
        print(f"#{idx} doc={ctx['doc_id']} chunks={ctx['chunk_ids']} fused={ctx['provenance']['fused_score']:.3f}")
