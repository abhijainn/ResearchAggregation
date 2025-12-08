"""
Hybrid retrieval orchestrator for the quote finder stack.

This module wires `_keywords` + `_es_index` + (optional) SPECTER2 embeddings
into a cohesive search pipeline with lexical + semantic fusion, adjacent
expansion, and simple completeness heuristics.
"""

from __future__ import annotations

import json
import hashlib
import math
import re
import statistics
from pathlib import Path
from textwrap import shorten
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from elasticsearch import Elasticsearch

try:
    from . import _es_index_v3 as _es_index  # type: ignore
    from . import _keywords_v3 as _keywords  # type: ignore
    from ._query_rewrite import rewrite_for_semantic_query  # type: ignore
    from ._llm_reranker import rerank_with_llm  # type: ignore
except ImportError:  # pragma: no cover - script mode fallback
    try:
        import _es_index_v3 as _es_index  # type: ignore
        import _keywords_v3 as _keywords  # type: ignore
    except ImportError:
        import _es_index as _es_index  # type: ignore
        import _keywords as _keywords  # type: ignore
    from _query_rewrite import rewrite_for_semantic_query  # type: ignore
    from _llm_reranker import rerank_with_llm  # type: ignore

try:
    from ._load_data import Specter2Embeddings, EMBED_MODEL_NAME as _DEFAULT_MODEL
except ImportError:  # pragma: no cover
    Specter2Embeddings = None  # type: ignore
    _DEFAULT_MODEL = None  # type: ignore

DEFAULT_EMBED_MODEL = _DEFAULT_MODEL or "allenai/specter2_base"
ENUMERATOR_RE = re.compile(r"(?:^|\n)\s*(?:[\-\u2022\*]|[0-9]+[.)])")

DEFAULT_SOURCE_FIELDS = [
    "id",
    "doc_id",
    "chunk_id",
    "text",
    "section_title",
    "section_level",
    "headings",
    "prev_chunk_id",
    "next_chunk_id",
    "page_num",
    "char_start",
    "char_end",
    "header_score",
    "model_names",
]

CompletenessLabel = str

_SECTION_PENALTIES = (
    "reference",
    "references",
    "bibliography",
    "acknowledgment",
    "acknowledgement",
    "supplement",
    "supplementary",
)
_SECTION_BONUSES = (
    "abstract",
    "introduction",
    "background",
    "method",
    "methods",
    "approach",
    "model",
    "experiment",
    "results",
    "finding",
    "analysis",
    "discussion",
)

__all__ = [
    "SearchConfig",
    "SearchResult",
    "search",
    "completeness_flag",
    "expand_with_adjacent",
]


@dataclass
class SearchConfig:
    mode: str = "hybrid"
    per_query_size_lex: int = 50
    semantic_k: int = 200
    fusion: str = "rrf"  # or "zscores"
    rrf_k: int = 60
    alpha: float = 0.5  # only used for z-score fusion
    return_k: int = 10
    mmr_lambda: float = 0.3
    mmr_k: int = 50
    max_expansion_hops: int = 2
    embed_model_name: str = DEFAULT_EMBED_MODEL
    cache_size: int = 128
    use_llm_rerank: bool = False
    llm_rerank_model: str = "gpt-4o-mini"
    llm_rerank_top_k: int = 5


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
    lists: Set[str] = field(default_factory=set)
    token_cache: Optional[Set[str]] = field(default=None, init=False, repr=False)
    section_bias: float = 1.0

    @property
    def text(self) -> str:
        return self.source.get("text", "")

    def best_lexical(self) -> float:
        if not self.lexical_scores:
            return 0.0
        return max(self.lexical_scores.values())

    def token_set(self) -> Set[str]:
        if self.token_cache is None:
            normalized = [token for token in self.text.lower().split()]
            self.token_cache = set(normalized)
        return self.token_cache


@dataclass
class SearchResult:
    contexts: List[Dict[str, Any]]
    queries_used: List[str]
    debug: Dict[str, Any]


class QuestionEmbedder:
    """Thin wrapper around Specter2Embeddings with small in-memory cache."""

    def __init__(self, model_name: str, cache_size: int = 128):
        if Specter2Embeddings is None:
            raise RuntimeError("Specter2 embeddings are unavailable in this environment.")
        self.model_name = model_name
        self.cache_size = cache_size
        self._embedder: Optional[Specter2Embeddings] = None
        self._cache: Dict[str, List[float]] = {}
        self._lru: List[str] = []

    def encode(self, text: str) -> List[float]:
        key = hashlib.md5(text.encode("utf-8")).hexdigest()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if self._embedder is None:
            self._embedder = Specter2Embeddings(self.model_name)
        vector = self._embedder.embed_query(text)
        self._cache[key] = vector
        self._lru.append(key)
        if len(self._lru) > self.cache_size:
            oldest = self._lru.pop(0)
            self._cache.pop(oldest, None)
        return vector


def search(
    es: Elasticsearch,
    question: str,
    index_name: str,
    *,
    config: Optional[SearchConfig] = None,
    doc_anchor_ids: Optional[Sequence[str]] = None,
    embedder: Optional[QuestionEmbedder] = None,
) -> SearchResult:
    config = config or SearchConfig()
    mode = config.mode.lower()
    normalized_question = question.strip()

    anchor_map = {}
    if doc_anchor_ids:
        anchor_map = _es_index.get_doc_anchors(es, index_name, doc_anchor_ids)

    keyword_queries, keyword_debug = _keywords.extract_queries(
        normalized_question,
        doc_anchors=anchor_map,
    )
    rewritten_question = rewrite_for_semantic_query(normalized_question, keyword_queries)

    candidate_map: Dict[str, Candidate] = {}
    lexical_debug: List[Dict[str, Any]] = []
    if mode in {"lexical", "hybrid"}:
        for idx, query_str in enumerate(keyword_queries):
            dsl = _es_index.build_lexical_dsl(query_str)
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
            lexical_debug.append(
                {
                    "query": query_str,
                    "query_id": query_id,
                    "hit_count": len(hits),
                    "took_ms": response.get("took"),
                }
            )

    semantic_debug: Dict[str, Any] = {"enabled": False}
    vector_hash = None
    semantic_hits: List[Dict[str, Any]] = []
    if mode in {"semantic", "hybrid"}:
        if embedder is None and Specter2Embeddings is not None:
            embedder = QuestionEmbedder(config.embed_model_name, cache_size=config.cache_size)
        if embedder is None:
            raise RuntimeError(
                "Semantic search requested but Specter2 embeddings are unavailable. "
                "Install the required embedding dependencies or set config.mode='lexical'."
            )
        vector = embedder.encode(rewritten_question)
        vector_hash = _hash_vector(vector)
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
            "vector_hash": vector_hash,
            "took_ms": response.get("took"),
            "rewritten_question": rewritten_question,
        }

    candidates = list(candidate_map.values())
    fusion_info = _apply_fusion(candidates, config)
    for candidate in candidates:
        bias = _section_bias(
            candidate.source.get("section_title"),
            candidate.source.get("headings"),
            candidate.source.get("is_reference"),
        )
        candidate.section_bias = bias
        candidate.fused_score *= bias

    if not candidates:
        return SearchResult(
            contexts=[],
            queries_used=keyword_queries,
            debug={
                "keyword_debug": keyword_debug,
                "lexical": lexical_debug,
                "semantic": semantic_debug,
                "fusion": fusion_info,
                "vector_hash": vector_hash,
            },
        )

    ranked_candidates = sorted(candidates, key=lambda cand: cand.fused_score, reverse=True)
    diversified = _apply_mmr(ranked_candidates, config.mmr_lambda, min(config.mmr_k, len(ranked_candidates)))
    top_candidates = diversified[: config.return_k]

    contexts: List[Dict[str, Any]] = []
    expansion_log: List[Dict[str, Any]] = []
    for candidate in top_candidates:
        # Temporarily disable adjacency expansion to isolate retrieval/reranking behavior.
        merged = {
            "text": candidate.text,
            "chunk_ids": [candidate.chunk_id],
            "char_start": candidate.source.get("char_start"),
            "char_end": candidate.source.get("char_end"),
            "completeness": completeness_flag(candidate.text),
            "expansions": [],
        }
        context = {
            "doc_id": candidate.doc_id,
            "chunk_ids": merged["chunk_ids"],
            "text": merged["text"],
            "section_title": candidate.source.get("section_title"),
            "headings": candidate.source.get("headings"),
            "page_num": candidate.source.get("page_num"),
            "char_span": (merged["char_start"], merged["char_end"]),
            "completeness": merged["completeness"],
            "provenance": {
                "lists": sorted(candidate.lists),
                "lexical_ranks": candidate.lexical_ranks,
                "lexical_scores": candidate.lexical_scores,
                "semantic_rank": candidate.semantic_rank,
                "semantic_score": candidate.semantic_score,
                "fused_score": candidate.fused_score,
                "section_bias": candidate.section_bias,
            },
        }
        contexts.append(context)
        if merged["expansions"]:
            expansion_log.append(
                {
                    "chunk_id": candidate.chunk_id,
                    "doc_id": candidate.doc_id,
                    "expansions": merged["expansions"],
                    "final_completeness": merged["completeness"],
                }
            )

    llm_debug: Dict[str, Any] = {"enabled": False}
    llm_top_k = min(config.llm_rerank_top_k, len(contexts)) if config.llm_rerank_top_k else None
    if config.use_llm_rerank and contexts:
        for idx, context in enumerate(contexts, start=1):
            context["_llm_index"] = idx
        contexts, llm_debug = rerank_with_llm(
            rewritten_question,
            contexts,
            model=config.llm_rerank_model,
            top_k=llm_top_k,
        )
        for context in contexts:
            context.pop("_llm_index", None)

    debug_payload = {
        "keyword_debug": keyword_debug,
        "lexical": lexical_debug,
        "semantic": semantic_debug,
        "fusion": fusion_info,
        "candidate_count": len(candidates),
        "vector_hash": vector_hash,
        "expansions": expansion_log,
        "rewritten_question": rewritten_question,
        "llm_rerank": llm_debug,
    }

    return SearchResult(contexts=contexts, queries_used=keyword_queries, debug=debug_payload)


def completeness_flag(text_span: str) -> CompletenessLabel:
    stripped = text_span.strip()
    if not stripped:
        return "INCOMPLETE"
    starts_lower = stripped[0].islower()
    ends_clean = stripped[-1] in ".?!"
    enumerator = _has_enumerator(stripped)
    dangling_end = stripped.endswith((";", ",", ":"))
    lowered = stripped.lower()
    words = lowered.split()
    last_word = words[-1] if words else ""
    dangling_refs = "as above" in lowered or last_word in {"these", "those", "following"}

    score = 0
    if enumerator:
        score += 1
    if starts_lower:
        score += 1
    if not ends_clean or dangling_end:
        score += 1
    if dangling_refs:
        score += 1

    if score >= 2:
        return "INCOMPLETE"
    if score == 1:
        return "PARTIAL"
    return "COMPLETE"


def expand_with_adjacent(
    es: Elasticsearch,
    index_name: str,
    chunk_id: str,
    *,
    direction: str,
) -> Optional[Dict[str, Any]]:
    """Fetch the prev/next neighbor for a chunk id."""
    record = _es_index.fetch_chunk_by_id(es, index_name, chunk_id)
    if record is None:
        return None
    return record


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _record_hits(candidate_map: Dict[str, Candidate], hits: Sequence[Mapping[str, Any]], *, query_id: str) -> None:
    for rank, hit in enumerate(hits, start=1):
        chunk_id = hit.get("_source", {}).get("id") or hit.get("_id")
        if not chunk_id:
            continue
        source = hit.get("_source", {})
        doc_id = source.get("doc_id")
        if not doc_id:
            continue
        candidate = candidate_map.get(chunk_id)
        if candidate is None:
            candidate = Candidate(chunk_id=chunk_id, doc_id=doc_id, source=source)
            candidate_map[chunk_id] = candidate
        score = hit.get("_score", 0.0)
        if query_id == "semantic":
            candidate.semantic_score = score
            candidate.semantic_rank = rank
        else:
            candidate.lexical_scores[query_id] = score
            candidate.lexical_ranks[query_id] = rank
        candidate.lists.add(query_id)


def _apply_fusion(candidates: List[Candidate], config: SearchConfig) -> Dict[str, Any]:
    if not candidates:
        return {"method": config.fusion}

    if config.fusion.lower() == "zscores":
        lex_scores = [candidate.best_lexical() for candidate in candidates if candidate.lexical_scores]
        sem_scores = [candidate.semantic_score for candidate in candidates if candidate.semantic_score is not None]
        lex_stats = _stats(lex_scores)
        sem_stats = _stats(sem_scores)
        for candidate in candidates:
            z_lex = _zscore(candidate.best_lexical(), lex_stats)
            z_sem = _zscore(candidate.semantic_score, sem_stats)
            candidate.fused_score = config.alpha * z_lex + (1.0 - config.alpha) * z_sem
        return {
            "method": "zscores",
            "lex_mean": lex_stats["mean"],
            "lex_std": lex_stats["std"],
            "sem_mean": sem_stats["mean"],
            "sem_std": sem_stats["std"],
            "alpha": config.alpha,
        }

    for candidate in candidates:
        score = 0.0
        for rank in candidate.lexical_ranks.values():
            score += 1.0 / (config.rrf_k + rank)
        if candidate.semantic_rank is not None:
            score += 1.0 / (config.rrf_k + candidate.semantic_rank)
        candidate.fused_score = score

    return {"method": "rrf", "rrf_k": config.rrf_k}


def _stats(values: Sequence[Optional[float]]) -> Dict[str, float]:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return {"mean": 0.0, "std": 1.0}
    if len(filtered) == 1:
        return {"mean": filtered[0] or 0.0, "std": 1.0}
    mean_val = statistics.mean(filtered)
    std_val = statistics.stdev(filtered) or 1.0
    return {"mean": mean_val, "std": std_val}


def _zscore(value: Optional[float], stats_payload: Dict[str, float]) -> float:
    if value is None:
        return 0.0
    return (value - stats_payload["mean"]) / stats_payload["std"]


def _apply_mmr(candidates: List[Candidate], lambda_param: float, top_k: int) -> List[Candidate]:
    if not candidates:
        return []
    selected: List[Candidate] = []
    remaining = candidates.copy()
    while remaining and len(selected) < top_k:
        best_candidate = None
        best_score = -math.inf
        for candidate in remaining:
            diversity = 0.0
            for picked in selected:
                diversity = max(diversity, _jaccard(candidate.token_set(), picked.token_set()))
            score = lambda_param * candidate.fused_score - (1.0 - lambda_param) * diversity
            if score > best_score:
                best_score = score
                best_candidate = candidate
        if best_candidate is None:
            break
        selected.append(best_candidate)
        remaining.remove(best_candidate)
    return selected


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


def _ensure_completeness(
    es: Elasticsearch,
    index_name: str,
    candidate: Candidate,
    max_hops: int,
) -> Dict[str, Any]:
    base_text = candidate.text
    chunk_ids = [candidate.chunk_id]
    char_start = candidate.source.get("char_start")
    char_end = candidate.source.get("char_end")
    completeness = completeness_flag(base_text)
    expansions: List[Dict[str, str]] = []
    hops = 0
    prev_id = candidate.source.get("prev_chunk_id")
    next_id = candidate.source.get("next_chunk_id")

    while completeness != "COMPLETE" and hops < max_hops:
        direction = "prev" if completeness == "INCOMPLETE" and prev_id else "next"
        target_id = prev_id if direction == "prev" else next_id
        if not target_id:
            break
        neighbor = _es_index.fetch_chunk_by_id(es, index_name, target_id)
        if not neighbor:
            break
        neighbor_text = neighbor.get("text", "")
        if direction == "prev":
            base_text = f"{neighbor_text}\n\n{base_text}"
            chunk_ids.insert(0, target_id)
            prev_id = neighbor.get("prev_chunk_id")
            char_start = _min_ignore_none(char_start, neighbor.get("char_start"))
        else:
            base_text = f"{base_text}\n\n{neighbor_text}"
            chunk_ids.append(target_id)
            next_id = neighbor.get("next_chunk_id")
            char_end = _max_ignore_none(char_end, neighbor.get("char_end"))
        expansions.append({"direction": direction, "chunk_id": target_id})
        completeness = completeness_flag(base_text)
        hops += 1

    return {
        "text": base_text,
        "chunk_ids": chunk_ids,
        "char_start": char_start,
        "char_end": char_end,
        "completeness": completeness,
        "expansions": expansions,
    }


def _has_enumerator(text: str) -> bool:
    return bool(ENUMERATOR_RE.search(text))


def _hash_vector(vector: Sequence[float]) -> str:
    digest = hashlib.md5()
    for value in vector:
        digest.update(f"{value:.6f}".encode("utf-8"))
    return digest.hexdigest()


def _min_ignore_none(*values: Optional[int]) -> Optional[int]:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return min(filtered)


def _max_ignore_none(*values: Optional[int]) -> Optional[int]:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return max(filtered)


def _section_bias(
    section_title: Optional[str],
    headings: Optional[str],
    is_reference: Optional[bool] = None,
) -> float:
    """Return a multiplicative bias based on section metadata."""
    if is_reference:
        return 0.05
    text = " ".join(part for part in [section_title, headings] if part)
    lowered = (text or "").lower()
    if any(term in lowered for term in _SECTION_PENALTIES):
        return 0.15
    bonus = 1.0
    if any(term in lowered for term in _SECTION_BONUSES):
        bonus = 1.15
    return bonus


if __name__ == "__main__":
    # Minimal CLI harness that exercises the search pipeline against the cached
    # `2402.09642.json` payload without requiring a live Elasticsearch instance.
    #
    # The harness:
    #   * chunks the document locally via `_es_index.chunk_document`
    #   * runs a lightweight keyword scorer + hashed “semantic” scorer
    #   * prints the top-k lexical and semantic hits separately
    #   * monkeypatches `_es_index` helpers with the in-memory scorers
    #   * runs the full `search` function and reports expansions + reranked hits

    default_payload = Path(__file__).resolve().parent.parent / "2402.09642.json"
    default_question = (
        "What problem does MoC Mixtures of Text Chunking Learners for Retrieval-Augmented Generation System aim to solve?"
    )

    def _prompt_path(prompt: str, default: Path) -> Path:
        raw = input(f"{prompt} [{default}]: ").strip()
        return Path(raw or str(default)).expanduser()

    def _prompt_text(prompt: str, default: str) -> str:
        raw = input(f"{prompt} [{default}]: ").strip()
        return raw or default

    def _prompt_int(prompt: str, default: int) -> int:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            print(f"Invalid integer '{raw}'. Using default {default}.")
            return default

    print("helpers._search smoke harness — press Enter to accept defaults.\n")

    payload_path = _prompt_path("Payload path", default_payload)
    if not payload_path.exists():
        raise SystemExit(f"Payload file not found: {payload_path}")
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse payload JSON '{payload_path}': {exc}") from exc

    document_text = payload.get("processed_text") or payload.get("full_text") or ""
    if not document_text:
        raise SystemExit("Payload file did not contain 'processed_text' or 'full_text'.")
    doc_id = payload.get("arxiv_id") or "sample-doc"
    structural_metadata = {
        key: payload.get(key)
        for key in ("sections", "headings", "section_spans", "page_spans", "pages")
        if payload.get(key)
    }

    tokens_per_chunk = _prompt_int("Tokens per chunk", 256)
    chunk_cfg = _es_index.ChunkingConfig(tokens_per_chunk=tokens_per_chunk, overlap_tokens=64)
    chunk_records = _es_index.chunk_document(doc_id=doc_id, text=document_text, metadata=structural_metadata, chunk_cfg=chunk_cfg)
    if not chunk_records:
        raise SystemExit("Chunking produced zero records; aborting.")
    chunk_docs = [chunk.to_document() for chunk in chunk_records]
    chunk_by_id: Dict[str, Dict[str, Any]] = {doc["id"]: doc for doc in chunk_docs}

    if Specter2Embeddings is None:
        raise SystemExit(
            "Specter2 embeddings are unavailable. Install the required dependencies to run semantic search."
        )
    embed_model = DEFAULT_EMBED_MODEL or "allenai/specter2_base"
    try:
        harness_embedder = Specter2Embeddings(embed_model)
    except Exception as exc:
        raise SystemExit(f"Failed to load Specter2 embeddings '{embed_model}': {exc}") from exc
    chunk_texts = [doc.get("text") or "" for doc in chunk_docs]
    print("[INFO] Computing chunk embeddings with Specter2...")
    chunk_vectors_list = harness_embedder.embed_documents(chunk_texts)
    chunk_vectors: Dict[str, List[float]] = {
        doc["id"]: vector for doc, vector in zip(chunk_docs, chunk_vectors_list)
    }
    question = _prompt_text("Question", default_question).strip()
    if not question:
        raise SystemExit("Question must be non-empty.")
    lower_question_tokens = re.findall(r"\w+", question.lower())

    def _compute_keyword_hits(query_obj: _keywords.Query, top_k: int) -> List[Tuple[str, float]]:
        terms = [term.lower() for term in (query_obj.boosted_terms + query_obj.normal_terms) if term]
        if not terms:
            terms = lower_question_tokens
        seen: List[Tuple[str, float]] = []
        for doc in chunk_docs:
            section = (doc.get("section_title") or "").strip().lower()
            if query_obj.section_filter and section != query_obj.section_filter.strip().lower():
                continue
            text_lower = (doc.get("text") or "").lower()
            if not text_lower:
                continue
            score = sum(text_lower.count(term) for term in terms if term)
            if score <= 0:
                continue
            seen.append((doc["id"], float(score)))
        if not seen:
            defaults = chunk_docs[:top_k]
            seen = [(doc["id"], 0.0) for doc in defaults]
        seen.sort(key=lambda item: (item[1], item[0]), reverse=True)
        return seen[:top_k]

    def _compute_semantic_hits_from_vector(vector: Sequence[float], top_k: int) -> List[Tuple[str, float]]:
        hits: List[Tuple[str, float]] = []
        for chunk_id, chunk_vector in chunk_vectors.items():
            score = sum(a * b for a, b in zip(vector, chunk_vector))
            hits.append((chunk_id, float(score)))
        hits.sort(key=lambda item: (item[1], item[0]), reverse=True)
        return hits[:top_k]

    def _format_hits(label: str, hits: List[Tuple[str, float]]) -> None:
        print(f"\n[{label}]")
        for rank, (chunk_id, score) in enumerate(hits, start=1):
            source = chunk_by_id[chunk_id]
            snippet = shorten(" ".join((source.get("text") or "").split()), width=200, placeholder=" …")
            section = source.get("section_title") or "Unknown section"
            print(f"  #{rank} chunk={chunk_id} section='{section}' score={score:.3f}")
            print(f"     snippet: {snippet}")

    class _HarnessQuestionEmbedder:
        """Thin wrapper that exposes the Specter2 embedder via an encode method."""

        def __init__(self, embedder: "Specter2Embeddings"):
            self.embedder = embedder
            self._cache: Dict[str, List[float]] = {}

        def encode(self, text: str) -> List[float]:
            normalized = text.strip()
            key = hashlib.md5(normalized.encode("utf-8")).hexdigest()
            cached = self._cache.get(key)
            if cached is not None:
                return cached
            vector = self.embedder.embed_query(normalized)
            self._cache[key] = vector
            if len(self._cache) > 64:
                self._cache.pop(next(iter(self._cache)))
            return vector

    def _patch_source(doc: Mapping[str, Any], includes: Optional[Sequence[str]]) -> Dict[str, Any]:
        if not includes:
            return dict(doc)
        return {field: doc.get(field) for field in includes}

    def _stub_build_lexical_dsl(query_obj: _keywords.Query) -> Dict[str, Any]:
        raw_terms = query_obj.normal_terms or query_obj.boosted_terms
        return {
            "_raw_query": query_obj,
            "_raw_text": " ".join(term.strip() for term in raw_terms if term.strip()) or question,
        }

    def _stub_search_index(
        es: Any,
        index_name: str,
        dsl: Mapping[str, Any],
        *,
        size: int = 10,
        source_includes: Optional[Sequence[str]] = None,
        track_total_hits: Optional[object] = None,
    ) -> Dict[str, Any]:
        del es, index_name, track_total_hits
        query_obj = dsl.get("_raw_query")
        if not isinstance(query_obj, _keywords.Query):
            raise RuntimeError("Stub lexical search received an unexpected query payload.")
        hits = _compute_keyword_hits(query_obj, size)
        records = []
        for chunk_id, score in hits:
            source = _patch_source(chunk_by_id[chunk_id], source_includes)
            records.append({"_id": chunk_id, "_score": score, "_source": source})
        return {"hits": {"hits": records}}

    def _stub_knn_search(
        es: Any,
        index_name: str,
        vector: Sequence[float],
        *,
        k: int = 200,
        num_candidates: Optional[int] = None,
        filter_query: Optional[Mapping[str, Any]] = None,
        source_includes: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        del es, index_name, num_candidates, filter_query
        hits = _compute_semantic_hits_from_vector(vector, k)
        records = []
        for chunk_id, score in hits:
            source = _patch_source(chunk_by_id[chunk_id], source_includes)
            records.append({"_id": chunk_id, "_score": score, "_source": source})
        return {"hits": {"hits": records}}

    def _stub_fetch_chunk_by_id(es: Any, index_name: str, chunk_id: str) -> Optional[Dict[str, Any]]:
        del es, index_name
        return chunk_by_id.get(chunk_id)

    # Monkeypatch Elasticsearch helpers so `search` can run without a cluster.
    _es_index.build_lexical_dsl = _stub_build_lexical_dsl  # type: ignore[assignment]
    _es_index.search_index = _stub_search_index  # type: ignore[assignment]
    _es_index.knn_search = _stub_knn_search  # type: ignore[assignment]
    _es_index.fetch_chunk_by_id = _stub_fetch_chunk_by_id  # type: ignore[assignment]

    question_embedder = _HarnessQuestionEmbedder(harness_embedder)
    print(f"Loaded payload '{payload_path.name}' -> {len(chunk_docs)} chunks for doc '{doc_id}'.")
    print(f"Question: {question}\n")

    keyword_queries, keyword_debug = _keywords.extract_queries(question)
    if not keyword_queries:
        raise SystemExit("Keyword extractor returned zero queries for the provided question.")

    print("[Keyword Queries]")
    for idx, query_obj in enumerate(keyword_queries, start=1):
        print(
            f"  #{idx}: boosted={query_obj.boosted_terms} "
            f"normal={query_obj.normal_terms} section={query_obj.section_filter}"
        )

    keyword_top_k = _prompt_int("Top-K keyword hits to show", 5)
    semantic_top_k = _prompt_int("Top-K semantic hits to show", 5)
    rerank_top_k = _prompt_int("Top-K final results", 5)

    output_path = _prompt_path("Output file for full contexts", Path("search_full_chunks.txt"))

    keyword_logs: List[Dict[str, Any]] = []
    for idx, query_obj in enumerate(keyword_queries, start=1):
        hits = _compute_keyword_hits(query_obj, keyword_top_k)
        keyword_logs.append({"query_index": idx, "query": query_obj, "hits": hits})
        _format_hits(f"Keyword results for query #{idx}", hits)

    semantic_vector = question_embedder.encode(question)
    semantic_hits = _compute_semantic_hits_from_vector(semantic_vector, semantic_top_k)
    _format_hits("Semantic results", semantic_hits)

    print("\n[Hybrid Search]")
    config = SearchConfig(
        mode="hybrid",
        per_query_size_lex=max(keyword_top_k, 10),
        semantic_k=max(semantic_top_k, 10),
        return_k=rerank_top_k,
    )
    result = search(
        es=object(),  # unused by the stubbed helpers
        question=question,
        index_name="in-memory",
        config=config,
        embedder=question_embedder,
    )

    expansions = result.debug.get("expansions", [])
    if expansions:
        print("\n[Expansion Log]")
        for record in expansions:
            base = record.get("chunk_id")
            chain = ", ".join(f"{step['direction']}->{step['chunk_id']}" for step in record.get("expansions", []))
            print(
                f"  chunk={base} doc={record.get('doc_id')} expansions=[{chain}] "
                f"final_completeness={record.get('final_completeness')}"
            )
    else:
        print("\n[Expansion Log] No expansions were required.")

    print("\n[Final Reranked Contexts]")
    if not result.contexts:
        print("No reranked contexts returned.")
    for rank, context in enumerate(result.contexts, start=1):
        snippet = shorten(" ".join((context.get("text") or "").split()), width=220, placeholder=" …")
        print(
            f"  #{rank} doc={context.get('doc_id')} chunks={context.get('chunk_ids')} "
            f"completeness={context.get('completeness')} fused={context['provenance']['fused_score']:.3f}"
        )
        print(f"     snippet: {snippet}")

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    def _write_hits_section(handle, label: str, hits: Sequence[Tuple[str, float]]) -> None:
        handle.write(f"{label}\n")
        for rank, (chunk_id, score) in enumerate(hits, start=1):
            source = chunk_by_id.get(chunk_id, {})
            handle.write(f"  #{rank} chunk={chunk_id} section={source.get('section_title')} score={score:.3f}\n")
            handle.write((source.get("text") or "").strip() + "\n")
        handle.write("-" * 80 + "\n")

    with output_path.open("w", encoding="utf-8") as fh:
        if not result.contexts:
            fh.write("No contexts returned.\n")
        fh.write("[Keyword Search Results]\n")
        if not keyword_logs:
            fh.write("  (no keyword queries)\n")
        for entry in keyword_logs:
            label = f"Query #{entry['query_index']} boosted={entry['query'].boosted_terms} normal={entry['query'].normal_terms}"
            _write_hits_section(fh, label, entry["hits"])
        fh.write("[Semantic Search Results]\n")
        if semantic_hits:
            _write_hits_section(fh, "Semantic leaderboard", semantic_hits)
        else:
            fh.write("  (semantic search disabled)\n" + "-" * 80 + "\n")
        fh.write("[Final Reranked Contexts]\n")
        for rank, context in enumerate(result.contexts, start=1):
            fh.write(
                f"#{rank} doc={context.get('doc_id')} chunks={context.get('chunk_ids')} "
                f"completeness={context.get('completeness')} fused={context['provenance']['fused_score']:.3f}\n"
            )
            fh.write((context.get("text") or "").strip())
            fh.write("\n" + ("-" * 80) + "\n")
    print(f"\n[Output] Saved full contexts to {output_path}")

    print("\n[Debug Summary]")
    print(
        json.dumps(
            {
                "keyword_debug": keyword_debug,
                "expansions": expansions,
                "llm_rerank": result.debug.get("llm_rerank"),
            },
            indent=2,
        )
    )
