"""
Combined v4 search helpers: query analysis, BM25 lookup, semantic k-NN, and reranking.

This module merges the essentials from `_keywords_v4`, `_es_index_v4`, and `_search_v4`
into three public functions:
    * `bm25_search`    - stopword-trimmed lexical search for quote strings
    * `semantic_search`- embed-and-search over specter2 vectors
    * `rerank`         - merge, dedupe, and OpenAI-rerank results against the original query
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

from elasticsearch import Elasticsearch

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore

try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
except Exception:  # pragma: no cover - keep running without sklearn
    ENGLISH_STOP_WORDS = frozenset(
        {
            "the",
            "a",
            "an",
            "and",
            "or",
            "of",
            "for",
            "to",
            "in",
            "on",
            "with",
            "at",
            "by",
            "from",
            "that",
            "this",
            "these",
            "those",
            "is",
            "are",
        }
    )

import torch
import torch.nn.functional as F
from adapters import AutoAdapterModel
from transformers import AutoModel, AutoTokenizer

try:
    from config import CONFIG  # type: ignore
except ImportError:  # pragma: no cover - optional config
    CONFIG = None  # type: ignore

from QuerySpec_v4 import analyze_query_llm_v4

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

DEFAULT_EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_ADAPTER_NAME = getattr(getattr(CONFIG, "vector", None), "adapter_name", None) if CONFIG else None
DEFAULT_DEVICE = getattr(getattr(CONFIG, "runtime", None), "device", None) if CONFIG else None

# Public knobs for retrieval depth.
top_k_bm25 = 10
top_k_semantic = 10


class _Specter2Embedder:
    """Local Specter2 embedder that avoids helper imports."""

    def __init__(
        self,
        model_name: str,
        *,
        adapter_name: Optional[str] = DEFAULT_ADAPTER_NAME,
        device: Optional[str] = DEFAULT_DEVICE,
        batch_size: int = 16,
    ):
        self.device = device or "cpu"
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        # Prefer adapter models when compatible; otherwise fall back to vanilla AutoModel (e.g., Qwen embeddings).
        try:
            self.model = AutoAdapterModel.from_pretrained(model_name)
            if adapter_name:
                self.model.load_adapter(adapter_name, source="hf", load_as="proximity", set_active=True)
        except Exception:
            self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def _prepare_inputs(self, texts: List[str]):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False,
            max_length=512,
        )

    def _embed_batch(self, inputs):
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        normalized = F.normalize(cls_embeddings, p=2, dim=1)
        return normalized

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings: List[List[float]] = []
        with torch.inference_mode():
            for start_idx in range(0, len(texts), self.batch_size):
                batch = texts[start_idx : start_idx + self.batch_size]
                inputs = self._prepare_inputs(batch)
                normalized = self._embed_batch(inputs)
                embeddings.extend(normalized.cpu().tolist())
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed_texts(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed_texts([text])[0]


def _load_specter2_embedder(model_name: str) -> _Specter2Embedder:
    return _Specter2Embedder(model_name)


class _QuestionEmbedder:
    """Minimal cache wrapper around a Specter2-style embedder."""

    def __init__(self, model_name: str, cache_size: int = 128):
        self.model_name = model_name
        self.cache_size = cache_size
        self._embedder = None
        self._cache: Dict[str, List[float]] = {}
        self._lru: List[str] = []

    def encode(self, text: str) -> List[float]:
        key = text.strip()
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


def _analyze_query(question: str, abstract: Optional[str] = None) -> str:
    spec = analyze_query_llm_v4(question, abstract=abstract)
    return spec.simulated_answer or question.strip()


def _remove_stopwords(text: str, stopwords: Set[str]) -> str:
    tokens = re.findall(r"[A-Za-z0-9']+", text.lower())
    filtered = [tok for tok in tokens if tok and tok not in stopwords]
    return " ".join(filtered).strip() or text.strip()


def _build_lexical_dsl(answer_text: str) -> Dict[str, Any]:
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


def _search_index(
    es: Elasticsearch,
    index_name: str,
    dsl: Mapping[str, Any],
    *,
    size: int,
    source_includes: Optional[Sequence[str]],
) -> Dict[str, Any]:
    body = dict(dsl)
    body.setdefault("size", size)
    if source_includes is not None:
        body["_source"] = {"includes": list(source_includes)}
    return es.search(index=index_name, body=body)


def _knn_search(
    es: Elasticsearch,
    index_name: str,
    vector: Sequence[float],
    *,
    k: int,
    source_includes: Optional[Sequence[str]],
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


def bm25_search(
    es: Elasticsearch,
    index_name: str,
    analyzed_text: str,
    *,
    top_k: Optional[int] = None,
    source_fields: Sequence[str] = DEFAULT_SOURCE_FIELDS,
) -> List[str]:
    """
    Run a BM25 search using the query analyzer output, with stopwords stripped via scikit-learn.
    Returns a list of quote strings (chunk text) sorted by ES score.
    """
    processed = _remove_stopwords(analyzed_text, set(ENGLISH_STOP_WORDS))
    dsl = _build_lexical_dsl(processed)
    response = _search_index(
        es,
        index_name,
        dsl,
        size=top_k or top_k_bm25,
        source_includes=source_fields,
    )
    hits = response.get("hits", {}).get("hits", []) or []
    quotes: List[str] = []
    for hit in hits:
        text = (hit.get("_source") or {}).get("text")
        if text:
            quotes.append(text)
    return quotes


def semantic_search(
    es: Elasticsearch,
    index_name: str,
    search_text: str,
    *,
    embedder: Optional[_QuestionEmbedder] = None,
    top_k: Optional[int] = None,
    source_fields: Sequence[str] = DEFAULT_SOURCE_FIELDS,
    embed_model: str = DEFAULT_EMBED_MODEL,
) -> List[str]:
    """
    Run a semantic search by embedding the provided (already-analyzed) string directly and
    querying the specter2 vector field. Returns quote strings.
    """
    embedder = embedder or _QuestionEmbedder(embed_model)
    vector = embedder.encode(search_text)
    response = _knn_search(
        es,
        index_name,
        vector,
        k=top_k or top_k_semantic,
        source_includes=source_fields,
    )
    hits = response.get("hits", {}).get("hits", []) or []
    quotes: List[str] = []
    for hit in hits:
        text = (hit.get("_source") or {}).get("text")
        if text:
            quotes.append(text)
    return quotes


def rerank(
    question: str,
    bm25_results: Sequence[str],
    semantic_results: Sequence[str],
    *,
    model: str = "gpt-4o-mini",
) -> List[str]:
    """
    Combine and dedupe lexical + semantic quotes, then rerank them by relevance to the
    original user query using OpenAI. Falls back to the combined list if OpenAI is unavailable.
    """
    combined: List[str] = []
    seen: Set[str] = set()
    for quote in list(semantic_results) + list(bm25_results):
        normalized = quote.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        combined.append(normalized)

    if not combined:
        return []

    if OpenAI is None:
        return combined

    prompt_lines = [
        "Rerank the following quotes by how well they answer the user's question.",
        f"Question: {question.strip()}",
        "Quotes:",
    ]
    for idx, quote in enumerate(combined, start=1):
        prompt_lines.append(f"{idx}. {quote}")
    prompt_lines.append("Return the quotes sorted from most to least relevant, one per line, unchanged.")
    prompt = "\n".join(prompt_lines)

    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        content = resp.choices[0].message.content if resp.choices else None
        if not content:
            return combined
        reranked: List[str] = []
        for line in content.splitlines():
            cleaned = line.strip()
            if not cleaned:
                continue
            cleaned = re.sub(r"^\d+[\).\s-]*", "", cleaned).strip()
            if cleaned and cleaned in seen:
                reranked.append(cleaned)
        return reranked or combined
    except Exception:
        return combined


if __name__ == "__main__":
    """
    Quick manual test:
        * Fetch arXiv:2310.11511
        * Analyze the query to get the simulated answer text
        * Build an in-memory ES stub where each sentence is a chunk with embeddings
        * Run BM25, semantic search, then rerank and print outputs
    """

    from _get_full_text import get_full_text

    TEST_ID = "arXiv:2310.11511"
    QUESTION = "Help me find papers that apply the agentic framework in RAG applications"

    payload = get_full_text(TEST_ID)
    document_text = payload.get("processed_text") or payload.get("full_text") or ""
    if not document_text:
        raise SystemExit("No text returned from get_full_text.")
    test_abstract = (
        "Despite their remarkable capabilities, large language models (LLMs) often produce responses containing factual inaccuracies due to their sole reliance on the parametric knowledge they encapsulate. "
        "Retrieval-Augmented Generation (RAG), an ad hoc approach that augments LMs with retrieval of relevant knowledge, decreases such issues. However, indiscriminately retrieving and incorporating a fixed number of retrieved passages, regardless of whether retrieval is necessary, or passages are relevant, diminishes LM versatility or can lead to unhelpful response generation. "
        "We introduce a new framework called Self-Reflective Retrieval-Augmented Generation (Self-RAG) that enhances an LM's quality and factuality through retrieval and self-reflection. Our framework trains a single arbitrary LM that adaptively retrieves passages on-demand, and generates and reflects on retrieved passages and its own generations using special tokens, called reflection tokens. "
        "Generating reflection tokens makes the LM controllable during the inference phase, enabling it to tailor its behavior to diverse task requirements. Experiments show that Self-RAG (7B and 13B parameters) significantly outperforms state-of-the-art LLMs and retrieval-augmented models on a diverse set of tasks. "
        "Specifically, Self-RAG outperforms ChatGPT and retrieval-augmented Llama2-chat on Open-domain QA, reasoning and fact verification tasks, and it shows significant gains in improving factuality and citation accuracy for long-form generations relative to these models."
    )

    # Split into sentences and treat each as a chunk.
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", document_text) if s.strip()]
    chunk_docs = []
    doc_id = payload.get("arxiv_id") or TEST_ID
    for idx, sentence in enumerate(sentences):
        chunk_id = f"{doc_id}#{idx:04d}"
        chunk_docs.append({"id": chunk_id, "doc_id": doc_id, "text": sentence})

    # Embed sentences for the semantic stub.
    embedder_backend = _load_specter2_embedder(DEFAULT_EMBED_MODEL)
    chunk_vectors = embedder_backend.embed_documents([doc["text"] for doc in chunk_docs])
    chunk_vector_map = {doc["id"]: vec for doc, vec in zip(chunk_docs, chunk_vectors)}
    chunk_by_id = {doc["id"]: doc for doc in chunk_docs}

    class _StubElasticsearch:
        """Minimal stub to satisfy bm25/semantic search calls."""

        def search(self, index: str, body: Mapping[str, Any]) -> Dict[str, Any]:
            del index
            if "knn" in body:
                query_vector = body["knn"]["query_vector"]
                k = body["knn"]["k"]
                hits = []
                for chunk_id, vec in chunk_vector_map.items():
                    score = float(sum(a * b for a, b in zip(query_vector, vec)))
                    src = {field: chunk_by_id[chunk_id].get(field) for field in DEFAULT_SOURCE_FIELDS}
                    hits.append({"_id": chunk_id, "_score": score, "_source": src})
                hits.sort(key=lambda h: h["_score"], reverse=True)
                return {"hits": {"hits": hits[:k]}, "took": 0}
            raw_query = body.get("_raw_query") or ""
            terms = [t.strip().lower() for t in raw_query.split() if t.strip()]
            hits = []
            for doc in chunk_docs:
                text_lower = (doc.get("text") or "").lower()
                score = sum(text_lower.count(t) for t in terms) or 0.0
                if score > 0:
                    src = {field: doc.get(field) for field in DEFAULT_SOURCE_FIELDS}
                    hits.append({"_id": doc["id"], "_score": float(score), "_source": src})
            hits.sort(key=lambda h: h["_score"], reverse=True)
            size = body.get("size") or len(hits)
            return {"hits": {"hits": hits[:size]}, "took": 0}

    stub_es = _StubElasticsearch()
    print(f"\n[Query Analyzer] question='{QUESTION}'")
    analyzed = _analyze_query(QUESTION, abstract=test_abstract)
    print(f"Simulated answer: {analyzed}")

    bm25_quotes = bm25_search(stub_es, index_name="stub", analyzed_text=analyzed, top_k=5)
    semantic_quotes = semantic_search(
        stub_es,
        index_name="stub",
        search_text=analyzed,
        top_k=5,
        embedder=_QuestionEmbedder(DEFAULT_EMBED_MODEL),
    )
    reranked_quotes = rerank(QUESTION, bm25_quotes, semantic_quotes)

    print("\n[BM25 Results]")
    for i, quote in enumerate(bm25_quotes, start=1):
        print(f"{i}. {quote}")

    print("\n[Semantic Results]")
    for i, quote in enumerate(semantic_quotes, start=1):
        print(f"{i}. {quote}")

    print("\n[Reranked Results]")
    for i, quote in enumerate(reranked_quotes, start=1):
        print(f"{i}. {quote}")
