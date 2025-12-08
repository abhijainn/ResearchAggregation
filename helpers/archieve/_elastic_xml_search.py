"""
Utilities for indexing TEI XML sentences into Elasticsearch and running searches.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import ssl
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from elasticsearch import Elasticsearch
from elasticsearch import helpers as es_helpers
from elasticsearch import exceptions as es_exceptions

try:
    from .._load_data import Specter2Embeddings, EMBED_MODEL_NAME as _DEFAULT_EMBED_MODEL
except ImportError:  # pragma: no cover - allow running as script
    try:
        from _load_data import Specter2Embeddings, EMBED_MODEL_NAME as _DEFAULT_EMBED_MODEL  # type: ignore
    except ImportError:  # pragma: no cover - embedding optional
        Specter2Embeddings = None  # type: ignore
        _DEFAULT_EMBED_MODEL = None  # type: ignore

DEFAULT_EMBED_MODEL = _DEFAULT_EMBED_MODEL if _DEFAULT_EMBED_MODEL else None

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def create_embedder(model_name: str, *, device: Optional[str] = None, batch_size: int = 16):
    if Specter2Embeddings is None:
        raise RuntimeError(
            "Embedding support requires 'helpers._load_data.Specter2Embeddings'. "
            "Install the necessary dependencies or disable embeddings."
        )
    return Specter2Embeddings(model_name=model_name, device=device, batch_size=batch_size)


@dataclass
class SentenceRecord:
    sentence_id: str
    text: str
    section: Optional[str]
    order: int
    embedding: Optional[List[float]] = None


def _strip_namespace(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _split_into_sentences(paragraph: str) -> List[str]:
    if not paragraph:
        return []
    pieces = _SENTENCE_SPLIT_RE.split(paragraph)
    sentences = [piece.strip() for piece in pieces if piece.strip()]
    if sentences:
        return sentences
    return [paragraph.strip()]


def parse_tei_sentences(path: str | Path) -> List[SentenceRecord]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"TEI file not found: {path}")

    records: List[SentenceRecord] = []
    section_index = 0
    sentence_index = 0
    current_section: Optional[str] = None
    current_section_key = f"s{section_index:04d}"

    for event, element in ET.iterparse(path, events=("start", "end")):
        tag = _strip_namespace(element.tag)

        if event == "end":
            if tag == "head":
                title = " ".join(text.strip() for text in element.itertext() if text).strip()
                if title:
                    section_index += 1
                    current_section = title
                    current_section_key = f"s{section_index:04d}"
                    sentence_index = 0
                element.clear()
            elif tag == "p":
                paragraph = " ".join(text.strip() for text in element.itertext() if text).strip()
                if paragraph:
                    for text in _split_into_sentences(paragraph):
                        sentence_index += 1
                        records.append(
                            SentenceRecord(
                                sentence_id=f"{current_section_key}-{sentence_index:04d}",
                                text=text,
                                section=current_section,
                                order=len(records),
                            )
                        )
                element.clear()

    return records


def ensure_index(es: Elasticsearch, index: str, recreate: bool = False, *, vector_dims: Optional[int] = None) -> None:
    try:
        exists = es.indices.exists(index=index)
    except es_exceptions.ConnectionError:
        raise
    except es_exceptions.TransportError as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to check index '{index}': {exc}") from exc

    if exists and recreate:
        try:
            es.indices.delete(index=index)
        except es_exceptions.ConnectionError:
            raise
        except es_exceptions.TransportError as exc:  # pragma: no cover
            raise RuntimeError(f"Failed to delete index '{index}': {exc}") from exc
        exists = False

    if exists:
        return

    mappings = {
        "properties": {
            "sentence_id": {"type": "keyword"},
            "text": {"type": "text", "analyzer": "english"},
            "section": {
                "type": "text",
                "fields": {"raw": {"type": "keyword"}},
            },
            "order": {"type": "integer"},
        }
    }

    if vector_dims is not None:
        mappings["properties"]["text_vector"] = {
            "type": "dense_vector",
            "dims": vector_dims,
            "index": True,
            "similarity": "cosine",
        }

    es.indices.create(
        index=index,
        settings={
            "analysis": {
                "analyzer": {
                    "default": {
                        "type": "standard",
                        "stopwords": "_english_",
                    }
                }
            }
        },
        mappings=mappings,
    )


def index_sentences(
    es: Elasticsearch,
    index: str,
    sentences: Iterable[SentenceRecord],
    *,
    chunk_size: int = 500,
    refresh: bool = False,
) -> None:
    actions = []
    for record in sentences:
        source = {
            "sentence_id": record.sentence_id,
            "text": record.text,
            "section": record.section,
            "order": record.order,
        }
        if record.embedding is not None:
            source["text_vector"] = record.embedding
        actions.append(
            {
                "_index": index,
                "_id": record.sentence_id,
                "_source": source,
            }
        )

    es_helpers.bulk(es, actions, chunk_size=chunk_size)

    if refresh:
        es.indices.refresh(index=index)


def search_sentences(
    es: Elasticsearch,
    index: str,
    query: str,
    *,
    size: int = 5,
    embedder: Optional[object] = None,
    knn_candidates: Optional[int] = None,
) -> List[dict]:
    params = {
        "index": index,
        "query": {"match": {"text": {"query": query}}},
        "highlight": {"fields": {"text": {}}},
        "size": size,
    }

    if embedder is not None:
        query_vector = embedder.embed_query(query)
        num_candidates = knn_candidates or max(size * 4, 100)
        params["knn"] = {
            "field": "text_vector",
            "query_vector": query_vector,
            "k": size,
            "num_candidates": num_candidates,
        }

    response = es.search(**params)

    hits = response.get("hits", {}).get("hits", [])
    results: List[dict] = []
    for hit in hits:
        source = hit.get("_source", {})
        highlight = hit.get("highlight", {}).get("text", [])
        results.append(
            {
                "sentence_id": source.get("sentence_id"),
                "text": source.get("text"),
                "section": source.get("section"),
                "score": hit.get("_score"),
                "highlight": " ".join(highlight) if highlight else None,
            }
        )
    return results


def build_index_from_file(
    es: Elasticsearch,
    index: str,
    tei_path: str | Path,
    *,
    embedder: Optional[object] = None,
    recreate: bool = True,
    refresh: bool = False,
) -> List[SentenceRecord]:
    print("[STEP] Parsing TEI XML into sentences...")
    try:
        sentences = parse_tei_sentences(tei_path)
    except Exception as exc:
        print(f"[FAIL] Parsing TEI XML failed: {exc}")
        raise
    print(f"[STEP] Parsed {len(sentences)} sentences.")

    vector_dims: Optional[int] = None
    if embedder is not None and sentences:
        print("[STEP] Computing sentence embeddings for hybrid retrieval...")
        texts = [record.text for record in sentences]
        embeddings = embedder.embed_documents(texts)
        if embeddings and len(embeddings) != len(sentences):
            raise ValueError(
                f"Embedding count ({len(embeddings)}) did not match sentence count ({len(sentences)})."
            )
        if embeddings:
            vector_dims = len(embeddings[0])
            for record, embedding in zip(sentences, embeddings):
                record.embedding = embedding
        print("[STEP] Embeddings computed.")
    elif embedder is None:
        print("[STEP] Embeddings disabled; proceeding with BM25 only.")

    print("[STEP] Ensuring Elasticsearch index exists...")
    try:
        ensure_index(es, index, recreate=recreate, vector_dims=vector_dims)
    except Exception as exc:
        print(f"[FAIL] Ensuring index '{index}' failed: {exc}")
        raise
    print("[STEP] Index ready.")

    if sentences:
        print("[STEP] Bulk indexing sentences into Elasticsearch...")
        try:
            index_sentences(es, index, sentences, refresh=refresh)
        except Exception as exc:
            print(f"[FAIL] Bulk indexing failed: {exc}")
            raise
        print("[STEP] Bulk indexing complete.")
    else:
        print("[STEP] No sentences parsed; skipping indexing.")

    return sentences


__all__ = [
    "create_embedder",
    "SentenceRecord",
    "parse_tei_sentences",
    "ensure_index",
    "index_sentences",
    "search_sentences",
    "build_index_from_file",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index TEI XML sentences into Elasticsearch and run a sample query.")
    default_tei = Path(__file__).resolve().parent / "test.xml"
    parser.add_argument("--tei", default=str(default_tei), help="Path to the TEI XML file to index.")
    parser.add_argument("--index", default="tei-sentences", help="Elasticsearch index name.")
    parser.add_argument("--query", default=" Most rumor detectors use content, user stats, or coarse diffusion stats, but miss the structure of how posts spread", help="Search query to run after indexing.")
    parser.add_argument("--host", default="https://localhost:9200", help="Elasticsearch host URL.")
    parser.add_argument("--recreate", action="store_true", default=True, help="Drop and recreate the index before indexing (default: enabled).")
    parser.add_argument("--no-recreate", dest="recreate", action="store_false", help="Do not drop and recreate the index before indexing.")
    parser.add_argument("--size", type=int, default=5, help="Number of search hits to return.")
    parser.add_argument("--user", default="elastic", help="Elasticsearch basic auth username (default: elastic).")
    parser.add_argument(
        "--password",
        help="Elasticsearch basic auth password. If omitted, the ELASTIC_PASSWORD environment variable is used.",
    )
    parser.add_argument(
        "--ca-cert",
        dest="ca_cert",
        default="~/http_ca.crt",
        help="Path to CA certificate for HTTPS connections (default: ~/http_ca.crt).",
    )
    parser.add_argument(
        "--allow-insecure",
        action="store_true",
        help="Disable certificate verification (use only for local testing).",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBED_MODEL,
        help=(
            "Sentence embedding model to use for hybrid retrieval."
            if DEFAULT_EMBED_MODEL
            else "Sentence embedding model to use for hybrid retrieval."
        ),
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
        help="Disable embedding computation and hybrid retrieval.",
    )
    parser.add_argument(
        "--knn-candidates",
        type=int,
        default=100,
        help="Number of vector candidates to consider for hybrid search.",
    )
    args = parser.parse_args()

    password = args.password or os.getenv("ELASTIC_PASSWORD")
    if args.user and not password:
        raise SystemExit("Password not provided. Supply --password or set ELASTIC_PASSWORD.")

    es_kwargs = {}
    if args.ca_cert:
        ca_cert_path = Path(args.ca_cert).expanduser()
        if not ca_cert_path.exists():
            raise SystemExit(f"CA certificate not found: {ca_cert_path}")
        ca_cert_path = ca_cert_path.resolve()
        try:
            ssl_context = ssl.create_default_context(cafile=str(ca_cert_path))
        except Exception as exc:
            raise SystemExit(f"Failed to load CA certificate '{ca_cert_path}': {exc}") from exc
        es_kwargs["ssl_context"] = ssl_context
    elif not args.allow_insecure:
        print("[WARN] No CA certificate supplied; relying on system trust store.")
    if args.user and password:
        es_kwargs["basic_auth"] = (args.user, password)
    if args.allow_insecure:
        es_kwargs["verify_certs"] = False
        print("[WARN] TLS verification disabled (allow-insecure).")

    print(f"[INFO] Connecting to Elasticsearch host: {args.host}")
    if args.ca_cert:
        print(f"[INFO] Using CA certificate: {ca_cert_path}")
    if args.user:
        print(f"[INFO] Authenticating as user: {args.user}")

    try:
        es_client = Elasticsearch(args.host, **es_kwargs)
    except es_exceptions.ConnectionError as exc:
        raise SystemExit(
            "Failed to connect to Elasticsearch during client creation; ensure it is running and accessible."
        ) from exc
    except Exception as exc:  # pragma: no cover - unexpected failure
        raise SystemExit(f"Unexpected error constructing Elasticsearch client: {exc}") from exc

    tei_path = Path(args.tei)
    if not tei_path.exists():
        raise SystemExit(f"TEI file not found: {tei_path}")

    embedding_model_name: Optional[str] = None
    embedder = None
    if args.no_embeddings:
        print("[INFO] Embeddings disabled via --no-embeddings.")
    else:
        embedding_model_name = args.embedding_model
        if embedding_model_name:
            print(f"[INFO] Loading embedding model: {embedding_model_name}")
            try:
                embedder = create_embedder(
                    embedding_model_name,
                    device=args.embedding_device,
                    batch_size=args.embedding_batch_size,
                )
            except Exception as exc:
                raise SystemExit(f"Failed to load embedding model '{embedding_model_name}': {exc}") from exc
        else:
            print("[INFO] No embedding model specified; hybrid retrieval disabled.")

    print(f"[INFO] Using TEI file: {tei_path}")
    print(f"[INFO] Target index: {args.index} (recreate={args.recreate})")

    try:
        print("[INFO] Parsing TEI XML and indexing sentences...")
        sentences = build_index_from_file(
            es_client,
            args.index,
            tei_path,
            embedder=embedder,
            recreate=args.recreate,
            refresh=True,
        )
        print(f"[INFO] Indexed {len(sentences)} sentences.")
    except es_exceptions.ConnectionError as exc:
        raise SystemExit(
            "Failed while communicating with Elasticsearch during indexing."
        ) from exc
    except Exception as exc:
        raise SystemExit(f"Unexpected error during indexing step: {exc}") from exc

    try:
        print(f"[INFO] Running search for query: {args.query!r}")
        results = search_sentences(
            es_client,
            args.index,
            args.query,
            size=args.size,
            embedder=embedder,
            knn_candidates=args.knn_candidates,
        )
        output = {
            "tei_path": str(tei_path.resolve()),
            "indexed_count": len(sentences),
            "query": args.query,
            "results": results,
        }
        print(json.dumps(output, indent=2))
    except es_exceptions.ConnectionError as exc:
        raise SystemExit(
            "Failed while communicating with Elasticsearch during search."
        ) from exc
    except Exception as exc:
        raise SystemExit(f"Unexpected error during search step: {exc}") from exc
