"""
Helpers for building a FAISS-backed QA index over logits-guided parent chunks.

Each parent chunk is split into a configurable number of sentence-aligned
sub-chunks. The sub-chunks are embedded and stored in a FAISS index. At query
time, parent chunks are scored by the best-performing sub-chunk and the top
parents are returned.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import sys

from langchain_community.vectorstores import FAISS

current_path = Path(__file__).resolve()
if __package__:
    from .._log import log  # type: ignore
    from ._LG_chunker import LGChunk  # type: ignore
    from .Archive import _LG_chunker as _lg_chunker  # type: ignore
    from .._load_data import EMBED_MODEL_NAME, Specter2Embeddings  # type: ignore
else:  # pragma: no cover - support running as a script
    parent_dir = current_path.parent
    if str(parent_dir) not in sys.path:
        sys.path.append(str(parent_dir))
    if str(parent_dir.parent) not in sys.path:
        sys.path.append(str(parent_dir.parent))
    from _log import log  # type: ignore
    from Archive._LG_chunker import LGChunk  # type: ignore
    import Archive._LG_chunker as _lg_chunker  # type: ignore
    from _load_data import EMBED_MODEL_NAME, Specter2Embeddings  # type: ignore


@dataclass(frozen=True)
class ParentChunkResult:
    """Returned record containing the parent chunk and the sub-chunk that matched."""

    chunk: LGChunk
    score: float
    sub_chunk: str
    metadata: Dict[str, int | float | str]


def _split_sentences_into_subchunks(
    sentences: Sequence[str],
    desired_count: int,
) -> List[List[str]]:
    """Split a sequence of sentences into contiguous sublists."""
    if desired_count <= 0:
        raise ValueError("desired_count must be at least 1.")

    total_sentences = len(sentences)
    if total_sentences == 0:
        return []

    count = min(desired_count, total_sentences)
    base_size = total_sentences // count
    remainder = total_sentences % count

    subchunks: List[List[str]] = []
    cursor = 0
    for idx in range(count):
        take = base_size + (1 if idx < remainder else 0)
        end = cursor + take
        subchunk = list(sentences[cursor:end])
        if subchunk:
            subchunks.append(subchunk)
        cursor = end

    if cursor < total_sentences and subchunks:
        subchunks[-1].extend(sentences[cursor:])

    return subchunks


def _chunk_to_subchunks(
    chunk: LGChunk,
    subchunks_per_chunk: int,
) -> List[Tuple[str, int, int]]:
    """Split a parent chunk into sub-chunks, return text and sentence indices."""
    sentences = _lg_chunker._sentence_tokenize(chunk.text)  # type: ignore[attr-defined]
    if not sentences:
        return []

    subchunks = _split_sentences_into_subchunks(sentences, subchunks_per_chunk)

    results: List[Tuple[str, int, int]] = []
    local_cursor = 0
    for sentences_block in subchunks:
        start_sentence = chunk.start_sentence + local_cursor
        local_length = len(sentences_block)
        end_sentence = start_sentence + local_length - 1
        text = " ".join(sentences_block).strip()
        if text:
            results.append((text, start_sentence, end_sentence))
        local_cursor += local_length
    return results


class FullTextQAStore:
    """FAISS-backed QA store over logits-guided parent chunks."""

    def __init__(
        self,
        vector_store: FAISS,
        embeddings: Specter2Embeddings,
        parent_chunks: Sequence[LGChunk],
        subchunks_per_chunk: int,
    ) -> None:
        self._vector_store = vector_store
        self._embeddings = embeddings
        self._parent_chunks = list(parent_chunks)
        self._subchunks_per_chunk = subchunks_per_chunk

    @classmethod
    def from_chunks(
        cls,
        parent_chunks: Sequence[LGChunk],
        subchunks_per_chunk: int,
        *,
        embeddings: Optional[Specter2Embeddings] = None,
    ) -> "FullTextQAStore":
        if subchunks_per_chunk <= 0:
            raise ValueError("subchunks_per_chunk must be a positive integer.")
        if not parent_chunks:
            raise ValueError("parent_chunks must contain at least one chunk.")

        embeddings = embeddings or Specter2Embeddings(EMBED_MODEL_NAME)

        texts: List[str] = []
        metadatas: List[Dict[str, int | float]] = []

        log(
            f"Building FAISS index from {len(parent_chunks)} parent chunks "
            f"with {subchunks_per_chunk} sub-chunks each"
        )

        for parent_idx, chunk in enumerate(parent_chunks):
            subchunks = _chunk_to_subchunks(chunk, subchunks_per_chunk)
            if not subchunks:
                log(f"Skipping empty chunk at index {parent_idx}")
                continue
            for sub_idx, (text, start_sentence, end_sentence) in enumerate(subchunks):
                texts.append(text)
                metadatas.append(
                    {
                        "parent_idx": parent_idx,
                        "parent_start": chunk.start_sentence,
                        "parent_end": chunk.end_sentence,
                        "subchunk_idx": sub_idx,
                        "subchunk_start": start_sentence,
                        "subchunk_end": end_sentence,
                        "eos_probability": chunk.eos_probability,
                    }
                )

        if not texts:
            raise ValueError("No valid sub-chunks were produced from the provided chunks.")

        vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
        log(f"FAISS index initialised with {len(texts)} sub-chunks")
        return cls(
            vector_store=vector_store,
            embeddings=embeddings,
            parent_chunks=parent_chunks,
            subchunks_per_chunk=subchunks_per_chunk,
        )

    def query(
        self,
        query: str,
        *,
        parent_top_k: int,
        subchunk_top_k: Optional[int] = None,
    ) -> List[ParentChunkResult]:
        if parent_top_k <= 0:
            raise ValueError("parent_top_k must be a positive integer.")
        if not query.strip():
            return []

        if subchunk_top_k is None:
            subchunk_top_k = max(parent_top_k * self._subchunks_per_chunk, parent_top_k)

        log(f"Querying FAISS index for top {parent_top_k} parent chunks")
        results = self._vector_store.similarity_search_with_score(query, k=subchunk_top_k)
        if not results:
            return []

        best_per_parent: Dict[int, ParentChunkResult] = {}

        for doc, distance in results:
            metadata = doc.metadata or {}
            parent_idx = int(metadata.get("parent_idx", -1))
            if parent_idx < 0 or parent_idx >= len(self._parent_chunks):
                continue
            similarity = -float(distance)
            existing = best_per_parent.get(parent_idx)
            if existing is None or similarity > existing.score:
                enriched_metadata: Dict[str, int | float | str] = {
                    **{k: v for k, v in metadata.items() if isinstance(v, (int, float, str))},
                    "distance": float(distance),
                }
                best_per_parent[parent_idx] = ParentChunkResult(
                    chunk=self._parent_chunks[parent_idx],
                    score=similarity,
                    sub_chunk=doc.page_content,
                    metadata=enriched_metadata,
                )

        ordered = sorted(best_per_parent.values(), key=lambda item: item.score, reverse=True)
        return ordered[:parent_top_k]


__all__ = ["FullTextQAStore", "ParentChunkResult"]


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    input_path = base_dir / "test_paper.json"

    if not input_path.exists():
        raise FileNotFoundError(f"Expected test input at {input_path}")

    log(f"Loading test document from {input_path}")
    import json

    data = input_path.read_text(encoding="utf-8")
    payload = json.loads(data)
    full_text = payload.get("full_text")
    if not isinstance(full_text, str) or not full_text.strip():
        raise ValueError("Expected a non-empty 'full_text' field in test payload.")

    chunker_config = _lg_chunker.LGChunkerConfig(window_size_words=300)
    log("Generating parent chunks with logits-guided chunker v2")
    chunks = _lg_chunker.logits_guided_parent_chunks_v2(full_text, config=chunker_config)

    if not chunks:
        raise RuntimeError("Chunker produced no parent chunks for test document.")

    log(f"Generated {len(chunks)} parent chunks; building QA store")
    store = FullTextQAStore.from_chunks(chunks, subchunks_per_chunk=3)

    log("Enter an empty line to exit.")
    while True:
        try:
            query = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            break
        results = store.query(query, parent_top_k=3)
        if not results:
            log("No results returned for query.")
            continue
        for idx, result in enumerate(results, start=1):
            log(
                f"Result {idx}: parent chunk sentences "
                f"{result.chunk.start_sentence}-{result.chunk.end_sentence} "
                f"score={result.score:.4f}"
            )
            log(f"    Parent chunk text: {result.chunk.text}")
            log(f"    Matching sub-chunk: {result.sub_chunk}")
