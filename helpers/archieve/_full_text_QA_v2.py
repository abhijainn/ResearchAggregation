"""
Question-answer retriever that prepares fixed-size document chunks, generates synthetic
questions with a Qwen 3 family model, and indexes them with FAISS for semantic lookup.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json
import sys
import time

import torch
from contextlib import contextmanager
from langchain_community.vectorstores import FAISS
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

current_path = Path(__file__).resolve()
if __package__:
    from .._load_data import EMBED_MODEL_NAME, Specter2Embeddings  # type: ignore
    from .._log import log  # type: ignore
else:  # pragma: no cover
    parent_dir = current_path.parent
    if str(parent_dir) not in sys.path:
        sys.path.append(str(parent_dir))
    if str(parent_dir.parent) not in sys.path:
        sys.path.append(str(parent_dir.parent))
    from _load_data import EMBED_MODEL_NAME, Specter2Embeddings  # type: ignore
    from _log import log  # type: ignore


GENERATION_MODEL_ID = "Qwen/Qwen3-4B"
_GENERATION_PROMPT = """Passage:
{chunk}

Write the single best quiz question about this passage. If no meaningful question can be formed, reply with INVALID."""
DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_QUERY = "What are the main contributions of the paper?"
DEFAULT_TOP_K = 5


@contextmanager
def log_timer(label: str) -> None:
    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        log(f"{label} completed in {elapsed:.2f}s")


@dataclass(frozen=True)
class ChunkRecord:
    idx: int
    text: str
    token_start: int
    token_end: int
    question: str


def _load_tokenizer(model_id: str = GENERATION_MODEL_ID) -> PreTrainedTokenizerBase:
    log(f"Loading tokenizer for {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_generation_components(
    model_id: str = GENERATION_MODEL_ID,
) -> Tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    tokenizer = _load_tokenizer(model_id)
    log(f"Loading generation model {model_id}")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
        )
        log("Loaded generation model with device_map='auto'")
        return tokenizer, model
    except Exception as exc:
        log(f"device_map='auto' unavailable: {exc}. Falling back to manual placement.")

    device = torch.device("cpu")
    model_kwargs: Dict[str, Any] = {}
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log("Using CUDA for generation model")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
        model_kwargs["torch_dtype"] = torch.float16
        log("Using Apple MPS for generation model")
    else:
        log("Falling back to CPU for generation model")

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.to(device)
    return tokenizer, model


@lru_cache(maxsize=1)
def _get_generation_components(
    model_id: str = GENERATION_MODEL_ID,
) -> Tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    return _load_generation_components(model_id)


def _chunk_sentences(
    text: str,
    sentences_per_chunk: int = 4,
) -> List[Tuple[int, int, str]]:
    if sentences_per_chunk <= 0:
        raise ValueError("sentences_per_chunk must be positive.")

    import re

    sentence_end_re = re.compile(r"(?<=[.!?])\s+")
    sentences = [s.strip() for s in sentence_end_re.split(text) if s.strip()]

    chunks: List[Tuple[int, int, str]] = []
    start_idx = 0
    while start_idx < len(sentences):
        end_idx = min(start_idx + sentences_per_chunk, len(sentences))
        chunk_text = " ".join(sentences[start_idx:end_idx]).strip()
        if chunk_text:
            chunks.append((start_idx, end_idx, chunk_text))
        start_idx = end_idx
    return chunks


def _generate_question(
    chunk_text: str,
    *,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    max_new_tokens: int = 256,
) -> str:
    if not chunk_text.strip():
        return "INVALID"

    system_prompt = (
        "You are a study guide author. Generate a single question about the passage. "
        "The question must be answerable using only the passage, focus on a key fact, "
        "avoid yes/no wording, and must not repeat sentences verbatim. If the passage "
        "is unintelligible or lacks factual content, respond with the single word INVALID."
    )
    user_prompt = _GENERATION_PROMPT.format(chunk=chunk_text.strip())
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking = False
    )

    model_inputs = tokenizer([chat_text], return_tensors="pt")
    target_device = getattr(model, "device", None)
    if target_device is not None:
        try:
            model_inputs = model_inputs.to(target_device)
        except Exception:
            pass

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    output_ids = generated_ids[0][model_inputs.input_ids.shape[-1]:].tolist()

    question_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    if not question_text:
        return "INVALID"

    if "Question:" in question_text:
        question_text = question_text.split("Question:", 1)[-1].strip()

    question_line = question_text.splitlines()[0].strip()
    if not question_line:
        return "INVALID"

    if question_line.upper().startswith("INVALID"):
        return "INVALID"

    if not question_line.endswith("?"):
        question_line = question_line.rstrip(". ")
        if not question_line.endswith("?"):
            question_line = f"{question_line}?"
    print(question_line)

    return question_line.strip()


def _prepare_chunk_records(
    document_text: str,
    *,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    chunk_size: int = 400,
    overlap: int = 50,
) -> List[ChunkRecord]:
    token_chunks = _chunk_sentences(document_text, sentences_per_chunk=4)
    records: List[ChunkRecord] = []
    total = len(token_chunks)
    log(f"Generating questions for {total} chunks")
    if total == 0:
        return records

    with log_timer("Question generation run"):
        for idx, (start, end, chunk_text) in enumerate(token_chunks):
            chunk_start = time.perf_counter()
            question = _generate_question(chunk_text, tokenizer=tokenizer, model=model)
            records.append(
                ChunkRecord(
                    idx=idx,
                    text=chunk_text,
                    token_start=start,
                    token_end=end,
                    question=question,
                )
            )
            chunk_elapsed = time.perf_counter() - chunk_start
            remaining = total - idx - 1
            log(
                f"Generated question {idx + 1}/{total} in {chunk_elapsed:.2f}s; "
                f"{remaining} chunks remaining"
            )
    return records


def _load_default_document_text() -> Tuple[str, Path]:
    json_path = current_path.parent / "test_paper.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Default test document not found: {json_path}")
    log(f"Reading document from {json_path}")
    with log_timer(f"Reading JSON file {json_path.name}"):
        raw = json_path.read_text(encoding="utf-8")
    with log_timer("Parsing JSON payload"):
        data = json.loads(raw)
    log(f"Loaded JSON ({len(raw)} bytes)")
    text = data.get("full_text") or data.get("summary")
    if not text or not isinstance(text, str):
        raise ValueError("The test document does not contain 'full_text' or 'summary'.")
    return text, json_path


class FullDocumentQAIndex:
    def __init__(
        self,
        records: Sequence[ChunkRecord],
        vector_store: FAISS,
        embeddings: Specter2Embeddings,
    ) -> None:
        self._records = list(records)
        self._vector_store = vector_store
        self._embeddings = embeddings

    @classmethod
    def from_document(
        cls,
        document_text: str,
        *,
        embeddings: Optional[Specter2Embeddings] = None,
        chunk_size: int = 400,
        overlap: int = 50,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        generator: Optional[Any] = None,
        generation_model: Optional[PreTrainedModel] = None,
    ) -> "FullDocumentQAIndex":
        if not document_text.strip():
            raise ValueError("document_text must be a non-empty string.")

        embeddings = embeddings or Specter2Embeddings(EMBED_MODEL_NAME)
        base_tokenizer, base_model = _get_generation_components()
        if generator is not None:
            log("Received deprecated 'generator' argument; ignoring in favour of built-in generation model.")

        active_tokenizer = tokenizer or base_tokenizer
        active_model = generation_model or base_model

        records = _prepare_chunk_records(
            document_text,
            tokenizer=active_tokenizer,
            model=active_model,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        if not records:
            raise ValueError("The document did not produce any non-empty chunks.")

        log("Embedding synthetic questions")
        questions = [record.question for record in records]
        metadatas = [
            {
                "chunk_idx": record.idx,
                "token_start": record.token_start,
                "token_end": record.token_end,
                "chunk_text": record.text,
            }
            for record in records
        ]
        with log_timer("Embedding questions into FAISS"):
            vector_store = FAISS.from_texts(questions, embeddings, metadatas=metadatas)
        log(f"Vector store created with {len(records)} entries")

        return cls(records=records, vector_store=vector_store, embeddings=embeddings)

    def query(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> List[dict]:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if not query.strip():
            return []

        results = self._vector_store.similarity_search_with_score(query, k=top_k)
        output: List[dict] = []
        for doc, score in results:
            metadata = doc.metadata or {}
            chunk_idx = metadata.get("chunk_idx")
            chunk_text = metadata.get("chunk_text")
            output.append(
                {
                    "chunk_idx": chunk_idx,
                    "chunk_text": chunk_text,
                    "question": doc.page_content,
                    "score": float(score),
                }
            )
        return output


__all__ = [
    "ChunkRecord",
    "FullDocumentQAIndex",
    "_prepare_chunk_records",
]  # Functions left public for testing.


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = list(argv if argv is not None else sys.argv[1:])
    top_k = DEFAULT_TOP_K
    if args:
        try:
            top_k = max(1, int(args[0]))
        except ValueError:
            log(f"Ignoring invalid top_k '{args[0]}'; using default {DEFAULT_TOP_K}.")

    chunk_size = DEFAULT_CHUNK_SIZE
    overlap = DEFAULT_CHUNK_OVERLAP

    document_text, json_path = _load_default_document_text()
    log(f"Using default chunk size {chunk_size} with {overlap} token overlap")

    log("Creating FullDocumentQAIndex")
    with log_timer("Building FullDocumentQAIndex"):
        index = FullDocumentQAIndex.from_document(
            document_text,
            chunk_size=chunk_size,
            overlap=overlap,
        )
    log(f"Index created from {json_path.name}")

    output_path = current_path.parent / "test_paper_questions.json"
    output_payload = [
        {
            "chunk_idx": record.idx,
            "chunk_text": record.text,
            "question": record.question,
        }
        for record in index._records
    ]
    with log_timer(f"Writing generated questions to {output_path.name}"):
        output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    log(f"Saved generated questions to {output_path}")

    log("Entering interactive query loop. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            user_query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            log("Exiting interactive loop.")
            break

        if not user_query:
            continue
        if user_query.lower() in {"exit", "quit"}:
            log("Received exit command.")
            break

        query_start = time.perf_counter()
        results = index.query(user_query, top_k=top_k)
        elapsed = time.perf_counter() - query_start
        log(f"Query completed in {elapsed:.2f}s with {len(results)} results")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
