"""
Logits-guided chunker (LG) for the LGMGC pipeline.

This module uses a small language model's understanding of when text "naturally
ends" to define parent chunks. It queries a Hugging Face causal LM (default:
Qwen3 0.6B Base) and scores candidate split points by the model's
End-of-Sequence (EOS) probability.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch

current_path = Path(__file__).resolve()
if __package__:
    from .._log import log  # type: ignore
else:
    parent_dir = current_path.parent
    if str(parent_dir) not in sys.path:
        sys.path.append(str(parent_dir))
    if str(parent_dir.parent) not in sys.path:
        sys.path.append(str(parent_dir.parent))
    from _log import log  # type: ignore

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
except ImportError as exc:  # pragma: no cover - deferred failure until runtime use
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    err = ImportError(
        "transformers is required for the logits-guided chunker. Install it with "
        "'pip install transformers torch'."
    )
    err.__cause__ = exc
    _IMPORT_ERROR: Optional[ImportError] = err
else:
    _IMPORT_ERROR = None


DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Base"
_SENTENCE_RE = re.compile(r"([^.!?]+[.!?]+)(?=\s|$)", flags=re.MULTILINE)


@dataclass(frozen=True)
class LGChunkerConfig:
    model_id: str = DEFAULT_MODEL_ID
    window_size_words: int = 300
    min_sentences: int = 1
    max_window_sentences: int = 30


@dataclass(frozen=True)
class LGChunk:
    text: str
    start_sentence: int
    end_sentence: int
    eos_probability: float


@dataclass(frozen=True)
class _ModelBundle:
    model: torch.nn.Module
    tokenizer: any
    eos_token_id: int
    device: torch.device


def _raise_if_unavailable() -> None:
    if _IMPORT_ERROR is not None:
        raise _IMPORT_ERROR


@lru_cache(maxsize=2)
def _load_model_bundle(model_id: str) -> _ModelBundle:
    _raise_if_unavailable()
    if AutoTokenizer is None or AutoModelForCausalLM is None:  # pragma: no cover - safety
        raise RuntimeError("transformers is unavailable.")

    log(f"Loading tokenizer for {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    log(f"Loading model {model_id} with dtype {torch_dtype}")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        log("Using CUDA execution for LG chunker")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
        log("Using Apple MPS execution for LG chunker")
    else:
        device = torch.device("cpu")
        log("Using CPU execution for LG chunker")

    log("Model loaded; moving to device")
    model.to(device)
    model.eval()
    log("Model initialised and ready")

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is None:
            raise ValueError(f"Tokenizer for '{model_id}' does not expose an EOS token.")
        eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
        
    if eos_token_id is None:
        raise ValueError(f"Could not resolve EOS token id for '{model_id}'.")
    return _ModelBundle(model=model, tokenizer=tokenizer, eos_token_id=int(eos_token_id), device=device)


def _sentence_tokenize(text: str) -> List[str]:
    stripped = (text or "").strip()
    if not stripped:
        return []

    sentences: List[str] = []
    end = 0
    for match in _SENTENCE_RE.finditer(stripped):
        sentence = match.group(1).strip()
        end = match.end()
        if sentence:
            sentences.append(sentence)
    tail = stripped[end:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def _word_count(sentence: str) -> int:
    return max(len(sentence.split()), 1)


def _collect_window(
    sentences: Sequence[str],
    start: int,
    *,
    window_words: int,
    min_sentences: int,
    max_sentences: int,
) -> Sequence[str]:
    count = 0
    collected: List[str] = []
    idx = start
    while idx < len(sentences) and len(collected) < max_sentences:
        sentence = sentences[idx]
        collected.append(sentence)
        count += _word_count(sentence)
        idx += 1
        if count >= window_words and len(collected) >= min_sentences:
            break
    return collected


def _eos_probability(bundle: _ModelBundle, text: str) -> float:
    if not text.strip():
        return 0.0
    encoded = bundle.tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    encoded = {key: value.to(bundle.device) for key, value in encoded.items()}

    generation = bundle.model.generate(
        **encoded,
        max_new_tokens=1,
        return_dict_in_generate=True,
        output_scores=True,
        eos_token_id=bundle.eos_token_id,
        pad_token_id=bundle.eos_token_id,
        output_hidden_states=False,
    )

    logits = generation.scores[0][0]
    probs = torch.softmax(logits, dim=-1)
    return float(probs[bundle.eos_token_id].item())


def _select_split(
    bundle: _ModelBundle,
    sentences: Sequence[str],
) -> Tuple[int, float]:
    best_idx = len(sentences) - 1
    best_score = -math.inf
    prefix_parts: List[str] = []

    for idx, sentence in enumerate(sentences):
        prefix_parts.append(sentence)
        candidate = " ".join(prefix_parts).strip()
        try:
            score = _eos_probability(bundle, candidate)
        except Exception as exc:  # pragma: no cover - runtime logging
            log(f"LG chunker fell back on heuristic split due to: {exc}")
            return best_idx, float("nan")
        if score >= best_score:
            best_score = score
            best_idx = idx
    return best_idx, best_score


def logits_guided_parent_chunks(
    text: str,
    config: Optional[LGChunkerConfig] = None,
) -> List[LGChunk]:
    config = config or LGChunkerConfig()
    log("Tokenizing document into sentences")
    sentences = _sentence_tokenize(text)
    if not sentences:
        return []

    try:
        bundle = _load_model_bundle(config.model_id)
    except ImportError as exc:
        log(f"Unable to load LG model: {exc}")
        raise

    chunks: List[LGChunk] = []
    cursor = 0
    total_sentences = len(sentences)

    while cursor < len(sentences):
        log(f"Collecting window starting at sentence {cursor}")
        window = _collect_window(
            sentences,
            cursor,
            window_words=config.window_size_words,
            min_sentences=config.min_sentences,
            max_sentences=config.max_window_sentences,
        )
        if not window:
            break

        log(f"Window size {len(window)} sentences; scoring EOS splits")
        split_idx, score = _select_split(bundle, window)
        chunk_sentences = window[: split_idx + 1]
        if not chunk_sentences:
            chunk_sentences = [window[0]]
            split_idx = 0
            log("Fall back to chunk of length 1")

        chunk_text = " ".join(chunk_sentences).strip()
        start_sentence = cursor
        end_sentence = cursor + split_idx
        remaining = total_sentences - (end_sentence + 1)

        chunks.append(
            LGChunk(
                text=chunk_text,
                start_sentence=start_sentence,
                end_sentence=end_sentence,
                eos_probability=score,
            )
        )
        log(
            f"Created chunk covering sentences {start_sentence}-{end_sentence} "
            f"with eos probability {score}"
        )
        log(f"Approximately {remaining} sentences remain to process")
        cursor = end_sentence + 1

    return chunks


def logits_guided_parent_chunks_v2(
    text: str,
    config: Optional[LGChunkerConfig] = None,
) -> List[LGChunk]:
    config = config or LGChunkerConfig()
    log("Tokenizing document into sentences")
    sentences = _sentence_tokenize(text)
    if not sentences:
        return []

    try:
        bundle = _load_model_bundle(config.model_id)
    except ImportError as exc:
        log(f"Unable to load LG model: {exc}")
        raise

    chunks: List[LGChunk] = []
    cursor = 0
    total_sentences = len(sentences)

    cached_sentences: List[str] = []
    cached_scores: List[float] = []

    while cursor < total_sentences:
        log(f"[v2] Collecting window starting at sentence {cursor}")
        window = _collect_window(
            sentences,
            cursor,
            window_words=config.window_size_words,
            min_sentences=config.min_sentences,
            max_sentences=config.max_window_sentences,
        )
        if not window:
            break

        prefix_len = 0
        if cached_sentences:
            if window[: len(cached_sentences)] == cached_sentences:
                prefix_len = min(len(cached_sentences), len(cached_scores))
            else:
                log("[v2] Cached sentences no longer align with window; discarding cache")
                cached_sentences = []
                cached_scores = []

        scores: List[float] = []
        prefix_parts: List[str] = []
        scoring_failed = False

        for idx, sentence in enumerate(window):
            prefix_parts.append(sentence)
            if idx < prefix_len:
                scores.append(cached_scores[idx])
                continue

            candidate = " ".join(prefix_parts).strip()
            try:
                score = _eos_probability(bundle, candidate)
            except Exception as exc:  # pragma: no cover - runtime logging
                log(f"[v2] LG chunker fell back on heuristic split due to: {exc}")
                scoring_failed = True
                break
            scores.append(score)

        if not scores:
            split_idx = 0
            best_score = float("nan")
            chunk_sentences = [window[0]]
        else:
            best_idx = len(scores) - 1
            best_score = -math.inf
            for idx, score in enumerate(scores):
                if score >= best_score:
                    best_score = score
                    best_idx = idx
            split_idx = best_idx
            chunk_sentences = window[: split_idx + 1]
            if not chunk_sentences:
                chunk_sentences = [window[0]]
                split_idx = 0

        chunk_text = " ".join(chunk_sentences).strip()
        start_sentence = cursor
        end_sentence = cursor + split_idx
        remaining = total_sentences - (end_sentence + 1)

        chunks.append(
            LGChunk(
                text=chunk_text,
                start_sentence=start_sentence,
                end_sentence=end_sentence,
                eos_probability=best_score,
            )
        )
        log(
            f"[v2] Created chunk covering sentences {start_sentence}-{end_sentence} "
            f"with eos probability {best_score}"
        )
        log(f"[v2] Approximately {remaining} sentences remain to process")

        if scoring_failed:
            cached_sentences = []
            cached_scores = []
        else:
            cached_sentences = window[split_idx + 1 :]
            cached_scores = scores[split_idx + 1 :]

        cursor = end_sentence + 1

    return chunks


__all__ = [
    "LGChunk",
    "LGChunkerConfig",
    "logits_guided_parent_chunks",
    "logits_guided_parent_chunks_v2",
]


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    input_path = base_dir / "test_paper.json"
    output_path = base_dir / "results_v2.txt"

    if not input_path.exists():
        raise FileNotFoundError(f"Expected test input at {input_path}")

    log(f"Loading test document from {input_path}")
    import json

    data = input_path.read_text(encoding="utf-8")
    payload = json.loads(data)
    full_text = payload.get("full_text", "")
    if not full_text:
        raise ValueError("No 'full_text' field available in test payload.")

    log("Generating logits-guided parent chunks")
    start_time = time.perf_counter()
    chunks = logits_guided_parent_chunks_v2(full_text)
    elapsed = time.perf_counter() - start_time
    log(f"Chunking completed in {elapsed:.2f} seconds")

    if not chunks:
        log("No chunks were produced.")

    lines: List[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        header = f"Chunk {idx} | sentences {chunk.start_sentence}-{chunk.end_sentence} | eos={chunk.eos_probability:.4f}"
        lines.append(header)
        lines.append(chunk.text)
        lines.append("")

    log(f"Writing {len(chunks)} chunks to {output_path}")
    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    log("Done. Inspect results.txt for chunk boundaries.")
