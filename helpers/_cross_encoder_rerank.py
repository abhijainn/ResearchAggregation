from functools import lru_cache
from typing import List, Dict, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
 
from ._log import log

try:
    from app.config import CONFIG  # type: ignore
except ImportError:
    from config import CONFIG  # type: ignore


DEFAULT_MODEL = CONFIG.rerank.cross_encoder_model


@lru_cache(maxsize=2)
def _load_model_and_tokenizer(model_name: str = DEFAULT_MODEL, device: Optional[str] = None):
    if device is None:
        device = CONFIG.runtime.device
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def _score_batch(model, tokenizer, device, pairs: List[tuple], batch_size: int = 16) -> List[float]:
    scores: List[float] = []
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start : start + batch_size]
            # tokenizer can take pairs as (text1, text2)
            texts1 = [p[0] for p in batch_pairs]
            texts2 = [p[1] for p in batch_pairs]
            inputs = tokenizer(texts1, texts2, padding=True, truncation=True, return_tensors='pt', max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            logits = outputs.logits
            # If binary classification with single logit, use that; else compute score for positive class
            if logits.shape[-1] == 1:
                batch_scores = logits.view(-1).cpu().tolist()
            else:
                probs = torch.softmax(logits, dim=1)
                # assume positive class is index 1
                batch_scores = probs[:, 1].cpu().tolist()
            scores.extend(batch_scores)
    return scores


def rerank_with_cross_encoder(
    query: str,
    candidates: List[Dict],
    model_name: str = DEFAULT_MODEL,
    device: Optional[str] = None,
    batch_size: int = 16,
    top_k: Optional[int] = None,
) -> List[Dict]:
    """
    Rerank candidate documents using a cross-encoder model.

    - query: user query string
    - candidates: list of dicts that should have 'title' and/or 'abstract' (or 'content')
    - returns: candidates augmented with 'cross_score' and sorted descending by cross_score
    """
    if not candidates:
        return []

    model, tokenizer, device = _load_model_and_tokenizer(model_name, device)

    logmsg = f"Loaded model and tokenizer to {device}"
    log(logmsg)

    pairs = []
    for cand in candidates:
        title = (cand.get('title') or '').strip()
        abstract = (cand.get('abstract') or cand.get('summary') or cand.get('content') or '').strip()
        text = title
        if title and abstract:
            text = f"{title} [SEP] {abstract}"
        elif abstract:
            text = abstract
        pairs.append((query, text))

    scores = _score_batch(model, tokenizer, device, pairs, batch_size=batch_size)

    for cand, score in zip(candidates, scores):
        try:
            cand['cross_score'] = float(score)
        except Exception:
            cand['cross_score'] = None

    candidates_sorted = sorted(candidates, key=lambda d: (d.get('cross_score') is not None, d.get('cross_score')), reverse=True)

    if top_k is not None:
        return candidates_sorted[:top_k]
    return candidates_sorted
