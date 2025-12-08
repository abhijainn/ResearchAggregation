"""
Deterministic keyword-query generator used by the hybrid quote finder stack.

Responsibilities:
    * normalize a natural-language question
    * surface high-signal noun phrases + intent cues
    * blend in optional document anchors (section headers / model names)
    * emit 3-5 Elasticsearch-friendly query strings with +must and quotes
"""

from __future__ import annotations

import re
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import wraps
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None  # type: ignore[assignment]

try:
    from nltk.corpus import wordnet as wn
except ImportError:  # pragma: no cover - optional dependency
    wn = None

try:
    import spacy
except ImportError:  # pragma: no cover - optional dependency
    spacy = None  # type: ignore[assignment]

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore[assignment]

_SENTENCE_MODEL_NAME = "all-MiniLM-L6-v2"
_sentence_model: Optional["SentenceTransformer"] = None  # type: ignore[name-defined]
_sentence_model_failed = False
_NER_MODEL_NAME = "en_core_web_sm"
_ner_model = None
_ner_model_failed = False
logger = logging.getLogger(__name__)

__all__ = [
    "KeywordConfig",
    "Query",
    "load_default_config",
    "extract_queries",
    "mine_doc_anchors_from_text",
    "normalize_keyword_terms",
]

DEFAULT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "tell",
    "me",
    "paper",
    "papers",
    "use",
    "uses",
    "used",
    "using",
    "that",
    "the",
    "their",
    "these",
    "this",
    "those",
    "to",
    "what",
    "which",
    "who",
    "why",
    "with",
}

INTENT_CUES: Mapping[str, Sequence[str]] = {
    "architecture": ("architecture", "framework"),
    "framework": ("framework", "design"),
    "pipeline": ("pipeline", "workflow"),
    "method": ("method", "approach"),
    "approach": ("approach", "strategy"),
    "compare": ("comparison", "baseline"),
    "evaluation": ("evaluation", "metrics"),
    "ablation": ("ablation", "analysis"),
    "result": ("result", "finding"),
}

ANCHOR_HEADING_RE = re.compile(r"^(?:\d+(\.\d+)*)?\s*[A-Z][A-Za-z0-9 ,:/\-]{2,}$")
MODEL_NAME_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9]*(?:Net|Former|BERT|GPT|GAN|LM|Transformer|Model|Network|System))\b"
)
WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]+")
MAX_RAKE_PHRASE_TOKENS = 5
MAX_ENTITY_SPAN_TOKENS = 8
MAX_RAW_TERM_CHARS = 64
MAX_PHRASE_CHAR_LEN = 96
MIN_ALPHA_RATIO = 0.35
PRIORITY_SECTION_TERMS = {
    "result",
    "results",
    "analysis",
    "architecture",
    "model",
    "method",
    "evaluation",
    "experiment",
}


@dataclass
class KeywordConfig:
    min_queries: int = 3
    max_queries: int = 5
    max_terms: int = 10
    max_anchor_queries: int = 2
    min_term_length: int = 3
    must_term_threshold: float = 1.6
    recall_term_threshold: float = 1.1
    stopwords: Set[str] = field(default_factory=lambda: set(DEFAULT_STOPWORDS))
    synonyms: Mapping[str, Sequence[str]] = field(default_factory=dict)
    intent_cues: Mapping[str, Sequence[str]] = field(default_factory=lambda: INTENT_CUES)


@dataclass
class TokenInfo:
    surface: str
    lemma: str
    start: int
    is_stop: bool
    is_capitalized: bool
    has_digit: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "surface": self.surface,
            "lemma": self.lemma,
            "start": self.start,
            "is_stop": self.is_stop,
            "is_capitalized": self.is_capitalized,
            "has_digit": self.has_digit,
        }


@dataclass
class CandidateTerm:
    text: str
    score: float
    sources: List[str] = field(default_factory=list)

    def bump(self, score: float, source: str) -> None:
        if score > self.score:
            self.score = score
        if source not in self.sources:
            self.sources.append(source)


@dataclass
class Query:
    boosted_terms: List[str]
    normal_terms: List[str]
    section_filter: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "boosted_terms": list(self.boosted_terms),
            "normal_terms": list(self.normal_terms),
            "section_filter": self.section_filter,
        }


@dataclass
class EntitySpan:
    text: str
    start: int
    end: int
    label: str
    score: float


def _timecall(label: Optional[str] = None):
    def decorator(func):
        func_label = label or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.debug("timing:%s %.2f ms", func_label, elapsed_ms)

        return wrapper

    return decorator


def load_default_config() -> KeywordConfig:
    return KeywordConfig()


@_timecall("extract_queries")
def extract_queries(
    question: str,
    doc_anchors: Optional[Mapping[str, Mapping[str, Sequence[str]] | Sequence[str]]] = None,
    config: Optional[KeywordConfig] = None,
) -> Tuple[List[Query], Dict[str, object]]:
    """
    Build Query objects capturing boosted/normal terms and section filters.
    """

    config = config or load_default_config()
    normalized_question = question.strip()
    entity_spans = _extract_entity_spans(normalized_question)
    tokens = _tokenize(normalized_question, config.stopwords)
    candidates = _gather_candidates(tokens, config)
    anchors = _normalize_anchors(doc_anchors)
    original_sections = list(anchors.get("section_titles", []))
    best_section, section_scores, section_rankings = _rerank_sections(
        normalized_question,
        anchors.get("section_titles", []),
        tokens,
    )
    if not best_section and original_sections:
        best_section = original_sections[0].strip() or original_sections[0]
    anchors["section_titles"] = [best_section] if best_section else []
    _boost_anchor_terms(candidates, anchors, source="anchor")
    intent_terms = _boost_intent_terms(candidates, normalized_question, config.intent_cues)

    ranked_terms = sorted(candidates.values(), key=lambda cand: cand.score, reverse=True)
    ranked_terms = ranked_terms[: config.max_terms]
    section_term_blocklist = {
        title.strip().lower() for title in section_rankings if title and title.strip()
    }
    if best_section:
        section_term_blocklist.add(best_section.strip().lower())
    if section_term_blocklist:
        ranked_terms = [term for term in ranked_terms if term.text.lower() not in section_term_blocklist]

    ranked_term_texts = [term.text for term in ranked_terms]
    synonyms_applied = _gather_synonyms(ranked_term_texts, config.synonyms)

    raw_terms = _build_raw_terms(normalized_question, entity_spans, config.stopwords)
    model_term = _select_model_name(normalized_question, anchors.get("model_names", []))
    boost_pool = _prepare_boost_pool(
        entity_spans,
        ranked_terms,
        model_term=model_term,
        intent_terms=intent_terms,
        disallowed_terms=section_term_blocklist,
    )

    queries: List[Query] = []
    queries.append(Query(boosted_terms=[], normal_terms=raw_terms, section_filter=None))
    queries.extend(_build_strong_queries(boost_pool, ranked_term_texts, raw_terms))
    seen_sections: Set[str] = set()
    section_sequence: List[Optional[str]] = []
    for title in section_rankings:
        normalized = title.strip() if title else None
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen_sections:
            continue
        seen_sections.add(key)
        section_sequence.append(normalized)
        if len(section_sequence) >= 2:
            break
    if not section_sequence and best_section:
        section_sequence = [best_section]
    queries.extend(
        _build_weak_queries(
            boost_pool,
            ranked_term_texts,
            raw_terms,
            synonyms_applied,
            section_sequence,
        )
    )

    debug_info = {
        "question": normalized_question,
        "tokens": [token.to_dict() for token in tokens],
        "ranked_terms": [{"text": term.text, "score": term.score, "sources": term.sources} for term in ranked_terms],
        "anchors_used": anchors,
        "best_section": best_section,
        "section_scores": section_scores,
        "section_rankings": section_rankings,
        "synonyms_applied": synonyms_applied,
        "entity_spans": [span.__dict__ for span in entity_spans],
        "queries": [query.to_dict() for query in queries],
    }

    return queries, debug_info


def mine_doc_anchors_from_text(
    doc_text: str,
    *,
    max_headings: int = 10,
    max_models: int = 10,
) -> Dict[str, List[str]]:
    """Lightweight heading/model mining fallback."""
    headings: List[str] = []
    for line in doc_text.splitlines():
        candidate = line.strip()
        if ANCHOR_HEADING_RE.match(candidate):
            headings.append(candidate.rstrip(":"))
    heading_list = list(dict.fromkeys(headings))[:max_headings]
    model_list = list(dict.fromkeys(MODEL_NAME_RE.findall(doc_text)))[:max_models]
    return {
        "section_titles": heading_list,
        "model_names": model_list,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _tokenize(question: str, stopwords: Set[str]) -> List[TokenInfo]:
    tokens: List[TokenInfo] = []
    for match in WORD_RE.finditer(question):
        surface = match.group(0)
        lemma = _lemmatize(surface)
        tokens.append(
            TokenInfo(
                surface=surface,
                lemma=lemma,
                start=match.start(),
                is_stop=lemma in stopwords,
                is_capitalized=surface[0].isupper(),
                has_digit=any(ch.isdigit() for ch in surface),
            )
        )
    return tokens


def _lemmatize(token: str) -> str:
    lowered = token.lower()
    if len(lowered) <= 3:
        return lowered
    if lowered.endswith("ies") and len(lowered) > 4:
        return lowered[:-3] + "y"
    if lowered.endswith("ses") and len(lowered) > 4:
        return lowered[:-2]
    if lowered.endswith("ing") and len(lowered) > 5:
        stem = lowered[:-3]
        if stem.endswith(stem[-1]):
            stem = stem[:-1]
        return stem
    if lowered.endswith("ed") and len(lowered) > 4:
        stem = lowered[:-2]
        if stem.endswith(stem[-1]):
            stem = stem[:-1]
        return stem
    if lowered.endswith("ly") and len(lowered) > 4:
        return lowered[:-2]
    if lowered.endswith("er") and len(lowered) > 4:
        return lowered[:-2]
    if lowered.endswith("est") and len(lowered) > 5:
        return lowered[:-3]
    if lowered.endswith("s") and len(lowered) > 3 and not lowered.endswith("ss"):
        return lowered[:-1]
    return lowered


def normalize_keyword_terms(text: str) -> List[str]:
    """Return the lemmatized tokens for ``text`` using the query rules."""
    if not text:
        return []
    return [_lemmatize(match.group(0)) for match in WORD_RE.finditer(text)]


@_timecall("extract_entity_spans")
def _extract_entity_spans(question: str) -> List[EntitySpan]:
    spans = _extract_entity_spans_with_spacy(question)
    if spans:
        return spans
    return _extract_capitalized_spans(question)


@_timecall("spacy_ner")
def _extract_entity_spans_with_spacy(question: str) -> List[EntitySpan]:
    model = _get_ner_model()
    if model is None:
        return []
    try:
        doc = model(question)
    except Exception:
        return []
    spans: List[EntitySpan] = []
    weights = {
        "PERSON": 1.2,
        "ORG": 1.0,
        "WORK_OF_ART": 0.8,
        "PRODUCT": 0.8,
        "GPE": 0.7,
        "EVENT": 0.6,
    }
    for ent in doc.ents:
        text = ent.text.strip()
        if not text:
            continue
        label = ent.label_
        token_count = max(1, len(text.split()))
        score = token_count + weights.get(label, 0.4)
        spans.append(EntitySpan(text=text, start=ent.start_char, end=ent.end_char, label=label, score=score))
    return spans


def _extract_capitalized_spans(question: str) -> List[EntitySpan]:
    spans: List[EntitySpan] = []
    for match in re.finditer(r"\b([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)+)\b", question):
        text = match.group(0).strip()
        if not text:
            continue
        score = len(text.split()) + 0.5
        spans.append(EntitySpan(text=text, start=match.start(), end=match.end(), label="HEURISTIC", score=score))
    return spans


@_timecall("build_raw_terms")
def _build_raw_terms(question: str, entity_spans: Sequence[EntitySpan], stopwords: Set[str]) -> List[str]:
    if not question:
        return []
    spans = sorted((span for span in entity_spans if span.text), key=lambda span: span.start)
    result: List[str] = []
    cursor = 0
    for span in spans:
        if span.start < cursor:
            continue
        if cursor < span.start:
            result.extend(_filter_tokens(_split_text_segment(question[cursor:span.start]), stopwords))
        normalized_span = _normalize_span_text(span.text)
        if normalized_span:
            result.append(normalized_span)
        cursor = span.end
    if cursor < len(question):
        result.extend(_filter_tokens(_split_text_segment(question[cursor:]), stopwords))
    cleaned = [token for token in result if token]
    fallback = _filter_tokens(_split_text_segment(question), stopwords)
    return cleaned or fallback


def _split_text_segment(text: str) -> List[str]:
    return WORD_RE.findall(text)


def _filter_tokens(tokens: Sequence[str], stopwords: Set[str]) -> List[str]:
    cleaned: List[str] = []
    for token in tokens:
        if not token:
            continue
        normalized = _clean_single_token(token)
        if not normalized:
            continue
        if normalized.lower() in stopwords:
            continue
        cleaned.append(normalized)
    return cleaned


def _clean_single_token(token: str) -> Optional[str]:
    stripped = token.strip().strip("\"'.,:;()[]{}")
    if not stripped:
        return None
    if len(stripped) > MAX_RAW_TERM_CHARS:
        return None
    if _alpha_ratio(stripped) < MIN_ALPHA_RATIO and not any(ch.isdigit() for ch in stripped):
        return None
    return stripped


def _normalize_span_text(text: str) -> Optional[str]:
    collapsed = " ".join((text or "").split())
    if not collapsed:
        return None
    tokens = collapsed.split()
    if len(tokens) > MAX_ENTITY_SPAN_TOKENS:
        tokens = tokens[:MAX_ENTITY_SPAN_TOKENS]
    collapsed = " ".join(tokens)
    if len(collapsed) > MAX_RAW_TERM_CHARS:
        trimmed = collapsed[:MAX_RAW_TERM_CHARS]
        last_space = trimmed.rfind(" ")
        collapsed = trimmed[:last_space] if last_space > 0 else trimmed
        collapsed = collapsed.strip()
    if not collapsed:
        return None
    if _alpha_ratio(collapsed) < MIN_ALPHA_RATIO:
        return None
    return collapsed


def _alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    alpha = sum(1 for ch in text if ch.isalpha())
    return alpha / max(1, len(text))


def _extract_candidate_phrases(tokens: Sequence[TokenInfo], stopwords: Set[str]) -> List[List[TokenInfo]]:
    phrases: List[List[TokenInfo]] = []
    current: List[TokenInfo] = []
    for token in tokens:
        if token.lemma in stopwords:
            if current:
                phrases.append(current)
                current = []
            continue
        current.append(token)
    if current:
        phrases.append(current)
    return phrases


def _compute_rake_word_scores(phrases: Sequence[Sequence[TokenInfo]]) -> Dict[str, float]:
    word_freq: Dict[str, int] = defaultdict(int)
    word_degree: Dict[str, int] = defaultdict(int)
    for phrase in phrases:
        lemmas = [token.lemma for token in phrase if token.lemma]
        if not lemmas:
            continue
        degree = len(lemmas) - 1
        for lemma in lemmas:
            word_freq[lemma] += 1
            word_degree[lemma] += degree
    word_scores: Dict[str, float] = {}
    for lemma, freq in word_freq.items():
        word_scores[lemma] = (word_degree[lemma] + freq) / max(freq, 1)
    return word_scores


def _phrase_text(tokens: Sequence[TokenInfo]) -> str:
    return " ".join(token.surface for token in tokens)


def _is_relevant_phrase(tokens: Sequence[TokenInfo], min_term_length: int) -> bool:
    if not tokens:
        return False
    if len(tokens) > MAX_RAKE_PHRASE_TOKENS:
        return False
    surface = _phrase_text(tokens)
    if len(surface) > MAX_PHRASE_CHAR_LEN:
        return False
    if _alpha_ratio(surface) < MIN_ALPHA_RATIO:
        return False
    has_long_token = any(len(token.lemma) >= min_term_length for token in tokens)
    all_caps_span = len(tokens) > 1 and all(token.surface[0].isupper() for token in tokens)
    acronym = any(token.surface.isupper() and len(token.surface) >= 2 for token in tokens)
    has_digit = any(token.has_digit for token in tokens)
    multiword_valid = len(tokens) == 1 or all_caps_span or has_digit
    return multiword_valid and (has_long_token or all_caps_span or acronym or has_digit)


def _adjust_phrase_score(base_score: float, tokens: Sequence[TokenInfo]) -> float:
    score = base_score
    if len(tokens) > 1:
        score += 0.3
    if all(token.surface[0].isupper() for token in tokens):
        score += 0.4
    if any(token.has_digit for token in tokens):
        score += 0.2
    earliest = min((token.start for token in tokens), default=0)
    score += max(0.0, 0.4 - earliest * 0.01)
    return score


@_timecall("gather_candidates")
def _gather_candidates(tokens: List[TokenInfo], config: KeywordConfig) -> Dict[str, CandidateTerm]:
    candidates: Dict[str, CandidateTerm] = {}
    phrases = _extract_candidate_phrases(tokens, config.stopwords)
    if phrases:
        word_scores = _compute_rake_word_scores(phrases)
        for phrase_tokens in phrases:
            if not _is_relevant_phrase(phrase_tokens, config.min_term_length):
                continue
            phrase_surface = _phrase_text(phrase_tokens)
            base = sum(word_scores.get(token.lemma, 0.0) for token in phrase_tokens)
            if base == 0.0:
                continue
            rake_score = _adjust_phrase_score(base, phrase_tokens)
            _store_candidate(candidates, phrase_surface, rake_score, source="rake")

    if not candidates:
        for token in tokens:
            if token.is_stop:
                continue
            if len(token.lemma) < config.min_term_length and not token.surface.isupper():
                continue
            _store_candidate(candidates, token.surface, score=1.0, source="fallback")

    return candidates


def _store_candidate(candidates: Dict[str, CandidateTerm], text: str, score: float, source: str) -> None:
    key = text.strip()
    if not key:
        return
    normalized = key.lower()
    existing = candidates.get(normalized)
    if existing:
        existing.bump(score, source)
    else:
        candidates[normalized] = CandidateTerm(text=key, score=score, sources=[source])


def _normalize_anchors(
    doc_anchors: Optional[Mapping[str, Mapping[str, Sequence[str]] | Sequence[str]]]
) -> Dict[str, List[str]]:
    if not doc_anchors:
        return {"section_titles": [], "model_names": []}
    if "section_titles" in doc_anchors or "model_names" in doc_anchors:
        section_titles = list(dict.fromkeys(doc_anchors.get("section_titles", [])))  # type: ignore[arg-type]
        model_names = list(dict.fromkeys(doc_anchors.get("model_names", [])))  # type: ignore[arg-type]
        return {"section_titles": section_titles, "model_names": model_names}

    collected_sections: List[str] = []
    collected_models: List[str] = []
    for doc_anchor in doc_anchors.values():  # type: ignore[attr-defined]
        if isinstance(doc_anchor, Mapping):
            collected_sections.extend(doc_anchor.get("section_titles", []))
            collected_models.extend(doc_anchor.get("model_names", []))
        elif isinstance(doc_anchor, Sequence) and not isinstance(doc_anchor, (str, bytes)):
            collected_sections.extend(doc_anchor)
    return {
        "section_titles": list(dict.fromkeys(collected_sections)),
        "model_names": list(dict.fromkeys(collected_models)),
    }


@_timecall("rerank_sections")
def _rerank_sections(
    question: str,
    section_titles: Sequence[str],
    tokens: Sequence[TokenInfo],
) -> Tuple[Optional[str], Dict[str, float], List[str]]:
    cleaned = [title.strip() for title in section_titles if title and title.strip()]
    if not cleaned:
        return None, {}, []
    scores: Dict[str, float] = {}
    model = _get_sentence_model()
    if model and np is not None:
        try:
            embeddings = model.encode([question] + cleaned, normalize_embeddings=True)
            matrix = np.asarray(embeddings, dtype=np.float32)
        except Exception:
            matrix = None
        if matrix is not None and len(matrix) > 1:
            query_vec = matrix[0]
            candidate_vecs = matrix[1:]
            try:
                similarity = np.dot(candidate_vecs, query_vec)
            except Exception:
                similarity = None
            if similarity is not None and len(similarity):
                best_index = int(np.argmax(similarity))
                best_score = float(similarity[best_index])
                for idx, title in enumerate(cleaned):
                    scores[title] = float(similarity[idx])
                if best_score > 0.15:
                    ordered = _order_sections_by_score(scores, cleaned)
                    return cleaned[best_index], scores, ordered
    fallback_best, fallback_scores = _score_sections_by_overlap(tokens, cleaned)
    scores.update(fallback_scores)
    ordered = _order_sections_by_score(scores, cleaned)
    return fallback_best, scores, ordered


def _score_sections_by_overlap(
    tokens: Sequence[TokenInfo],
    titles: Sequence[str],
) -> Tuple[Optional[str], Dict[str, float]]:
    question_terms = {token.lemma for token in tokens if not token.is_stop}
    best_title: Optional[str] = None
    best_score = float("-inf")
    scores: Dict[str, float] = {}
    for title in titles:
        lemmas = [_lemmatize(match.group(0)) for match in WORD_RE.finditer(title)]
        if not lemmas:
            continue
        overlap = sum(1 for lemma in lemmas if lemma in question_terms)
        coverage = overlap / len(lemmas)
        capital_bonus = 0.2 if any(word[:1].isupper() for word in title.split()) else 0.0
        length_bonus = min(len(lemmas), 4) * 0.05
        priority_bonus = sum(
            0.8 for lemma in lemmas if lemma in question_terms and lemma in PRIORITY_SECTION_TERMS
        )
        score = overlap + coverage + capital_bonus + length_bonus + priority_bonus
        scores[title] = score
        if score > best_score:
            best_score = score
            best_title = title
    if best_title:
        return best_title, scores
    return (titles[0] if titles else None), scores


def _order_sections_by_score(scores: Mapping[str, float], candidates: Sequence[str]) -> List[str]:
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    seen: Set[str] = set()
    result: List[str] = []
    for title, _ in ordered:
        normalized = title.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    for title in candidates:
        normalized = title.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


def _select_model_name(question: str, model_names: Sequence[str]) -> Optional[str]:
    if "model" not in question.lower():
        return None
    for name in model_names:
        clean = name.strip()
        if clean:
            return clean
    return None


@_timecall("prepare_boost_pool")
def _prepare_boost_pool(
    entity_spans: Sequence[EntitySpan],
    ranked_terms: Sequence[CandidateTerm],
    *,
    model_term: Optional[str],
    intent_terms: Sequence[str],
    disallowed_terms: Optional[Set[str]] = None,
) -> List[str]:
    pool: List[str] = []
    seen: Set[str] = set()
    disallowed_terms = disallowed_terms or set()

    def _add(term: Optional[str]) -> None:
        if not term:
            return
        clean = term.strip()
        if not clean:
            return
        key = clean.lower()
        if key in seen or key in disallowed_terms:
            return
        seen.add(key)
        pool.append(clean)

    if model_term:
        _add(model_term)

    for term in intent_terms:
        _add(term)

    for span in sorted(entity_spans, key=lambda sp: sp.score, reverse=True):
        _add(span.text)

    for candidate in ranked_terms:
        _add(candidate.text)

    return pool


@_timecall("build_strong_queries")
def _build_strong_queries(
    boost_pool: Sequence[str],
    ranked_term_texts: Sequence[str],
    fallback_terms: Sequence[str],
    *,
    desired: int = 2,
) -> List[Query]:
    queries: List[Query] = []
    for idx in range(desired):
        boosted = _pick_terms(boost_pool, needed=2, start=idx * 2)
        if len(boosted) < 2:
            boosted = _ensure_boost_coverage(boosted, [ranked_term_texts, fallback_terms], needed=2)
        normals = _select_normal_terms(ranked_term_texts, boosted, fallback_terms, limit=4)
        queries.append(Query(boosted_terms=boosted, normal_terms=normals, section_filter=None))
    return queries


@_timecall("build_weak_queries")
def _build_weak_queries(
    boost_pool: Sequence[str],
    ranked_term_texts: Sequence[str],
    fallback_terms: Sequence[str],
    synonyms: Mapping[str, Sequence[str]],
    section_filters: Sequence[Optional[str]],
    *,
    desired: int = 2,
) -> List[Query]:
    queries: List[Query] = []
    for idx in range(desired):
        boosted = _pick_terms(boost_pool, needed=1, start=idx)
        if len(boosted) < 1:
            boosted = _ensure_boost_coverage(boosted, [ranked_term_texts, fallback_terms], needed=1)
        normals = _select_normal_terms(ranked_term_texts, boosted, fallback_terms, limit=3)
        swapped = [_swap_with_synonym(term, synonyms) for term in normals]
        section = None
        if idx < len(section_filters):
            section = section_filters[idx]
        queries.append(Query(boosted_terms=boosted, normal_terms=swapped, section_filter=section))
    return queries


def _pick_terms(source: Sequence[str], *, needed: int, start: int = 0) -> List[str]:
    terms: List[str] = []
    for term in source[start:]:
        if term not in terms:
            terms.append(term)
        if len(terms) >= needed:
            break
    if len(terms) < needed:
        for term in source:
            if term not in terms:
                terms.append(term)
            if len(terms) >= needed:
                break
    return terms[:needed]


def _ensure_boost_coverage(
    current: Sequence[str],
    fallback_sources: Sequence[Sequence[str]],
    *,
    needed: int,
) -> List[str]:
    result = [term for term in current if term]
    for pool in fallback_sources:
        for term in pool:
            if not term:
                continue
            if term in result:
                continue
            result.append(term)
            if len(result) >= needed:
                break
        if len(result) >= needed:
            break
    return result[:needed]


def _select_normal_terms(
    ranked_term_texts: Sequence[str],
    exclude_terms: Sequence[str],
    fallback_terms: Sequence[str],
    *,
    limit: int,
) -> List[str]:
    exclude = {term.lower() for term in exclude_terms}
    normals: List[str] = []
    for term in ranked_term_texts:
        if term.lower() in exclude:
            continue
        normals.append(term)
        if len(normals) >= limit:
            break
    if len(normals) < limit:
        for term in fallback_terms:
            if term.lower() in exclude:
                continue
            normals.append(term)
            if len(normals) >= limit:
                break
    if not normals:
        normals = list(ranked_term_texts[:limit])
    return normals[:limit]


def _swap_with_synonym(term: str, synonyms: Mapping[str, Sequence[str]]) -> str:
    options = synonyms.get(term)
    if options:
        return options[0]
    options = synonyms.get(term.lower())
    if options:
        return options[0]
    return term


def _boost_anchor_terms(
    candidates: Dict[str, CandidateTerm],
    anchors: Mapping[str, Sequence[str]],
    *,
    source: str,
) -> None:
    for heading in anchors.get("section_titles", []):
        clean = heading.strip()
        if clean:
            _store_candidate(candidates, clean, score=2.4, source=source)
    for model in anchors.get("model_names", []):
        clean = model.strip()
        if clean:
            _store_candidate(candidates, clean, score=2.2, source=source)


def _boost_intent_terms(
    candidates: Dict[str, CandidateTerm],
    question: str,
    intent_cues: Mapping[str, Sequence[str]],
) -> List[str]:
    question_lower = question.lower()
    hits: List[str] = []
    for cue, expansions in intent_cues.items():
        if cue in question_lower:
            for term in expansions:
                hits.append(term)
                _store_candidate(candidates, term, score=2.0, source=f"intent:{cue}")
    return list(dict.fromkeys(hits))


@_timecall("gather_synonyms")
def _gather_synonyms(
    terms: Sequence[str],
    synonym_table: Mapping[str, Sequence[str]],
) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    for term in terms:
        collected: List[str] = []
        for synonym in _lookup_wordnet_synonyms(term):
            collected.append(synonym)
        if synonym_table:
            extras = synonym_table.get(term.lower(), [])
            collected.extend(extras)
        seen: Set[str] = set()
        cleaned: List[str] = []
        for synonym in collected:
            normalized = synonym.strip()
            if not normalized or normalized.lower() == term.lower():
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(normalized)
        if cleaned:
            result[term] = cleaned[:3]
    return result


def _get_sentence_model():
    global _sentence_model, _sentence_model_failed
    if SentenceTransformer is None or _sentence_model_failed:
        return None
    if _sentence_model is None:
        try:
            _sentence_model = SentenceTransformer(_SENTENCE_MODEL_NAME)
        except Exception:
            _sentence_model_failed = True
            return None
    return _sentence_model


def _get_ner_model():
    global _ner_model, _ner_model_failed
    if spacy is None or _ner_model_failed:
        return None
    if _ner_model is None:
        try:
            _ner_model = spacy.load(_NER_MODEL_NAME)
        except Exception:
            _ner_model_failed = True
            return None
    return _ner_model


def _lookup_wordnet_synonyms(term: str) -> List[str]:
    if wn is None:
        return []
    try:
        synsets = wn.synsets(term)
    except LookupError:
        return []
    synonyms: Set[str] = set()
    for synset in synsets:
        for lemma in synset.lemma_names():
            candidate = lemma.replace("_", " ").strip()
            if not candidate or candidate.lower() == term.lower():
                continue
            synonyms.add(candidate)
    return sorted(synonyms)


if __name__ == "__main__":
    sample_question = "What problem does MoC Mixtures of Text Chunking Learners for Retrieval-Augmented Generation System aim to solve"
    sample_anchors = {
        "doc123": {
            "section_titles": [
                "3. Methodology",
                "4. Experiments",
                "5. Results and Analysis",
            ],
            "model_names": ["MoC"],
        }
    }
    queries, debug = extract_queries(sample_question, doc_anchors=sample_anchors)
    print("Question:", sample_question)
    print("Generated queries:")
    for idx, query in enumerate(queries, start=1):
        print(
            f"  {idx}. boosted={query.boosted_terms} normal={query.normal_terms} section={query.section_filter}"
        )
    print("\nDebug info keys:", list(debug.keys()))
