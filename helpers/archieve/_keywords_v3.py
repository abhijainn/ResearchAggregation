"""
LLM-driven keyword-query builder that consumes boosted/regular terms from the
QuerySpec analyzer and emits up to 5 queries (mixing boosted/regular and
synonym variants) with section filters to stay out of references by default.
"""

from __future__ import annotations

import json
import re
import os
import sys
from dataclasses import dataclass, field, asdict, is_dataclass
from types import SimpleNamespace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

# current_dir = os.path.dirname(os.path.abspath(__file__))

# parent_dir = os.path.dirname(current_dir)

# sys.path.append(parent_dir)

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional; analyzer may be mocked in tests
    OpenAI = None  # type: ignore[assignment]

try:
    # Prefer the local helper version
    from QuerySpec import analyze_query_llm
except ImportError:  # pragma: no cover
    try:
        from helpers.QuerySpec import analyze_query_llm  # type: ignore
    except ImportError:
        from helpers.QuerySpec import analyze_query_llm  # type: ignore

__all__ = [
    "KeywordConfig",
    "Query",
    "load_default_config",
    "extract_queries",
    "mine_doc_anchors_from_text",
    "normalize_keyword_terms",
]

WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]+")
DEFAULT_SYNONYMS: Mapping[str, Sequence[str]] = {
    "retrieval": ("search",),
    "qa": ("question answering", "q&a"),
    "method": ("approach",),
    "result": ("finding",),
    "experiment": ("study",),
}


@dataclass
class KeywordConfig:
    max_section_suggestions: int = 2
    max_queries: int = 5
    synonyms: Mapping[str, Sequence[str]] = field(default_factory=lambda: dict(DEFAULT_SYNONYMS))


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


def load_default_config() -> KeywordConfig:
    return KeywordConfig()


def normalize_keyword_terms(text: str) -> List[str]:
    """Lowercase tokenization used for keyword overlap debugging."""
    if not text:
        return []
    return [match.group(0).lower() for match in WORD_RE.finditer(text)]


def mine_doc_anchors_from_text(doc_text: str, *, max_headings: int = 10) -> Dict[str, List[str]]:
    """
    Lightweight heading miner for documents without structured metadata.
    """
    headings: List[str] = []
    for line in doc_text.splitlines():
        candidate = line.strip()
        if candidate and candidate[0].isupper():
            headings.append(candidate.rstrip(":"))
    return {"section_titles": list(dict.fromkeys(headings))[:max_headings]}


def _dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _swap_with_synonym(term: str, synonyms: Mapping[str, Sequence[str]]) -> str:
    options = synonyms.get(term) or synonyms.get(term.lower())
    if options:
        for opt in options:
            if opt:
                return str(opt)
    return term


def _choose_headers_with_llm(question: str, headers: Sequence[str], max_choices: int) -> List[str]:
    """
    Use the OpenAI API to pick the most relevant headers for the question.
    Falls back to the first headers if the API is unavailable.
    """
    headers = [h for h in headers if h and h.strip()]
    if not headers:
        return []
    if OpenAI is None:
        return list(headers)[:max_choices]

    client = OpenAI()
    prompt = (
        "You select document section headers that are most relevant to a query.\n"
        f"User query: {question.strip()}\n"
        f"Available headers: {headers}\n"
        f"Return up to {max_choices} headers from the list in order of relevance. "
        "If none are relevant, return an empty list."
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        content = resp.choices[0].message.content
        data = json.loads(content) if isinstance(content, str) else content
        chosen = data.get("headers") if isinstance(data, dict) else None
        if isinstance(chosen, list):
            return [h for h in chosen if isinstance(h, str) and h.strip()][:max_choices]
    except Exception:
        pass
    return list(headers)[:max_choices]


def _filter_headers(question: str, headers: Sequence[str]) -> List[str]:
    """
    Remove reference-like sections unless the user explicitly asks for references.
    """
    refs_allowed = bool(re.search(r"\breferenc|\bcitation", question.lower()))
    filtered: List[str] = []
    for h in headers:
        if not h:
            continue
        lower = h.lower()
        if not refs_allowed and ("reference" in lower or "bibliography" in lower):
            continue
        filtered.append(h)
    return filtered


def _ensure_headers(headers: Sequence[str]) -> List[str]:
    defaults = ["Introduction", "Method", "Results"]
    ordered = [h for h in headers if h and h.strip()]
    if not ordered:
        ordered = defaults
    return ordered


def _build_queries_from_terms(keywords: List[str]) -> List[Query]:
    """
    Build a single Query using the provided keywords (no boosted/regular split).
    """
    filtered = [kw for kw in keywords if kw]
    return [Query(boosted_terms=[], normal_terms=filtered, section_filter=None)]


def extract_queries(
    question: str,
    doc_anchors: Optional[Mapping[str, Mapping[str, Sequence[str]] | Sequence[str]]] = None,
    config: Optional[KeywordConfig] = None,
) -> Tuple[List[Query], Dict[str, object]]:
    """
    Build Query objects using the LLM query analyzer outputs:
    - keywords are taken directly from the analyzer's keyword_query field (no boosted/regular split)
    """
    config = config or load_default_config()
    spec = analyze_query_llm(question)

    keywords = _dedupe_preserve_order(spec.keyword_query or [])
    if not keywords:
        keywords = _dedupe_preserve_order(spec.keyword_boosted + spec.keyword_regular)
    if not keywords:
        keywords = _dedupe_preserve_order(normalize_keyword_terms(question))

    queries = _build_queries_from_terms(keywords)

    debug_info = {
        "spec": asdict(spec) if is_dataclass(spec) else getattr(spec, "__dict__", {}),
        "section_titles": list((doc_anchors or {}).get("section_titles", [])) if isinstance(doc_anchors, Mapping) else [],
        "queries": [q.to_dict() for q in queries],
    }
    return queries, debug_info


if __name__ == "__main__":
    mock_question = "Find recent RAG papers that improve quote-level retrieval for document QA"
    mock_anchors = {
        "section_titles": [
            "Abstract",
            "Introduction",
            "Related Works",
            "Methodology",
            "Experiment",
            "Conclusion",
            "Limitations",
            "References",
        ]
    }

    mock_spec = SimpleNamespace(
        keyword_boosted=["rag", "retrieval", "quote"],
        keyword_regular=["document", "qa"],
        keyword_query=["rag", "retrieval", "quote", "document", "qa"],
        semantic_query=mock_question,
        intent="find_papers",
        metadata_filters={},
        qualifiers=["recent"],
    )

    def _mock_analyze_query_llm(question: str):
        return mock_spec

    original_analyzer = analyze_query_llm
    try:
        globals()["analyze_query_llm"] = _mock_analyze_query_llm
        queries, debug = extract_queries(mock_question, doc_anchors=mock_anchors)
    finally:
        globals()["analyze_query_llm"] = original_analyzer

    print("Queries:")
    for q in queries:
        print(q.to_dict())
    print("\nDebug:")
    print(json.dumps(debug, indent=2))
