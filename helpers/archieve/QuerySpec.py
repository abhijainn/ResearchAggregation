# ---------- LLM Query Analyzer (ASTA-style Query Decomposer) ----------
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - allow running local tests without OpenAI SDK
    OpenAI = None  # type: ignore[assignment]

@dataclass
class QuerySpec:
    intent: str = "find_papers"                   # "find_papers" | "find_specific_paper" | "find_author"
    keyword_query: List[str] = field(default_factory=list)  # re-formatted query for keyword-based retrieval
    keyword_boosted: List[str] = field(default_factory=list)  # top 2-3 terms to weight higher
    keyword_regular: List[str] = field(default_factory=list)  # remaining keyword terms
    semantic_query: str = ""                       # re-formatted query for semantic/vector-based retrieval
    metadata_filters: Dict[str, Any] = field(default_factory=dict)  # e.g., {"year_min": 2022, "venue": "arxiv", "author": "Hinton", "field_of_study": "cs.CL"}
    qualifiers: List[str] = field(default_factory=list)             # e.g., ["recent","popular","classic","central"]
    
    # Backward compatibility property
    @property
    def semantic_criteria(self) -> str:
        """Backward compatibility: returns semantic_query."""
        return self.semantic_query or ""


_KEYWORD_SPLIT_RE = re.compile(r"[\n,;|]+")
_WORD_SPLIT_RE = re.compile(r"[^A-Za-z0-9]+")
_STOP_WORDS = {
    "a",
    "an",
    "the",
    "of",
    "and",
    "or",
    "in",
    "on",
    "for",
    "to",
    "from",
    "by",
    "with",
    "at",
}


def _normalize_keyword_query(raw_keywords: Any) -> List[str]:
    """
    Coerce keyword_query into a list of discrete keyword strings.
    Handles strings (including comma-separated), JSON-encoded lists, and iterables.
    """
    if raw_keywords is None:
        return []
    if isinstance(raw_keywords, str):
        text = raw_keywords.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                return _normalize_keyword_query(parsed)
            except json.JSONDecodeError:
                pass
        segments = _KEYWORD_SPLIT_RE.split(text) if _KEYWORD_SPLIT_RE.search(text) else [text]
        return [segment.strip() for segment in segments if segment.strip()]
    if isinstance(raw_keywords, (list, tuple, set)):
        normalized: List[str] = []
        for value in raw_keywords:
            normalized.extend(_normalize_keyword_query(value))
        return normalized
    fallback = str(raw_keywords).strip()
    return [fallback] if fallback else []


def _single_word_keywords(keywords: List[str]) -> List[str]:
    """
    Force keywords into single-word tokens to make keyword search more precise.
    Splits on whitespace/punctuation and preserves order.
    """
    single_words: List[str] = []
    for kw in keywords:
        for token in _WORD_SPLIT_RE.split(kw):
            token = token.strip()
            if token:
                if token.lower() in _STOP_WORDS:
                    continue
                single_words.append(token)
    return single_words


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    deduped: List[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def _split_boosted_keywords(keywords: List[str]) -> Tuple[List[str], List[str]]:
    """
    Split keywords into boosted (top 2-3) and regular buckets.
    Prioritizes the earliest terms from the LLM (assumed most salient).
    """
    deduped = _dedupe_preserve_order(keywords)
    if not deduped:
        return [], []
    boost_count = min(3, len(deduped))
    if boost_count == 1 and len(deduped) > 1:
        boost_count = 2
    boosted = deduped[:boost_count]
    regular = deduped[boost_count:]
    return boosted, regular

ANALYZER_SCHEMA = {
    "intent": "find_papers | find_specific_paper | find_author",
    "keyword_query": "list[string] - query optimized for keyword/title search (extract key terms, acronyms)",
    "semantic_query": "string - query optimized for semantic similarity search (natural language paraphrase)",
    "metadata_filters": {
        "year_min": "int (optional) - minimum publication year",
        "year_max": "int (optional) - maximum publication year",
        "venue": "string (optional) - conference/journal name (e.g., 'NeurIPS', 'ICLR', 'arXiv')",
        "author": "string (optional) - author name",
        "field_of_study": "string (optional) - arXiv category code (e.g., 'cs.CL', 'cs.IR', 'cs.LG', 'cs.CV')"
    },
    "qualifiers": ["recent | popular | classic | central (optional)"],
    # Richer decomposition fields (all optional)
    "content": "string - content-only extraction of the topic (no metadata)",
    "authors": "array[string] - author names requested in the query",
    "venues": "array[string] - venue names requested in the query",
    "recency": "'recent' | 'early' | null",
    "centrality": "'first' | 'last' | null",
    "time_range": {"start": "int|null", "end": "int|null"},
    "broad_or_specific": "'broad' | 'specific' | null",
    "by_name_or_title": "'name' | 'title' | null",
    "relevance_criteria": {
        "required_relevance_critieria": "array[{name:string, description:string, weight:number}] (weights sum to 1)",
        "nice_to_have_relevance_criteria": "array[{name:string, description:string, weight:number}]",
        "clarification_questions": "array[string]"
    },
    "domains": {"main": "string|null", "other": "array[string]"},
    "possible_refusal": {"type": "string|null"}
}

ANALYZER_PROMPT = """Goal
You are part of a search engine for academic papers. Read the user's query and produce a single JSON object that captures both a practical search spec and a richer decomposition. If the input is not a direct paper-finding request, assume it is a question that should be answered via academic literature.

User query:
{query}

Output JSON schema (keys and intent):
{schema}

Instructions for key behavior:
- keyword_query: return an array of discrete keywords (preserve named entities); do not return a comma-separated string
- content: Extract only topical content; exclude metadata (authors, years, venues, impact terms). If only metadata exists, leave empty.
- authors: Only names explicitly requested as authors of the desired papers (not people to discuss in the paper).
- venues: Only venues explicitly requested. Do not include years in the venue strings (years must go to time_range).
- recency: 'recent' if explicitly asked for latest/most recent; 'early' if asked for early/classic. Otherwise null.
- centrality: 'first' for central/seminal/highly cited/top; 'last' for least/less cited; otherwise null.
- time_range: Extract only explicit years or ranges. Current year is 2025.
- domains: Map to a main field if possible and optional others; if unknown, main can be null.
- broad_or_specific: 'specific' if the user provided a unique identifier like an exact title or a unique short name; otherwise 'broad'.
- by_name_or_title: 'title' if the query looks for a paper exactly by its title; otherwise 'name'. If unsure, 'name'.
- relevance_criteria: Provide required criteria with weights summing to 1 if the query is sufficiently specific; otherwise ask minimal clarification questions.
- possible_refusal: If the query requires unsupported behavior (e.g., web URL-only lookup), set a category; else null.

Also fill the practical fields to be used in retrieval:
- intent: find_papers | find_specific_paper | find_author
- keyword_query: array of keywords that are likely to appear in the chunk corresponding to the user's query
- semantic_query: paraphrase-oriented natural language of the content
- metadata_filters: year_min/year_max from time_range, single author/venue strings if applicable, field_of_study from domains.main
- qualifiers: include 'recent'/'classic'/'central' when recency/centrality imply them

Return only valid JSON.
"""

def _call_llm_for_json(prompt: str) -> Dict[str, Any]:
    if OpenAI is None:
        raise ImportError("OpenAI SDK is required to analyze queries. Install the latest 'openai' package.")
    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    content = resp.choices[0].message.content
    if content is None:
        raise ValueError("LLM returned empty content for JSON response")
    if isinstance(content, str):
        # response_format=json_object ensures a JSON string; parse it
        return json.loads(content)
    if isinstance(content, dict):
        # In case future SDK versions return a dict directly
        return content
    # Last resort: stringify and attempt to parse
    return json.loads(str(content))


def analyze_query_llm(user_query: str) -> QuerySpec:
    """
    Analyzes a user query using LLM to decompose it into:
    - Keyword query (for keyword-based retrieval)
    - Semantic query (for semantic similarity search)
    - Metadata filters (year, venue, author, field_of_study)
    - Qualifiers (recent, popular, classic, central)
    
    This mimics ASTA's Query Decomposer approach.
    """
    prompt = ANALYZER_PROMPT.format(query=user_query.strip(), schema=json.dumps(ANALYZER_SCHEMA))
    raw = _call_llm_for_json(prompt)  # expects dict
    
    # Extract intent
    intent = raw.get("intent", "find_papers")
    if intent not in ["find_papers", "find_specific_paper", "find_author"]:
        intent = "find_papers"
    
    # Rich fields (optional)
    content = (raw.get("content") or "").strip() if isinstance(raw.get("content"), str) else ""
    authors_list = raw.get("authors") if isinstance(raw.get("authors"), list) else []
    venues_list = raw.get("venues") if isinstance(raw.get("venues"), list) else []
    recency = raw.get("recency") if raw.get("recency") in ("recent", "early", None) else None
    centrality = raw.get("centrality") if raw.get("centrality") in ("first", "last", None) else None
    time_range = raw.get("time_range") if isinstance(raw.get("time_range"), dict) else {}
    domains = raw.get("domains") if isinstance(raw.get("domains"), dict) else {}
    broad_or_specific = raw.get("broad_or_specific") if raw.get("broad_or_specific") in ("broad", "specific", None) else None

    # Extract keyword and semantic queries (prefer content when present)
    keyword_terms = _single_word_keywords(_normalize_keyword_query(raw.get("keyword_query")))
    semantic_raw = raw.get("semantic_query", "")
    if isinstance(semantic_raw, str):
        semantic_query = semantic_raw.strip()
    else:
        semantic_query = str(semantic_raw or "").strip()
    if content:
        if not keyword_terms:
            keyword_terms = _single_word_keywords(_normalize_keyword_query(content))
        if not semantic_query:
            semantic_query = content
    
    # Fallback: if new fields not present, use old semantic_criteria or original query
    if not semantic_query:
        semantic_query = raw.get("semantic_criteria", user_query.strip())
    if not keyword_terms:
        keyword_terms = _single_word_keywords(_normalize_keyword_query(user_query.strip()))

    keyword_boosted, keyword_regular = _split_boosted_keywords(keyword_terms)
    keyword_query = keyword_boosted + keyword_regular
    
    # Extract metadata filters
    meta = raw.get("metadata_filters") or {}
    if not isinstance(meta, dict):
        meta = {}
    
    # Map richer fields into metadata
    if isinstance(time_range, dict):
        if isinstance(time_range.get("start"), int):
            meta["year_min"] = time_range.get("start")
        if isinstance(time_range.get("end"), int):
            meta["year_max"] = time_range.get("end")
    if authors_list:
        meta["author"] = ", ".join([str(a) for a in authors_list if isinstance(a, str) and a.strip()])
    if venues_list:
        meta["venue"] = ", ".join([str(v) for v in venues_list if isinstance(v, str) and v.strip()])
    if isinstance(domains, dict) and domains.get("main"):
        meta.setdefault("field_of_study", str(domains.get("main")))

    # Normalize field_of_study to lowercase for consistency
    if "field_of_study" in meta and meta["field_of_study"]:
        meta["field_of_study"] = str(meta["field_of_study"]).strip().lower()
    
    # Extract qualifiers
    quals = raw.get("qualifiers") or []
    if not isinstance(quals, list):
        quals = []
    
    # Validate qualifiers
    valid_quals = ["recent", "popular", "classic", "central"]
    quals = [q for q in quals if q in valid_quals]
    # Enrich qualifiers based on recency/centrality
    if recency == "recent" and "recent" not in quals:
        quals.append("recent")
    if recency == "early" and "classic" not in quals:
        quals.append("classic")
    if centrality == "first" and "central" not in quals:
        quals.append("central")
    
    # Intent refinement from broad_or_specific
    if broad_or_specific == "specific":
        intent = "find_specific_paper"

    return QuerySpec(
        intent=intent,
        keyword_query=keyword_query,
        keyword_boosted=keyword_boosted,
        keyword_regular=keyword_regular,
        semantic_query=semantic_query,
        metadata_filters=meta,
        qualifiers=quals
    )

def simple_post_filter(results: List[Dict[str, Any]], spec: QuerySpec) -> List[Dict[str, Any]]:
    """
    Filters and sorts results based on metadata filters from QuerySpec.
    Supports filtering by year, venue, author, and field_of_study (arXiv categories).
    """
    if not results:
        return results
    meta = spec.metadata_filters or {}
    y_min = meta.get("year_min")
    y_max = meta.get("year_max")
    venue = (meta.get("venue") or "").lower()
    author = (meta.get("author") or "").lower()
    field_of_study = (meta.get("field_of_study") or "").lower()

    def extract_year(p: Dict[str, Any]) -> Optional[int]:
        for k in ("year", "published", "updated", "date"):
            v = p.get(k)
            if not v:
                continue
            s = str(v)
            m = __import__("re").search(r"(19|20)\d{2}", s)
            if m:
                return int(m.group(0))
        return None

    def keep(p: Dict[str, Any]) -> bool:
        y = extract_year(p)
        if y_min is not None and (y is None or y < y_min): return False
        if y_max is not None and (y is None or y > y_max): return False
        if venue:
            in_fields = [p.get("venue"), p.get("journal"), p.get("source"), p.get("url"), p.get("source_url")]
            if not any(isinstance(v, str) and venue in v.lower() for v in in_fields):
                return False
        if author:
            a = p.get("authors") or p.get("author") or p.get("creators")
            a_str = a if isinstance(a, str) else " ".join(map(str, a)) if isinstance(a, (list, tuple, set)) else str(a or "")
            if author not in a_str.lower():
                return False
        if field_of_study:
            # Check arXiv categories field
            categories = p.get("categories") or p.get("category") or p.get("fields") or []
            if isinstance(categories, str):
                cats = categories.lower().split()
            elif isinstance(categories, list):
                cats = [str(c).lower() for c in categories]
            else:
                cats = []
            # Check if field_of_study matches any category (exact or prefix match for cs.CL, cs.IR, etc.)
            if not any(field_of_study in cat or cat.startswith(field_of_study + ".") for cat in cats):
                return False
        return True

    filtered = [p for p in results if keep(p)]

    # Optional: light qualifier sort (leave as-is if none)
    quals = set(spec.qualifiers or [])
    if "recent" in quals:
        filtered.sort(key=lambda p: extract_year(p) or -1, reverse=True)
    elif "classic" in quals:
        filtered.sort(key=lambda p: extract_year(p) or 9999)
    elif "popular" in quals or "central" in quals:
        def cites(p):
            for k in ("citation_count","num_citations","cited_by","citations"):
                v = p.get(k)
                if isinstance(v, (int, float)): return int(v)
            return 0
        filtered.sort(key=cites, reverse=True)

    return filtered


if __name__ == "__main__":
    mock_query = (
        "Help me find papers on RAG that talk about techniques used to find specific quotes within "
        "documents to aid document-level QA. The paper should focus on how to construct semantically "
        "complete chunks for retrieval."
    )
    try:
        spec = analyze_query_llm(mock_query)
    except Exception as exc:
        print(f"Failed to analyze mock query: {exc}")
    else:
        output = {
            "intent": spec.intent,
            "keyword_query": spec.keyword_query,
            "keyword_boosted": spec.keyword_boosted,
            "keyword_regular": spec.keyword_regular,
            "semantic_query": spec.semantic_query,
            "metadata_filters": spec.metadata_filters,
            "qualifiers": spec.qualifiers,
        }
        print(json.dumps(output, indent=2))

        output_path = Path(__file__).with_name("mock_queryspec_output.json")
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=2)
        print(f"Mock QuerySpec written to {output_path}")
