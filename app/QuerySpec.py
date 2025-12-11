# ---------- LLM Query Analyzer (ASTA-style Query Decomposer) ----------
from dataclasses import dataclass, field
import json
from typing import Any, Dict, List, Optional
from openai import OpenAI

@dataclass
class QuerySpec:
    intent: str = "find_papers"                   # "find_papers" | "find_specific_paper" | "find_author"
    keyword_query: str = ""                        # re-formatted query for keyword-based retrieval
    semantic_query: str = ""                       # re-formatted query for semantic/vector-based retrieval
    metadata_filters: Dict[str, Any] = field(default_factory=dict)  # e.g., {"year_min": 2022, "venue": "arxiv", "author": "Hinton", "field_of_study": "cs.CL"}
    qualifiers: List[str] = field(default_factory=list)             # e.g., ["recent","popular","classic","central"]
    
    # Backward compatibility property
    @property
    def semantic_criteria(self) -> str:
        """Backward compatibility: returns semantic_query."""
        return self.semantic_query or ""

ANALYZER_SCHEMA = {
    "intent": "find_papers | find_specific_paper | find_author",

    "keyword_query": "string - optimized for keyword/title search",
    "semantic_query": "string - optimized for semantic similarity search",

    "metadata_filters": {
        "year_min": "int (optional)",
        "year_max": "int (optional)",
        "author": "string (optional)"
    },

    "content": "string - topic-only content",
    "authors": "array[string] - explicitly requested author names",
    "title": "string - explicit title if present",

    "date": {
        "year_min": "int|null",
        "year_max": "int|null"
    },

    "qualifiers": ["recent | classic (optional)"]
}


ANALYZER_PROMPT = """Goal
You are part of a search engine for academic papers. Read the user's query and produce a single JSON object describing:
1. A practical search spec (intent, queries, metadata filters)
2. A minimal metadata decomposition (content, authors, title, date range)

The user may ask for papers, an author, or a specific paper.

User query:
{query}

Output JSON schema:
{schema}

-----------------------------
Field Instructions
-----------------------------

content:
- Extract only topical content.
- Exclude metadata (authors, years, titles).
- If the query contains only metadata, leave this empty.

authors:
- Only author names the user is explicitly requesting papers *by*.

title:
- Only if a full title explicitly appears.

date.year_min / date.year_max:
Interpret publication time expressions:
- "2018" → year_min=2018, year_max=2018
- "since 2022" / "after 2022" → year_min=2022, year_max=2025 (current year: 2025)
- "2014-2019" → year_min=2014, year_max=2019
- "any time" or no explicit years → both null

intent:
- "find_papers" for general topic queries
- "find_specific_paper" if an explicit title is provided
- "find_author" if explicitly asking for papers by a person

keyword_query:
- A short keyword-oriented summary of the topical content.

semantic_query:
- A paraphrased natural-language version of the topical content.

metadata_filters:
- Include year_min/year_max (if extracted).
- Include author (if extracted).
- No other metadata allowed.

qualifiers:
- "recent" for queries asking for the latest work.
- "classic" for queries asking for foundational/early work.

Return ONLY valid JSON.
"""


def _call_llm_for_json(prompt: str) -> Dict[str, Any]:
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
    
        # -------------------------------
    # Extract minimal metadata fields
    # -------------------------------
    content = (raw.get("content") or "").strip()

    authors_list = raw.get("authors")
    if not isinstance(authors_list, list):
        authors_list = []

    title = raw.get("title") or ""

    # -------------------------------
    # Date range extraction
    # -------------------------------
    date_block = raw.get("date") or {}
    year_min = date_block.get("year_min")
    year_max = date_block.get("year_max")

    # -------------------------------
    # Keyword + semantic query
    # -------------------------------
    keyword_query = raw.get("keyword_query", "").strip()
    semantic_query = raw.get("semantic_query", "").strip()

    if content:
        if not keyword_query:
            keyword_query = content
        if not semantic_query:
            semantic_query = content

    if not semantic_query:
        semantic_query = user_query.strip()
    if not keyword_query:
        keyword_query = user_query.strip()

    # -------------------------------
    # Metadata filters
    # -------------------------------
    meta = raw.get("metadata_filters") or {}
    if not isinstance(meta, dict):
        meta = {}

    # Overwrite with LLM-parsed date range
    if year_min is not None:
        meta["year_min"] = year_min
    if year_max is not None:
        meta["year_max"] = year_max

    # Author
    if authors_list:
        meta["author"] = ", ".join(a for a in authors_list if isinstance(a, str))

    # -------------------------------
    # Qualifiers
    # -------------------------------
    qualifiers = raw.get("qualifiers") or []
    if not isinstance(qualifiers, list):
        qualifiers = []
    qualifiers = [q for q in qualifiers if q in ("recent", "classic")]

    # -------------------------------
    # Intent refinement
    # -------------------------------
    if title:
        intent = "find_specific_paper"

    return QuerySpec(
        intent=intent,
        keyword_query=keyword_query,
        semantic_query=semantic_query,
        metadata_filters=meta,
        qualifiers=qualifiers,
    )


def simple_post_filter(results: List[Dict[str, Any]], spec: QuerySpec) -> List[Dict[str, Any]]:
    """
    Filters and sorts results based on:
    - year_min / year_max (year ranges)
    - author
    Sorting:
    - 'recent' (descending year)
    - 'classic' (ascending year)
    """
    if not results:
        return results

    meta = spec.metadata_filters or {}
    year_min = meta.get("year_min")
    year_max = meta.get("year_max")
    author_filter = (meta.get("author") or "").lower()

    # -------------------------------
    # Year extraction helper
    # -------------------------------
    def extract_year(p: Dict[str, Any]) -> Optional[int]:
        """
        Extract a year from typical metadata keys.
        """
        for k in ("year", "published", "updated", "date"):
            v = p.get(k)
            if not v:
                continue
            m = __import__("re").search(r"(19|20)\d{2}", str(v))
            if m:
                return int(m.group(0))
        return None

    # -------------------------------
    # Filtering logic
    # -------------------------------
    def keep(p: Dict[str, Any]) -> bool:
        y = extract_year(p)

        # --- Year range filtering ---
        if year_min is not None:
            if y is None or y < year_min:
                return False
        if year_max is not None:
            if y is None or y > year_max:
                return False

        # --- Author filtering ---
        if author_filter:
            a = p.get("authors") or p.get("author")
            if isinstance(a, str):
                a_str = a.lower()
            elif isinstance(a, (list, tuple, set)):
                a_str = " ".join(str(x).lower() for x in a)
            else:
                a_str = ""
            if author_filter not in a_str:
                return False

        return True

    filtered = [p for p in results if keep(p)]

    # -------------------------------
    # Qualifier sorting
    # -------------------------------
    quals = set(spec.qualifiers or [])

    if "recent" in quals:
        filtered.sort(key=lambda p: extract_year(p) or -1, reverse=True)
    elif "classic" in quals:
        filtered.sort(key=lambda p: extract_year(p) or 9999)

    return filtered
