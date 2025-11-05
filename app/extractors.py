from __future__ import annotations

from typing import Dict, Any, List
from pydantic import BaseModel

from .llm_wrappers import call_prompt_json, call_chat_json
from .spec_models import (
    ExtractedContent,
    ExtractedAuthors,
    ExtractedVenues,
    ExtractedRecency,
    ExtractedCentrality,
    ExtractedYearlyTimeRange,
    BroadOrSpecificType,
    ByNameOrTitleType,
    DomainsIdentified,
    PossibleRefusal,
    RelevanceCriteria,
)

# ------------------ Prompts ------------------

CONTENT_PROMPT = (
    """
# Task Definition

Given a query for finding papers about a specific topic, extract only the content of the query, ignoring all metadata. Metadata includes:
* Author/Coauthor name(s)
* Year(s), or words describing time (e.g., "recent", "latest")
* Impact words (e.g., "central", "seminal", "influential")
* Venues (e.g., ACL, EMNLP, AAAI)
* Search-process words (e.g., "run an exhaustive search")

Rules:
* Keep phrases like "papers using", "papers proposing", "survey on" as part of content; drop redundant prefixes like "papers about/on".
* If the query is a question, extract a coherent representation focusing on the topic.
* Do not invent content; if unsure, return the original query minus metadata. If only metadata is present, return an empty string.

Return JSON: {"content": string}. Use "" when there is no content.

Examples:
{"query": "Graph-based Neural Multi-Document Summarization Yasunaga et al., 2017"}
{"content": "Graph-based Neural Multi-Document Summarization"}

{"query": "classic or early papers on pretrained transformer models"}
{"content": "pretrained transformer models"}

{"query": "good paper about CRISPR gene editing"}
{"content": "CRISPR gene editing"}

{"query": "papers about LLM chains"}
{"content": "LLM chains"}

{"query": "multi document summarization methods"}
{"content": "multi document summarization methods"}

{"query": "latest research on using annotation disagreements in classification models"}
{"content": "using annotation disagreements in classification models"}

{"query": "papers from ICLR 2024"}
{"content": ""}
"""
).strip()

AUTHOR_PROMPT = (
    """
# Task Definition

Given a query for finding papers, identify the authors whose papers are being requested. Extract only author name(s) explicitly requested as authors of the desired papers.

Return JSON: {"authors": [...]}.
If there are no such author names, return {"authors": []}.

Guidance and Examples:
{"query": "Graph-based Neural Multi-Document Summarization Yasunaga et al., 2017"}
{"authors": ["Yasunaga"]}

{"query": "papers on planning by Dan Weld"}
{"authors": ["Dan Weld"]}

{"query": "papers about transformer models by Google"}
{"authors": []}
Reason: Organizations are not authors.

{"query": "papers by author with scopus ID 123456789"}
{"authors": []}
Reason: IDs are not author names.

{"query": "papers on LLM chains"}
{"authors": []}

{"query": "papers discussing Henry David Thoreau's ideas about nature"}
{"authors": []}
Reason: The person mentioned is not required to be an author of the desired papers.
"""
).strip()

VENUE_PROMPT = (
    """
# Task Definition
Extract venue names requested (conference/journal). Do not include years. Return JSON: {"venues": [..]}.
"""
).strip()

RECENCY_PROMPT = (
    """
# Task Definition
Return {"recency": "recent"} if explicitly asks for latest/most recent; {"recency": "early"} if explicitly asks for early/classic; else {"recency": null}.
"""
).strip()

CENTRALITY_PROMPT = (
    """
# Task Definition
Return {"centrality": "first"} for central/seminal/highly-cited/top; {"centrality": "last"} for less/least cited; else {"centrality": null}.
"""
).strip()

TIME_RANGE_PROMPT = (
    """
# Task Definition
Current year is 2025. Extract explicit time range. Return {"start": int|null, "end": int|null}.
"""
).strip()

BROAD_OR_SPECIFIC_PROMPT = (
    """
# Task Definition
Return {"type": "specific"} if the query includes a unique identifier (exact title or unique short name), else {"type": "broad"}.
"""
).strip()

BY_NAME_OR_TITLE_PROMPT = (
    """
# Task Definition
If looking for a paper by exact title, return {"type": "title"}; otherwise {"type": "name"}. If unsure, {"type": "name"}.
"""
).strip()

REL_CRITERIA_SYSTEM = (
    """
Identify relevance criteria for the query. Output one of:
- {"required_relevance_critieria": [{name, description, weight}...]}
- {"clarification_questions": ["..."]}
Optionally include {"nice_to_have_relevance_criteria": [...]}. Sum of required weights must be 1.
"""
).strip()

DOMAIN_IDENT_SYSTEM = (
    """
Given a paper finding query, identify fields of study. Output {"main": string|null, "other": [..]}.
"""
).strip()

REFUSAL_SYSTEM = (
    """
Check if the query falls into unsupported categories: not paper finding, similar to, web access, affiliation, author ID. Output {"type": string|null}.
"""
).strip()

NAME_EXTRACTION_PROMPT = (
    """
Extract the most central resource name in the query (dataset, model, method, etc.). Return {"name": string|null, "corrected": string|null}.
"""
).strip()

# ------------------ Extractor functions ------------------

def extract_content(query: str) -> ExtractedContent:
    prompt = f"{CONTENT_PROMPT}\n\n{{\"query\": {__import__('json').dumps(query)} }}"
    return call_prompt_json(prompt, ExtractedContent)

def extract_authors(query: str) -> ExtractedAuthors:
    prompt = f"{AUTHOR_PROMPT}\n\n{{\"query\": {__import__('json').dumps(query)} }}"
    return call_prompt_json(prompt, ExtractedAuthors)

def extract_venues(query: str) -> ExtractedVenues:
    prompt = f"{VENUE_PROMPT}\n\n{{\"query\": {__import__('json').dumps(query)} }}"
    return call_prompt_json(prompt, ExtractedVenues)

def extract_recency(query: str) -> ExtractedRecency:
    messages = [
        {"role": "system", "content": RECENCY_PROMPT},
        {"role": "user", "content": __import__('json').dumps({"query": query})},
    ]
    # Map recent/early to first/last to align with sorter downstream
    class _R(BaseModel):
        recency: str | None
    r = call_chat_json(messages, _R)
    if r.recency == "recent":
        return ExtractedRecency(recency="first")
    if r.recency == "early":
        return ExtractedRecency(recency="last")
    return ExtractedRecency(recency=None)

def extract_centrality(query: str) -> ExtractedCentrality:
    messages = [
        {"role": "system", "content": CENTRALITY_PROMPT},
        {"role": "user", "content": __import__('json').dumps({"query": query})},
    ]
    return call_chat_json(messages, ExtractedCentrality)

def extract_time_range(query: str) -> ExtractedYearlyTimeRange:
    messages = [
        {"role": "system", "content": TIME_RANGE_PROMPT},
        {"role": "user", "content": __import__('json').dumps({"query": query})},
    ]
    return call_chat_json(messages, ExtractedYearlyTimeRange)

def extract_broad_or_specific(query: str) -> BroadOrSpecificType:
    messages = [
        {"role": "system", "content": BROAD_OR_SPECIFIC_PROMPT},
        {"role": "user", "content": __import__('json').dumps({"query": query})},
    ]
    return call_chat_json(messages, BroadOrSpecificType)

def extract_by_name_or_title(query: str) -> ByNameOrTitleType:
    messages = [
        {"role": "system", "content": BY_NAME_OR_TITLE_PROMPT},
        {"role": "user", "content": __import__('json').dumps({"query": query})},
    ]
    return call_chat_json(messages, ByNameOrTitleType)

def identify_domains(query: str) -> DomainsIdentified:
    messages = [
        {"role": "system", "content": DOMAIN_IDENT_SYSTEM},
        {"role": "user", "content": __import__('json').dumps({"query": query})},
    ]
    return call_chat_json(messages, DomainsIdentified)

def check_refusal(query: str) -> PossibleRefusal:
    messages = [
        {"role": "system", "content": REFUSAL_SYSTEM},
        {"role": "user", "content": query},
    ]
    return call_chat_json(messages, PossibleRefusal)

# Relevance criteria: accept flexible LLM output then validate via RelevanceCriteria
class _RelOut(BaseModel):
    required_relevance_critieria: List[dict] | None = None
    nice_to_have_relevance_criteria: List[dict] | None = None
    clarification_questions: List[str] | None = None

def identify_relevance_criteria(query: str) -> RelevanceCriteria:
    messages = [
        {"role": "system", "content": REL_CRITERIA_SYSTEM},
        {"role": "user", "content": query},
    ]
    raw = call_chat_json(messages, _RelOut)
    # Ensure validator preconditions: at least required or questions must be present
    req = raw.required_relevance_critieria
    nice = raw.nice_to_have_relevance_criteria
    qns = raw.clarification_questions
    if not req and not qns:
        qns = ["Could you clarify the key relevance criteria you care about?"]
    data: dict[str, Any] = {
        "query": query,
        "required_relevance_critieria": req,
        "nice_to_have_relevance_criteria": nice,
        "clarification_questions": qns,
    }
    return RelevanceCriteria.model_validate(data)
