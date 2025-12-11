from __future__ import annotations
from typing import Dict, Any
from pydantic import BaseModel
from .llm_wrappers import call_prompt_json, call_chat_json

from .spec_models import (
    ExtractedContent,
    ExtractedAuthors,
    ExtractedTitle,
    ExtractedDate,
)

# ---------------------------------------------------------
# Simplified Prompts (Aligned With New System)
# ---------------------------------------------------------

CONTENT_PROMPT = ( 
    """ # Task Definition Given a query for finding papers about a specific topic, extract only the 
    content of the query, ignoring all metadata. Metadata includes: * Author/Coauthor name(s) * Year(s), 
    or words describing time (e.g., "recent", "latest") * Impact words (e.g., "central", "seminal", 
    "influential") * Venues (e.g., ACL, EMNLP, AAAI) * Search-process words (e.g., "run an exhaustive search") 
    Rules: * Keep phrases like "papers using", "papers proposing", "survey on" as part of content; drop 
    redundant prefixes like "papers about/on". * If the query is a question, extract a coherent representation 
    focusing on the topic. * Do not invent content; if unsure, return the original query minus metadata. 
    If only metadata is present, return an empty string. Return JSON: {"content": string}. Use "" when there 
    is no content. 
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
    {"content": ""} """ ).strip()

AUTHOR_PROMPT = """
Extract author names explicitly requested as authors of the papers.
Do NOT infer authors, and ignore organizations.

Guidance and Examples: 
{"query": "Graph-based Neural Multi-Document Summarization Yasunaga et al., 2017"} 
{"authors": ["Yasunaga"]} 
{"query": "papers on planning by Dan Weld"} 
{"authors": ["Dan Weld"]} 
{"query": "papers about transformer models by Google"} 
{"authors": []} Reason: Organizations are not authors.

Return JSON: {"authors": [ ... ]}
If none, return {"authors": []}.
""".strip()

TITLE_PROMPT = """
Extract an explicit paper title, if present.
If no title is explicitly given, return {"title": ""}.

Return JSON: {"title": string}
""".strip()

DATE_PROMPT = """
Extract a publication year range from the query.

Rules:
- "2018" → year_min=2018, year_max=2018
- "2014-2019" → start/end
- "since 2020" or "after 2020" → year_min=2020, year_max=2025 (current year)
- "any time" or no explicit years → both null

Return JSON:
{
  "year_min": int|null,
  "year_max": int|null
}
""".strip()

# ---------------------------------------------------------
# Extractor Functions
# ---------------------------------------------------------

def extract_content(query: str) -> ExtractedContent:
    prompt = f"{CONTENT_PROMPT}\n\n{{\"query\": {__import__('json').dumps(query)} }}"
    return call_prompt_json(prompt, ExtractedContent)

def extract_authors(query: str) -> ExtractedAuthors:
    prompt = f"{AUTHOR_PROMPT}\n\n{{\"query\": {__import__('json').dumps(query)} }}"
    return call_prompt_json(prompt, ExtractedAuthors)

def extract_title(query: str) -> ExtractedTitle:
    prompt = f"{TITLE_PROMPT}\n\n{{\"query\": {__import__('json').dumps(query)} }}"
    return call_prompt_json(prompt, ExtractedTitle)

def extract_date_range(query: str) -> ExtractedDate:
    messages = [
        {"role": "system", "content": DATE_PROMPT},
        {"role": "user", "content": __import__('json').dumps({"query": query})},
    ]
    return call_chat_json(messages, ExtractedDate)
