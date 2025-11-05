from __future__ import annotations

from .spec_models import ExtractedFields
from .extractors import (
    extract_content,
    extract_authors,
    extract_venues,
    extract_recency,
    extract_centrality,
    extract_time_range,
    extract_broad_or_specific,
    extract_by_name_or_title,
    identify_relevance_criteria,
    identify_domains,
    check_refusal,
)


def decompose_query(query: str) -> ExtractedFields:
    content = extract_content(query)
    authors = extract_authors(query)
    venues = extract_venues(query)
    recency = extract_recency(query)
    centrality = extract_centrality(query)
    time_range = extract_time_range(query)
    broad_or_specific = extract_broad_or_specific(query)
    by_name_or_title = extract_by_name_or_title(query)
    relevance_criteria = identify_relevance_criteria(query)
    domains = identify_domains(query)
    possible_refusal = check_refusal(query)

    return ExtractedFields(
        content=content,
        authors=authors,
        venues=venues,
        recency=recency,
        centrality=centrality,
        time_range=time_range,
        broad_or_specific=broad_or_specific,
        by_name_or_title=by_name_or_title,
        relevance_criteria=relevance_criteria,
        domains=domains,
        possible_refusal=possible_refusal,
    )
