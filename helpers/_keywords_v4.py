"""
Keyword builder v4: use a simulated answer string directly for retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .QuerySpec_v4 import analyze_query_llm_v4

__all__ = ["Query", "extract_queries"]


@dataclass
class Query:
    text: str

    def to_dict(self) -> Dict[str, str]:
        return {"text": self.text}


def extract_queries(question: str) -> Tuple[List[Query], Dict[str, object]]:
    spec = analyze_query_llm_v4(question)
    answer_text = spec.simulated_answer or question.strip()
    queries = [Query(text=answer_text)]
    debug = {"spec": spec.raw, "queries": [q.to_dict() for q in queries]}
    return queries, debug
