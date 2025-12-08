import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore


@dataclass
class QuerySpecV4:
    simulated_answer: str
    raw: Dict[str, Any]


PROMPT_TEMPLATE = """You are helping with semantic search over academic papers.
Your task is to rewrite a user’s research query as a single sentence that looks like a verbatim quote from a relevant academic paper. This quote will be used as a query to find matching passages. Ground your rewrite in the provided abstract when possible.

Requirements:
	•	Preserve all important technical terms from the user’s query (e.g., “agentic framework”, “RAG”, “transformers”, “variational inference”).
	•	Write in a formal, academic tone, as if it appears in the body of a research article.
	•	Do not mention the user, queries, questions, or “this paper”. Just state the content.
	•	The sentence should be self-contained and understandable without any external context.
	•	Maximum length: 1–3 sentences (no more than 60 words total).

Output:
	•	Return only the quote text, with no explanations or additional formatting.

User query:
“{USER_QUERY}”

Paper abstract (context):
“{ABSTRACT}”

Example behavior (for you, the model, not to be repeated in the output):
	•	User query: “find me papers that incorporate the agentic framework with RAG”
	•	Good quote: “In this work, we integrate an agentic framework with retrieval-augmented generation (RAG) to enable autonomous decision-making and tool use grounded in external knowledge sources.”

Respond with pure JSON: {{"simulated_answer": "<text>"}}"""


def _call_llm(prompt: str) -> Dict[str, Any]:
    if OpenAI is None:
        raise ImportError("OpenAI SDK not installed for QuerySpec_v4.")
    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    content = resp.choices[0].message.content
    if not content:
        raise ValueError("Empty response from LLM")
    return json.loads(content)


def analyze_query_llm_v4(user_query: str, *, abstract: Optional[str] = None) -> QuerySpecV4:
    prompt = PROMPT_TEMPLATE.format(
        USER_QUERY=user_query.strip(),
        ABSTRACT=(abstract or "").strip(),
    )
    raw = _call_llm(prompt)
    answer = raw.get("simulated_answer") if isinstance(raw, dict) else None
    if not isinstance(answer, str) or not answer.strip():
        answer = user_query.strip()
    return QuerySpecV4(simulated_answer=answer.strip(), raw=raw if isinstance(raw, dict) else {})
