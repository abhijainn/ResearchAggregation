# llm_summary.py
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

# Load .env so OPENAI_API_KEY is available in os.environ
load_dotenv()

@lru_cache(maxsize=1)
def _client() -> OpenAI:
    # Reads OPENAI_API_KEY from environment
    return OpenAI()

def summarize_title(title: str, max_tokens: int = 220) -> Optional[str]:
    """
    Given a paper title, generate a concise technical summary.
    Returns None on failure.
    """
    title = (title or "").strip()
    if not title:
        return None

    prompt = (
        "You are a precise research assistant. Given the paper title, "
        " produce a concise 4–6 sentence summary that infers the problem, method, "
        "and contributions.\n\n"
        f"Title: {title}\n\n"
        "Return only the summary and ensure it describes the full paper well"
    )

    try:
        resp = _client().responses.create(
            model="gpt-4o-mini",
            input=prompt,
            max_output_tokens=max_tokens,
        )
        return (resp.output_text or "").strip() or None
    except Exception:
        return None
    
__all__ = [
    "summarize_title"
]
