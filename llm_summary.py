# llm_summary.py
from functools import lru_cache
from typing import Optional

import requests
from io import BytesIO
from PyPDF2 import PdfReader


from dotenv import load_dotenv
from openai import OpenAI

# Load .env so OPENAI_API_KEY is available in os.environ
load_dotenv()

def fetch_paper_metadata(title: str) -> Optional[dict]:
    """
    Searches Semantic Scholar for a paper by title and returns metadata
    including abstract and PDF link if available.
    """
    try:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": title,
            "limit": 1,
            "fields": "title,abstract,openAccessPdf,url"
        }
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        if data.get("data"):
            return data["data"][0]
        return None
    except Exception as e:
        print(f"[Semantic Scholar] Error fetching metadata: {e}")
        return None
def extract_pdf_text(pdf_url: str, max_chars: int = 20000) -> str:
    """
    Downloads and extracts text from the first ~20k characters of a PDF.
    """
    if not pdf_url:
        return ""
    try:
        r = requests.get(pdf_url, timeout=20)
        reader = PdfReader(BytesIO(r.content))
        text = " ".join(page.extract_text() or "" for page in reader.pages)
        return text[:max_chars]
    except Exception as e:
        print(f"[PDF Extract] Failed: {e}")
        return ""

@lru_cache(maxsize=1)
def _client() -> OpenAI:
    
    return OpenAI()

def summarize_title(title: str, max_tokens: int = 220) -> Optional[str]:
    """
    Given a paper title, generate a concise technical summary.
    Returns None on failure.
    """
    title = (title or "").strip()
    if not title:
        return None
    
    metadata = fetch_paper_metadata(title)
    if metadata:
        pdf_url = metadata.get("openAccessPdf", {}).get("url")
        abstract = metadata.get("abstract") or ""
        text = extract_pdf_text(pdf_url) if pdf_url else abstract
    else:
        text = ""

    context = text[:12000] or "(No full text or abstract found.)"
    prompt = (
        "You are a precise research assistant. Given the paper title and text, "
        "produce a concise 4–6 sentence summary describing the problem, method, "
        "and contributions.\n\n"
        f"Title: {title}\n\n"
        f"Context:\n{context}\n\n"
        "Return only the summary."
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
