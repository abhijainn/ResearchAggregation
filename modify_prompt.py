"""
========================================================================================================================================
Usage: get_claim(query) takes a query str as input and outputs a str that contains the body of the hypothesis.
========================================================================================================================================

Change model as needed.

Notes: 
 - gpt-5-nano does not take temp and max tokens as parameters.
 - v1 on 10/12/2025: implemented basic functions
"""

# Imports
import os
from typing import Dict, List, Optional
from openai import OpenAI

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=False)
except Exception:
    pass

# Globals
MODEL_NAME = "gpt-5-nano"

_client: Optional[OpenAI] = None

SYSTEM_PROMPT = (
    "You are a meticulous scientific writing assistant.\n"
    "Write plausible research paper abstracts in 2-3 complete sentences.\n"
    "Mirror the tone of peer-reviewed scientific literature, focusing on objectives, methodology, and key findings.\n"
    "Align your response with the user's claim. Do not attempt to correct the user if their claim is wrong.\n"
    "Avoid citations, hedging, and unnecessary background context.\n"
    "Do not invent overly specific experimental details."
)


def _get_openai_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("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")
        if api_key:
            _client = OpenAI(api_key=api_key)
        else:
            try:
                _client = OpenAI()
            except Exception as e:
                raise EnvironmentError(
                    "OPENAI_API_KEY is not set. Create a .env file in your project root with 'OPENAI_API_KEY=sk-...' or export it in your shell before running."
                ) from e
    return _client

def _normalize_query(query: str) -> str:
    cleaned = (query or "").strip()
    if not cleaned:
        raise ValueError("Query must be a non-empty string.")
    return cleaned

def _build_messages(query: str) -> List[Dict[str, List[Dict[str, str]]]]:
    normalized_query = _normalize_query(query)

    user_prompt = (
        f"User Claim:\n{normalized_query}\n\n"
        "Write a 3-4 sentence abstract that aligns with the chosen stance."
        "ABSTRACT: "
    )

    return [
        {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
    ]

def _call_model(messages: List[Dict[str, List[Dict[str, str]]]]) -> str:
    client = _get_openai_client()
    request_messages = list(messages)
    response = client.responses.create(
        model=MODEL_NAME,
        input=request_messages,
    )

    output_text = response.output_text
    if output_text == None:
        raise RuntimeError("Model response did not contain any text output.")
    return output_text.strip()
    

def get_claim(query: str) -> str:
    messages = _build_messages(query)
    raw_text = _call_model(messages)
    return raw_text

__all__ = [
    "MODEL_NAME",
    "SYSTEM_PROMPT",
    "get_claim",
]

# Test
if __name__ == "__main__":

    from pprint import pprint

    test_claim = "RAG hinders output accuracy of LLMs"

    print("ENV check OPENAI_API_KEY present:", bool(os.getenv("OPENAI_API_KEY")))
    print("=== MODEL ===")
    print(MODEL_NAME)
    print("\n=== MESSAGES ===")
    pprint(_build_messages(test_claim))

    print("\n=== ABSTRACT ===")
    try:
        abstract = get_claim(
            test_claim
        )
        print(abstract)
    except Exception as e:
        print("Error while generating abstract:", repr(e))
