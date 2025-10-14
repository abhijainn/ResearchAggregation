"""
========================================================================================================================================
Usage: get_claim(query) takes a query str as input and outputs a str that contains the body of the hypothesis.
========================================================================================================================================

Change model as needed.

Notes: 
 - gpt-5-nano does not take temp and max tokens as parameters.
 - v1 on 10/12/2025: implemented basic functions
 - v2 on 10/13/2025: added code to pull from config file, removed API key
"""

# Imports
import os
from typing import Dict, List, Optional
from openai import OpenAI

from ._log import log

try:
    from app.config import CONFIG  # type: ignore
except ImportError:
    from config import CONFIG  # type: ignore

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=False)
except Exception:
    pass

# Globals
MODEL_NAME = CONFIG.llm.model_name
SYSTEM_PROMPT = CONFIG.prompts.abstract_system_prompt
USER_PROMPT_TEMPLATE = CONFIG.prompts.abstract_user_prompt_template

_client: Optional[OpenAI] = None

def _get_openai_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            _client = OpenAI(api_key=api_key)
        else:
            try:
                _client = OpenAI()
            except Exception as e:
                raise EnvironmentError(
                    "OPENAI_API_KEY is not set. Create a .env file in your project root with 'OPENAI_API_KEY=...' or export it in your shell before running."
                ) from e
    return _client

def _normalize_query(query: str) -> str:
    cleaned = (query or "").strip()
    if not cleaned:
        raise ValueError("Query must be a non-empty string.")
    return cleaned

def _build_messages(query: str) -> List[Dict[str, List[Dict[str, str]]]]:
    normalized_query = _normalize_query(query)

    user_prompt = USER_PROMPT_TEMPLATE.format(claim=normalized_query)

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
    "USER_PROMPT_TEMPLATE",
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
