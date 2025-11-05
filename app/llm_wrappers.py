from __future__ import annotations

from typing import Dict, List, Any, Type, Sequence
from pydantic import BaseModel
from openai import OpenAI

_client: OpenAI | None = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client

def call_prompt_json(prompt: str, output_model: Type[BaseModel], temperature: float = 0.0) -> BaseModel:
    client = _get_client()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Return a valid json object only."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    content = resp.choices[0].message.content
    if content is None:
        raise ValueError("LLM returned empty content for JSON response")
    if isinstance(content, dict):
        data = content
    else:
        # Some SDKs return string for json_object; parse defensively
        data = __import__("json").loads(str(content))
    return output_model.model_validate(data)

def call_chat_json(messages: Sequence[Dict[str, str]], output_model: Type[BaseModel], temperature: float = 0.0) -> BaseModel:
    client = _get_client()
    # Prepend a system cue that contains the word 'json' to satisfy API requirements
    msgs: list[Dict[str, str]] = [{"role": "system", "content": "Always respond with a valid json object only."}] + list(messages)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=msgs,
        temperature=temperature,
    )
    content = resp.choices[0].message.content
    if content is None:
        raise ValueError("LLM returned empty content for JSON response")
    if isinstance(content, dict):
        data = content
    else:
        data = __import__("json").loads(str(content))
    return output_model.model_validate(data)
