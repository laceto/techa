"""
agents/_llm.py — Centralized OpenAI client and structured invocation helper.

Single source of truth for model name, client instantiation, and structured output calls.
Imported by all ask_* tool files. Do not instantiate openai.OpenAI() anywhere else.
"""

from __future__ import annotations

from typing import Type

from dotenv import load_dotenv
from pydantic import BaseModel
import openai

load_dotenv()

MODEL = "gpt-4.1-nano"   # structured-output worker model (all ask_* tools)
SYNTHESIS_MODEL = "gpt-4o"  # synthesis LLM used in _call_synthesis_llm nodes
_client = openai.OpenAI()


def invoke_structured(
    schema: Type[BaseModel],
    messages: list,
    max_tokens: int = 1024,
) -> BaseModel:
    """
    Call the OpenAI model with structured output and return the parsed Pydantic object.

    Args:
        schema:     Pydantic BaseModel class to use as response_format.
        messages:   List of OpenAI message dicts (role/content).
        max_tokens: Maximum tokens for the response (default 1024).

    Returns:
        Parsed Pydantic model instance.
    """
    response = _client.beta.chat.completions.parse(
        model=MODEL,
        max_tokens=max_tokens,
        messages=messages,
        response_format=schema,
    )
    return response.choices[0].message.parsed
