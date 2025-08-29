from __future__ import annotations

import os
from importlib import import_module
from typing import Any, Callable

from dotenv import load_dotenv

MODEL = "gpt-3.5-turbo"


def _load_completion() -> Callable[..., Any]:
    load_dotenv()
    llm = import_module("litellm")
    llm.api_key = os.getenv("OPENAI_API_KEY")
    return llm.completion


def _build_prompt(word: str, snippet: str) -> str:
    return (
        f"The following snippet contains the possibly erroneous word '{word}':\n\n"
        f'"{snippet}"\n\n'
        "If the word results from words glued together accidentally, correct it. "
        "Otherwise, return the original word unchanged.\n\n"
        "Reply ONLY with the corrected or original word."
    )


def correct_word(word: str, snippet: str) -> str:
    completion = _load_completion()
    response = completion(
        model=MODEL,
        messages=[{"role": "user", "content": _build_prompt(word, snippet)}],
        temperature=0.0,
        max_tokens=100,
    )
    return response.choices[0].message.content.strip()
