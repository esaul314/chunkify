"""Offline request counting; tokenizer I/O stays out of the enrichment pass."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


def _positive_integer(value: Any, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"request_budget.{name} must be an integer >= {minimum}")
    return value


def prepare_request_counter(
    budget: Mapping[str, Any], *, model: str
) -> Callable[[Sequence[Mapping[str, str]]], int]:
    """Load only hash-pinned cached Qwen files; never download or load weights.

    The caller must verify this explicit revision/context/template contract
    against its serving profile. Tokenizer model_max_length is not admission.
    """
    if not isinstance(budget, Mapping):
        raise ValueError("request_budget must be a mapping")
    tokenizer_model = budget.get("model")
    revision = budget.get("revision")
    if tokenizer_model != "Qwen/Qwen3-8B-AWQ" or model not in {
        "qwen3-8b-local",
        "Qwen/Qwen3-8B-AWQ",
    }:
        raise ValueError("request_budget model does not match the configured Qwen 8B provider")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("request_budget requires an exact cached tokenizer revision")
    context = _positive_integer(budget.get("context_tokens"), "context_tokens")
    margin = _positive_integer(budget.get("margin_tokens", 64), "margin_tokens", 0)
    if margin >= context:
        raise ValueError("request_budget margin must be smaller than the context")
    if budget.get("enable_thinking") is not False:
        raise ValueError("request_budget requires the verified disabled-thinking template")

    hf_home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    cache = Path(os.environ.get("HF_HUB_CACHE", str(hf_home / "hub")))
    snapshot = cache / "models--Qwen--Qwen3-8B-AWQ" / "snapshots" / revision

    def pinned_bytes(filename: str, field: str) -> bytes:
        expected = budget.get(field)
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"request_budget.{field} requires an exact SHA256")
        data = (snapshot / filename).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"request_budget cached {filename} hash mismatch")
        return data

    # These optional enrichment dependencies are loaded only for opted-in callers.
    from jinja2 import Environment
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_str(pinned_bytes("tokenizer.json", "tokenizer_sha256").decode())
    # Counting must never honor a saved truncation/padding policy.
    tokenizer.no_truncation()
    tokenizer.no_padding()
    config = json.loads(pinned_bytes("tokenizer_config.json", "tokenizer_config_sha256"))
    template_text = config.get("chat_template")
    if not isinstance(template_text, str) or not template_text:
        raise ValueError("request_budget cached chat template is missing")
    template = Environment().from_string(template_text)

    def count(messages: Sequence[Mapping[str, str]]) -> int:
        rendered = template.render(
            messages=messages,
            tools=None,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return len(tokenizer.encode(rendered, add_special_tokens=False).ids)

    return count
