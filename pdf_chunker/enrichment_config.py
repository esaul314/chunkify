"""Pure, credential-free serving settings shared by enrichment adapters.

Resolve explicit options against an explicit environment mapping. These values
describe requests, not model availability, prompt/vocabulary identity or reuse
authority. File discovery, credentials, tokenizer loading and transport stay in
the adapter. Functions are data-first and support ``functools.partial``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

DEFAULT_AI_ENRICH_MODEL = "qwen3-8b-local"
DEFAULT_AI_ENRICH_API_BASE = "http://localhost:8100/v1"
DEFAULT_AI_ENRICH_MAX_TOKENS = 256
DEFAULT_AI_ENRICH_FALLBACK_MODEL = "gpt-4.1-nano"
DEFAULT_AI_ENRICH_FALLBACK_MAX_TOKENS = 128


@dataclass(frozen=True)
class CompletionSettings:
    """Resolved non-secret request settings, detached from ambient state.

    A fallback without ``api_base`` keeps the provider's existing default route.
    Fallback requests never receive the primary Qwen ``/no_think`` directive.
    """

    model: str
    api_base: str | None
    max_tokens: int
    no_think: bool


def _settings(model: Any, api_base: Any, max_tokens: int, no_think: Any) -> CompletionSettings:
    if not isinstance(model, str) or not model.strip():
        raise ValueError("AI model must be a nonempty string")
    if api_base is not None and (not isinstance(api_base, str) or not api_base.strip()):
        raise ValueError("AI api_base must be a nonempty string or None")
    if type(no_think) is not bool:
        raise ValueError("AI no_think must be a boolean")
    return CompletionSettings(model, api_base, max_tokens, no_think)


def _token_limit(explicit: Any, raw_default: str | int) -> int:
    # Only environment strings are parsed; explicit values must already be ints.
    try:
        value = int(raw_default) if explicit is None else explicit
    except (TypeError, ValueError) as exc:
        raise ValueError("AI max_tokens must be a positive integer") from exc
    if type(value) is not int or value < 1:
        raise ValueError("AI max_tokens must be a positive integer")
    return value


def resolve_primary_settings(
    options: Mapping[str, Any], *, environment: Mapping[str, str]
) -> CompletionSettings:
    """Resolve options → service/OpenAI environment aliases → local defaults.

    Preserve existing alias precedence and environment boolean spelling. Explicit
    token limits and booleans are strict: zero/False limits and string booleans
    refuse instead of silently selecting defaults or Python truthiness.
    Credentials and unrelated options are neither inspected nor retained.
    """
    raw_no_think = environment.get("PDF_CHUNKER_AI_ENRICH_NO_THINK")
    no_think = options.get("no_think")
    return _settings(
        options.get("model")
        or environment.get("PDF_CHUNKER_AI_ENRICH_MODEL")
        or environment.get("OPENAI_MODEL")
        or DEFAULT_AI_ENRICH_MODEL,
        options.get("api_base")
        or options.get("base_url")
        or environment.get("PDF_CHUNKER_AI_ENRICH_API_BASE")
        or environment.get("PDF_CHUNKER_AI_ENRICH_BASE_URL")
        or environment.get("OPENAI_API_BASE_URL")
        or DEFAULT_AI_ENRICH_API_BASE,
        _token_limit(
            options.get("max_tokens"),
            environment.get("PDF_CHUNKER_AI_ENRICH_MAX_TOKENS", DEFAULT_AI_ENRICH_MAX_TOKENS),
        ),
        (
            (raw_no_think is None or raw_no_think.strip().lower() in {"1", "true", "yes", "on"})
            if no_think is None
            else no_think
        ),
    )


def resolve_fallback_settings(
    options: Mapping[str, Any], *, environment: Mapping[str, str]
) -> CompletionSettings:
    """Resolve fallback-local option names against only fallback environment keys.

    This describes settings even without credentials; the adapter still decides
    whether fallback is available. Use ``model``, ``api_base`` and ``max_tokens``
    as for ``init_fallback_llm``, not the outer pass's ``fallback_*`` names.
    """
    return _settings(
        options.get("model")
        or environment.get("PDF_CHUNKER_AI_ENRICH_FALLBACK_MODEL")
        or DEFAULT_AI_ENRICH_FALLBACK_MODEL,
        options.get("api_base")
        or environment.get("PDF_CHUNKER_AI_ENRICH_FALLBACK_API_BASE")
        or None,
        _token_limit(
            options.get("max_tokens"),
            environment.get(
                "PDF_CHUNKER_AI_ENRICH_FALLBACK_MAX_TOKENS", DEFAULT_AI_ENRICH_FALLBACK_MAX_TOKENS
            ),
        ),
        False,
    )


def completion_request(prompt: str, *, settings: CompletionSettings) -> dict[str, Any]:
    """Build a fresh complete request without credentials or transport effects.

    Keep prompt text exact apart from the existing opt-in Qwen directive. The
    adapter adds its credential; budget counting and transport use these same
    messages. No token counting, truncation or provider defaults are inferred.
    """
    no_think = settings.no_think and settings.model in {
        "qwen3-8b-local",
        "Qwen/Qwen3-8B-AWQ",
        "qwen3-14b-local",
        "Qwen/Qwen3-14B-AWQ",
    }
    return {
        "model": settings.model,
        "messages": [{"role": "user", "content": f"/no_think\n{prompt}" if no_think else prompt}],
        "temperature": 0.0,
        "max_tokens": settings.max_tokens,
        **(
            {"api_base": settings.api_base, "custom_llm_provider": "openai"}
            if settings.api_base
            else {}
        ),
    }
