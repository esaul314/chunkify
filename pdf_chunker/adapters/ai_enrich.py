import json
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import reduce
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from pdf_chunker.passes.ai_enrich import classify_chunk_utterance

yaml = cast(Any, import_module("yaml"))
DEFAULT_AI_ENRICH_MODEL = "qwen3-8b-local"
DEFAULT_AI_ENRICH_API_BASE = "http://localhost:8100/v1"
DEFAULT_AI_ENRICH_API_KEY = "sk-local"
DEFAULT_AI_ENRICH_MAX_TOKENS = 256
DEFAULT_AI_ENRICH_FALLBACK_MODEL = "gpt-4.1-nano"
DEFAULT_AI_ENRICH_FALLBACK_MAX_TOKENS = 128


class Client:
    """Simple adapter wrapping ``classify_chunk_utterance`` for the pass."""

    def __init__(
        self,
        *,
        completion_fn: Callable[[str], str],
        tag_configs: dict[str, list[str]] | None = None,
        fallback_completion_fn: Callable[[str], str] | None = None,
        fallback_min_chars: int = 200,
    ) -> None:
        self._completion_fn = completion_fn
        self._tag_configs: dict[str, list[str]] = tag_configs or _load_tag_configs()
        self._fallback_completion_fn = fallback_completion_fn
        self._fallback_min_chars = fallback_min_chars

    def classify_chunk_utterance(
        self, text: str, *, tag_configs: dict[str, list[str]] | None = None
    ) -> dict:
        return classify_chunk_utterance(
            text,
            tag_configs=tag_configs or self._tag_configs,
            completion_fn=self._completion_fn,
            fallback_completion_fn=self._fallback_completion_fn,
            fallback_min_chars=self._fallback_min_chars,
        )


def _resolve_config_path(config_dir: str) -> Path:
    """Resolve tag config directory for repo and installed-package layouts."""
    config_path = Path(config_dir)
    if config_path.is_absolute():
        return config_path

    package_root = Path(__file__).resolve().parent.parent
    repo_root = package_root.parent
    candidates = (
        package_root / config_path,
        repo_root / config_path,
        Path.cwd() / config_path,
    )
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def _load_tag_configs(config_dir: str = "config/tags") -> dict[str, list[str]]:
    """Merge YAML tag configurations into a single dictionary."""
    config_path = _resolve_config_path(config_dir)
    if not config_path.exists():
        return {}

    def load_yaml(path: Path) -> dict[str, list[str]]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
                return {k: v for k, v in data.items() if isinstance(v, list)}
        except FileNotFoundError:
            return {}

    def merge_dicts(
        acc: dict[str, list[str]],
        nxt: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        keys = set(acc).union(nxt)
        return {key: acc.get(key, []) + nxt.get(key, []) for key in keys}

    merged: dict[str, list[str]] = reduce(
        merge_dicts,
        map(load_yaml, config_path.glob("*.yaml")),
        cast(dict[str, list[str]], {}),
    )

    return {
        k: sorted({tag.strip().lower() for tag in v if isinstance(tag, str)})
        for k, v in merged.items()
    }


def init_llm(
    api_key: str | None = None,
    model: str | None = None,
    api_base: str | None = None,
    max_tokens: int | None = None,
) -> Callable[[str], str]:
    """Return a completion function configured with an API key."""
    from dotenv import load_dotenv  # lazy import

    load_dotenv()
    resolved_model = (
        model
        or os.environ.get("PDF_CHUNKER_AI_ENRICH_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or DEFAULT_AI_ENRICH_MODEL
    )
    resolved_api_base = (
        api_base
        or os.environ.get("PDF_CHUNKER_AI_ENRICH_API_BASE")
        or os.environ.get("PDF_CHUNKER_AI_ENRICH_BASE_URL")
        or os.environ.get("OPENAI_API_BASE_URL")
        or DEFAULT_AI_ENRICH_API_BASE
    )
    prefers_local_key = (
        "localhost:8100" in resolved_api_base
        or "127.0.0.1:8100" in resolved_api_base
        or "host.docker.internal:8100" in resolved_api_base
    )
    key = (
        api_key
        or os.environ.get("PDF_CHUNKER_AI_ENRICH_API_KEY")
        or (DEFAULT_AI_ENRICH_API_KEY if prefers_local_key else None)
        or os.environ.get("OPENAI_API_KEY")
    )
    resolved_max_tokens = (
        max_tokens
        or int(os.environ.get("PDF_CHUNKER_AI_ENRICH_MAX_TOKENS", DEFAULT_AI_ENRICH_MAX_TOKENS))
    )
    if not key:
        raise ValueError(
            "No API key found for AI enrichment. Set PDF_CHUNKER_AI_ENRICH_API_KEY or OPENAI_API_KEY."
        )
    try:
        import litellm
    except Exception as exc:  # pragma: no cover
        raise ImportError("litellm is required for init_llm") from exc

    def completion(prompt: str) -> str:
        if resolved_model in {"qwen3-8b-local", "Qwen/Qwen3-8B-AWQ"}:
            prompt = f"/no_think\n{prompt}"
        kwargs = {
            "model": resolved_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": resolved_max_tokens,
            "api_key": key,
        }
        if resolved_api_base:
            kwargs["api_base"] = resolved_api_base
            kwargs["custom_llm_provider"] = "openai"
        response = litellm.completion(**kwargs)
        return response.choices[0].message.content

    return completion


def init_fallback_llm(
    api_key: str | None = None,
    model: str | None = None,
    api_base: str | None = None,
    max_tokens: int | None = None,
) -> Callable[[str], str] | None:
    """Return a fallback completion function or ``None`` when disabled."""
    from dotenv import load_dotenv  # lazy import

    load_dotenv()
    resolved_key = (
        api_key
        or os.environ.get("PDF_CHUNKER_AI_ENRICH_FALLBACK_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not resolved_key:
        return None

    resolved_model = (
        model
        or os.environ.get("PDF_CHUNKER_AI_ENRICH_FALLBACK_MODEL")
        or DEFAULT_AI_ENRICH_FALLBACK_MODEL
    )
    resolved_api_base = (
        api_base
        or os.environ.get("PDF_CHUNKER_AI_ENRICH_FALLBACK_API_BASE")
        or None
    )
    resolved_max_tokens = (
        max_tokens
        or int(
            os.environ.get(
                "PDF_CHUNKER_AI_ENRICH_FALLBACK_MAX_TOKENS",
                DEFAULT_AI_ENRICH_FALLBACK_MAX_TOKENS,
            )
        )
    )
    try:
        import litellm
    except Exception as exc:  # pragma: no cover
        raise ImportError("litellm is required for init_fallback_llm") from exc

    def completion(prompt: str) -> str:
        kwargs = {
            "model": resolved_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": resolved_max_tokens,
            "api_key": resolved_key,
        }
        if resolved_api_base:
            kwargs["api_base"] = resolved_api_base
            kwargs["custom_llm_provider"] = "openai"
        response = litellm.completion(**kwargs)
        return response.choices[0].message.content

    return completion


def _process_chunk_for_file(
    chunk: dict,
    *,
    tag_configs: dict[str, list[str]],
    completion_fn: Callable[[str], str],
) -> dict:
    """Helper to wrap utterance classification and tagging for file processing."""
    result = classify_chunk_utterance(
        chunk.get("text", ""), tag_configs=tag_configs, completion_fn=completion_fn
    )
    chunk = {
        **chunk,
        "utterance_type": result["classification"],
        "tags": result["tags"],
    }
    meta = {
        **chunk.get("metadata", {}),
        "utterance_type": result["classification"],
        "tags": result["tags"],
    }
    return {**chunk, "metadata": meta}


def _process_jsonl_file(
    input_path: str,
    output_path: str,
    completion_fn: Callable[[str], str],
    tag_configs: dict[str, list[str]] | None = None,
    max_workers: int = 10,
) -> None:
    """Read ``input_path`` JSONL, classify chunks, and write to ``output_path``."""
    tag_configs = tag_configs or _load_tag_configs()
    import logging

    logger = logging.getLogger(__name__)

    with (
        open(input_path, encoding="utf-8") as infile,
        open(output_path, "w", encoding="utf-8") as outfile,
    ):
        chunks = [json.loads(line) for line in infile]
        total = len(chunks)
        processed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for chunk in ex.map(
                lambda c: _process_chunk_for_file(
                    c, tag_configs=tag_configs, completion_fn=completion_fn
                ),
                chunks,
            ):
                outfile.write(json.dumps(chunk) + "\n")
                processed += 1
                if processed % 10 == 0 or processed == total:
                    logger.info("Processed %d/%d chunks", processed, total)
