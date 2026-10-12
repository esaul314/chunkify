import json
import logging
import os
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from functools import reduce
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from pdf_chunker.adapters.request_budget import prepare_request_counter
from pdf_chunker.passes.ai_enrich import classify_chunk_utterance

yaml = cast(Any, import_module("yaml"))
DEFAULT_AI_ENRICH_MODEL = "qwen3-8b-local"
DEFAULT_AI_ENRICH_API_BASE = "http://localhost:8100/v1"
DEFAULT_AI_ENRICH_API_KEY = "sk-local"
DEFAULT_AI_ENRICH_MAX_TOKENS = 256
DEFAULT_AI_ENRICH_FALLBACK_MODEL = "gpt-4.1-nano"
DEFAULT_AI_ENRICH_FALLBACK_MAX_TOKENS = 128
logger = logging.getLogger(__name__)


class CompletionError(RuntimeError):
    """Raised when the external completion provider cannot answer."""


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


def _sanitize_tag_configs(raw: Mapping[str, Any]) -> dict[str, list[str]]:
    """Normalize caller-supplied tag configuration at the adapter boundary."""
    return {
        str(category): sorted({str(tag).strip().lower() for tag in values if tag is not None})
        for category, values in raw.items()
        if isinstance(values, (list, tuple, set)) and values
    }


def resolve_completion(
    options: Mapping[str, Any],
) -> tuple[Callable[[str], str] | None, str | None]:
    """Resolve a completion callback using adapter-owned configuration and I/O."""
    completion = options.get("completion_fn")
    if callable(completion):
        return completion, "completion_fn"
    try:
        return (
            init_llm(
                api_key=options.get("api_key"),
                model=options.get("model"),
                api_base=options.get("api_base") or options.get("base_url"),
                max_tokens=options.get("max_tokens"),
                no_think=options.get("no_think"),
                **(
                    {"request_budget": options["request_budget"]}
                    if "request_budget" in options
                    else {}
                ),
            ),
            "init_llm",
        )
    except (ImportError, OSError, ValueError) as exc:
        return None, f"init_llm_error:{exc.__class__.__name__}"


def resolve_fallback_completion(
    options: Mapping[str, Any],
) -> tuple[Callable[[str], str] | None, str | None]:
    """Resolve the optional fallback completion callback in the adapter."""
    if options.get("fallback_enabled") is False:
        return None, "fallback_disabled"
    try:
        completion = init_fallback_llm(
            api_key=options.get("fallback_api_key"),
            model=options.get("fallback_model"),
            api_base=options.get("fallback_api_base"),
            max_tokens=options.get("fallback_max_tokens"),
        )
    except (ImportError, OSError, ValueError) as exc:
        return None, f"init_fallback_llm_error:{exc.__class__.__name__}"
    return completion, "init_fallback_llm" if completion else "fallback_unconfigured"


def _tag_configs_for_options(options: Mapping[str, Any]) -> dict[str, list[str]]:
    explicit = options.get("tag_configs")
    if isinstance(explicit, Mapping):
        sanitized = _sanitize_tag_configs(explicit)
        if sanitized:
            return sanitized

    path_value = next(
        (options.get(key) for key in ("tags_dir", "tags_path", "tags_file") if options.get(key)),
        None,
    )
    if path_value:
        path = Path(str(path_value))
        if path.is_dir():
            loaded = _load_tag_configs(str(path))
        elif path.is_file():
            try:
                loaded = _sanitize_tag_configs(
                    yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                )
            except (OSError, TypeError, ValueError):
                loaded = {}
        else:
            loaded = {}
        if loaded:
            return loaded
    return _load_tag_configs()


def prepare_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Inject the configured enrichment client before a pass is executed."""
    prepared = dict(options)
    tag_configs = _tag_configs_for_options(prepared)
    prepared["tag_configs"] = tag_configs
    if prepared.get("client") is not None:
        return prepared

    completion, reason = resolve_completion(prepared)
    if completion is None:
        raise RuntimeError(
            "ai_enrich enabled but no completion client could be constructed "
            f"(reason={reason or 'unknown'})"
        )
    fallback_completion, fallback_reason = resolve_fallback_completion(prepared)
    prepared["client"] = Client(
        completion_fn=completion,
        tag_configs=tag_configs or None,
        fallback_completion_fn=fallback_completion,
        fallback_min_chars=int(prepared.get("fallback_min_chars", 200)),
    )
    prepared["client_source"] = (
        "adapter"
        if fallback_reason in {None, "fallback_unconfigured", "fallback_disabled"}
        else f"adapter+{fallback_reason}"
    )
    return prepared


def init_llm(
    api_key: str | None = None,
    model: str | None = None,
    api_base: str | None = None,
    max_tokens: int | None = None,
    no_think: bool | None = None,
    request_budget: Mapping[str, Any] | None = None,
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
    resolved_max_tokens = max_tokens or int(
        os.environ.get("PDF_CHUNKER_AI_ENRICH_MAX_TOKENS", DEFAULT_AI_ENRICH_MAX_TOKENS)
    )
    if no_think is None:
        raw_no_think = os.environ.get("PDF_CHUNKER_AI_ENRICH_NO_THINK")
        if raw_no_think is None:
            resolved_no_think = True
        else:
            resolved_no_think = raw_no_think.strip().lower() in {"1", "true", "yes", "on"}
    else:
        resolved_no_think = bool(no_think)
    if not key:
        raise ValueError(
            "No API key found for AI enrichment. "
            "Set PDF_CHUNKER_AI_ENRICH_API_KEY or OPENAI_API_KEY."
        )
    if type(resolved_max_tokens) is not int or resolved_max_tokens < 1:
        raise ValueError("AI max_tokens must be a positive integer")
    if request_budget is not None and not isinstance(request_budget, Mapping):
        raise ValueError("request_budget must be a mapping")
    budget_contract = dict(request_budget) if request_budget is not None else None
    request_counter = (
        prepare_request_counter(budget_contract, model=resolved_model)
        if budget_contract is not None
        else None
    )
    try:
        import litellm
    except ImportError as exc:  # pragma: no cover
        raise ImportError("litellm is required for init_llm") from exc

    def messages_for(prompt: str) -> list[dict[str, str]]:
        if resolved_no_think and resolved_model in {
            "qwen3-8b-local",
            "Qwen/Qwen3-8B-AWQ",
            "qwen3-14b-local",
            "Qwen/Qwen3-14B-AWQ",
        }:
            prompt = f"/no_think\n{prompt}"
        return [{"role": "user", "content": prompt}]

    def request_fits(prompt: str) -> bool:
        if request_counter is None or budget_contract is None:
            return True
        return (
            request_counter(messages_for(prompt))
            + resolved_max_tokens
            + budget_contract.get("margin_tokens", 64)
            <= budget_contract["context_tokens"]
        )

    def completion(prompt: str) -> str:
        messages = messages_for(prompt)
        kwargs = {
            "model": resolved_model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": resolved_max_tokens,
            "api_key": key,
        }
        if request_counter is not None and budget_contract is not None:
            input_tokens = request_counter(messages)
            context = budget_contract["context_tokens"]
            margin = budget_contract.get("margin_tokens", 64)
            if input_tokens + resolved_max_tokens + margin > context:
                logger.warning(
                    "ai_enrich request budget refused: input=%d output=%d margin=%d context=%d",
                    input_tokens,
                    resolved_max_tokens,
                    margin,
                    context,
                )
                raise CompletionError(
                    "AI request exceeds verified context budget: "
                    f"input={input_tokens}, output={resolved_max_tokens}, "
                    f"margin={margin}, context={context}"
                )
        if resolved_api_base:
            kwargs["api_base"] = resolved_api_base
            kwargs["custom_llm_provider"] = "openai"
        try:
            response = litellm.completion(**kwargs)
        except Exception as exc:
            raise CompletionError("AI completion request failed") from exc
        return response.choices[0].message.content

    # Optional pure capability: planning and transport use the SAME wire contract.
    # Unbudgeted/custom callbacks keep their previous one-request behavior.
    if request_counter is not None:
        cast(Any, completion).request_fits = request_fits
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
        api_base or os.environ.get("PDF_CHUNKER_AI_ENRICH_FALLBACK_API_BASE") or None
    )
    resolved_max_tokens = max_tokens or int(
        os.environ.get(
            "PDF_CHUNKER_AI_ENRICH_FALLBACK_MAX_TOKENS",
            DEFAULT_AI_ENRICH_FALLBACK_MAX_TOKENS,
        )
    )
    try:
        import litellm
    except ImportError as exc:  # pragma: no cover
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
        try:
            response = litellm.completion(**kwargs)
        except Exception as exc:
            raise CompletionError("fallback AI completion request failed") from exc
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
