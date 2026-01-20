import ast
import logging
import os
import re
from collections.abc import Callable, Iterable, Mapping
from collections.abc import Iterable as TypingIterable
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from os import PathLike
from pathlib import Path
from typing import Any, Protocol

from pdf_chunker.framework import Artifact, register

logger = logging.getLogger(__name__)
_DEBUG_SAMPLE_COUNT = 0
_DEBUG_SAMPLE_LIMIT: int | None = None

Chunk = dict[str, Any]
Chunks = list[Chunk]


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _normalized_tags(values: Any) -> list[str]:
    if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
        return []
    return [
        normalized
        for normalized in (str(item).strip().lower() for item in values if item is not None)
        if normalized
    ]


def _sanitize_tag_configs(raw: Mapping[str, Any]) -> dict[str, list[str]]:
    return {
        str(category): _normalized_tags(values)
        for category, values in raw.items()
        if _normalized_tags(values)
    }


def _copy_tag_configs(configs: Mapping[str, Iterable[str]]) -> dict[str, list[str]]:
    return {str(category): [*tags] for category, tags in configs.items()}


@cache
def _load_default_tag_configs() -> dict[str, list[str]]:
    from pdf_chunker.adapters.ai_enrich import _load_tag_configs as load_dir

    return _copy_tag_configs(load_dir())


def _load_tag_configs_from_path(path: Path) -> dict[str, list[str]]:
    if path.is_dir():
        from pdf_chunker.adapters.ai_enrich import _load_tag_configs as load_dir

        return _copy_tag_configs(load_dir(config_dir=str(path)))
    if path.is_file():
        try:
            import yaml
        except Exception:  # pragma: no cover - optional dependency missing
            return {}
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:  # pragma: no cover - defensive read guard
            return {}
        return _sanitize_tag_configs(_as_mapping(data))
    return {}


def _debug_enabled() -> bool:
    return os.getenv("PDF_CHUNKER_AI_ENRICH_DEBUG", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _debug_sample_limit() -> int:
    global _DEBUG_SAMPLE_LIMIT
    if _DEBUG_SAMPLE_LIMIT is not None:
        return _DEBUG_SAMPLE_LIMIT
    raw = os.getenv("PDF_CHUNKER_AI_ENRICH_DEBUG_SAMPLES", "1").strip()
    try:
        _DEBUG_SAMPLE_LIMIT = max(0, int(raw))
    except ValueError:
        _DEBUG_SAMPLE_LIMIT = 1
    return _DEBUG_SAMPLE_LIMIT


def _debug_next_sample() -> bool:
    global _DEBUG_SAMPLE_COUNT
    limit = _debug_sample_limit()
    if limit <= 0 or limit <= _DEBUG_SAMPLE_COUNT:
        return False
    _DEBUG_SAMPLE_COUNT += 1
    return True


def _debug_preview(text: str, limit: int = 300) -> str:
    cleaned = " ".join(text.split())
    return cleaned[:limit] + ("…" if len(cleaned) > limit else "")


def _tag_config_stats(configs: Mapping[str, Iterable[str]]) -> tuple[int, int]:
    return len(configs), sum(len(tags) for tags in configs.values())


def _resolve_tag_configs(
    options: Mapping[str, Any],
    cached: dict[str, list[str]] | None,
) -> tuple[dict[str, list[str]], str]:
    explicit = options.get("tag_configs")
    if isinstance(explicit, Mapping):
        sanitized = _sanitize_tag_configs(explicit)
        if sanitized:
            return sanitized, "explicit"

    path_value = next(
        (options.get(key) for key in ("tags_dir", "tags_path", "tags_file") if options.get(key)),
        None,
    )
    if isinstance(path_value, (str, Path, PathLike)):
        loaded = _load_tag_configs_from_path(Path(path_value))
        if loaded:
            return loaded, f"path:{path_value}"

    if cached:
        return _copy_tag_configs(cached), "cached"

    return _load_default_tag_configs(), "default"


def _resolve_options(meta: Mapping[str, Any] | None) -> dict[str, Any]:
    base = _as_mapping(meta)
    nested = _as_mapping(base.get("options"))
    staged = _as_mapping(nested.get("ai_enrich"))
    direct = _as_mapping(base.get("ai_enrich"))
    return {**staged, **direct}


def _resolve_completion(
    options: Mapping[str, Any],
) -> tuple[Callable[[str], str] | None, str | None]:
    completion = options.get("completion_fn")
    if callable(completion):
        return completion, "completion_fn"
    try:
        from pdf_chunker.adapters.ai_enrich import init_llm
    except Exception:  # pragma: no cover - adapters unavailable
        return None, "missing_adapters"
    try:
        return init_llm(api_key=options.get("api_key")), "init_llm"
    except Exception as exc:  # pragma: no cover - missing credentials
        return None, f"init_llm_error:{exc.__class__.__name__}"


def _ensure_client(
    options: Mapping[str, Any],
    fallback: "ClassifyClient | None",
    tag_configs: dict[str, list[str]],
    *,
    completion_reason: str | None,
) -> tuple["ClassifyClient | None", str | None]:
    client = options.get("client")
    if client:
        return client, "client_option"  # type: ignore[return-value]
    if fallback:
        return fallback, "client_cached"

    completion, reason = _resolve_completion(options)
    if not completion:
        if _debug_enabled():
            logger.warning(
                "ai_enrich debug: no completion client (reason=%s)",
                reason or completion_reason or "unknown",
            )
        return None, reason or completion_reason or "missing_completion"
    try:
        from pdf_chunker.adapters.ai_enrich import Client
    except Exception:  # pragma: no cover - adapters unavailable
        if _debug_enabled():
            logger.warning("ai_enrich debug: adapters unavailable for Client")
        return None, "missing_adapters"
    try:
        return Client(completion_fn=completion, tag_configs=tag_configs or None), "client_ready"
    except Exception as exc:  # pragma: no cover - client init failure
        if _debug_enabled():
            logger.warning("ai_enrich debug: Client initialization failed")
        return None, f"client_init_error:{exc.__class__.__name__}"


def _iterable_items(value: Any) -> list[Any]:
    return (
        list(value) if isinstance(value, Iterable) and not isinstance(value, (str, bytes)) else []
    )


def _payload_chunks(payload: Any) -> tuple[Chunks, Callable[[Chunks], Any]]:
    if isinstance(payload, list):
        items = list(payload)
        indices = [i for i, item in enumerate(items) if isinstance(item, Mapping)]
        if not indices:
            return [], lambda _: items
        selected = set(indices)

        def rebuild(enriched: Chunks) -> list[Any]:
            replacements = iter(enriched)
            return [
                next(replacements) if index in selected else item
                for index, item in enumerate(items)
            ]

        chunks = [dict(items[i]) for i in indices]
        return chunks, rebuild

    if isinstance(payload, Mapping):
        items = _iterable_items(payload.get("items"))
        if not items:
            return [], lambda _: payload
        indices = [i for i, item in enumerate(items) if isinstance(item, Mapping)]
        if not indices:
            base = {k: v for k, v in payload.items() if k != "items"}
            return [], lambda _: {**base, "items": items}
        selected = set(indices)
        base = {k: v for k, v in payload.items() if k != "items"}

        def rebuild(enriched: Chunks) -> dict[str, Any]:
            replacements = iter(enriched)
            updated = [
                next(replacements) if index in selected else item
                for index, item in enumerate(items)
            ]
            return {**base, "items": updated}

        chunks = [dict(items[i]) for i in indices]
        return chunks, rebuild

    return [], lambda _: payload


UTTERANCE_TYPES = [
    "definition",
    "explanation",
    "instruction",
    "example",
    "opinion",
    "statement_of_fact",
    "question",
    "summary",
    "critique",
    "unclassified",
]


def _normalized_classification(value: str) -> str:
    candidate = value.strip().lower()
    return candidate if candidate in UTTERANCE_TYPES else "unclassified"


def _valid_tags(tag_configs: Mapping[str, TypingIterable[str]]) -> set[str]:
    return {tag for tags in tag_configs.values() for tag in tags if isinstance(tag, str)}


def _response_lines(response_text: str) -> list[str]:
    return [line.strip() for line in response_text.splitlines()]


def _should_continue_tag_block(line: str) -> bool:
    if not line:
        return False
    lower = line.lower()
    if lower.startswith("classification:") or lower.startswith("response:"):
        return False
    if ":" in line and not line.startswith(("-", "*", "•")):
        return False
    return True


def _collect_tag_block(lines: list[str], start_index: int) -> list[str]:
    first = lines[start_index]
    remainder = first.split(":", 1)[1].strip() if ":" in first else ""
    block = [remainder] if remainder else []
    for line in lines[start_index + 1 :]:
        if not _should_continue_tag_block(line):
            break
        block.append(line)
    return block


def _literal_tags(text: str) -> list[str]:
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return []
    if isinstance(parsed, str):
        return [parsed]
    if isinstance(parsed, TypingIterable) and not isinstance(parsed, (bytes, Mapping)):
        return [str(item) for item in parsed if isinstance(item, (str, int, float))]
    return []


_BULLET_PREFIX = re.compile(r"^(?:[-*•]+\s*|\d+[.)]\s*)")


def _split_tag_line(line: str) -> list[str]:
    cleaned = _BULLET_PREFIX.sub("", line).strip().strip(",")
    if not cleaned:
        return []
    if ":" in cleaned:
        prefix, suffix = cleaned.split(":", 1)
        cleaned = suffix.strip() or prefix.strip()
    cleaned = cleaned.strip("[]")
    tokens = [
        re.sub(r"\(.*?\)$", "", token).strip().strip("\"'")
        for token in cleaned.split(",")
        if token.strip()
    ]
    return [token for token in tokens if token]


def _tag_candidates(block: list[str]) -> list[str]:
    literal = _literal_tags(" ".join(block).strip())
    if literal:
        return literal
    return [candidate for line in block for candidate in _split_tag_line(line)]


def _dedupe(sequence: TypingIterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _normalize_tags(candidates: list[str], valid: set[str]) -> list[str]:
    normalized = _dedupe(token.strip().lower() for token in candidates if token and token.strip())
    filtered = [tag for tag in normalized if not valid or tag in valid]
    return filtered or normalized


def _parse_completion(
    response_text: str, tag_configs: Mapping[str, TypingIterable[str]]
) -> tuple[str, list[str]]:
    classification = "unclassified"
    tags: list[str] = []
    lines = _response_lines(response_text)
    valid = _valid_tags(tag_configs)
    for index, line in enumerate(lines):
        lower = line.lower()
        if lower.startswith("classification:"):
            value = line.split(":", 1)[1] if ":" in line else ""
            classification = _normalized_classification(value)
        elif lower.startswith("tags:"):
            block = _collect_tag_block(lines, index)
            tags = _normalize_tags(_tag_candidates(block), valid)
    return classification, tags


def classify_chunk_utterance(
    text_chunk: str,
    *,
    tag_configs: dict[str, list[str]],
    completion_fn: Callable[[str], str],
) -> dict[str, Any]:
    """Classify ``text_chunk`` and assign relevant tags using ``completion_fn``."""
    if not text_chunk or not text_chunk.strip():
        return {"classification": "unclassified", "tags": []}

    tag_lines = [f"- {cat}: {', '.join(tags)}\n" for cat, tags in tag_configs.items()]
    available_tags = (
        (
            "\n\nAvailable tags by category:\n"
            + "".join(tag_lines)
            + "\nSelect 2-4 most relevant tags from the available categories."
        )
        if tag_configs
        else ""
    )

    prompt = (
        "Given the following text, classify its primary utterance type and "
        "assign relevant tags.\n\n"
        f"Classification: Choose the best fit from this list: {UTTERANCE_TYPES}.\n\n"
        f"{available_tags}\n\n"
        "Respond in this exact format:\n"
        "Classification: [chosen_type]\n"
        "Tags: [tag1, tag2, tag3]\n\n"
        f'Text: "{text_chunk}"\n\n'
        "Response:"
    )

    try:
        response_text = completion_fn(prompt).strip()
        classification, tags = _parse_completion(response_text, tag_configs)
        if _debug_enabled() and _debug_next_sample():
            has_tags = "tags:" in response_text.lower()
            logger.warning(
                "ai_enrich debug: response_preview='%s' has_tags=%s classification=%s tags=%s",
                _debug_preview(response_text),
                has_tags,
                classification,
                tags,
            )
        return {"classification": classification, "tags": tags}
    except Exception as exc:
        if _debug_enabled():
            logger.warning(
                "ai_enrich debug: completion failed (%s)",
                exc.__class__.__name__,
            )
        return {"classification": "error", "tags": []}


class ClassifyClient(Protocol):
    def classify_chunk_utterance(
        self, text_chunk: str, *, tag_configs: dict[str, list[str]]
    ) -> dict[str, Any]: ...


def _meta_payload(chunk: Chunk, key: str) -> Mapping[str, Any]:
    return _as_mapping(chunk.get(key))


def _merge_meta(payload: Mapping[str, Any], classification: str, tags: list[str]) -> dict[str, Any]:
    return {
        **payload,
        "utterance_type": classification,
        "tags": tags,
    }


def _classify_chunk(
    chunk: Chunk, *, client: ClassifyClient, tag_configs: dict[str, list[str]]
) -> Chunk:
    result = client.classify_chunk_utterance(
        chunk.get("text", ""),
        tag_configs=tag_configs,
    )
    classification = result.get("classification", "unclassified")
    tags = result.get("tags", [])

    meta_source = _meta_payload(chunk, "meta") or _meta_payload(chunk, "metadata")
    meta = _merge_meta(meta_source, classification, tags)
    metadata_source = _meta_payload(chunk, "metadata") or meta
    metadata = _merge_meta(metadata_source, classification, tags)

    return {
        **chunk,
        "utterance_type": classification,
        "tags": tags,
        "meta": meta,
        "metadata": metadata,
    }


def _classify_all(
    chunks: Chunks, *, client: ClassifyClient, tag_configs: dict[str, list[str]]
) -> Chunks:
    count = len(chunks)
    if count == 0:
        return []

    logger.info("ai_enrich: enriching %d chunks...", count)

    # Sequential for single chunk to avoid overhead
    if count == 1:
        return [_classify_chunk(chunks[0], client=client, tag_configs=tag_configs)]

    max_workers = min(count, 10)
    results: list[Chunk] = []
    processed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = executor.map(
            lambda c: _classify_chunk(c, client=client, tag_configs=tag_configs),
            chunks,
        )
        for result in futures:
            results.append(result)
            processed += 1
            if processed % 10 == 0 or processed == count:
                logger.info("ai_enrich: processed %d/%d chunks", processed, count)

    return results


def _update_meta(meta: dict[str, Any] | None, count: int) -> dict[str, Any]:
    base = dict(meta or {})
    base.setdefault("metrics", {}).setdefault("ai_enrich", {})["chunks"] = count
    return base


class _AiEnrichPass:
    name = "ai_enrich"
    input_type = list
    output_type = list

    def __init__(
        self,
        client: ClassifyClient | None = None,
        tag_configs: dict[str, list[str]] | None = None,
    ) -> None:
        self._client = client
        self._tag_configs = _copy_tag_configs(tag_configs) if tag_configs else None

    def __call__(self, a: Artifact) -> Artifact:
        options = _resolve_options(a.meta)
        enabled = bool(options.get("enabled"))
        if _debug_enabled():
            logger.warning(
                "ai_enrich debug: enabled=%s api_key_set=%s option_keys=%s",
                enabled,
                bool(os.getenv("OPENAI_API_KEY") or options.get("api_key")),
                sorted(options.keys()),
            )
        if not enabled:
            if _debug_enabled():
                logger.warning("ai_enrich debug: pass disabled")
            return a
        chunks, rebuild_payload = _payload_chunks(a.payload)
        tag_configs, source = _resolve_tag_configs(options, self._tag_configs)
        if _debug_enabled():
            categories, total = _tag_config_stats(tag_configs)
            logger.warning(
                "ai_enrich debug: tag configs source=%s categories=%d total_tags=%d",
                source,
                categories,
                total,
            )
            if not tag_configs:
                logger.warning("ai_enrich debug: no tag configs loaded")
        client, client_reason = _ensure_client(
            options,
            self._client,
            tag_configs,
            completion_reason=source,
        )
        if not client:
            raise RuntimeError(
                f"ai_enrich enabled but no completion client (reason={client_reason or 'unknown'})"
            )
        if not chunks:
            if _debug_enabled() and not chunks:
                logger.warning("ai_enrich debug: no chunks to enrich")
            return a
        self._client = client
        self._tag_configs = _copy_tag_configs(tag_configs)
        enriched = _classify_all(chunks, client=client, tag_configs=tag_configs)
        payload = rebuild_payload(enriched)
        meta = _update_meta(a.meta, len(enriched))
        return Artifact(payload=payload, meta=meta)


ai_enrich = register(_AiEnrichPass())
