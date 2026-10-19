import ast
import json
import logging
import os
import re
from collections.abc import Callable, Iterable, Mapping
from collections.abc import Iterable as TypingIterable
from concurrent.futures import ThreadPoolExecutor
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
    return len(configs), sum(sum(1 for _ in tags) for tags in configs.values())


def _resolve_tag_configs(
    options: Mapping[str, Any],
    cached: dict[str, list[str]] | None,
) -> tuple[dict[str, list[str]], str]:
    """Resolve only caller-supplied tag configuration in the pure pass."""
    explicit = options.get("tag_configs")
    if isinstance(explicit, Mapping):
        sanitized = _sanitize_tag_configs(explicit)
        if sanitized:
            return sanitized, "explicit"

    if cached:
        return _copy_tag_configs(cached), "cached"

    return {}, "injected"


def _resolve_options(meta: Mapping[str, Any] | None) -> dict[str, Any]:
    base = _as_mapping(meta)
    nested = _as_mapping(base.get("options"))
    staged = _as_mapping(nested.get("ai_enrich"))
    direct = _as_mapping(base.get("ai_enrich"))
    return {**staged, **direct}


def _resolve_completion(
    options: Mapping[str, Any],
) -> tuple[Callable[[str], str] | None, str | None]:
    """Compatibility shim; client construction belongs to the adapter."""
    from pdf_chunker.adapters.ai_enrich import resolve_completion

    return resolve_completion(options)


def _resolve_fallback_completion(
    options: Mapping[str, Any],
) -> tuple[Callable[[str], str] | None, str | None]:
    """Compatibility shim; fallback client construction belongs to the adapter."""
    from pdf_chunker.adapters.ai_enrich import resolve_fallback_completion

    return resolve_fallback_completion(options)


def _serialized_record_size(text: str, metadata: Mapping[str, Any], metadata_key: str) -> int:
    """Return the emitter-compatible JSON character count for one record."""
    return len(
        json.dumps(
            {"text": text, metadata_key: dict(metadata)},
            ensure_ascii=False,
        )
    )


def _largest_fitting_prefix(
    text: str,
    *,
    metadata: Mapping[str, Any],
    metadata_key: str,
    max_chars: int,
) -> int:
    """Find the longest exact text prefix whose serialized row fits."""
    low, high = 1, len(text)
    best = 0
    while low <= high:
        middle = (low + high) // 2
        if _serialized_record_size(text[:middle], metadata, metadata_key) <= max_chars:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    return best


def _admit_chunks_for_jsonl(
    chunks: Chunks,
    *,
    max_chars: int,
    metadata_key: str,
    tag_configs: Mapping[str, Iterable[str]],
) -> Chunks:
    """Losslessly split chunks against a conservative enriched JSONL budget."""
    tags = sorted(_valid_tags(_effective_tag_configs(tag_configs)))
    result: Chunks = []
    for chunk in chunks:
        text = chunk.get("text", "")
        if not isinstance(text, str):
            raise TypeError("JSONL admission requires chunk text to be a string")
        source_meta = dict(_meta_payload(chunk, "meta") or _meta_payload(chunk, "metadata"))
        projected = {
            **source_meta,
            "utterance_type": "statement_of_fact",
            "tags": tags,
            "enrichment_status": "complete",
        }
        if _serialized_record_size(text, projected, metadata_key) <= max_chars:
            result.append(chunk)
            continue

        remaining = text
        part = 0
        while remaining:
            part_meta = {**projected, "chunk_part": part}
            size = _largest_fitting_prefix(
                remaining,
                metadata=part_meta,
                metadata_key=metadata_key,
                max_chars=max_chars,
            )
            if size == 0:
                raise ValueError("serialized metadata exceeds JSONL record limit before enrichment")
            piece, remaining = remaining[:size], remaining[size:]
            result.append(
                {
                    **chunk,
                    "text": piece,
                    "meta": {**source_meta, "chunk_part": part},
                }
            )
            part += 1
    return result


def _validate_enriched_jsonl_size(chunks: Chunks, *, max_chars: int, metadata_key: str) -> None:
    """Refuse unexpected enrichment output that exceeds the admitted record size."""
    for index, chunk in enumerate(chunks):
        metadata = _meta_payload(chunk, "meta") or _meta_payload(chunk, "metadata")
        if _serialized_record_size(chunk.get("text", ""), metadata, metadata_key) > max_chars:
            raise ValueError(
                f"enriched JSONL record {index} exceeds the admitted {max_chars}-character limit"
            )


def _iterable_items(value: Any) -> list[Any]:
    return (
        list(value) if isinstance(value, Iterable) and not isinstance(value, (str, bytes)) else []
    )


def _payload_chunks(payload: Any) -> tuple[Chunks, Callable[[Chunks], Any]]:
    def indexed_chunks(items: list[Any]) -> tuple[Chunks, Callable[[Chunks], list[Any]]]:
        indices = [i for i, item in enumerate(items) if isinstance(item, Mapping)]
        if not indices:
            return [], lambda _: items

        selected = set(indices)
        marker = "__pdf_chunker_original_item_index__"
        while any(marker in item for item in items if isinstance(item, Mapping)):
            marker += "_"

        chunks: Chunks = []
        for index in indices:
            chunk = dict(items[index])
            chunk[marker] = index
            chunks.append(chunk)

        def rebuild(enriched: Chunks) -> list[Any]:
            grouped: dict[int, list[dict[str, Any]]] = {}
            for chunk in enriched:
                if not isinstance(chunk, Mapping) or marker not in chunk:
                    raise ValueError("enrichment output lost its original chunk index")
                indexed = dict(chunk)
                index = indexed.pop(marker)
                if type(index) is not int or index not in selected:
                    raise ValueError("enrichment output has an invalid original chunk index")
                grouped.setdefault(index, []).append(indexed)
            if set(grouped) != selected:
                raise ValueError("enrichment output is missing an original chunk group")

            updated: list[Any] = []
            for index, item in enumerate(items):
                if index in selected:
                    updated.extend(grouped[index])
                else:
                    updated.append(item)
            return updated

        return chunks, rebuild

    if isinstance(payload, list):
        return indexed_chunks(list(payload))

    if isinstance(payload, Mapping):
        items = _iterable_items(payload.get("items"))
        if not items:
            return [], lambda _: payload
        base = {key: value for key, value in payload.items() if key != "items"}
        chunks, rebuild_items = indexed_chunks(items)
        return chunks, lambda enriched: {**base, "items": rebuild_items(enriched)}

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


_EXCLUDED_TAG_CATEGORIES = {"content_type"}
_EXCLUDED_TAGS = set(UTTERANCE_TYPES)


def _effective_tag_configs(tag_configs: Mapping[str, TypingIterable[str]]) -> dict[str, list[str]]:
    effective: dict[str, list[str]] = {}
    for category, tags in tag_configs.items():
        if category in _EXCLUDED_TAG_CATEGORIES:
            continue
        filtered = _dedupe(
            tag for tag in tags if isinstance(tag, str) and tag not in _EXCLUDED_TAGS
        )
        if filtered:
            effective[str(category)] = filtered
    return effective


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
    return not (":" in line and not line.startswith(("-", "*", "•")))


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
    except (SyntaxError, TypeError, ValueError):
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
        cleaned = suffix.strip()
        if not cleaned:
            return []
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
    if not valid:
        return [tag for tag in normalized if tag not in _EXCLUDED_TAGS]

    def normalize_candidate(tag: str) -> str | None:
        if tag in _EXCLUDED_TAGS:
            return None
        if tag in valid:
            return tag
        if tag.endswith("s") and tag[:-1] in valid:
            return tag[:-1]
        return None

    filtered = _dedupe(
        normalized_tag
        for tag in normalized
        if (normalized_tag := normalize_candidate(tag)) is not None
    )
    return filtered


def _grounded_tags(tags: list[str], text_chunk: str) -> list[str]:
    lowered = text_chunk.lower()
    words = set(re.findall(r"[a-z]+", lowered))

    def is_grounded(tag: str) -> bool:
        parts = tag.split("_")
        if len(parts) == 1:
            return len(parts[0]) >= 5 and parts[0] in words
        phrase = " ".join(parts)
        return phrase in lowered or all(part in words for part in parts)

    return [tag for tag in tags if is_grounded(tag)]


def _lexical_fallback_tags(text_chunk: str, valid: set[str], limit: int = 4) -> list[str]:
    if not valid:
        return []
    lowered = text_chunk.lower()
    words = set(re.findall(r"[a-z]+", lowered))
    title_window = lowered[:400]
    scored: list[tuple[int, str]] = []
    for tag in valid:
        parts = tag.split("_")
        if len(parts) < 2:
            continue
        score = 0
        phrase = " ".join(parts)
        if phrase in lowered:
            score += 4
        if phrase in title_window:
            score += 2
        if all(part in words for part in parts):
            score += len(parts)
        if score >= 3:
            scored.append((score, tag))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [tag for _, tag in scored[:limit]]


def _classification_repair_prompt(text_chunk: str) -> str:
    return (
        "Given the following text, choose the single best utterance type.\n\n"
        f"Classification: Choose exactly one from this list: {UTTERANCE_TYPES}.\n\n"
        "Only use unclassified for front matter, bibliographic entries, indexes, "
        "tables of contents, or text with no clear communicative act.\n"
        "Otherwise choose the nearest valid type.\n\n"
        "Respond in this exact format:\n"
        "Classification: [chosen_type]\n\n"
        f'Text: "{text_chunk}"\n\n'
        "Response:"
    )


def _repair_classification(
    text_chunk: str,
    *,
    completion_fn: Callable[[str], str],
    classification: str,
) -> str:
    if classification != "unclassified":
        return classification
    response_text = completion_fn(_classification_repair_prompt(text_chunk)).strip()
    repaired, _ = _parse_completion(response_text, {})
    return classification if repaired == "unclassified" else repaired


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


def _classification_prompt(text_chunk: str, tag_configs: dict[str, list[str]]) -> str:
    """Build the unchanged grounded prompt without shortening canonical text."""
    # Advertise only candidates that the unchanged result grounding can accept.
    # Keep the full vocabulary for parsing and fallback; never shorten source text.
    prompt_tag_configs = {
        category: grounded
        for category, tags in tag_configs.items()
        if (grounded := _grounded_tags(tags, text_chunk))
    }
    tag_lines = [f"- {cat}: {', '.join(tags)}\n" for cat, tags in prompt_tag_configs.items()]
    available_tags = (
        (
            "\n\nAllowed topical tags by category:\n"
            + "".join(tag_lines)
            + "\nChoose 1-4 tags only from this list."
        )
        if prompt_tag_configs
        else ""
    )

    return (
        "Given the following text, identify its primary utterance type and assign topical tags.\n\n"
        f"Classification: Choose the best fit from this list: {UTTERANCE_TYPES}.\n\n"
        f"{available_tags}\n\n"
        "Important rules for tags:\n"
        "- Tags must describe subject matter, concepts, methods, technologies, or "
        "artifacts discussed in the text.\n"
        "- Prefer tags that are explicitly grounded in the title or body text.\n"
        "- Do not use utterance-type labels as tags, such as explanation, instruction, "
        "example, summary, critique, definition, opinion, or question.\n"
        "- Do not invent new tags.\n"
        "- If no allowed topical tag is clearly grounded in the text, return an empty list.\n\n"
        "Good tag examples:\n"
        "- For a chunk about Python design patterns: Tags: [python, design_patterns]\n"
        "- For a chunk about configuration syntax: Tags: [configuration]\n"
        "Bad tag examples:\n"
        "- Tags: [explanation, summary]\n\n"
        "Respond in this exact format:\n"
        "Classification: [chosen_type]\n"
        "Tags: [tag1, tag2, tag3]\n\n"
        f'Text: "{text_chunk}"\n\n'
        "Response:"
    )


def _enrichment_windows(
    text: str,
    *,
    tag_configs: dict[str, list[str]],
    request_fits: Callable[[str], bool],
) -> list[str]:
    """Partition contiguous text losslessly; no token monotonicity assumption.

    Every leaf must fit BOTH primary and possible repair messages. Halving
    terminates even for whitespace-free multilingual text; boundaries prefer
    whitespace near the midpoint without removing it. Stored parent is untouched.
    """
    pending = [text]
    windows: list[str] = []
    while pending:
        part = pending.pop()
        if request_fits(_classification_prompt(part, tag_configs)) and request_fits(
            _classification_repair_prompt(part)
        ):
            windows.append(part)
            continue
        if len(part) < 2:
            raise ValueError("Enrichment instructions or single character exceed request budget")
        midpoint = len(part) // 2
        boundary = next(
            (i + 1 for i in range(midpoint - 1, midpoint // 2 - 1, -1) if part[i].isspace()),
            midpoint,
        )
        pending.extend((part[boundary:], part[:boundary]))
    return windows


def classify_chunk_utterance(
    text_chunk: str,
    *,
    tag_configs: dict[str, list[str]],
    completion_fn: Callable[[str], str],
    fallback_completion_fn: Callable[[str], str] | None = None,
    fallback_min_chars: int = 200,
) -> dict[str, Any]:
    """Classify full text, using lossless windows only for budgeted callbacks."""
    if not text_chunk or not text_chunk.strip():
        return {"classification": "unclassified", "tags": []}

    tag_configs = _effective_tag_configs(tag_configs)
    request_fits = getattr(completion_fn, "request_fits", None)
    try:
        windows = (
            _enrichment_windows(text_chunk, tag_configs=tag_configs, request_fits=request_fits)
            if callable(request_fits)
            else [text_chunk]
        )
    except (RuntimeError, TimeoutError, ConnectionError, OSError, ValueError):
        return {"classification": "error", "tags": []}
    if len(windows) > 1:
        # Each window follows the existing parsing/grounding/repair/fallback path.
        # No recursive planning: transport still guards every actual request.
        results = []
        for window in windows:
            result = _classify_window(
                window,
                tag_configs=tag_configs,
                completion_fn=completion_fn,
                fallback_completion_fn=fallback_completion_fn,
                fallback_min_chars=fallback_min_chars,
                require_fallback_success=True,
            )
            if result["classification"] == "error":
                return {"classification": "error", "tags": []}
            results.append(result)
        scores: dict[str, int] = {}
        for window, result in zip(windows, results, strict=True):
            label = result["classification"]
            scores[label] = scores.get(label, 0) + len(window.strip())
        meaningful = {key: value for key, value in scores.items() if key != "unclassified"}
        classification = max(meaningful or scores, key=lambda key: (meaningful or scores)[key])
        return {
            "classification": classification,
            "tags": _grounded_tags(
                _dedupe(tag for result in results for tag in result["tags"]), text_chunk
            ),
        }
    return _classify_window(
        text_chunk,
        tag_configs=tag_configs,
        completion_fn=completion_fn,
        fallback_completion_fn=fallback_completion_fn,
        fallback_min_chars=fallback_min_chars,
    )


def _classify_window(
    text_chunk: str,
    *,
    tag_configs: dict[str, list[str]],
    completion_fn: Callable[[str], str],
    fallback_completion_fn: Callable[[str], str] | None,
    fallback_min_chars: int,
    require_fallback_success: bool = False,
) -> dict[str, Any]:
    """Existing single-window classification with full validation vocabulary."""
    if not text_chunk.strip():
        return {"classification": "unclassified", "tags": []}
    prompt = _classification_prompt(text_chunk, tag_configs)
    try:
        response_text = completion_fn(prompt).strip()
        classification, tags = _parse_completion(response_text, tag_configs)
        tags = _grounded_tags(tags, text_chunk)
        if classification == "unclassified":
            classification = _repair_classification(
                text_chunk,
                completion_fn=completion_fn,
                classification=classification,
            )
    except (RuntimeError, TimeoutError, ConnectionError, OSError, ValueError) as exc:
        if _debug_enabled():
            cause = exc.__cause__
            status = getattr(cause, "status_code", None) or getattr(cause, "status", None)
            code = getattr(cause, "code", None)
            # Provider error text is useful for diagnosing HTTP failures. Keep it
            # debug-only and bounded; never log the prompt or request kwargs.
            detail = str(cause)[:240] if cause is not None else ""
            logger.warning(
                "ai_enrich debug: completion failed type=%s cause=%s status=%s code=%s detail=%r",
                exc.__class__.__name__,
                cause.__class__.__name__ if cause is not None else None,
                status,
                code,
                detail,
            )
        return {"classification": "error", "tags": []}

    if (
        not tags
        and tag_configs
        and fallback_completion_fn
        and len(text_chunk.strip()) >= fallback_min_chars
    ):
        try:
            fallback_response = fallback_completion_fn(prompt).strip()
            fallback_classification, fallback_tags = _parse_completion(
                fallback_response,
                tag_configs,
            )
            fallback_tags = _grounded_tags(fallback_tags, text_chunk)
            if fallback_tags:
                tags = fallback_tags
            if classification == "error" and fallback_classification != "unclassified":
                classification = fallback_classification
        except (RuntimeError, TimeoutError, ConnectionError, OSError, ValueError) as exc:
            if require_fallback_success:
                return {"classification": "error", "tags": []}
            if _debug_enabled():
                logger.warning(
                    "ai_enrich debug: fallback completion failed (%s)",
                    exc.__class__.__name__,
                )

    if not tags and tag_configs:
        tags = _lexical_fallback_tags(text_chunk, _valid_tags(tag_configs))
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
        "enrichment_status": "complete",
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
    if classification == "error":
        raise RuntimeError("AI enrichment completion failed for a chunk")

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
        client = options.get("client") or self._client
        if not client:
            raise RuntimeError(
                "ai_enrich enabled but no completion client was injected; "
                "construct one in the adapter or pass it in options"
            )
        if not chunks:
            if _debug_enabled() and not chunks:
                logger.warning("ai_enrich debug: no chunks to enrich")
            return a
        max_chars = options.get("jsonl_max_chars")
        metadata_key = options.get("jsonl_metadata_key", "metadata")
        if isinstance(max_chars, int) and max_chars > 0:
            chunks = _admit_chunks_for_jsonl(
                chunks,
                max_chars=max_chars,
                metadata_key=str(metadata_key),
                tag_configs=tag_configs,
            )
        enriched = _classify_all(chunks, client=client, tag_configs=tag_configs)
        if isinstance(max_chars, int) and max_chars > 0:
            _validate_enriched_jsonl_size(
                enriched,
                max_chars=max_chars,
                metadata_key=str(metadata_key),
            )
        payload = rebuild_payload(enriched)
        meta = _update_meta(a.meta, len(enriched))
        return Artifact(payload=payload, meta=meta)


ai_enrich = register(_AiEnrichPass())
