from collections.abc import Iterable, Mapping
from functools import lru_cache
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Dict, List, Protocol

from pdf_chunker.framework import Artifact, register

Chunk = Dict[str, Any]
Chunks = List[Chunk]


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _normalized_tags(values: Any) -> List[str]:
    if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
        return []
    return [
        normalized
        for normalized in (
            str(item).strip().lower()
            for item in values
            if item is not None
        )
        if normalized
    ]


def _sanitize_tag_configs(raw: Mapping[str, Any]) -> Dict[str, List[str]]:
    return {
        str(category): _normalized_tags(values)
        for category, values in raw.items()
        if _normalized_tags(values)
    }


def _copy_tag_configs(configs: Mapping[str, Iterable[str]]) -> Dict[str, List[str]]:
    return {str(category): [*tags] for category, tags in configs.items()}


@lru_cache(maxsize=None)
def _load_default_tag_configs() -> Dict[str, List[str]]:
    from pdf_chunker.adapters.ai_enrich import _load_tag_configs as load_dir

    return _copy_tag_configs(load_dir())


def _load_tag_configs_from_path(path: Path) -> Dict[str, List[str]]:
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


def _resolve_tag_configs(
    options: Mapping[str, Any],
    cached: Dict[str, List[str]] | None,
) -> Dict[str, List[str]]:
    explicit = options.get("tag_configs")
    if isinstance(explicit, Mapping):
        sanitized = _sanitize_tag_configs(explicit)
        if sanitized:
            return sanitized

    path_value = next(
        (
            options.get(key)
            for key in ("tags_dir", "tags_path", "tags_file")
            if options.get(key)
        ),
        None,
    )
    if isinstance(path_value, (str, Path, PathLike)):
        loaded = _load_tag_configs_from_path(Path(path_value))
        if loaded:
            return loaded

    if cached:
        return _copy_tag_configs(cached)

    return _load_default_tag_configs()


def _resolve_options(meta: Mapping[str, Any] | None) -> Dict[str, Any]:
    base = _as_mapping(meta)
    nested = _as_mapping(base.get("options"))
    staged = _as_mapping(nested.get("ai_enrich"))
    direct = _as_mapping(base.get("ai_enrich"))
    return {**staged, **direct}


def _resolve_completion(options: Mapping[str, Any]) -> Callable[[str], str] | None:
    completion = options.get("completion_fn")
    if callable(completion):
        return completion
    try:
        from pdf_chunker.adapters.ai_enrich import init_llm
    except Exception:  # pragma: no cover - adapters unavailable
        return None
    try:
        return init_llm(api_key=options.get("api_key"))
    except Exception:  # pragma: no cover - missing credentials
        return None


def _ensure_client(
    options: Mapping[str, Any],
    fallback: "ClassifyClient | None",
    tag_configs: Dict[str, List[str]],
) -> "ClassifyClient | None":
    client = options.get("client")
    if client:
        return client  # type: ignore[return-value]
    if fallback:
        return fallback

    completion = _resolve_completion(options)
    if not completion:
        return None
    try:
        from pdf_chunker.adapters.ai_enrich import Client
    except Exception:  # pragma: no cover - adapters unavailable
        return None
    try:
        return Client(completion_fn=completion, tag_configs=tag_configs or None)
    except Exception:  # pragma: no cover - client init failure
        return None


def _iterable_items(value: Any) -> List[Any]:
    return (
        list(value)
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes))
        else []
    )


def _payload_chunks(payload: Any) -> tuple[Chunks, Callable[[Chunks], Any]]:
    if isinstance(payload, list):
        items = list(payload)
        indices = [i for i, item in enumerate(items) if isinstance(item, Mapping)]
        if not indices:
            return [], lambda _: items
        selected = set(indices)

        def rebuild(enriched: Chunks) -> List[Any]:
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

        def rebuild(enriched: Chunks) -> Dict[str, Any]:
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


def classify_chunk_utterance(
    text_chunk: str,
    *,
    tag_configs: Dict[str, List[str]],
    completion_fn: Callable[[str], str],
) -> Dict[str, Any]:
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
        classification, tags = "unclassified", []
        for line in response_text.split("\n"):
            line = line.strip()
            if line.startswith("Classification:"):
                classification = line.split(":", 1)[1].strip().lower()
                if classification not in UTTERANCE_TYPES:
                    classification = "unclassified"
            elif line.startswith("Tags:"):
                tags_text = line.split(":", 1)[1].strip()
                raw_tags = [
                    tag.strip().lower()
                    for tag in tags_text.replace("[", "").replace("]", "").split(",")
                    if tag.strip()
                ]
                valid = {t for v in tag_configs.values() for t in v}
                tags = [tag for tag in raw_tags if tag in valid]
        return {"classification": classification, "tags": tags}
    except Exception:
        return {"classification": "error", "tags": []}


class ClassifyClient(Protocol):
    def classify_chunk_utterance(
        self, text_chunk: str, *, tag_configs: Dict[str, List[str]]
    ) -> Dict[str, Any]: ...


def _meta_payload(chunk: Chunk, key: str) -> Mapping[str, Any]:
    return _as_mapping(chunk.get(key))


def _merge_meta(
    payload: Mapping[str, Any], classification: str, tags: List[str]
) -> Dict[str, Any]:
    return {
        **payload,
        "utterance_type": classification,
        "tags": tags,
    }


def _classify_chunk(
    chunk: Chunk, *, client: ClassifyClient, tag_configs: Dict[str, List[str]]
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
    chunks: Chunks, *, client: ClassifyClient, tag_configs: Dict[str, List[str]]
) -> Chunks:
    return [_classify_chunk(c, client=client, tag_configs=tag_configs) for c in chunks]


def _update_meta(meta: Dict[str, Any] | None, count: int) -> Dict[str, Any]:
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
        tag_configs: Dict[str, List[str]] | None = None,
    ) -> None:
        self._client = client
        self._tag_configs = _copy_tag_configs(tag_configs) if tag_configs else None

    def __call__(self, a: Artifact) -> Artifact:
        options = _resolve_options(a.meta)
        if not bool(options.get("enabled")):
            return a
        chunks, rebuild_payload = _payload_chunks(a.payload)
        tag_configs = _resolve_tag_configs(options, self._tag_configs)
        client = _ensure_client(options, self._client, tag_configs)
        if not client or not chunks:
            return a
        self._client = client
        self._tag_configs = _copy_tag_configs(tag_configs)
        enriched = _classify_all(chunks, client=client, tag_configs=tag_configs)
        payload = rebuild_payload(enriched)
        meta = _update_meta(a.meta, len(enriched))
        return Artifact(payload=payload, meta=meta)


ai_enrich = register(_AiEnrichPass())
