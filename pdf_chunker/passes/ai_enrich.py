from typing import Any, Callable, Dict, List, Protocol

from pdf_chunker.framework import Artifact, register

Chunk = Dict[str, Any]
Chunks = List[Chunk]


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


def _classify_chunk(
    chunk: Chunk, *, client: ClassifyClient, tag_configs: Dict[str, List[str]]
) -> Chunk:
    result = client.classify_chunk_utterance(
        chunk.get("text", ""),
        tag_configs=tag_configs,
    )
    meta = {
        **chunk.get("metadata", {}),
        "utterance_type": result["classification"],
        "tags": result["tags"],
    }
    return {
        **chunk,
        "utterance_type": result["classification"],
        "tags": result["tags"],
        "metadata": meta,
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

    def __init__(self, client: ClassifyClient | None = None) -> None:
        self._client = client

    def __call__(self, a: Artifact) -> Artifact:
        chunks = a.payload if isinstance(a.payload, list) else []
        options = (a.meta or {}).get("ai_enrich", {})
        if not options.get("enabled", False):
            return a
        client = options.get("client") or self._client
        if not client:
            return a
        tag_configs = options.get("tag_configs", {})
        enriched = _classify_all(chunks, client=client, tag_configs=tag_configs)
        meta = _update_meta(a.meta, len(enriched))
        return Artifact(payload=enriched, meta=meta)


ai_enrich = register(_AiEnrichPass())
