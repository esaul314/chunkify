from typing import Any, Callable, Dict, List

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

    available_tags = (
        "\n\nAvailable tags by category:\n" +
        "".join(
            f"- {cat}: {', '.join(tags)}\n" for cat, tags in tag_configs.items()
        ) +
        "\nSelect 2-4 most relevant tags from the available categories."
        if tag_configs
        else ""
    )

    prompt = f"""Given the following text, classify its primary utterance type and assign relevant tags.

Classification: Choose the best fit from this list: {UTTERANCE_TYPES}.

{available_tags}

Respond in this exact format:
Classification: [chosen_type]
Tags: [tag1, tag2, tag3]

Text: \"{text_chunk}\"

Response:"""

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


def _classify_chunk(
    chunk: Chunk,
    *,
    classify: Callable[..., Dict[str, Any]],
    tag_configs: Dict[str, List[str]],
    completion_fn: Callable[[str], str],
) -> Chunk:
    result = classify(
        chunk.get("text", ""),
        tag_configs=tag_configs,
        completion_fn=completion_fn,
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
    chunks: Chunks,
    *,
    classify: Callable[..., Dict[str, Any]],
    tag_configs: Dict[str, List[str]],
    completion_fn: Callable[[str], str],
) -> Chunks:
    return [
        _classify_chunk(
            c,
            classify=classify,
            tag_configs=tag_configs,
            completion_fn=completion_fn,
        )
        for c in chunks
    ]


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
        classify: Callable[..., Dict[str, Any]] = classify_chunk_utterance,
    ) -> None:
        self._classify = classify

    def __call__(self, a: Artifact) -> Artifact:
        chunks = a.payload if isinstance(a.payload, list) else []
        options = (a.meta or {}).get("ai_enrich", {})
        completion_fn = options.get("completion_fn")
        tag_configs = options.get("tag_configs", {})
        if not completion_fn:
            return a
        enriched = _classify_all(
            chunks,
            classify=self._classify,
            tag_configs=tag_configs,
            completion_fn=completion_fn,
        )
        meta = _update_meta(a.meta, len(enriched))
        return Artifact(payload=enriched, meta=meta)


ai_enrich = register(_AiEnrichPass())
