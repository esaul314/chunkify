from typing import Any, Callable, Dict, List

from pdf_chunker.ai_enrichment import classify_chunk_utterance
from pdf_chunker.framework import Artifact, register

Chunk = Dict[str, Any]
Chunks = List[Chunk]


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
