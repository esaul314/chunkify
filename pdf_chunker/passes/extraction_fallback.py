from __future__ import annotations

from typing import Any

from pdf_chunker.framework import Artifact, register

Block = dict[str, Any]


def _blocks_doc(blocks: list[Block], source: str) -> dict[str, Any]:
    return {"type": "blocks", "blocks": blocks, "source_path": source}


def _score(blocks: list[Block]) -> float:
    from pdf_chunker.extraction_fallbacks import _assess_text_quality

    text = "\n".join(b.get("text", "") for b in blocks)
    return float(_assess_text_quality(text).get("quality_score", 0.0))


def _metrics(reason: str | None, blocks: list[Block]) -> dict[str, Any]:
    metrics: dict[str, Any] = {"score": _score(blocks)}
    return metrics if reason is None else {**metrics, "reason": reason}


def _extract(path: str, reason: str | None) -> tuple[list[Block], dict[str, Any]]:
    from pdf_chunker.extraction_fallbacks import execute_fallback_extraction

    blocks = execute_fallback_extraction(path, fallback_reason=reason)
    return blocks, _metrics(reason, blocks)


def _meta(meta: dict[str, Any] | None, metrics: dict[str, Any]) -> dict[str, Any]:
    """Return a new meta dict with fallback metrics merged immutably."""

    metrics_root = (meta or {}).get("metrics", {})
    fallback = {**metrics_root.get("extraction_fallback", {}), **metrics}
    return {**(meta or {}), "metrics": {**metrics_root, "extraction_fallback": fallback}}


class _ExtractionFallbackPass:
    name = "extraction_fallback"
    input_type = dict
    output_type = dict

    def __call__(self, a: Artifact) -> Artifact:
        doc = a.payload if isinstance(a.payload, dict) else {}
        path = doc.get("source_path", "")
        reason = (a.meta or {}).get("fallback_reason")
        blocks, metrics = _extract(path, reason)
        meta = _meta(a.meta, metrics)
        return Artifact(payload=_blocks_doc(blocks, path), meta=meta)


extraction_fallback = register(_ExtractionFallbackPass())
