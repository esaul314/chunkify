from typing import Any, Dict, List

from pdf_chunker.framework import Artifact, register

Block = Dict[str, Any]


def _blocks_doc(blocks: List[Block], source: str) -> Dict[str, Any]:
    return {"type": "blocks", "blocks": blocks, "source_path": source}


def _quality(blocks: List[Block]) -> Dict[str, float]:
    from pdf_chunker.extraction_fallbacks import _assess_text_quality

    text = "\n".join(b.get("text", "") for b in blocks)
    return _assess_text_quality(text)


def _extract(path: str, reason: str | None) -> List[Block]:
    from pdf_chunker.extraction_fallbacks import execute_fallback_extraction

    return execute_fallback_extraction(path, fallback_reason=reason)


def _meta(
    meta: Dict[str, Any] | None,
    reason: str | None,
    quality: Dict[str, float],
) -> Dict[str, Any]:
    base = dict(meta or {})
    metrics = base.setdefault("metrics", {}).setdefault("extraction_fallback", {})
    if reason:
        metrics["reason"] = reason
    metrics["quality_score"] = quality.get("quality_score", 0.0)
    return base


class _ExtractionFallbackPass:
    name = "extraction_fallback"
    input_type = dict
    output_type = dict

    def __call__(self, a: Artifact) -> Artifact:
        doc = a.payload if isinstance(a.payload, dict) else {}
        path = doc.get("source_path", "")
        reason = (a.meta or {}).get("fallback_reason")
        blocks = _extract(path, reason)
        quality = _quality(blocks)
        meta = _meta(a.meta, reason, quality)
        return Artifact(payload=_blocks_doc(blocks, path), meta=meta)


extraction_fallback = register(_ExtractionFallbackPass())
