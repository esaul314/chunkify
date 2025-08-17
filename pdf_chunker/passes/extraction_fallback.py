from typing import Any, Dict, List

from pdf_chunker.framework import Artifact, register

Block = Dict[str, Any]


def _blocks_doc(blocks: List[Block], source: str) -> Dict[str, Any]:
    return {"type": "blocks", "blocks": blocks, "source_path": source}


def _score(blocks: List[Block]) -> float:
    from pdf_chunker.extraction_fallbacks import _assess_text_quality

    text = "\n".join(b.get("text", "") for b in blocks)
    return _assess_text_quality(text).get("quality_score", 0.0)


def _extract(path: str, reason: str | None) -> List[Block]:
    from pdf_chunker.extraction_fallbacks import execute_fallback_extraction

    return execute_fallback_extraction(path, fallback_reason=reason)


def _meta(meta: Dict[str, Any] | None, reason: str | None, score: float) -> Dict[str, Any]:
    metrics = (meta or {}).get("metrics", {})
    fallback = metrics.get("extraction_fallback", {})
    update = {"score": score, **({"reason": reason} if reason else {})}
    return {
        **(meta or {}),
        "metrics": {"extraction_fallback": {**fallback, **update}, **metrics},
    }


class _ExtractionFallbackPass:
    name = "extraction_fallback"
    input_type = dict
    output_type = dict

    def __call__(self, a: Artifact) -> Artifact:
        doc = a.payload if isinstance(a.payload, dict) else {}
        path = doc.get("source_path", "")
        reason = (a.meta or {}).get("fallback_reason")
        blocks = _extract(path, reason)
        score = _score(blocks)
        meta = _meta(a.meta, reason, score)
        return Artifact(payload=_blocks_doc(blocks, path), meta=meta)


extraction_fallback = register(_ExtractionFallbackPass())
