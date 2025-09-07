"""Block-level extraction quality assessment and fallback orchestration."""

from __future__ import annotations

from typing import Iterable, Optional, Callable, Sequence, Tuple

from .pdf_blocks import Block

try:
    from .extraction_fallbacks import (
        _assess_text_quality,
        _extract_with_pdftotext as _extract_with_pdftotext_impl,
        _extract_with_pdfminer as _extract_with_pdfminer_impl,
        PDFMINER_AVAILABLE,
    )
except Exception:  # pragma: no cover - fallback when dependency missing
    def _assess_text_quality(text: str) -> dict[str, float]:
        return {"quality_score": 0.0}

    def _extract_with_pdftotext_impl(
        filepath: str, exclude_pages: Optional[str] = None
    ) -> list[dict]:
        return []

    def _extract_with_pdfminer_impl(
        filepath: str, exclude_pages: Optional[str] = None
    ) -> list[dict]:
        return []

    PDFMINER_AVAILABLE = False

_extract_with_pdftotext: Callable[[str, Optional[str]], list[dict]] = (
    _extract_with_pdftotext_impl
)
_extract_with_pdfminer: Callable[[str, Optional[str]], list[dict]] = (
    _extract_with_pdfminer_impl
)


def _page_count(blocks: Sequence[Block]) -> int:
    return len({b.source.get("page") for b in blocks})


def apply_fallbacks(
    blocks: Iterable[Block], filepath: str, excluded: set[int]
) -> Iterable[Block]:
    """Assess text quality and optionally replace with a fallback extraction."""

    primary = list(blocks)
    blob = "\n".join(b.text for b in primary)
    quality = _assess_text_quality(blob).get("quality_score", 0.0)
    if quality >= 0.7:
        return primary

    candidates: Sequence[Tuple[str, Callable[[str, Optional[str]], list[dict]]]] = (
        ("pdftotext", _extract_with_pdftotext),
        (
            "pdfminer",
            _extract_with_pdfminer if PDFMINER_AVAILABLE else (lambda *_: []),
        ),
    )
    base_pages = _page_count(primary)
    exclude_str = ",".join(str(p) for p in sorted(excluded)) if excluded else None
    for _name, extractor in candidates:
        raw = extractor(filepath, exclude_str)
        fallback = [
            Block(**b)
            for b in raw
            if b.get("source", {}).get("page") not in excluded
        ]
        if len(fallback) <= 1:
            continue
        if _page_count(fallback) < base_pages:
            continue
        score = _assess_text_quality("\n".join(b.text for b in fallback)).get(
            "quality_score", 0.0
        )
        if score <= quality:
            continue
        return fallback

    return primary
