"""Block-level extraction quality assessment and fallback orchestration."""

from __future__ import annotations

import os
from subprocess import TimeoutExpired
from typing import Callable, Iterable, Sequence, Tuple

from .adapters.io_pdf import run_pdftotext
from .language import default_language
from .page_artifacts import remove_page_artifact_lines
from .page_utils import parse_page_ranges
from .pdf_blocks import Block
from .text_cleaning import clean_text

try:  # pragma: no cover - optional dependency
    from pdfminer.high_level import extract_text
    from pdfminer.layout import LAParams
    PDFMINER_AVAILABLE = True
except Exception:  # pragma: no cover
    PDFMINER_AVAILABLE = False


def _assess_text_quality(text: str) -> dict[str, float]:
    """Return a trivial quality assessment favoring primary extraction."""
    score = 1.0 if text.strip() else 0.0
    return {"avg_line_length": 0.0, "space_density": 0.0, "quality_score": score}


def _clean_fallback_text(text: str) -> str:
    pages = text.split("\f")
    cleaned = [
        remove_page_artifact_lines(page, i + 1) for i, page in enumerate(pages)
    ]
    return "\f".join(cleaned)


def _filter_text_by_pages(text: str, excluded: set[int]) -> str:
    if not excluded:
        return text
    return "\f".join(
        page for i, page in enumerate(text.split("\f"), start=1) if i not in excluded
    )


def _is_heading(text: str) -> bool:
    return len(text.split()) < 15 and text.istitle() and not text.endswith(".")


def _text_to_blocks(text: str, filepath: str, method: str) -> list[dict]:
    return [
        {
            "type": "heading" if _is_heading(t) else "paragraph",
            "text": t,
            "language": default_language(),
            "source": {"filename": os.path.basename(filepath), "method": method},
        }
        for t in (clean_text(p) for p in text.split("\n\n"))
        if t
    ]


def _extract_with_pdftotext(
    filepath: str, exclude_pages: str | None = None
) -> list[dict]:
    try:
        excluded = parse_page_ranges(exclude_pages) if exclude_pages else set()
        cmd = ["pdftotext", "-layout", filepath, "-"]
        result = run_pdftotext(cmd)
        if result.returncode != 0:
            return []
        raw_text = _filter_text_by_pages(result.stdout, excluded)
        if _assess_text_quality(raw_text)["quality_score"] < 0.7:
            return []
        pages = _clean_fallback_text(raw_text).split("\f")
        return [
            {
                "type": "paragraph",
                "text": page_text,
                "source": {
                    "filename": os.path.basename(filepath),
                    "page": i + 1,
                    "method": "pdftotext",
                },
            }
            for i, page_text in enumerate(pages)
            if page_text.strip()
        ]
    except TimeoutExpired:
        return []
    except FileNotFoundError:
        return []
    except Exception:
        return []


def _extract_with_pdfminer(
    filepath: str, exclude_pages: str | None = None
) -> list[dict]:
    if not PDFMINER_AVAILABLE:
        return []
    try:
        excluded = parse_page_ranges(exclude_pages) if exclude_pages else set()
        laparams = LAParams(all_texts=True)
        text = extract_text(filepath, laparams=laparams)
        text = _filter_text_by_pages(text, excluded)
        if _assess_text_quality(text)["quality_score"] < 0.7:
            return []
        repaired = _clean_fallback_text(text)
        return _text_to_blocks(repaired, filepath, "pdfminer")
    except Exception:
        return []


def execute_fallback_extraction(
    filepath: str,
    exclude_pages: str | None = None,
    fallback_reason: str | None = None,
) -> list[dict]:
    blocks = _extract_with_pdftotext(filepath, exclude_pages)
    if blocks:
        return blocks
    return _extract_with_pdfminer(filepath, exclude_pages)


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

    candidates: Sequence[Tuple[str, Callable[[str, str | None], list[dict]]]] = (
        ("pdftotext", _extract_with_pdftotext),
        ("pdfminer", _extract_with_pdfminer if PDFMINER_AVAILABLE else (lambda *_: [])),
    )
    base_pages = _page_count(primary)
    exclude_str = ",".join(map(str, sorted(excluded))) if excluded else None
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
