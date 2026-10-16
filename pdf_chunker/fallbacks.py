"""Block-level extraction quality assessment and fallback orchestration."""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from subprocess import TimeoutExpired
from typing import Any

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


_WORD_RE = re.compile(r"[\w]+(?:['’-][\w]+)*", re.UNICODE)
_REPLACEMENT_GLYPHS = frozenset({"�", "□", "▯"})
_FALLBACK_THRESHOLD = 0.7


def _assess_text_quality(text: str) -> dict[str, float]:
    """Measure text plausibility using deterministic, in-memory signals."""
    stripped = text.strip()
    if not stripped:
        return {
            "avg_line_length": 0.0,
            "space_density": 0.0,
            "word_density": 0.0,
            "word_plausibility": 0.0,
            "replacement_glyph_ratio": 0.0,
            "repeated_line_ratio": 0.0,
            "quality_score": 0.0,
        }

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    words = _WORD_RE.findall(stripped)
    char_count = len(stripped)
    space_density = sum(char.isspace() for char in stripped) / char_count
    avg_line_length = sum(map(len, lines)) / len(lines) if lines else 0.0
    repeated_lines = len(lines) - len(set(lines))
    repeated_line_ratio = repeated_lines / len(lines) if lines else 0.0
    replacement_glyph_ratio = sum(char in _REPLACEMENT_GLYPHS for char in stripped) / char_count

    plausible_words = sum(1 <= len(word) <= 40 for word in words)
    word_plausibility = plausible_words / len(words) if words else 0.0
    word_density = min(len(words) / max(char_count / 6, 1), 1.0)

    # Normal prose generally contains some whitespace, but very high whitespace
    # density is also a sign of layout debris rather than readable text.
    space_score = (
        min(space_density / 0.08, 1.0)
        if space_density <= 0.16
        else max(0.0, 1.0 - (space_density - 0.16) / 0.24)
    )
    replacement_score = max(0.0, 1.0 - replacement_glyph_ratio * 20)
    repetition_score = max(0.0, 1.0 - repeated_line_ratio)
    quality_score = (
        0.30 * word_density
        + 0.25 * space_score
        + 0.20 * word_plausibility
        + 0.15 * replacement_score
        + 0.10 * repetition_score
    )
    return {
        "avg_line_length": avg_line_length,
        "space_density": space_density,
        "word_density": word_density,
        "word_plausibility": word_plausibility,
        "replacement_glyph_ratio": replacement_glyph_ratio,
        "repeated_line_ratio": repeated_line_ratio,
        "quality_score": quality_score,
    }


def _block_text(block: Block | Mapping[str, Any]) -> str:
    if isinstance(block, Mapping):
        return str(block.get("text", ""))
    return str(getattr(block, "text", ""))


def _block_pages(block: Block | Mapping[str, Any]) -> set[int]:
    source = block.get("source", {}) if isinstance(block, Mapping) else getattr(block, "source", {})
    if not isinstance(source, Mapping):
        return set()
    page_range = source.get("page_range")
    if (
        isinstance(page_range, (list, tuple))
        and len(page_range) == 2
        and all(isinstance(page, int) for page in page_range)
    ):
        start, end = page_range
        return set(range(min(start, end), max(start, end) + 1))
    page = source.get("page")
    return {page} if isinstance(page, int) else set()


def _assess_block_quality(
    blocks: Sequence[Block | Mapping[str, Any]],
    expected_pages: set[int] | None = None,
) -> dict[str, Any]:
    """Assess text and page coverage for one extraction candidate."""
    text_metrics = _assess_text_quality("\n".join(_block_text(block) for block in blocks))
    pages = set().union(*(_block_pages(block) for block in blocks)) if blocks else set()
    coverage = None
    missing_pages: list[int] = []
    score = text_metrics.get("quality_score", 0.0)
    if expected_pages is not None:
        coverage = len(pages & expected_pages) / len(expected_pages) if expected_pages else 1.0
        missing_pages = sorted(expected_pages - pages)
        score = 0.8 * score + 0.2 * coverage
    return {
        **text_metrics,
        "page_count": float(len(pages)),
        "page_coverage": coverage,
        "missing_pages": missing_pages,
        "quality_score": score,
    }


def _record_quality_report(
    target: MutableMapping[str, Any] | None,
    candidates: Mapping[str, Mapping[str, Any]],
    selected_engine: str,
    reason: str,
) -> None:
    if target is None:
        return
    selected = candidates[selected_engine]
    target.clear()
    target.update(
        {
            "selected_engine": selected_engine,
            "score": selected.get("quality_score", 0.0),
            "candidate_scores": {
                name: metrics.get("quality_score", 0.0) for name, metrics in candidates.items()
            },
            "candidates": {name: dict(metrics) for name, metrics in candidates.items()},
            "reason": reason,
            "page_coverage": selected.get("page_coverage"),
            "missing_pages": selected.get("missing_pages", []),
        }
    )


def _clean_fallback_text(text: str) -> str:
    pages = text.split("\f")
    cleaned = [
        remove_page_artifact_lines(page, i + 1, keep_shipping=True) for i, page in enumerate(pages)
    ]
    return "\f".join(cleaned)


def _filter_text_by_pages(text: str, excluded: set[int]) -> str:
    if not excluded:
        return text
    return "\f".join(
        "" if i in excluded else page for i, page in enumerate(text.split("\f"), start=1)
    )


def _is_heading(text: str) -> bool:
    return len(text.split()) < 15 and text.istitle() and not text.endswith(".")


def _text_to_blocks(text: str, filepath: str, method: str) -> list[dict]:
    blocks: list[dict] = []
    for page_number, page_text in enumerate(text.split("\f"), start=1):
        for paragraph in page_text.split("\n\n"):
            cleaned = clean_text(paragraph)
            if not cleaned:
                continue
            blocks.append(
                {
                    "type": "heading" if _is_heading(cleaned) else "paragraph",
                    "text": cleaned,
                    "language": default_language(),
                    "source": {
                        "filename": os.path.basename(filepath),
                        "page": page_number,
                        "method": method,
                    },
                }
            )
    return blocks


def _extract_with_pdftotext(
    filepath: str,
    exclude_pages: str | None = None,
    *,
    expected_page_count: int | None = None,
    preserve_layout: bool = True,
) -> list[dict]:
    try:
        excluded = parse_page_ranges(exclude_pages) if exclude_pages else set()
        cmd = ["pdftotext", *(["-layout"] if preserve_layout else []), filepath, "-"]
        result = run_pdftotext(cmd)
        if result.returncode != 0:
            return []
        if expected_page_count is not None:
            # Poppler terminates each page with form feed, including blank pages.
            # Validate BEFORE exclusions/cleanup so a partial body cannot pass.
            raw_pages = result.stdout.split("\f")
            if raw_pages[-1] != "" or len(raw_pages) - 1 != expected_page_count:
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


def _extract_with_pdfminer(filepath: str, exclude_pages: str | None = None) -> list[dict]:
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


def _page_count(blocks: Sequence[Block | Mapping[str, Any]]) -> int:
    return len(set().union(*(_block_pages(block) for block in blocks))) if blocks else 0


def select_best_extraction(
    primary: Sequence[Block | Mapping[str, Any]],
    candidates: Mapping[str, Sequence[Block | Mapping[str, Any]]],
    excluded: set[int],
    *,
    expected_pages: set[int] | None = None,
    quality_report: MutableMapping[str, Any] | None = None,
) -> list[Block | Mapping[str, Any]]:
    """Select the best already-extracted candidate without performing I/O.

    Extraction adapters own the work of producing ``candidates``.  This function
    only scores and compares in-memory values so it can be used by both the PDF
    adapter and the compatibility pass without crossing the side-effect boundary.
    """
    primary_blocks = list(primary)
    primary_metrics = _assess_block_quality(primary_blocks, expected_pages)
    candidate_metrics: dict[str, dict[str, Any]] = {"pymupdf": primary_metrics}
    quality = primary_metrics.get("quality_score", 0.0)
    if quality >= _FALLBACK_THRESHOLD:
        _record_quality_report(
            quality_report,
            candidate_metrics,
            "pymupdf",
            "primary_quality_acceptable",
        )
        return primary_blocks

    base_pages = _page_count(primary_blocks)
    for name, raw_candidate in candidates.items():
        fallback = [block for block in raw_candidate if not (_block_pages(block) & excluded)]
        fallback_metrics = _assess_block_quality(fallback, expected_pages)
        candidate_metrics[name] = fallback_metrics
        if len(fallback) <= 1:
            continue
        if expected_pages is not None:
            primary_coverage = primary_metrics.get("page_coverage") or 0.0
            fallback_coverage = fallback_metrics.get("page_coverage") or 0.0
            if fallback_coverage < primary_coverage:
                continue
        elif _page_count(fallback) < base_pages:
            continue
        score = fallback_metrics.get("quality_score", 0.0)
        if score <= quality:
            continue
        _record_quality_report(
            quality_report,
            candidate_metrics,
            name,
            "primary_quality_below_threshold",
        )
        return fallback

    _record_quality_report(
        quality_report,
        candidate_metrics,
        "pymupdf",
        "no_better_fallback",
    )
    return primary_blocks


def apply_fallbacks(
    blocks: Iterable[Block],
    filepath: str,
    excluded: set[int],
    *,
    expected_pages: set[int] | None = None,
    quality_report: MutableMapping[str, Any] | None = None,
) -> Iterable[Block]:
    """Assess candidates and optionally replace degraded primary extraction.

    ``quality_report`` is an optional adapter-owned sink. Keeping it out of the
    returned block stream preserves the historical API while allowing callers to
    expose the decision in run reports.
    """
    primary = list(blocks)
    primary_metrics = _assess_block_quality(primary, expected_pages)
    if primary_metrics.get("quality_score", 0.0) >= _FALLBACK_THRESHOLD:
        _record_quality_report(
            quality_report,
            {"pymupdf": primary_metrics},
            "pymupdf",
            "primary_quality_acceptable",
        )
        return primary

    candidates: Sequence[tuple[str, Callable[[str, str | None], list[dict]]]] = (
        ("pdftotext", _extract_with_pdftotext),
        ("pdfminer", _extract_with_pdfminer if PDFMINER_AVAILABLE else (lambda *_: [])),
    )
    exclude_str = ",".join(map(str, sorted(excluded))) if excluded else None
    extracted = {name: extractor(filepath, exclude_str) for name, extractor in candidates}
    selected = select_best_extraction(
        primary,
        extracted,
        excluded,
        expected_pages=expected_pages,
        quality_report=quality_report,
    )
    return [block if isinstance(block, Block) else Block(**block) for block in selected]
