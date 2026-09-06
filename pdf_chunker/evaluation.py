"""Deterministic quality metrics for corpus conversion reports.

The evaluator deliberately measures observable output rather than assigning a single
quality score. This keeps text preservation, structure, metadata, and redundancy
regressions reviewable by category.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from itertools import pairwise
from statistics import mean, median
from typing import Any

_WHITESPACE_RE = re.compile(r"\s+")
_PAGE_RANGE_RE = re.compile(r"^\s*(\d+)\s*(?:[-–]\s*(\d+))?\s*$")

REQUIRED_METADATA_FIELDS: tuple[str, ...] = (
    "chunk_id",
    "source",
    "source_file",
    "utterance_type",
    "tags",
    "language",
    "block_type",
    "readability",
    "importance",
)


def _normalize(text: str) -> str:
    """Normalize text for comparisons without changing reported source text."""

    return _WHITESPACE_RE.sub(" ", text).strip().casefold()


def _tokens(text: str) -> list[str]:
    """Return whitespace-delimited comparison tokens."""

    return _normalize(text).split()


def _metadata(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return the public metadata mapping used by a row."""

    for key in ("metadata", "meta"):
        value = row.get(key)
        if isinstance(value, Mapping):
            return value
    return None


def _page_numbers(metadata: Mapping[str, Any] | None) -> set[int]:
    """Extract page numbers from public chunk metadata."""

    if metadata is None:
        return set()
    page_range = metadata.get("page_range")
    if (
        isinstance(page_range, (list, tuple))
        and len(page_range) == 2
        and all(isinstance(page, int) for page in page_range)
    ):
        start, end = page_range
        return set(range(min(start, end), max(start, end) + 1))
    if isinstance(page_range, str):
        match = _PAGE_RANGE_RE.match(page_range)
        if match:
            start = int(match.group(1))
            end = int(match.group(2) or start)
            return set(range(min(start, end), max(start, end) + 1))
    page = metadata.get("page")
    return {page} if isinstance(page, int) else set()


def _stats(values: Sequence[int]) -> dict[str, float | int]:
    """Return stable descriptive statistics for a non-empty integer sequence."""

    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0, "max": 0}
    return {
        "count": len(values),
        "mean": round(mean(values), 3),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def _suffix_prefix_overlap(previous: Sequence[str], current: Sequence[str]) -> int:
    """Return the largest exact token overlap between adjacent chunks."""

    limit = min(len(previous), len(current))
    return next(
        (size for size in range(limit, 0, -1) if list(previous[-size:]) == list(current[:size])),
        0,
    )


def _preservation_checks(
    text: str,
    sentinels: Iterable[str],
    ordered_fragments: Iterable[str],
) -> dict[str, Any]:
    """Check configured sentinel phrases and their relative order."""

    normalized = _normalize(text)
    sentinel_results = [
        {"phrase": phrase, "found": _normalize(phrase) in normalized} for phrase in sentinels
    ]
    fragments = list(ordered_fragments)
    positions = [normalized.find(_normalize(fragment)) for fragment in fragments]
    order_results = [
        {"phrase": phrase, "position": position, "found": position >= 0}
        for phrase, position in zip(fragments, positions, strict=True)
    ]
    found_positions = [position for position in positions if position >= 0]
    order_passed = len(found_positions) == len(positions) and found_positions == sorted(
        found_positions
    )
    return {
        "sentinels": {
            "expected": len(sentinel_results),
            "found": sum(item["found"] for item in sentinel_results),
            "passed": all(item["found"] for item in sentinel_results),
            "items": sentinel_results,
        },
        "text_order": {
            "expected": len(order_results),
            "passed": order_passed,
            "items": order_results,
        },
    }


def _metadata_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Report required-field completeness without hiding missing fields."""

    missing_counts: Counter[str] = Counter()
    complete = 0
    for row in rows:
        metadata = _metadata(row)
        missing = [
            field for field in REQUIRED_METADATA_FIELDS if metadata is None or field not in metadata
        ]
        if missing:
            missing_counts.update(missing)
        else:
            complete += 1
    total = len(rows)
    return {
        "complete_chunks": complete,
        "total_chunks": total,
        "completeness": round(complete / total, 4) if total else 1.0,
        "missing_fields": dict(sorted(missing_counts.items())),
    }


def _chunk_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Measure chunk sizes, exact duplicates, and adjacent overlap."""

    texts = [str(row.get("text", "")) for row in rows]
    words = [_tokens(text) for text in texts]
    char_counts = [len(text) for text in texts]
    word_counts = [len(item) for item in words]
    normalized = [_normalize(text) for text in texts]
    duplicate_count = len(normalized) - len(set(normalized))

    overlaps = [_suffix_prefix_overlap(previous, current) for previous, current in pairwise(words)]
    overlap_tokens = sum(overlaps)
    total_tokens = sum(word_counts)
    return {
        "count": len(rows),
        "characters": _stats(char_counts),
        "words": _stats(word_counts),
        "duplicate_rate": round(duplicate_count / len(rows), 4) if rows else 0.0,
        "duplicate_chunks": duplicate_count,
        "overlap": {
            "adjacent_pairs": len(overlaps),
            "pairs_with_overlap": sum(value > 0 for value in overlaps),
            "total_tokens": overlap_tokens,
            "average_tokens": round(mean(overlaps), 3) if overlaps else 0.0,
            "max_tokens": max(overlaps, default=0),
            "redundancy_rate": round(overlap_tokens / total_tokens, 4) if total_tokens else 0.0,
        },
    }


def _artifact_metrics(run_metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Collect removal and fallback signals already emitted by the runner."""

    text_clean = run_metrics.get("text_clean")
    if not isinstance(text_clean, Mapping):
        text_clean = {}
    page_artifacts = run_metrics.get("detect_page_artifacts")
    if not isinstance(page_artifacts, Mapping):
        page_artifacts = {}
    document_end = run_metrics.get("detect_doc_end")
    if not isinstance(document_end, Mapping):
        document_end = {}
    split = run_metrics.get("split_semantic")
    if not isinstance(split, Mapping):
        split = {}
    return {
        "inline_footers_stripped": int(text_clean.get("inline_footers_stripped", 0) or 0),
        "page_artifact_blocks_cleaned": int(page_artifacts.get("blocks_cleaned", 0) or 0),
        "truncated_pages": int(document_end.get("truncated_pages", 0) or 0),
        "deduplicated_chunks": int(split.get("deduplicated", 0) or 0),
    }


def evaluate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    page_count: int | None = None,
    run_metrics: Mapping[str, Any] | None = None,
    sentinels: Iterable[str] = (),
    ordered_fragments: Iterable[str] = (),
) -> dict[str, Any]:
    """Return deterministic quality metrics for emitted rows and run metadata."""

    materialized_rows = list(rows)
    combined_text = "\n\n".join(str(row.get("text", "")) for row in materialized_rows)
    covered_pages = (
        set().union(*(_page_numbers(_metadata(row)) for row in materialized_rows))
        if materialized_rows
        else set()
    )
    coverage = (
        None
        if page_count is None
        else round(len(covered_pages) / page_count, 4) if page_count else 1.0
    )
    return {
        "chunks": _chunk_metrics(materialized_rows),
        "page_coverage": {
            "page_count": page_count,
            "covered_pages": len(covered_pages),
            "coverage": coverage,
            "pages": sorted(covered_pages),
        },
        "metadata": _metadata_metrics(materialized_rows),
        "artifact_removals": _artifact_metrics(run_metrics or {}),
        "preservation": _preservation_checks(
            combined_text,
            sentinels,
            ordered_fragments,
        ),
    }


def quality_passed(metrics: Mapping[str, Any]) -> bool:
    """Return whether configured preservation checks passed."""

    preservation = metrics.get("preservation")
    if not isinstance(preservation, Mapping):
        return False
    return all(
        bool(check.get("passed")) for check in preservation.values() if isinstance(check, Mapping)
    )


def extraction_summary(run_metrics: Mapping[str, Any], *, kind: str) -> dict[str, Any]:
    """Return engine and fallback information from a runner report."""

    parser_metrics: Mapping[str, Any] = {}
    for name in ("pdf_parse", "epub_parse", "text_parse"):
        candidate = run_metrics.get(name)
        if isinstance(candidate, Mapping):
            parser_metrics = candidate
            break
    quality = parser_metrics.get("extraction_quality")
    quality_map = quality if isinstance(quality, Mapping) else {}
    fallback_used = kind == "pdf" and quality_map.get("reason") == (
        "primary_quality_below_threshold"
    )
    return {
        "engine": quality_map.get("selected_engine") or kind,
        "fallback_used": bool(fallback_used),
        "fallback_reason": quality_map.get("reason"),
        "candidate_scores": (
            dict(quality_map.get("candidate_scores", {}))
            if isinstance(quality_map.get("candidate_scores"), Mapping)
            else {}
        ),
    }


def parser_page_count(run_metrics: Mapping[str, Any]) -> int | None:
    """Return the page/spine count reported by the parser pass."""

    for name in ("pdf_parse", "epub_parse", "text_parse"):
        parser_metrics = run_metrics.get(name)
        if isinstance(parser_metrics, Mapping) and isinstance(parser_metrics.get("pages"), int):
            return parser_metrics["pages"]
    return None
