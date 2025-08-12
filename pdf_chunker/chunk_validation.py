from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List
import importlib


@dataclass(frozen=True)
class ValidationReport:
    """Structured result from :func:`validate_chunks`.

    Attributes capture anomaly counts and duplication analysis while
    remaining hashable for deterministic comparisons in tests.
    """

    total_chunks: int
    empty_text: int
    mid_sentence_starts: int
    overlong: int
    duplications: List[Dict[str, Any]]
    boundary_overlaps: List[Dict[str, Any]]

    def is_empty(self) -> bool:
        """Return ``True`` when no chunks were provided."""

        return self.total_chunks == 0

    def has_issues(self) -> bool:
        """Return ``True`` if the report contains anomalies or is empty."""

        return self.is_empty() or any(
            (
                self.empty_text,
                self.mid_sentence_starts,
                self.overlong,
                self.duplications,
                self.boundary_overlaps,
            )
        )


def _extract_texts(chunks: Iterable[Dict[str, Any]]) -> List[str]:
    return [chunk.get("text", "") for chunk in chunks]


def _count(predicate, texts: Iterable[str]) -> int:
    return sum(1 for text in texts if predicate(text))


def _is_empty(text: str) -> bool:
    return not text.strip()


def _starts_mid_sentence(text: str) -> bool:
    stripped = text.lstrip()
    return bool(stripped) and stripped[0].islower()


def _is_overlong(text: str, limit: int = 8000) -> bool:
    return len(text) > limit


def validate_chunks(chunks: Iterable[Dict[str, Any]]) -> ValidationReport:
    """Validate chunk structures and content.

    Parameters
    ----------
    chunks:
        Iterable of chunk dictionaries containing at minimum a ``text`` key.

    Returns
    -------
    ValidationReport
        Dataclass summarising counts and duplication analysis.
    """

    chunk_list = list(chunks)
    texts = _extract_texts(chunk_list)
    detect_duplicates = importlib.import_module("scripts.detect_duplicates")

    return ValidationReport(
        total_chunks=len(chunk_list),
        empty_text=_count(_is_empty, texts),
        mid_sentence_starts=_count(_starts_mid_sentence, texts),
        overlong=_count(_is_overlong, texts),
        duplications=detect_duplicates.detect_duplications(
            chunk_list, window_size=50, min_overlap=10
        ),
        boundary_overlaps=detect_duplicates.analyze_chunk_boundaries(chunk_list)[
            "boundary_overlaps"
        ],
    )
