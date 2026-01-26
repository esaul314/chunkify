"""Transformation audit trail for text fragments.

This module provides lightweight logging of all transformations applied to text
as it flows through the pipeline. Each merge, split, clean, or other operation
records an entry explaining what happened and why.

Usage:
    log = TransformationLog.create(original_text)
    log.record("merged", "split_semantic", "Q&A sequence continuation",
               source=text_a, result=merged_text)

    # Later, for debugging:
    print(log.debug_view())
    # [split_semantic] merged: Q&A sequence continuation (a1b2c3d4 → e5f6g7h8)

Design philosophy:
- Zero overhead when not tracing (logs are optional dataclass)
- Immutable entries for audit integrity
- Human-readable output for debugging
- Hash-based identity for deduplication detection
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Literal

TransformKind = Literal[
    "extracted",  # Initial extraction from PDF/EPUB
    "cleaned",  # text_clean pass (ligatures, quotes, etc.)
    "merged",  # Cross-page merge, Q&A sequence, list continuation
    "split",  # Chunk boundary split
    "heading_attach",  # Heading attached to block
    "deduplicated",  # Duplicate sentence removed
    "pattern_match",  # Pattern registry matched and applied rule
    "interactive",  # User made decision via interactive mode
    "footer_strip",  # Footer artifact removed
    "overlap_trim",  # Boundary overlap trimmed
]


def _short_hash(text: str) -> str:
    """Return first 8 chars of MD5 hash for text identity."""
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()[:8]


@dataclass(frozen=True)
class TransformEntry:
    """Single transformation event in the audit trail.

    Attributes:
        kind: Category of transformation (merge, split, clean, etc.)
        pass_name: Which pipeline pass performed this transformation
        reason: Human-readable explanation of why this happened
        source_hash: Short hash of input text (for identity tracking)
        result_hash: Short hash of output text
        details: Optional extra context (pattern name, confidence, etc.)
    """

    kind: TransformKind
    pass_name: str
    reason: str
    source_hash: str
    result_hash: str
    details: dict[str, str | float | bool] | None = None

    def __str__(self) -> str:
        """Format as single-line log entry."""
        detail_str = ""
        if self.details:
            detail_str = f" {self.details}"
        return (
            f"[{self.pass_name}] {self.kind}: {self.reason} "
            f"({self.source_hash} → {self.result_hash}){detail_str}"
        )


@dataclass
class TransformationLog:
    """Audit trail for a text fragment through the pipeline.

    Tracks all transformations applied to a piece of text from extraction
    to final emission. Used by --trace mode for debugging and by regression
    tests to verify transformation behavior.

    Attributes:
        fragment_id: Stable identifier (hash of original extracted text)
        entries: Ordered list of transformation events
        original_text: First 100 chars of original text (for identification)
    """

    fragment_id: str
    entries: list[TransformEntry] = field(default_factory=list)
    original_preview: str = ""

    @classmethod
    def create(cls, original_text: str) -> TransformationLog:
        """Create a new log from the original extracted text."""
        return cls(
            fragment_id=_short_hash(original_text),
            original_preview=original_text[:100].replace("\n", "\\n"),
        )

    def record(
        self,
        kind: TransformKind,
        pass_name: str,
        reason: str,
        source: str,
        result: str,
        *,
        details: dict[str, str | float | bool] | None = None,
    ) -> None:
        """Record a transformation event.

        Args:
            kind: Category of transformation
            pass_name: Which pass performed this
            reason: Human-readable explanation
            source: Input text (hashed for storage)
            result: Output text (hashed for storage)
            details: Optional extra context
        """
        self.entries.append(
            TransformEntry(
                kind=kind,
                pass_name=pass_name,
                reason=reason,
                source_hash=_short_hash(source),
                result_hash=_short_hash(result),
                details=details,
            )
        )

    def debug_view(self) -> str:
        """Return human-readable transformation history."""
        lines = [
            f"Fragment {self.fragment_id}: {self.original_preview!r}",
            "-" * 60,
        ]
        for entry in self.entries:
            lines.append(str(entry))
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize for JSON output (e.g., in trace snapshots)."""
        return {
            "fragment_id": self.fragment_id,
            "original_preview": self.original_preview,
            "entries": [
                {
                    "kind": e.kind,
                    "pass_name": e.pass_name,
                    "reason": e.reason,
                    "source_hash": e.source_hash,
                    "result_hash": e.result_hash,
                    "details": e.details,
                }
                for e in self.entries
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> TransformationLog:
        """Deserialize from JSON (e.g., loading trace snapshots)."""
        log = cls(
            fragment_id=data["fragment_id"],
            original_preview=data.get("original_preview", ""),
        )
        for e in data.get("entries", []):
            log.entries.append(
                TransformEntry(
                    kind=e["kind"],
                    pass_name=e["pass_name"],
                    reason=e["reason"],
                    source_hash=e["source_hash"],
                    result_hash=e["result_hash"],
                    details=e.get("details"),
                )
            )
        return log


# ---------------------------------------------------------------------------
# Optional log holder for blocks/records that may or may not be traced
# ---------------------------------------------------------------------------


def maybe_record(
    log: TransformationLog | None,
    kind: TransformKind,
    pass_name: str,
    reason: str,
    source: str,
    result: str,
    *,
    details: dict[str, str | float | bool] | None = None,
) -> None:
    """Record transformation if log exists, otherwise no-op.

    This is the primary API for passes to use — it handles the common case
    where tracing may or may not be enabled without requiring conditionals
    at every call site.
    """
    if log is not None:
        log.record(kind, pass_name, reason, source, result, details=details)
