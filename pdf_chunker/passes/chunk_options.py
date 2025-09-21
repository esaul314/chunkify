"""Utilities for configuring semantic chunking passes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from pdf_chunker.passes.sentence_fusion import _compute_limit


def derive_min_chunk_size(chunk_size: int, min_size: int | None) -> int:
    """Return ``min_size`` or derive it as a fraction of ``chunk_size``."""

    return min_size if min_size is not None else max(8, chunk_size // 10)


@dataclass(frozen=True)
class SplitOptions:
    """Resolved configuration for a semantic chunking pass."""

    chunk_size: int
    overlap: int
    min_chunk_size: int

    @classmethod
    def from_base(cls, chunk_size: int, overlap: int, min_chunk_size: int | None) -> SplitOptions:
        """Instantiate options from baseline pass settings."""

        return cls(
            int(chunk_size),
            int(overlap),
            derive_min_chunk_size(int(chunk_size), min_chunk_size),
        )

    def with_meta(self, meta: Mapping[str, Any] | None) -> SplitOptions:
        """Merge artifact metadata overrides into the option record."""

        opts = ((meta or {}).get("options") or {}).get("split_semantic", {})
        if not opts:
            return self
        chunk = int(opts.get("chunk_size", self.chunk_size))
        overlap = int(opts.get("overlap", self.overlap))
        min_chunk = (
            int(opts["min_chunk_size"])
            if "min_chunk_size" in opts
            else (
                self.min_chunk_size
                if "chunk_size" not in opts
                else derive_min_chunk_size(chunk, None)
            )
        )
        return SplitOptions(chunk, overlap, min_chunk)

    def compute_limit(self) -> int | None:
        """Return the maximum word count allowed for continuation stitching."""

        return _compute_limit(self.chunk_size, self.overlap, self.min_chunk_size)


@dataclass(frozen=True)
class SplitMetrics:
    """Capture chunk metrics and merge them back into artifact metadata."""

    chunk_count: int
    extra: Mapping[str, int | bool] | None = None

    def apply(self, meta: Mapping[str, Any] | None) -> dict[str, Any]:
        """Return ``meta`` updated with split metrics."""

        extra_metrics = dict(self.extra or {})
        metrics = {"chunks": self.chunk_count, **extra_metrics}
        existing = ((meta or {}).get("metrics") or {}).get("split_semantic", {})
        merged = {**existing, **metrics}
        all_metrics = (meta or {}).get("metrics") or {}
        return {
            **(meta or {}),
            "metrics": {**all_metrics, "split_semantic": merged},
        }


__all__ = ["SplitMetrics", "SplitOptions", "derive_min_chunk_size"]
