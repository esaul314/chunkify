from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pdf_chunker.fallbacks import select_best_extraction
from pdf_chunker.framework import Artifact, register


class _ExtractionFallbackPass:
    name = "extraction_fallback"
    input_type = dict
    output_type = dict

    def __call__(self, a: Artifact) -> Artifact:
        """Select an already-extracted candidate without touching the filesystem."""
        payload = a.payload if isinstance(a.payload, Mapping) else {}
        primary = payload.get("primary", payload.get("blocks", []))
        candidates = payload.get("candidates", {})
        excluded = payload.get("excluded_pages", set())
        expected_pages = payload.get("expected_pages")

        if not isinstance(primary, Sequence) or isinstance(primary, (str, bytes)):
            primary = []
        if not isinstance(candidates, Mapping):
            candidates = {}
        candidate_map = {
            str(name): list(values)
            for name, values in candidates.items()
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes))
        }
        excluded_set = (
            {int(page) for page in excluded}
            if isinstance(excluded, (set, frozenset, list, tuple))
            else set()
        )
        expected_set = (
            {int(page) for page in expected_pages}
            if isinstance(expected_pages, (set, frozenset, list, tuple))
            else None
        )
        quality_report: dict[str, Any] = {}
        selected = select_best_extraction(
            list(primary),
            candidate_map,
            excluded_set,
            expected_pages=expected_set,
            quality_report=quality_report,
        )
        result: dict[str, Any] = {"type": "blocks", "blocks": selected}
        if payload.get("source_path") is not None:
            result["source_path"] = payload["source_path"]
        meta: dict[str, Any] = {**(a.meta or {})}
        metrics_value = meta.get("metrics")
        metrics = dict(metrics_value) if isinstance(metrics_value, Mapping) else {}
        metrics["extraction_fallback"] = quality_report
        meta["metrics"] = metrics
        return Artifact(payload=result, meta=meta)


extraction_fallback = register(_ExtractionFallbackPass())
