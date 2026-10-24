"""Read-only preparation of verified base article conversions for the runner."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import unicodedata
from collections.abc import Mapping
from importlib.metadata import distributions
from pathlib import Path
from typing import Any

from pdf_chunker.adapters.io_joplin_epub import describe_bound_article
from pdf_chunker.article_conversion import PreparedArticleConversion, resolve_article_spec
from pdf_chunker.config import PipelineSpec
from pdf_chunker.conversion_descriptor import _snapshot
from pdf_chunker.core_new import _load_document_metadata
from pdf_chunker.framework import registry
from pdf_chunker.passes.chunk_options import SplitOptions
from pdf_chunker.passes.emit_jsonl import EmitConfig
from pdf_chunker.passes.split_semantic import _SplitSemanticPass


def collect_article_runtime() -> dict[str, object]:
    """Fingerprint installed Python sources and distribution versions afresh.

    Include the whole package and environment conservatively: helper edits or
    optional readability dictionaries must not hide behind the package version.
    Only package-relative names and version labels are recorded, never credentials,
    host paths, Git messages or imported client representations. Missing/unreadable
    source files fail rather than silently omit a known implementation input.
    This is not an attestation of in-memory patches or external resource bytes.
    """
    root = Path(__file__).resolve().parents[1]
    sources = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.py"))
    }
    if not sources or "core_new.py" not in sources:
        raise ValueError("readable converter Python sources required")
    versions = sorted((dist.metadata["Name"], dist.version) for dist in distributions())
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": sys.version,
            "unicode": unicodedata.unidata_version,
        },
        "distributions": versions,
        "package_sources": sources,
    }


def prepare_bound_conversion(
    binding: Mapping[str, Any], spec: PipelineSpec, *, corpus: str
) -> PreparedArticleConversion:
    """Resolve actual defaults/versions and verify evidence without converting.

    Use ``convert(binding['path'], prepared.spec())`` to execute with those explicit
    settings and fresh source verification. The metadata file selected by the
    active parser must match the binding's native metadata exactly. No cache,
    receipt, source mutation, enrichment client or output file is created here.
    """
    splitter = registry()["split_semantic"]
    if not isinstance(splitter, _SplitSemanticPass):
        raise ValueError("canonical article splitter required")
    split_defaults = SplitOptions.from_base(
        splitter.chunk_size,
        splitter.overlap,
        splitter.min_chunk_size,
    )
    resolved = resolve_article_spec(
        spec,
        suffix=Path(binding["path"]).suffix,
        emit_defaults=EmitConfig.from_env(),
        split_defaults=split_defaults,
    )
    runtime = collect_article_runtime()
    implementation = hashlib.sha256(_snapshot(runtime).encode("utf-8")).hexdigest()
    descriptor = describe_bound_article(
        binding,
        corpus=corpus,
        contract={
            "implementation": implementation,
            "normalization": "approved-markdown-utf8-v1",
            "chunk_schema": "canonical-v1/contiguous_markdown_v3",
            "serialization": "python-json-ensure-ascii-false-v1",
            "configuration": {**resolved.model_dump(), "runtime": runtime},
        },
    )
    parser_options = resolved.options.get(resolved.pipeline[0], {})
    native = _load_document_metadata(parser_options.get("metadata_path"))
    if native != json.loads(descriptor.metadata_json):
        raise ValueError("native metadata differs from verified article evidence")
    return PreparedArticleConversion(descriptor, _snapshot(resolved.model_dump()))
