"""Preparation and guarded in-memory execution of verified base articles."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import unicodedata
from collections.abc import Mapping
from importlib.metadata import distributions
from pathlib import Path
from typing import Any, cast

from pdf_chunker.adapters.emit_jsonl import _serialize
from pdf_chunker.adapters.io_joplin_epub import describe_bound_article
from pdf_chunker.adapters.readability_runtime import collect_readability_runtime
from pdf_chunker.article_completion import (
    ArticleCompletion,
    describe_article_completion,
    read_article_completion,
    validate_completed_article,
)
from pdf_chunker.article_conversion import PreparedArticleConversion, resolve_article_spec
from pdf_chunker.article_output import ValidatedArticleOutput, validate_article_output
from pdf_chunker.config import PipelineSpec
from pdf_chunker.conversion_descriptor import _snapshot
from pdf_chunker.core_new import _load_document_metadata, convert
from pdf_chunker.framework import registry
from pdf_chunker.passes.chunk_options import SplitOptions
from pdf_chunker.passes.emit_jsonl import EmitConfig
from pdf_chunker.passes.split_semantic import _SplitSemanticPass


def collect_article_runtime() -> dict[str, object]:
    """Fingerprint installed Python sources and distribution versions afresh.

    Include the whole package and environment conservatively: helper edits and
    readability resource/table changes must not hide behind the package version.
    Only package-relative names and version labels are recorded, never credentials,
    host paths, Git messages or imported client representations. Missing/unreadable
    source files fail rather than silently omit a known implementation input.
    Readability pins include resource bytes and loaded tables; inconsistent word
    caches refuse. This is not an attestation of imported code or concurrent edits.
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
        "readability": collect_readability_runtime(),
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


def execute_bound_conversion(
    binding: Mapping[str, Any], *, prepared: PreparedArticleConversion, corpus: str
) -> ValidatedArticleOutput:
    """Return validated bytes only after normal completion and matching snapshots.

    Re-verify evidence, metadata, resolved settings and observed runtime before
    and after the existing synchronous runner. Failures propagate without a
    completion value, file write or shared-state repair. Ordinary cache warming
    is allowed. This detects persistent drift, not transient edits/replaced
    callables or changes after import; callers must keep the environment stable.
    No child process or durable receipt is involved, and no cached output is reused.
    """
    spec = prepared.spec()
    before = prepare_bound_conversion(binding, spec, corpus=corpus)
    if before != prepared:
        raise ValueError("prepared article contract changed before conversion")
    rows = convert(str(binding["path"]), spec)
    data = "".join(line + "\n" for line in _serialize(cast(list[dict[str, Any]], rows))).encode(
        "utf-8"
    )
    after = prepare_bound_conversion(binding, spec, corpus=corpus)
    if after != prepared:
        raise ValueError("prepared article contract changed during conversion")
    return validate_article_output(
        data, descriptor=prepared.descriptor, expected_sha256=hashlib.sha256(data).hexdigest()
    )


def complete_bound_conversion(
    binding: Mapping[str, Any], *, prepared: PreparedArticleConversion, corpus: str
) -> tuple[ArticleCompletion, ValidatedArticleOutput]:
    """Return completion facts/output after guarded execution, without persistence.

    The caller owns storing these artifacts and independently retaining the
    completion hash. Exceptions return neither completion nor partial output.
    This synchronous record has no child exit code or resumable stage lifecycle.
    """
    output = execute_bound_conversion(binding, prepared=prepared, corpus=corpus)
    return describe_article_completion(output, descriptor=prepared.descriptor), output


def read_bound_conversion(
    binding: Mapping[str, Any],
    spec: PipelineSpec,
    *,
    corpus: str,
    completion_path: str | Path,
    output_path: str | Path,
    expected_completion_sha256: str,
) -> ValidatedArticleOutput:
    """Read existing completed artifacts with fresh preparation and strict checks.

    Require a hash independently retained from trusted completed execution, never
    freshly derived from arbitrary cached records. Re-verify source/settings and
    observed runtime before/after reads. Invalid/missing/stale artifacts raise;
    there is no implicit conversion, repair, write or path-based authority.
    Keep runtime stable as for guarded execution; only lasting drift is observed.
    """
    before = prepare_bound_conversion(binding, spec, corpus=corpus)
    completion = read_article_completion(
        Path(completion_path).read_bytes(),
        descriptor=before.descriptor,
        expected_sha256=expected_completion_sha256,
    )
    output = validate_completed_article(Path(output_path).read_bytes(), completion=completion)
    after = prepare_bound_conversion(binding, spec, corpus=corpus)
    if after != before:
        raise ValueError("prepared article contract changed during readback")
    return output
