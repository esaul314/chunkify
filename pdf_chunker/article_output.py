"""Pure validation of bounded base article JSONL, separate from completion proof.

Inputs are explicit: bytes, their independently retained hash and a descriptor.
No source, environment, dictionary, cache or receipt is read or written here.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePath
from typing import Any

from pdf_chunker.conversion_descriptor import ConversionDescriptor, _snapshot
from pdf_chunker.passes.emit_jsonl import _compat_chunk_id
from pdf_chunker.utils import _generate_chunk_id, _normalize_list_block


@dataclass(frozen=True)
class ValidatedArticleOutput:
    """Detached output facts; validation alone is not a completion receipt.

    Keep the hash from a trusted completed run independently of cached bytes.
    Public construction of this value does not grant source or proof authority.
    """

    conversion_key: str
    jsonl: bytes
    row_count: int

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.jsonl).hexdigest()

    @property
    def size_bytes(self) -> int:
        return len(self.jsonl)

    def rows(self) -> list[dict[str, Any]]:
        """Return fresh JSON containers without exposing mutable stored state."""
        return [json.loads(line) for line in self.jsonl.decode("utf-8").split("\n")[:-1]]


@dataclass(frozen=True)
class ArticleOutputPreflight:
    """Byte-budget facts for validated base output, with caller-declared headroom.

    Row sizes include the complete serialized JSON object, excluding its LF
    delimiter. Projected file size includes every LF and reserves the same net
    enrichment growth per row. Oversized row numbers are one-based. ``fits``
    applies only to these bytes and reserve, not to future enrichment results.
    """

    output: ValidatedArticleOutput
    row_sizes_bytes: tuple[int, ...]
    max_row_bytes: int
    enrichment_headroom_bytes: int

    @property
    def largest_row_bytes(self) -> int:
        return max(self.row_sizes_bytes)

    @property
    def projected_output_bytes(self) -> int:
        return self.output.size_bytes + self.enrichment_headroom_bytes * self.output.row_count

    @property
    def oversized_rows(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, size in enumerate(self.row_sizes_bytes, 1)
            if size + self.enrichment_headroom_bytes > self.max_row_bytes
        )

    @property
    def fits(self) -> bool:
        return not self.oversized_rows


def preflight_article_output(
    data: bytes,
    *,
    descriptor: ConversionDescriptor,
    expected_sha256: str,
    max_row_bytes: int,
    enrichment_headroom_bytes: int = 0,
) -> ArticleOutputPreflight:
    """Validate retained base output and report complete UTF-8 row admission.

    Data-first and usable with ``partial``. Require a positive integer byte limit
    and a nonnegative integer reserve; booleans/coercions refuse. Over-budget
    output returns ``fits=False`` and all one-based ``oversized_rows`` for review,
    without changing text, IDs, provenance or bytes. Invalid output still raises
    through the existing validator. Callers must check the report before expensive
    enrichment and validate actual enriched rows afterwards. No I/O, automatic
    splitting, enrichment execution or new completion authority is introduced.
    """
    if type(max_row_bytes) is not int or max_row_bytes <= 0:
        raise ValueError("max_row_bytes must be a positive integer number of bytes")
    if type(enrichment_headroom_bytes) is not int or enrichment_headroom_bytes < 0:
        raise ValueError("enrichment_headroom_bytes must be a nonnegative integer number of bytes")
    output = validate_article_output(data, descriptor=descriptor, expected_sha256=expected_sha256)
    return ArticleOutputPreflight(
        output,
        tuple(len(line) for line in output.jsonl.split(b"\n")[:-1]),
        max_row_bytes,
        enrichment_headroom_bytes,
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    if len({key for key, _ in pairs}) != len(pairs):
        raise ValueError("duplicate article JSON key")
    return dict(pairs)


def _decode_output(data: bytes) -> list[dict[str, Any]]:
    if not data or not data.endswith(b"\n"):
        raise ValueError("nonempty article JSONL with a terminal newline required")
    rows = [
        json.loads(line, object_pairs_hook=_unique_object)
        for line in data.decode("utf-8").split("\n")[:-1]
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("article JSONL rows must be objects")
    _snapshot(rows)  # Refuse nonfinite values and invalid Unicode, without coercion.
    expected = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
    if expected != data:
        raise ValueError("article JSONL differs from the pinned serialization policy")
    return rows


def _emission_settings(descriptor: ConversionDescriptor) -> Mapping[str, Any]:
    contract = json.loads(descriptor.contract_json)
    versions = {
        "normalization": "approved-markdown-utf8-v1",
        "chunk_schema": "canonical-v1/contiguous_markdown_v3",
        "serialization": "python-json-ensure-ascii-false-v1",
    }
    if any(contract.get(key) != value for key, value in versions.items()):
        raise ValueError("unsupported article output contract")
    emit = contract.get("configuration", {}).get("options", {}).get("emit_jsonl", {})
    if (
        not isinstance(emit.get("metadata_key"), str)
        or emit["metadata_key"] in {"", "text"}
        or type(emit.get("max_chars")) is not int
        or emit["max_chars"] <= 0
    ):
        raise ValueError("resolved article emission settings required")
    byte_limit = emit.get("max_row_bytes")
    if byte_limit is not None and (type(byte_limit) is not int or byte_limit <= 0):
        raise ValueError("article max_row_bytes must be a positive integer or None")
    return emit


def _expected_metadata(
    native: Mapping[str, Any], *, text: str, filename: str, index: int, start: int, end: int
) -> dict[str, Any]:
    """Declare the existing bounded schema's generated-field precedence.

    Readability values are schema-checked separately; validating them must not
    invoke lazy dictionaries or re-run an effectful converter from the pure core.
    """
    normalized = _normalize_list_block({}, chunk_text=text)
    generated: dict[str, Any] = {
        "source": filename if native.get("source") is None else native["source"],
        "source_file": filename if native.get("source_file") is None else native["source_file"],
        "chunk_id": _compat_chunk_id(_generate_chunk_id(filename, 1, index)),
        "page": 1,
        "page_range": "1",
        "location": None,
        "block_type": normalized.get("type", "paragraph"),
        "language": "un",
        "utterance_type": "unclassified",
        "tags": [],
        "importance": "medium",
        "source_char_start": start,
        "source_char_end": end,
        "source_text_strategy": "contiguous_markdown_v3",
        **({"list_kind": normalized["list_kind"]} if normalized.get("list_kind") else {}),
    }
    provenance = {
        key: value
        for key, value in native.items()
        if value is not None and key not in {*generated, "filename", "readability"}
    }
    return {**provenance, **generated}


def _validate_readability(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {"flesch_kincaid_grade", "difficulty"}:
        raise ValueError("article readability schema required")
    grade = value["flesch_kincaid_grade"]
    if (
        type(grade) not in {int, float}
        or not isinstance(value["difficulty"], str)
        or value["difficulty"]
        not in {"elementary", "middle_school", "high_school", "college_level", "graduate_level"}
    ):
        raise ValueError("invalid article readability fields")


def validate_article_output(
    data: bytes, *, descriptor: ConversionDescriptor, expected_sha256: str
) -> ValidatedArticleOutput:
    """Validate complete base JSONL, data-first and usable with ``partial``.

    Check the caller's retained output hash before parsing, strict literal-Unicode
    JSONL, declared complete-row character/optional UTF-8 budgets, ordered IDs and
    metadata/provenance, contiguous character offsets and exact source hash.
    Nonfinite JSON values have already been rejected before schema checks.
    A matching hash supplied alongside corrupt bytes is not an integrity anchor;
    this function does not establish approval, execution or terminal completion.
    """
    if not isinstance(data, bytes) or hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("article output SHA-256 mismatch")
    emit = _emission_settings(descriptor)
    rows = _decode_output(data)
    native = json.loads(descriptor.metadata_json)
    filename = PurePath(descriptor.source_path).name
    start = 0
    for index, row in enumerate(rows):
        metadata_key = emit["metadata_key"]
        text, meta = row.get("text"), row.get(metadata_key)
        if (
            set(row) != {"text", metadata_key}
            or not isinstance(text, str)
            or not text
            or not isinstance(meta, dict)
        ):
            raise ValueError(f"invalid article row schema: {index}")
        if any(
            not isinstance(meta.get(key), str) or not meta[key] for key in ("source", "source_file")
        ):
            raise ValueError(f"article source labels required: {index}")
        end = start + len(text)
        expected = _expected_metadata(
            native, text=text, filename=filename, index=index, start=start, end=end
        )
        actual = {key: value for key, value in meta.items() if key != "readability"}
        if _snapshot(actual) != _snapshot(expected):
            raise ValueError(f"article metadata, order or source offsets differ: {index}")
        _validate_readability(meta.get("readability"))
        serialized = json.dumps(row, ensure_ascii=False)
        if len(serialized) > emit["max_chars"]:
            raise ValueError(f"article row exceeds JSONL size limit: {index}")
        byte_limit = emit.get("max_row_bytes")
        if byte_limit is not None and len(serialized.encode("utf-8")) > byte_limit:
            raise ValueError(f"article row exceeds JSONL byte limit: {index}")
        start = end
    text_hash = hashlib.sha256("".join(row["text"] for row in rows).encode("utf-8")).hexdigest()
    if text_hash != descriptor.content_sha256:
        raise ValueError("article output does not reconstruct the complete source")
    return ValidatedArticleOutput(descriptor.key, data, len(rows))
