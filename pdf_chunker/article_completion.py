"""Pure, hash-pinned completion records for the bounded synchronous article API.

These are caller-retained facts, not signatures or child-process receipts. Keep
their trusted hash separately; public construction never proves execution.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from pdf_chunker.article_output import ValidatedArticleOutput, validate_article_output
from pdf_chunker.conversion_descriptor import (
    ConversionDescriptor,
    _read_mapping_snapshot,
    _require_sha256,
    _snapshot,
    read_conversion_descriptor,
)


@dataclass(frozen=True)
class ArticleCompletion:
    """Immutable full input descriptor and completed output facts (schema 1)."""

    descriptor: ConversionDescriptor
    output_sha256: str
    size_bytes: int
    row_count: int

    def to_json(self) -> str:
        """Serialize canonical literal-Unicode JSON, with no terminal newline."""
        return _snapshot(
            {
                "schema": 1,
                "status": "succeeded",
                "execution": "guarded-synchronous-v1",
                "descriptor": json.loads(self.descriptor.to_json()),
                "output": {
                    "sha256": self.output_sha256,
                    "size_bytes": self.size_bytes,
                    "row_count": self.row_count,
                },
            }
        )

    @property
    def sha256(self) -> str:
        """Hash to retain independently at a trusted completed-run boundary."""
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


def describe_article_completion(
    output: ValidatedArticleOutput, *, descriptor: ConversionDescriptor
) -> ArticleCompletion:
    """Snapshot revalidated output facts; this pure function cannot prove execution."""
    restored = read_conversion_descriptor(descriptor.to_json())
    checked = validate_article_output(
        output.jsonl, descriptor=restored, expected_sha256=output.sha256
    )
    if type(output.row_count) is not int or checked != output:
        raise ValueError("article output facts differ from validated bytes")
    return ArticleCompletion(restored, checked.sha256, checked.size_bytes, checked.row_count)


def read_article_completion(
    data: bytes, *, descriptor: ConversionDescriptor, expected_sha256: str
) -> ArticleCompletion:
    """Read pinned canonical completion bytes and require the current descriptor.

    Check the independently retained hash before parsing. Only the declared
    synchronous success form is supported; no filename, log or exit code supplies
    completion authority. This does not read or validate the output artifact.
    """
    if not isinstance(data, bytes) or hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("article completion SHA-256 mismatch")
    value = _read_mapping_snapshot(data.decode("utf-8"))
    if (
        set(value) != {"schema", "status", "execution", "descriptor", "output"}
        or type(value["schema"]) is not int
        or value["schema"] != 1
        or value["status"] != "succeeded"
        or value["execution"] != "guarded-synchronous-v1"
    ):
        raise ValueError("supported successful article completion required")
    restored = read_conversion_descriptor(_snapshot(value["descriptor"]))
    if restored != descriptor:
        raise ValueError("article completion descriptor differs from current preparation")
    facts = value["output"]
    if (
        not isinstance(facts, dict)
        or set(facts) != {"sha256", "size_bytes", "row_count"}
        or any(
            type(facts[key]) is not int or facts[key] <= 0 for key in ("size_bytes", "row_count")
        )
    ):
        raise ValueError("positive integer article output facts required")
    _require_sha256(facts["sha256"])
    return ArticleCompletion(restored, facts["sha256"], facts["size_bytes"], facts["row_count"])


def validate_completed_article(
    data: bytes, *, completion: ArticleCompletion
) -> ValidatedArticleOutput:
    """Revalidate actual JSONL bytes and counts against previously pinned completion."""
    output = validate_article_output(
        data, descriptor=completion.descriptor, expected_sha256=completion.output_sha256
    )
    if (
        type(completion.size_bytes) is not int
        or type(completion.row_count) is not int
        or (output.size_bytes, output.row_count) != (completion.size_bytes, completion.row_count)
    ):
        raise ValueError("article output counts differ from completion")
    return output
