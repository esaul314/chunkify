"""Pure contiguous Markdown slicing for verified article source artifacts."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import PurePath
from typing import Any, cast

from pdf_chunker.framework import Artifact
from pdf_chunker.passes.split_semantic_metadata import build_chunk_with_meta

PROOF_KEY = "verified_joplin_markdown"


def verified_text(artifact: Artifact) -> str:
    """Reject text mutations between verified I/O and semantic chunking."""
    proof = cast(dict[str, Any], (artifact.meta or {})[PROOF_KEY])
    doc = cast(dict[str, Any], artifact.payload)
    text = "".join(block["text"] for page in doc["pages"] for block in page["blocks"])
    if hashlib.sha256(text.encode()).hexdigest() != proof["content_sha256"]:
        raise ValueError("verified Joplin Markdown changed before chunking")
    return text


def slices(text: str, target_words: int) -> list[tuple[int, int]]:
    """Group complete paragraphs; fenced code/table lines stay verbatim.

    No overlap or deduplication: the ordered source slices concatenate exactly.
    An oversized atom remains intact for real serialized admission downstream.
    """
    ends = []
    fence = None
    offset = 0
    for line in text.splitlines(keepends=True):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token
            elif (
                token[0] == fence[0]
                and len(token) >= len(fence)
                and not line[marker.end() :].strip()
            ):
                fence = None
        offset += len(line)
        if fence is None and not line.strip():
            ends.append(offset)
    if not ends or ends[-1] != len(text):
        ends.append(len(text))
    result = []
    start = 0
    previous = 0
    for end in ends:
        if previous > start and len(text[start:end].split()) > target_words:
            result.append((start, previous))
            start = previous
        previous = end
    if start < len(text):
        result.append((start, len(text)))
    return result


def split_verified(artifact: Artifact, target_words: int) -> Artifact:
    text = verified_text(artifact)
    doc = cast(dict[str, Any], artifact.payload)
    metadata = doc.get("document_metadata")
    items = []
    for index, (start, end) in enumerate(slices(text, max(1, target_words))):
        chunk = build_chunk_with_meta(
            text[start:end],
            {},
            1,
            PurePath(doc["source_path"]).name,
            index,
            document_metadata=metadata if isinstance(metadata, Mapping) else None,
        )
        chunk["meta"].update(
            {
                "source_char_start": start,
                "source_char_end": end,
                "source_text_strategy": "contiguous_markdown_v1",
            }
        )
        items.append(chunk)
    meta: dict[str, Any] = {**(artifact.meta or {}), "semantic_pipeline": "canonical-v1"}
    meta["metrics"] = {
        **meta.get("metrics", {}),
        "split_semantic": {
            "chunks": len(items),
            "strategy": "contiguous_markdown_v1",
            "overlap": 0,
        },
    }
    return Artifact(payload={"type": "chunks", "items": items}, meta=meta)
