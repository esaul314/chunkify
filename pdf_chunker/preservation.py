"""Bounded ordered native-text accounting, separate from semantic role decisions.

Full original/derived text stays in this private sidecar model. Rows carry only
fixed-size record references. No word sets, whole-book diffs or inferred approvals.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any, NamedTuple

VERSION = "native-source-spans-v1"
MAX_CHARACTERS = 8_000_000
MAX_UNITS = 50_000
MAX_UNIT_CHARACTERS = 256_000
MAX_CONTEXT_TOKENS = 8192
MAX_ROW_TRANSITIONS = 64
_LETTER_HYPHEN = re.compile(r"(?<=[^\W\d_])[-\u00ad]\s*\n\s*(?=[^\W\d_])")
_QUOTES = str.maketrans("\u2018\u2019\u201c\u201d", "''\"\"")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def identity(value: Any) -> str:
    import json

    return sha(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def settings_identity(options: Mapping[str, Any], pipeline: Sequence[str], reader: Any) -> str:
    """Bind actual stage order, effective reader and accounting algorithm."""
    ignored = {
        "answers",
        "answers_path",
        "questions_path",
        "callback",
        "interactive",
        "output_path",
        "preservation_path",
    }
    bound = {
        name: {key: value for key, value in values.items() if key not in ignored}
        for name, values in options.items()
    }
    return identity(
        {"options": bound, "pipeline": list(pipeline), "reader": reader, "accounting": VERSION}
    )


def normalize_source_text(text: str) -> str:
    """Only declared, content-preserving glyph/line normalization."""
    return unicodedata.normalize("NFKC", _LETTER_HYPHEN.sub("", text)).translate(_QUOTES)


class Token(NamedTuple):
    text: str
    start: int
    end: int
    positions: tuple[tuple[int, int], ...] | None


def tokens(text: str) -> list[Token]:
    """Canonical atoms with exact original character positions; digits stay atomic."""
    if len(text) > MAX_UNIT_CHARACTERS:
        raise ValueError("source accounting unit exceeds character limit")
    deleted: set[int] = set()
    for match in _LETTER_HYPHEN.finditer(text):
        deleted.update(range(match.start(), match.end()))
    chars: list[str] = []
    positions: list[tuple[int, int]] = []
    clusters: list[list[tuple[int, str]]] = []
    for index, char in enumerate(text):
        if index in deleted:
            continue
        if clusters and unicodedata.combining(char):
            clusters[-1].append((index, char))
        else:
            clusters.append([(index, char)])
    for cluster in clusters:
        value = "".join(char for _, char in cluster)
        normalized = unicodedata.normalize("NFKC", value).translate(_QUOTES)
        chars.extend(normalized)
        positions.extend([(cluster[0][0], cluster[-1][0] + 1)] * len(normalized))
    normalized = "".join(chars)
    result = []
    for match in re.finditer(r"\w+|[^\w\s]", normalized):
        indexes = positions[match.start() : match.end()]
        start, end = indexes[0][0], indexes[-1][1]
        result.append(
            Token(
                match.group(),
                start,
                end,
                None if text[start:end] == match.group() else tuple(indexes),
            )
        )
    return result


def _values(text: str) -> list[str]:
    return [token.text for token in tokens(text)]


def _put(texts: dict[str, str], text: str) -> str:
    key = sha(text)
    texts[key] = text
    return key


def _raw_key(raw: Mapping[str, Any]) -> str:
    return identity(dict(raw))


def _metadata(item: Mapping[str, Any]) -> dict[str, Any]:
    value = item.get("metadata") or item.get("meta")
    if not isinstance(value, Mapping):
        value = next(
            (
                value
                for value in item.values()
                if isinstance(value, Mapping) and "chunk_id" in value
            ),
            {},
        )
    return dict(value)


def _page_units(payload: Mapping[str, Any], texts: dict[str, str]) -> list[dict[str, Any]]:
    result = []
    for page in payload.get("pages", ()):
        for block in page.get("blocks", ()):
            source = block.get("source") or {}
            raw = source.get("raw_source")
            result.append(
                {
                    "source_id": _raw_key(raw) if isinstance(raw, Mapping) else None,
                    "page": page.get("page"),
                    "source_page": source.get("page"),
                    "text": _put(texts, str(block.get("text", ""))),
                    "section_id": (block.get("section") or {}).get("section_id"),
                    "section_role": (block.get("section") or {}).get("section_role"),
                    "section_span": block.get("_section_span"),
                    "parent_text": block.get("_section_parent_sha256"),
                    "block_ids": [
                        entry["id"] for entry in source.get("block_provenance", ()) if "id" in entry
                    ],
                    "method": source.get("method", "native"),
                }
            )
    if len(result) > MAX_UNITS:
        raise ValueError("source accounting unit limit exceeded")
    return result


def initialize(doc: Mapping[str, Any], settings: Mapping[str, Any]) -> dict[str, Any]:
    """Bind raw glyph blocks before adapter cleanup; annotations remain explicit gaps."""
    structure = doc.get("source_structure") or {}
    texts: dict[str, str] = {}
    raw, pending, exclusions = [], [], []
    from pdf_chunker.page_utils import parse_page_ranges

    parse = settings.get("pdf_parse") or {}
    declared = parse.get("exclude_pages")
    excluded_pages = (
        set(declared)
        if isinstance(declared, (list, tuple, set))
        else set(parse_page_ranges(declared)) if declared else set()
    )
    total = 0
    for page in structure.get("pages", ()):
        number = page["page"]
        if number in excluded_pages:
            exclusions.append({"page": number, "reason": "explicit_page_exclusion"})
            continue
        for block in page.get("blocks", ()):
            text = str(block.get("raw_text") or block["text"])
            total += len(text)
            origin = {
                "page": number,
                "block_index": block["index"],
                "bbox": block["bbox"],
                "text_sha256": sha(text),
                "span": [0, len(text)],
            }
            key = _raw_key(origin)
            excluded = False
            bbox = block["bbox"]
            height = page.get("height")
            if parse.get("header_margin") is not None and bbox[1] < parse["header_margin"]:
                excluded = True
            if (
                height
                and parse.get("footer_margin") is not None
                and bbox[3] > height - parse["footer_margin"]
            ):
                excluded = True
            raw.append(
                {
                    "source_id": key,
                    "origin": origin,
                    "text": _put(texts, text),
                    "excluded": excluded,
                }
            )
            if excluded:
                exclusions.append({"source_id": key, "reason": "configured_geometric_exclusion"})
            layout_text = block.get("layout_text")
            if layout_text is not None and _values(text) != _values(str(layout_text)):
                pending.append({"source_id": key, "reason": "reader_font_text_disagreement"})
        for annotation in page.get("annotations", ()):
            if annotation.get("text"):
                pending.append(
                    {
                        "page": number,
                        "reason": "annotation_object_not_in_glyph_stream",
                        "annotation": annotation,
                    }
                )
        if not page.get("blocks") and page.get("image_count"):
            pending.append({"page": number, "reason": "image_only_page_not_text_coverage"})
    if total > MAX_CHARACTERS or len(raw) > MAX_UNITS:
        raise ValueError("source accounting document limit exceeded")
    report = {
        "schema": VERSION,
        "source_sha256": structure.get("source_sha256"),
        "raw_manifest_sha256": identity(structure),
        "settings_sha256": structure.get("pipeline_settings_sha256"),
        "reader": structure.get(
            "reader", {"mode": "native-physical-blocks-v1", "version": "unbound"}
        ),
        "source_filename": str(doc.get("source_path") or "").replace("\\", "/").rsplit("/", 1)[-1],
        "overlap_words": int((settings.get("split_semantic") or {}).get("overlap", 50)),
        "scope": (
            "Native extracted glyph text; annotation objects and image-only "
            "surfaces require review."
        ),
        "limits": {
            "characters": MAX_CHARACTERS,
            "units": MAX_UNITS,
            "unit_characters": MAX_UNIT_CHARACTERS,
            "context_tokens": MAX_CONTEXT_TOKENS,
        },
        "texts": texts,
        "source_units": raw,
        "exclusions": exclusions,
        "pending": pending,
        "stages": [],
    }
    return record_stage(report, "adapter", doc)


def _page_check(report: dict[str, Any], units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    raw = {unit["source_id"]: unit for unit in report["source_units"]}
    actual_order: list[str] = []
    pending = []
    for unit in units:
        key = unit["source_id"]
        if key not in raw:
            pending.append({"reason": "unbound_source_identity", "page": unit["page"]})
            continue
        if (
            type(unit["page"]) is not int
            or type(unit["source_page"]) is not int
            or unit["page"] != raw[key]["origin"]["page"]
            or unit["source_page"] != unit["page"]
        ):
            pending.append({"reason": "source_page_identity_mismatch", "source_id": key})
        span = unit.get("section_span")
        if span is not None:
            parent = report["texts"].get(unit.get("parent_text"))
            if (
                not isinstance(span, (list, tuple))
                or len(span) != 2
                or any(type(value) is not int for value in span)
                or not isinstance(parent, str)
                or not 0 <= span[0] <= span[1] <= len(parent)
                or parent[span[0] : span[1]] != report["texts"][unit["text"]]
            ):
                pending.append({"reason": "section_parent_span_mismatch", "source_id": key})
        grouped[key].append(report["texts"][unit["text"]])
        if not actual_order or actual_order[-1] != key:
            actual_order.append(key)
    expected_order = [unit["source_id"] for unit in report["source_units"] if not unit["excluded"]]
    if actual_order != expected_order:
        pending.append(
            {
                "reason": "missing_duplicate_or_reordered_source_units",
                "expected": expected_order,
                "actual": actual_order,
            }
        )
    for key, unit in raw.items():
        if unit["excluded"]:
            continue
        original = report["texts"][unit["text"]]
        actual = "\n".join(grouped.get(key, ()))
        if _values(original) != _values(actual):
            pending.append(
                {
                    "reason": "unexplained_source_text_edit",
                    "source_id": key,
                    "before": unit["text"],
                    "after": _put(report["texts"], actual),
                }
            )
    return pending


def _logical_rows(payload: Any) -> list[dict[str, Any]]:
    items = payload.get("items", ()) if isinstance(payload, Mapping) else payload
    result: list[dict[str, Any]] = []
    for index, item in enumerate(items or ()):
        if not isinstance(item, Mapping) or not isinstance(item.get("text"), str):
            continue
        meta = _metadata(item)
        key = meta.get("chunk_id", str(index))
        part = meta.get("chunk_part")
        if part is not None and part != 0 and result and result[-1]["id"] == key:
            if part != len(result[-1]["parts"]):
                raise ValueError("serialized siblings have stale/reordered part identities")
            result[-1]["parts"].append(item)
            result[-1]["text"] += item["text"]
        else:
            if part not in (None, 0):
                raise ValueError("serialized sibling starts without its first part")
            result.append({"id": key, "metadata": meta, "text": item["text"], "parts": [item]})
    return result


def _longest_prefix(pattern: list[str], haystack: list[str]) -> tuple[int, int]:
    """KMP: longest pattern prefix occurring in a bounded, already consumed window."""
    if not pattern or not haystack:
        return 0, 0
    sequence: list[Any] = [*pattern, object(), *haystack]
    prefix = [0] * len(sequence)
    best, end = 0, 0
    for index in range(1, len(sequence)):
        length = prefix[index - 1]
        while length and sequence[index] != sequence[length]:
            length = prefix[length - 1]
        if sequence[index] == sequence[length]:
            length += 1
        prefix[index] = length
        if index > len(pattern) and length >= best:
            best, end = length, index - len(pattern)
    return best, end - best


def _match_output(
    report: dict[str, Any], payload: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    page_stage = next((stage for stage in reversed(report["stages"]) if "units" in stage), None)
    if page_stage is None:
        return [], [{"reason": "missing_classified_source_stage"}]
    raw = {unit["source_id"]: unit for unit in report["source_units"]}
    source_tokens = {key: tokens(report["texts"][value["text"]]) for key, value in raw.items()}
    offsets: dict[str, int] = defaultdict(int)
    expected: dict[Any, list[tuple[str, str, Token, str | None]]] = defaultdict(list)
    for unit in page_stage["units"]:
        key = unit["source_id"]
        if key not in raw:
            continue
        unit_tokens = tokens(report["texts"][unit["text"]])
        for token in unit_tokens:
            position = offsets[key]
            originals = source_tokens[key]
            if position >= len(originals) or originals[position].text != token.text:
                return [], [{"reason": "unaccounted_classified_source_tokens", "source_id": key}]
            expected[unit["section_id"]].append(
                (
                    token.text,
                    key,
                    originals[position],
                    unit.get("block_ids", [None])[0] if unit.get("block_ids") else None,
                )
            )
            offsets[key] += 1
    values_by_section = {
        section: [value[0] for value in stream] for section, stream in expected.items()
    }
    roles_by_section = {
        unit["section_id"]: unit.get("section_role") for unit in page_stage["units"]
    }
    cursor: dict[Any, int] = defaultdict(int)
    records: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    logical_rows = _logical_rows(payload)
    actual_sections: list[Any] = []
    for logical in logical_rows:
        section_id = logical["metadata"].get("section_id")
        if tokens(logical["text"]) and (not actual_sections or actual_sections[-1] != section_id):
            actual_sections.append(section_id)
    expected_sections = [section for section, stream in expected.items() if stream]
    if actual_sections != expected_sections:
        pending.append(
            {
                "reason": "section_document_order_mismatch",
                "expected": expected_sections,
                "actual": actual_sections,
            }
        )
    for logical in logical_rows:
        section = logical["metadata"].get("section_id")
        stream = expected.get(section, [])
        values = values_by_section.get(section, [])
        output = tokens(logical["text"])
        at, transitions = 0, 0
        pairs: list[tuple[str, int, int, int, int, str, str | None]] = []
        initial_cursor = cursor[section]
        context_words = 0
        while at < len(output):
            transitions += 1
            if transitions > MAX_ROW_TRANSITIONS:
                pending.append({"reason": "accounting_transition_limit", "chunk_id": logical["id"]})
                break
            start = cursor[section]
            count = 0
            while (
                start + count < len(values)
                and at + count < len(output)
                and values[start + count] == output[at + count].text
            ):
                count += 1
            kind = "source"
            if not count and report.get("overlap_words", 0) > 0:
                lower = max(0, start - min(MAX_CONTEXT_TOKENS, report["overlap_words"] * 8 + 32))
                count, relative = _longest_prefix(
                    [token.text for token in output[at:]], values[lower:start]
                )
                start = lower + relative
                kind = "overlap_or_context"
            if not count:
                pending.append(
                    {
                        "reason": "unexplained_output_text",
                        "chunk_id": logical["id"],
                        "output_span": [output[at].start, output[at].end],
                    }
                )
                break
            if kind == "overlap_or_context":
                context_words += sum(
                    bool(re.match(r"\w", token.text)) for token in output[at : at + count]
                )
                if context_words > report["overlap_words"]:
                    pending.append({"reason": "overlap_budget_exceeded", "chunk_id": logical["id"]})
                    break
            for offset in range(count):
                _, key, original, block_id = stream[start + offset]
                current = output[at + offset]
                for char in range(len(current.text)):
                    raw_pair = (
                        original.positions[char]
                        if original.positions
                        else (original.start + char, original.start + char + 1)
                    )
                    output_pair = (
                        current.positions[char]
                        if current.positions
                        else (current.start + char, current.start + char + 1)
                    )
                    pairs.append(
                        (
                            key,
                            raw_pair[0],
                            raw_pair[1],
                            output_pair[0],
                            output_pair[1],
                            kind,
                            block_id,
                        )
                    )
            at += count
            if kind == "source":
                cursor[section] = start + count
        if output and cursor[section] == initial_cursor:
            pending.append(
                {"reason": "output_record_without_source_progress", "chunk_id": logical["id"]}
            )
        logical_pages = {raw[pair[0]]["origin"]["page"] for pair in pairs}
        begin = 0
        for part in logical["parts"]:
            end = begin + len(part["text"])
            spans: list[dict[str, Any]] = []
            for key, before, before_end, after, after_end, kind, block_id in pairs:
                if after_end <= begin or after >= end:
                    continue
                local = max(after, begin) - begin
                local_end = min(after_end, end) - begin
                if (
                    spans
                    and spans[-1]["source_id"] == key
                    and spans[-1]["kind"] == kind
                    and spans[-1].get("block_id") == block_id
                    and before >= spans[-1]["source_span"][0]
                    and local >= spans[-1]["output_span"][0]
                ):
                    spans[-1]["source_span"][1] = max(spans[-1]["source_span"][1], before_end)
                    spans[-1]["output_span"][1] = max(spans[-1]["output_span"][1], local_end)
                else:
                    spans.append(
                        {
                            "source_id": key,
                            "block_id": block_id,
                            "source_span": [before, before_end],
                            "output_span": [local, local_end],
                            "kind": kind,
                        }
                    )
            meta = _metadata(part)
            pages = logical_pages
            wanted_role = roles_by_section.get(section)
            if wanted_role is not None and meta.get("section_role") not in (None, wanted_role):
                pending.append(
                    {"reason": "serialized_section_role_mismatch", "chunk_id": logical["id"]}
                )
            filename = report.get("source_filename")
            if meta.get("source_file") is not None and meta["source_file"] != filename:
                pending.append(
                    {"reason": "serialized_source_filename_mismatch", "chunk_id": logical["id"]}
                )
            if (
                meta.get("page") is not None
                and pages
                and (type(meta["page"]) is not int or meta["page"] not in pages)
            ):
                pending.append(
                    {"reason": "serialized_source_page_mismatch", "chunk_id": logical["id"]}
                )
            page_range = meta.get("page_range")
            if page_range is not None and pages:
                limits = re.findall(r"\d+", str(page_range))
                if not limits or [int(limits[0]), int(limits[-1])] != [min(pages), max(pages)]:
                    pending.append(
                        {
                            "reason": "serialized_source_page_range_mismatch",
                            "chunk_id": logical["id"],
                        }
                    )
            record = {
                "source_sha256": report["source_sha256"],
                "settings_sha256": report["settings_sha256"],
                "contract": VERSION,
                "metadata_scope": "logical_parent",
                "block_ids": list(
                    dict.fromkeys(span["block_id"] for span in spans if span.get("block_id"))
                ),
                "logical_text_sha256": sha(logical["text"]),
                "logical_span": [begin, end],
                "chunk_id": logical["id"],
                "chunk_part": meta.get("chunk_part"),
                "section_id": section,
                "text_sha256": _put(report["texts"], part["text"]),
                "spans": spans,
            }
            records.append({"record_id": identity(record)[:24], **record})
            begin = end
    for section, section_stream in expected.items():
        if cursor[section] != len(section_stream):
            pending.append(
                {
                    "reason": "unemitted_source_tail",
                    "section_id": section,
                    "remaining_tokens": len(section_stream) - cursor[section],
                }
            )
    return records, pending


def _normalization_witnesses(
    report: dict[str, Any], units: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    transformations = []
    grouped: dict[str, list[str]] = defaultdict(list)
    for unit in units:
        if unit["source_id"]:
            grouped[unit["source_id"]].append(report["texts"][unit["text"]])
    for origin in report["source_units"]:
        key = origin["source_id"]
        before = report["texts"][origin["text"]]
        after = "\n".join(grouped.get(key, ()))
        if before != after and _values(before) == _values(after):
            operations = ["whitespace"]
            if unicodedata.normalize("NFKC", before) != before:
                operations.append("unicode_nfkc")
            if _LETTER_HYPHEN.search(before):
                operations.append("letter_only_line_hyphenation")
            if before.translate(_QUOTES) != before:
                operations.append("curly_quotes")
            transformations.append(
                {
                    "source_id": key,
                    "source_span": [0, len(before)],
                    "before": origin["text"],
                    "after": _put(report["texts"], after),
                    "reason": "declared_normalization",
                    "operations": operations,
                }
            )
    return transformations


def record_stage(report: Mapping[str, Any], name: str, payload: Any) -> dict[str, Any]:
    """Append detached witnesses; no stage name by itself approves a text edit."""
    result = {
        **report,
        "texts": dict(report["texts"]),
        "pending": list(report["pending"]),
        "stages": list(report["stages"]),
    }
    if isinstance(payload, Mapping) and payload.get("type") == "page_blocks":
        if payload.get("source_block_identities") is not None:
            result["block_identities"] = dict(payload["source_block_identities"])
            result["block_identities_sha256"] = identity(result["block_identities"])
        units = _page_units(payload, result["texts"])
        pending = _page_check(result, units)
        transformations = _normalization_witnesses(result, units)
        result["stages"].append(
            {"name": name, "units": units, "pending": pending, "transformations": transformations}
        )
    else:
        records, pending = _match_output(result, payload)
        result["stages"].append({"name": name, "records": records, "pending": pending})
        result["records"] = records
    result["pending"].extend({"stage": name, **item} for item in pending)
    result["accepted"] = not result["pending"]
    return result


def bind_source_metadata(report: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    """Attach only actual contributors before serialized-size admission."""
    records, _ = _match_output(dict(report), payload)
    items = list(payload.get("items", ()))
    if len(records) != len(items):
        raise ValueError("source contributor binding requires logical chunks")
    raw = {unit["source_id"]: unit for unit in report["source_units"]}
    result = []
    for item, record in zip(items, records, strict=True):
        meta = _metadata(item)
        actual = set(record.get("block_ids", ()))
        if actual and isinstance(meta.get("block_provenance"), list):
            meta["block_provenance"] = [
                value for value in meta["block_provenance"] if value.get("id") in actual
            ]
        pages = {raw[span["source_id"]]["origin"]["page"] for span in record["spans"]}
        if pages:
            if meta.get("page") not in pages:
                meta["page"] = min(pages)
            meta["page_range"] = (
                str(min(pages)) if len(pages) == 1 else f"{min(pages)}-{max(pages)}"
            )
        meta["preservation_record_id"] = "0" * 24
        meta["preservation_status"] = "pending"
        meta["preservation_manifest_sha256"] = report["raw_manifest_sha256"]
        meta["preservation_unit_map_sha256"] = report.get("block_identities_sha256")
        key = "meta" if "meta" in item else "metadata"
        result.append({**item, key: meta})
    return {**payload, "items": result}


def admit_source_chunks(
    chunks: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
    *,
    fits: Callable[[str, Mapping[str, Any]], bool],
) -> list[dict[str, Any]]:
    """Partition internally bound provenance together with exact text slices."""
    bound = {
        (record["chunk_id"], record["text_sha256"]): record
        for record in report.get("records", ())
        if record.get("chunk_part") is None
    }
    result = []
    for chunk in chunks:
        text = str(chunk["text"])
        meta = _metadata(chunk)
        record = bound.get((meta.get("chunk_id"), sha(text)))
        if record is None or meta.get("preservation_unit_map_sha256") != report.get(
            "block_identities_sha256"
        ):
            raise ValueError("unbound source metadata cannot use provenance partitioning")
        if fits(text, meta):
            result.append(dict(chunk))
            continue
        if "chunk_part" in meta:
            raise ValueError("cannot split a chunk that already has chunk_part metadata")
        begin, part = 0, 0

        def portion(
            end: int,
            begin: int,
            part: int,
            record: Mapping[str, Any] = record,
            meta: Mapping[str, Any] = meta,
        ) -> dict[str, Any]:
            ids = {
                span["block_id"]
                for span in record["spans"]
                if span.get("block_id")
                and span["output_span"][1] > begin
                and span["output_span"][0] < end
            }
            return {
                **meta,
                "chunk_part": part,
                "block_provenance": [
                    entry for entry in meta.get("block_provenance", ()) if entry.get("id") in ids
                ],
            }

        while begin < len(text):
            low, high, best = begin + 1, len(text), begin
            while low <= high:
                middle = (low + high) // 2
                if fits(text[begin:middle], portion(middle, begin, part)):
                    best, low = middle, middle + 1
                else:
                    high = middle - 1
            if best == begin:
                raise ValueError("serialized metadata exceeds JSONL record limit before enrichment")
            fields = portion(best, begin, part)
            value = {**chunk, "text": text[begin:best]}
            for key in ("meta", "metadata"):
                if key in chunk:
                    value[key] = dict(fields)
            if "meta" not in value and "metadata" not in value:
                value["meta"] = fields
            result.append(value)
            begin, part = best, part + 1
    return result


def finalize_rows(
    report: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    max_chars: int = 8000,
    max_row_bytes: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Bind actual serialized siblings, including words split across record boundaries."""
    import json

    if type(max_chars) is not int or max_chars <= 0:
        raise ValueError("positive serialized character limit required")
    if max_row_bytes is not None and (type(max_row_bytes) is not int or max_row_bytes <= 0):
        raise ValueError("positive serialized byte limit required")
    result = record_stage(report, "serialized_rows", rows)
    result["serialization_contract"] = {"max_chars": max_chars, "max_row_bytes": max_row_bytes}
    records = result.get("records", [])
    if len(records) != len(rows):
        raise ValueError("source accounting row count mismatch")
    output = []
    for row, record in zip(rows, records, strict=True):
        meta = _metadata(row)
        if record.get("block_ids") and isinstance(meta.get("block_provenance"), list):
            actual_ids = set(record["block_ids"])
            meta["block_provenance"] = [
                entry for entry in meta["block_provenance"] if entry.get("id") in actual_ids
            ]
        meta["preservation_record_id"] = record["record_id"]
        meta["preservation_status"] = "pending"
        key = next(
            (
                key
                for key, value in row.items()
                if isinstance(value, Mapping) and "chunk_id" in value
            ),
            "metadata",
        )
        output.append({**row, key: meta})
    for index, row in enumerate(output):
        encoded = json.dumps(row, ensure_ascii=False)
        if len(encoded) > max_chars or (
            max_row_bytes is not None and len(encoded.encode("utf-8")) > max_row_bytes
        ):
            result["pending"].append({"reason": "serialized_record_budget_exceeded", "row": index})
    result["accepted"] = not result["pending"]
    for row in output:
        key = next(
            (
                key
                for key, value in row.items()
                if isinstance(value, Mapping) and "chunk_id" in value
            ),
            "metadata",
        )
        row[key]["preservation_status"] = "checked" if result["accepted"] else "pending"
    return result, output


def verify_preservation(
    report: Mapping[str, Any],
    *,
    source_structure: Mapping[str, Any],
    adapter_document: Mapping[str, Any],
    classified_document: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute witnesses from independently read raw facts, never producer flags."""
    from pdf_chunker.sections import block_identity

    options = settings.get("options") or {}
    pipeline = settings.get("pipeline") or ()
    expected_settings = settings_identity(options, pipeline, source_structure.get("reader"))
    raw = {**source_structure, "pipeline_settings_sha256": expected_settings}
    expected = initialize({**adapter_document, "source_structure": raw}, options)
    if report.get("schema") != VERSION:
        raise ValueError("source accounting version mismatch")
    for key in (
        "source_sha256",
        "settings_sha256",
        "raw_manifest_sha256",
        "source_filename",
        "reader",
        "limits",
        "source_units",
        "exclusions",
        "overlap_words",
    ):
        if report.get(key) != expected.get(key):
            raise ValueError(f"source accounting raw/settings mismatch: {key}")
    text_table = report.get("texts")
    if not isinstance(text_table, Mapping):
        raise ValueError("source accounting text table missing")
    total = 0
    for key, value in text_table.items():
        if not isinstance(value, str) or sha(value) != key:
            raise ValueError("source accounting text hash mismatch")
        total += len(value)
    if total > MAX_CHARACTERS * 8:
        raise ValueError("source accounting retained-text limit exceeded")
    expected["texts"] = dict(text_table)
    stages = report.get("stages") or []
    wanted_names = [
        "adapter",
        *(name for name in pipeline if name != "emit_jsonl"),
        "serialized_rows",
    ]
    if [stage.get("name") for stage in stages] != wanted_names:
        raise ValueError("source accounting stage order mismatch")
    for stage in stages:
        if "units" in stage:
            units = stage["units"]
            if not isinstance(units, list) or len(units) > MAX_UNITS:
                raise ValueError("source accounting stage unit limit exceeded")
            for unit in units:
                if unit.get("text") not in text_table:
                    raise ValueError("source accounting stage text is unbound")
            pending = _page_check(expected, units)
            if pending != stage.get("pending"):
                raise ValueError("source accounting page-stage pending witnesses differ")
            if _normalization_witnesses(expected, units) != stage.get("transformations"):
                raise ValueError("source accounting normalization witnesses differ")
            expected["pending"].extend({"stage": stage["name"], **value} for value in pending)
            expected["stages"].append(stage)
        elif "records" in stage:
            intermediate = []
            for record in stage["records"]:
                key = record.get("text_sha256")
                if key not in text_table:
                    raise ValueError("source accounting derived text is unbound")
                intermediate.append(
                    {
                        "text": text_table[key],
                        "metadata": {
                            "chunk_id": record.get("chunk_id"),
                            "chunk_part": record.get("chunk_part"),
                            "section_id": record.get("section_id"),
                        },
                    }
                )
            computed, pending = _match_output(expected, intermediate)
            if computed != stage["records"] or pending != stage.get("pending"):
                raise ValueError("source accounting intermediate span witnesses differ")
            expected["pending"].extend({"stage": stage["name"], **value} for value in pending)
            expected["stages"].append(stage)
        else:
            raise ValueError("source accounting unsupported stage witness")
    actual_units = _page_units(classified_document, expected["texts"])
    page_stages = [stage for stage in stages if "units" in stage]
    if not page_stages or page_stages[-1]["units"] != actual_units:
        raise ValueError("source accounting classified snapshot differs")
    full_map = {}
    for page in classified_document.get("pages", ()):
        for index, block in enumerate(page.get("blocks", ())):
            if not (block.get("source") or {}).get("raw_source"):
                continue
            entry = block_identity(page["page"], index, block)
            key = identity(entry)[:24]
            full_map[key] = entry
            declared = (block.get("source") or {}).get("block_provenance") or []
            if len(declared) != 1 or declared[0].get("id") != key:
                raise ValueError("source accounting block reference mismatch")
    if full_map != report.get("block_identities", {}):
        raise ValueError("source accounting private unit map differs")
    if identity(full_map) != report.get("block_identities_sha256"):
        raise ValueError("source accounting private unit map hash differs")
    expected["block_identities"] = full_map
    expected["block_identities_sha256"] = identity(full_map)
    expected["stages"] = page_stages
    emission = options.get("emit_jsonl") or {}
    verified, rebound = finalize_rows(
        expected,
        rows,
        max_chars=emission.get("max_chars", 8000),
        max_row_bytes=emission.get("max_row_bytes"),
    )
    if verified.get("records") != report.get("records"):
        raise ValueError("source accounting serialized span witnesses differ")
    for original, computed_row, record in zip(rows, rebound, verified["records"], strict=True):
        before, after = _metadata(original), _metadata(computed_row)
        for key in (
            "preservation_record_id",
            "preservation_manifest_sha256",
            "preservation_unit_map_sha256",
            "preservation_status",
        ):
            if before.get(key) != after.get(key):
                raise ValueError(f"source accounting serialized reference differs: {key}")
        actual = {
            entry.get("id") for entry in before.get("block_provenance", ()) if entry.get("id")
        }
        if actual != set(record.get("block_ids", ())):
            raise ValueError("source accounting serialized contributor set differs")
        for entry in before.get("block_provenance", ()):
            bound = full_map.get(entry.get("id"))
            if bound is None or any(
                entry.get(key) != bound.get(key)
                for key in ("page", "block_index", "bbox", "text_sha256")
            ):
                raise ValueError("source accounting serialized legacy geometry differs")
    return {
        "accepted": verified["accepted"],
        "pending": verified["pending"],
        "source_sha256": verified["source_sha256"],
        "settings_sha256": verified["settings_sha256"],
        "raw_manifest_sha256": verified["raw_manifest_sha256"],
        "records": len(rows),
        "source_units": len(expected["source_units"]),
        "raw_and_serialized_witnesses_recomputed": True,
    }
