"""Completed base article output, using synthetic evidence and no cache writes."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from functools import partial

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pdf_chunker.adapters import article_conversion as adapter
from pdf_chunker.adapters.article_conversion import (
    execute_bound_conversion,
    prepare_bound_conversion,
)
from pdf_chunker.adapters.emit_jsonl import _serialize
from pdf_chunker.article_output import validate_article_output
from pdf_chunker.core_new import convert
from pdf_chunker.utils import _normalize_list_block
from tests.verified_markdown_ingress_test import SOURCE, bind_article


def digest(data):
    return hashlib.sha256(data).hexdigest()


def wire(rows):
    return "".join(line + "\n" for line in _serialize(rows)).encode("utf-8")


@pytest.fixture
def case(tmp_path, monkeypatch):
    runtime = {"fixture": "stable"}
    monkeypatch.setattr(adapter, "collect_article_runtime", lambda: copy.deepcopy(runtime))
    bound = bind_article(tmp_path / "article")
    config = bound.config()
    prepared = prepare_bound_conversion(bound.binding, config, corpus="technology")
    rows = convert(str(bound.source), prepared.spec())
    return bound, prepared, rows, runtime


def validate(data, descriptor):
    return validate_article_output(data, descriptor=descriptor, expected_sha256=digest(data))


def test_output_is_detached_and_matches_existing_writer(case):
    _, prepared, rows, _ = case
    before = copy.deepcopy(rows)
    data = wire(rows)
    stage = partial(
        validate_article_output,
        descriptor=prepared.descriptor,
        expected_sha256=digest(data),
    )
    result = stage(data)
    assert result.jsonl == data
    assert result.sha256 == digest(data)
    assert result.size_bytes == len(data)
    assert result.row_count == len(rows)
    assert result.conversion_key == prepared.descriptor.key
    assert result.rows() == rows == before
    result.rows()[0]["metadata"]["joplin_references"].clear()
    assert result.rows() == before
    assert stage(data) == result
    with pytest.raises(FrozenInstanceError):
        result.jsonl = b""


@pytest.mark.parametrize("suffix", [".md", ".epub"])
def test_guarded_execution_custom_metadata_and_no_writes(tmp_path, monkeypatch, suffix):
    monkeypatch.setattr(adapter, "collect_article_runtime", lambda: {"fixture": "stable"})
    bound = bind_article(tmp_path / "article", suffix=suffix)
    config = bound.config()
    config.options["emit_jsonl"].update(
        metadata_key="article_meta", output_path=str(tmp_path / "not-written.jsonl")
    )
    prepared = prepare_bound_conversion(bound.binding, config, corpus="technology")
    expected = wire(convert(str(bound.source), prepared.spec()))
    result = execute_bound_conversion(bound.binding, prepared=prepared, corpus="technology")
    assert result.jsonl == expected
    assert "".join(row["text"] for row in result.rows()) == SOURCE
    assert all("article_meta" in row for row in result.rows())
    assert not list(tmp_path.rglob("*.jsonl"))


@pytest.mark.parametrize(
    "text",
    [
        "- First item.\n- Second item.\n",
        "1. First item.\n2. Second item.\n",
        "# Heading\n\n- Item after a heading.\n",
        "```text\n- Existing metadata heuristics also inspect code.\n```\n",
    ],
)
def test_existing_list_metadata_is_accepted_without_changing_text(tmp_path, monkeypatch, text):
    monkeypatch.setattr(adapter, "collect_article_runtime", lambda: {"fixture": "stable"})
    bound = bind_article(tmp_path / "article", text=text)
    prepared = prepare_bound_conversion(bound.binding, bound.config(), corpus="technology")
    expected = convert(str(bound.source), prepared.spec())
    assert any(row["metadata"]["block_type"] == "list_item" for row in expected)
    result = execute_bound_conversion(bound.binding, prepared=prepared, corpus="technology")
    assert result.jsonl == wire(expected)
    assert "".join(row["text"] for row in result.rows()) == text


def test_pinned_output_hash_is_checked_before_parsing(case):
    _, prepared, rows, _ = case
    original = wire(rows)
    with pytest.raises(ValueError, match="output SHA-256"):
        validate_article_output(
            original + b"broken", descriptor=prepared.descriptor, expected_sha256=digest(original)
        )


@pytest.mark.parametrize(
    "transform",
    [
        lambda raw: raw[:-1],
        lambda raw: raw + b"\n",
        lambda raw: b"\xef\xbb\xbf" + raw,
        lambda raw: raw.replace(b"\n", b"\r\n"),
        lambda raw: b"[]\n",
        lambda raw: b'{"text":"one","text":"two"}\n',
        lambda raw: b'{"text":NaN}\n',
        lambda raw: b'{"text":"\\ud800"}\n',
        lambda raw: b"\xff\n",
        lambda raw: b"",
        lambda raw: json.dumps(json.loads(raw.splitlines()[0]), ensure_ascii=True).encode() + b"\n",
    ],
)
def test_invalid_wire_refuses_even_with_matching_hash(case, transform):
    _, prepared, rows, _ = case
    with pytest.raises((ValueError, UnicodeError)):
        validate(transform(wire(rows)), prepared.descriptor)


@pytest.mark.parametrize(
    "field,value",
    [
        ("chunk_id", "wrong-but-unique"),
        ("source", "other.md"),
        ("source_file", "other.md"),
        ("source_char_start", True),
        ("source_char_end", 1),
        ("source_text_strategy", "other-strategy"),
        ("joplin_references", []),
        ("title", "replaced title"),
        ("page", True),
        ("tags", "example"),
        ("readability", {"flesch_kincaid_grade": "high", "difficulty": "college_level"}),
        ("readability", {"flesch_kincaid_grade": True, "difficulty": "college_level"}),
        ("readability", {"flesch_kincaid_grade": 1.0, "difficulty": []}),
        ("readability", {"flesch_kincaid_grade": float("inf"), "difficulty": "college_level"}),
    ],
)
def test_invalid_metadata_refuses_even_with_matching_hash(case, field, value):
    _, prepared, rows, _ = case
    rows[0]["metadata"][field] = value
    with pytest.raises(ValueError):
        validate(wire(rows), prepared.descriptor)


@pytest.mark.parametrize(
    "change", ["missing-text", "empty-text", "non-string", "missing-meta", "extra-field"]
)
def test_row_shape_is_required(case, change):
    _, prepared, rows, _ = case
    if change == "missing-text":
        rows[0].pop("text")
    elif change == "empty-text":
        rows[0]["text"] = ""
    elif change == "non-string":
        rows[0]["text"] = 1
    elif change == "missing-meta":
        rows[0].pop("metadata")
    else:
        rows[0]["unexpected"] = "unbound payload"
    with pytest.raises(ValueError, match="row schema"):
        validate(wire(rows), prepared.descriptor)


def test_unsupported_output_schema_refuses(case):
    _, prepared, rows, _ = case
    contract = json.loads(prepared.descriptor.contract_json)
    contract["chunk_schema"] = "future-schema"
    descriptor = replace(prepared.descriptor, contract_json=json.dumps(contract))
    with pytest.raises(ValueError, match="unsupported article output contract"):
        validate(wire(rows), descriptor)


def test_generated_collisions_and_null_passthrough_follow_existing_precedence(case):
    _, prepared, rows, _ = case
    native = json.loads(prepared.descriptor.metadata_json)
    native.update(readability={"caller": "ignored"}, source_char_start=-1, unused=None)
    descriptor = replace(prepared.descriptor, metadata_json=json.dumps(native))
    # Native page/chunk_id/readability/offset collisions never own generated fields.
    result = validate(wire(rows), descriptor)
    assert result.rows() == rows
    assert all(row["metadata"]["page"] == 1 for row in result.rows())


@pytest.mark.parametrize("change", ["missing", "duplicate", "reordered", "changed-text", "gap"])
def test_complete_ordered_source_reconstruction_required(case, change):
    _, prepared, rows, _ = case
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows.insert(1, copy.deepcopy(rows[0]))
    elif change == "reordered":
        rows.reverse()
    elif change == "changed-text":
        rows[0]["text"] = "x" + rows[0]["text"][1:]
    else:
        rows[1]["metadata"]["source_char_start"] += 1
    with pytest.raises(ValueError):
        validate(wire(rows), prepared.descriptor)


def test_complete_serialized_character_limit_required(case):
    _, prepared, rows, _ = case
    contract = json.loads(prepared.descriptor.contract_json)
    contract["configuration"]["options"]["emit_jsonl"]["max_chars"] = 1
    descriptor = replace(prepared.descriptor, contract_json=json.dumps(contract))
    with pytest.raises(ValueError, match="size limit"):
        validate(wire(rows), descriptor)


@given(
    st.lists(
        st.text(alphabet=st.characters(blacklist_categories=("Cs",)), min_size=1),
        min_size=1,
        max_size=8,
    )
)
def test_unicode_partition_reconstruction_and_input_preservation(parts):
    # Output validation is independent of I/O and does not reimplement chunking.
    from pdf_chunker.conversion_descriptor import describe_conversion

    parts = ["# Article\n", *parts]
    text = "".join(parts)
    descriptor = describe_conversion(
        text,
        source_path="/private/article.md",
        source_format="md",
        source_sha256="a" * 64,
        metadata={},
        references={"joplin:fixture": {}},
        corpus="technology",
        contract={
            "implementation": "fixture",
            "normalization": "approved-markdown-utf8-v1",
            "chunk_schema": "canonical-v1/contiguous_markdown_v3",
            "serialization": "python-json-ensure-ascii-false-v1",
            "configuration": {
                "options": {"emit_jsonl": {"metadata_key": "metadata", "max_chars": 100000}}
            },
        },
    )
    rows = []
    start = 0
    for index, part in enumerate(parts):
        normalized = _normalize_list_block({}, chunk_text=part)
        rows.append(
            {
                "text": part,
                "metadata": {
                    "source": "article.md",
                    "source_file": "article.md",
                    "chunk_id": f"article.md_p0_c{index}",
                    "page": 1,
                    "page_range": "1",
                    "location": None,
                    "block_type": normalized.get("type", "paragraph"),
                    "language": "un",
                    "utterance_type": "unclassified",
                    "tags": [],
                    "importance": "medium",
                    "readability": {"flesch_kincaid_grade": 1.0, "difficulty": "elementary"},
                    "source_char_start": start,
                    "source_char_end": start + len(part),
                    "source_text_strategy": "contiguous_markdown_v3",
                    **(
                        {"list_kind": normalized["list_kind"]}
                        if normalized.get("list_kind")
                        else {}
                    ),
                },
            }
        )
        start += len(part)
    before = copy.deepcopy(rows)
    result = validate(wire(rows), descriptor)
    assert "".join(row["text"] for row in result.rows()) == text
    assert rows == before


def test_stale_runtime_refuses_before_conversion(case, monkeypatch):
    bound, prepared, _, runtime = case
    runtime["fixture"] = "changed"
    monkeypatch.setattr(adapter, "convert", lambda *args: pytest.fail("stale inputs executed"))
    with pytest.raises(ValueError, match="before conversion"):
        execute_bound_conversion(bound.binding, prepared=prepared, corpus="technology")


def test_mutated_prepared_spec_refuses_before_conversion(case, monkeypatch):
    bound, prepared, _, _ = case
    spec = json.loads(prepared.spec_json)
    spec["options"]["split_semantic"]["chunk_size"] += 1
    altered = replace(prepared, spec_json=json.dumps(spec))
    monkeypatch.setattr(adapter, "convert", lambda *args: pytest.fail("altered spec executed"))
    with pytest.raises(ValueError, match="before conversion"):
        execute_bound_conversion(bound.binding, prepared=altered, corpus="technology")


def test_native_metadata_file_order_preserves_output_and_preparation(case):
    bound, prepared, rows, _ = case
    path = prepared.spec().options["text_parse"]["metadata_path"]
    with open(path, encoding="utf-8") as stream:
        native = json.load(stream)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(dict(reversed(list(native.items()))), stream, ensure_ascii=False)
    # The verified reader emits evidence-derived metadata, not the supplied file's order.
    assert wire(convert(str(bound.source), prepared.spec())) == wire(rows)
    changed = prepare_bound_conversion(bound.binding, prepared.spec(), corpus="technology")
    assert changed == prepared
    result = execute_bound_conversion(bound.binding, prepared=prepared, corpus="technology")
    assert result.jsonl == wire(rows)


@pytest.mark.parametrize("change", ["runtime", "source", "metadata", "proof", "rows", "failure"])
def test_execution_changes_or_failure_never_return_completion(case, monkeypatch, change):
    bound, prepared, _, runtime = case

    def changing_convert(path, spec):
        rows = convert(path, spec)
        if change == "runtime":
            runtime["fixture"] = "changed-during-execution"
        elif change == "source":
            bound.source.write_text("unapproved replacement", encoding="utf-8")
        elif change in {"metadata", "proof"}:
            path = (
                spec.options["text_parse"]["metadata_path"]
                if change == "metadata"
                else bound.binding["quality_path"]
            )
            with open(path, "w", encoding="utf-8") as stream:
                stream.write("{}")
        elif change == "rows":
            rows.pop()
        else:
            raise RuntimeError("conversion did not finish")
        return rows

    monkeypatch.setattr(adapter, "convert", changing_convert)
    with pytest.raises((ValueError, RuntimeError)):
        execute_bound_conversion(bound.binding, prepared=prepared, corpus="technology")
    assert not list(bound.source.parent.glob("*.jsonl"))


def test_real_runtime_warming_and_intervening_article_keep_output_identical(tmp_path):
    bound = bind_article(tmp_path / "article")
    prepared = prepare_bound_conversion(bound.binding, bound.config(), corpus="technology")
    first = execute_bound_conversion(bound.binding, prepared=prepared, corpus="technology")
    other = bind_article(tmp_path / "other", text="# Another\n\nDifferent words to warm caches.\n")
    convert(str(other.source), other.config())
    repeated = execute_bound_conversion(bound.binding, prepared=prepared, corpus="technology")
    assert repeated == first
    assert repeated.jsonl == wire(convert(str(bound.source), prepared.spec()))
