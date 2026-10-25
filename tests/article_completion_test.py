"""Pinned completion readback with synthetic approved articles and no inference."""

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
    complete_bound_conversion,
    prepare_bound_conversion,
    read_bound_conversion,
)
from pdf_chunker.article_completion import (
    describe_article_completion,
    read_article_completion,
    validate_completed_article,
)
from pdf_chunker.conversion_descriptor import read_conversion_descriptor
from tests.conversion_descriptor_test import descriptor
from tests.verified_markdown_ingress_test import SOURCE, bind_article


def digest(data):
    return hashlib.sha256(data).hexdigest()


def wire(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


@pytest.fixture
def case(tmp_path, monkeypatch):
    runtime = {"fixture": "stable"}
    monkeypatch.setattr(adapter, "collect_article_runtime", lambda: copy.deepcopy(runtime))
    bound = bind_article(tmp_path / "article")
    prepared = prepare_bound_conversion(bound.binding, bound.config(), corpus="technology")
    completion, output = complete_bound_conversion(
        bound.binding, prepared=prepared, corpus="technology"
    )
    return bound, prepared, completion, output, runtime


def test_completion_roundtrip_and_partial_validation(case):
    _, prepared, completion, output, _ = case
    raw = completion.to_json().encode("utf-8")
    read = partial(
        read_article_completion, descriptor=prepared.descriptor, expected_sha256=completion.sha256
    )
    restored = read(raw)
    assert restored == completion
    assert restored.descriptor == read_conversion_descriptor(prepared.descriptor.to_json())
    assert restored.output_sha256 == output.sha256
    assert restored.size_bytes == output.size_bytes
    assert restored.row_count == output.row_count
    check = partial(validate_completed_article, completion=restored)
    assert check(output.jsonl) == output
    check(output.jsonl).rows()[0]["metadata"].clear()
    assert check(output.jsonl) == output
    with pytest.raises(FrozenInstanceError):
        restored.row_count = 0


@given(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=100))
def test_descriptor_roundtrip_preserves_unicode_and_identity(text):
    original = descriptor("# Article\n" + text)
    restored = read_conversion_descriptor(original.to_json())
    assert restored == original
    assert restored.key == original.key
    assert restored.to_json() == original.to_json()


@pytest.mark.parametrize(
    "change",
    [
        "schema",
        "boolean-schema",
        "extra",
        "missing",
        "bad-hash",
        "empty-label",
        "metadata-list",
        "noncanonical-metadata",
        "bad-contract",
        "noncanonical-contract",
        "empty-associations",
        "duplicate-reference",
        "unordered-references",
        "wrong-content",
        "empty-corpus",
        "mixed-corpus",
        "provenance-list",
        "noncanonical-provenance",
        "extra-association",
    ],
)
def test_malformed_descriptor_refuses(change):
    value = json.loads(descriptor().to_json())
    if change == "schema":
        value["schema"] = 2
    elif change == "boolean-schema":
        value["schema"] = True
    elif change == "extra":
        value["unexpected"] = 1
    elif change == "missing":
        value.pop("content_sha256")
    elif change == "bad-hash":
        value["source_sha256"] = "A" * 64
    elif change == "empty-label":
        value["source_path"] = " "
    elif change == "metadata-list":
        value["metadata_json"] = "[]"
    elif change == "noncanonical-metadata":
        value["metadata_json"] = '{ "a": 1 }'
    elif change == "bad-contract":
        value["contract_json"] = "{}"
    elif change == "noncanonical-contract":
        value["contract_json"] = json.dumps(json.loads(value["contract_json"]))
    elif change == "empty-associations":
        value["associations"] = []
    elif change == "duplicate-reference":
        value["associations"].append(copy.deepcopy(value["associations"][0]))
    elif change == "unordered-references":
        value["associations"].reverse()
    elif change == "wrong-content":
        value["associations"][0]["content_sha256"] = "0" * 64
    elif change == "empty-corpus":
        value["associations"][0]["corpus"] = ""
    elif change == "mixed-corpus":
        value["associations"][0]["corpus"] = "different"
    elif change == "provenance-list":
        value["associations"][0]["provenance_json"] = "[]"
    elif change == "noncanonical-provenance":
        value["associations"][0]["provenance_json"] = '{ "a": 1 }'
    else:
        value["associations"][0]["extra"] = None
    with pytest.raises(ValueError):
        read_conversion_descriptor(wire(value).decode())


@pytest.mark.parametrize(
    "raw", ["[]", "null", '{"schema":1,"schema":1}', '{"x":NaN}', '{"x":"\\ud800"}']
)
def test_invalid_descriptor_json_refuses(raw):
    with pytest.raises(ValueError):
        read_conversion_descriptor(raw)


def test_descriptor_requires_exact_canonical_wire():
    raw = descriptor().to_json()
    for changed in (raw + "\n", " " + raw, raw.replace('"schema":1', '"schema":1,"schema":1')):
        with pytest.raises(ValueError):
            read_conversion_descriptor(changed)


@pytest.mark.parametrize(
    "change",
    [
        "schema",
        "boolean-schema",
        "status",
        "execution",
        "extra",
        "missing",
        "hash",
        "size",
        "rows",
        "boolean-count",
        "float-count",
        "unknown-output",
        "descriptor",
    ],
)
def test_malformed_completion_refuses_even_with_rehashed_record(case, change):
    _, prepared, completion, _, _ = case
    value = json.loads(completion.to_json())
    if change == "schema":
        value["schema"] = 2
    elif change == "boolean-schema":
        value["schema"] = True
    elif change == "status":
        value["status"] = "running"
    elif change == "execution":
        value["execution"] = "child-exit-zero"
    elif change == "extra":
        value["extra"] = None
    elif change == "missing":
        value.pop("status")
    elif change == "hash":
        value["output"]["sha256"] = "x" * 64
    elif change in {"size", "rows", "boolean-count", "float-count"}:
        key = "size_bytes" if change == "size" else "row_count"
        value["output"][key] = {"size": 0, "rows": -1, "boolean-count": True, "float-count": 1.0}[
            change
        ]
    elif change == "unknown-output":
        value["output"]["extra"] = None
    else:
        value["descriptor"]["source_path"] = "/changed/article.md"
    raw = wire(value)
    with pytest.raises(ValueError):
        read_article_completion(raw, descriptor=prepared.descriptor, expected_sha256=digest(raw))


@pytest.mark.parametrize("kind", ["newline", "duplicate", "noncanonical", "invalid-utf8"])
def test_completion_requires_exact_canonical_utf8(case, kind):
    _, prepared, completion, _, _ = case
    raw = completion.to_json().encode()
    if kind == "newline":
        raw += b"\n"
    elif kind == "duplicate":
        raw = raw.replace(b'"schema":1', b'"schema":1,"schema":1', 1)
    elif kind == "noncanonical":
        raw = json.dumps(json.loads(raw)).encode()
    else:
        raw = b"\xff"
    with pytest.raises(ValueError):
        read_article_completion(raw, descriptor=prepared.descriptor, expected_sha256=digest(raw))


def test_retained_completion_hash_is_checked_before_parsing(case):
    _, prepared, completion, _, _ = case
    with pytest.raises(ValueError, match="completion SHA-256"):
        read_article_completion(
            b"broken", descriptor=prepared.descriptor, expected_sha256=completion.sha256
        )


def test_replacing_both_output_and_record_cannot_replace_the_retained_anchor(case, tmp_path):
    bound, prepared, completion, output, _ = case
    paths = persist(tmp_path, completion, output)
    rows = output.rows()
    rows[0]["metadata"]["readability"]["flesch_kincaid_grade"] += 1
    changed = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    value = json.loads(completion.to_json())
    value["output"]["sha256"] = digest(changed)
    value["output"]["size_bytes"] = len(changed)
    paths["completion_path"].write_bytes(wire(value))
    paths["output_path"].write_bytes(changed)
    with pytest.raises(ValueError, match="completion SHA-256"):
        read_bound_conversion(bound.binding, prepared.spec(), corpus="technology", **paths)


@pytest.mark.parametrize("field", ["conversion_key", "row_count", "jsonl"])
def test_public_output_construction_does_not_bypass_completion_validation(case, field):
    _, prepared, _, output, _ = case
    value = {
        "conversion_key": "wrong",
        "row_count": output.row_count + 1,
        "jsonl": output.jsonl[:-1],
    }[field]
    with pytest.raises(ValueError):
        describe_article_completion(
            replace(output, **{field: value}), descriptor=prepared.descriptor
        )


def test_completion_factory_refuses_a_malformed_public_descriptor(case):
    _, prepared, _, output, _ = case
    original = prepared.descriptor
    altered = replace(original.associations[0], content_sha256="0" * 64)
    malformed = replace(original, associations=(altered, *original.associations[1:]))
    # Even self-consistent public output facts cannot legitimize a malformed descriptor.
    altered_output = replace(output, conversion_key=malformed.key)
    with pytest.raises(ValueError, match="association content"):
        describe_article_completion(altered_output, descriptor=malformed)


def test_completion_factory_detaches_public_descriptor_containers(case):
    _, prepared, _, output, _ = case
    associations = list(prepared.descriptor.associations)
    supplied = replace(prepared.descriptor, associations=associations)
    before = supplied.to_json()
    completion = describe_article_completion(output, descriptor=supplied)
    assert supplied.to_json() == before
    associations.clear()
    assert completion.descriptor == prepared.descriptor
    assert completion.descriptor.to_json() == before


@pytest.mark.parametrize("field", ["size_bytes", "row_count", "output_sha256", "boolean-count"])
def test_readback_checks_actual_output_against_completion_facts(case, field):
    _, _, completion, output, _ = case
    value = (
        "0" * 64
        if field == "output_sha256"
        else True if field == "boolean-count" else getattr(completion, field) + 1
    )
    field = "row_count" if field == "boolean-count" else field
    with pytest.raises(ValueError):
        validate_completed_article(output.jsonl, completion=replace(completion, **{field: value}))


def persist(tmp_path, completion, output):
    record_path, output_path = tmp_path / "completion.json", tmp_path / "output.jsonl"
    record_path.write_bytes(completion.to_json().encode())
    output_path.write_bytes(output.jsonl)
    return {
        "completion_path": record_path,
        "output_path": output_path,
        "expected_completion_sha256": completion.sha256,
    }


@pytest.mark.parametrize("suffix", [".md", ".epub"])
def test_read_only_reuse_reverifies_and_never_runs_converter(tmp_path, monkeypatch, suffix):
    monkeypatch.setattr(adapter, "collect_article_runtime", lambda: {"fixture": "stable"})
    bound = bind_article(tmp_path / "article", suffix=suffix)
    config = bound.config()
    config.options["emit_jsonl"]["metadata_key"] = "article_meta"
    prepared = prepare_bound_conversion(bound.binding, config, corpus="technology")
    completion, output = complete_bound_conversion(
        bound.binding, prepared=prepared, corpus="technology"
    )
    paths = persist(tmp_path, completion, output)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    monkeypatch.setattr(adapter, "convert", lambda *a: pytest.fail("cached output reconverted"))
    result = read_bound_conversion(bound.binding, config, corpus="technology", **paths)
    assert result == output
    assert "".join(row["text"] for row in result.rows()) == SOURCE
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


@pytest.mark.parametrize(
    "change",
    [
        "output",
        "completion",
        "missing-output",
        "missing-completion",
        "source",
        "proof",
        "metadata",
        "runtime",
        "configuration",
        "corpus",
    ],
)
def test_stale_missing_or_corrupt_cache_refuses_without_fallback(
    case, tmp_path, monkeypatch, change
):
    bound, prepared, completion, output, runtime = case
    paths = persist(tmp_path, completion, output)
    spec = prepared.spec()
    corpus = "technology"
    if change in {"output", "completion"}:
        paths[f"{change}_path"].write_bytes(b"tampered")
    elif change.startswith("missing-"):
        paths[f"{change.removeprefix('missing-')}_path"].unlink()
    elif change == "source":
        bound.source.write_text("not approved")
    elif change == "proof":
        with open(bound.binding["quality_path"], "w") as stream:
            stream.write("{}")
    elif change == "metadata":
        with open(spec.options["text_parse"]["metadata_path"], "w") as stream:
            stream.write("{}")
    elif change == "runtime":
        runtime["fixture"] = "changed"
    elif change == "configuration":
        spec.options["emit_jsonl"]["max_chars"] += 1
    else:
        corpus = "changed-corpus"
    monkeypatch.setattr(adapter, "convert", lambda *a: pytest.fail("invalid cache reconverted"))
    with pytest.raises((ValueError, FileNotFoundError)):
        read_bound_conversion(bound.binding, spec, corpus=corpus, **paths)


def test_bad_completion_refuses_before_output_read(case, tmp_path):
    bound, prepared, completion, output, _ = case
    paths = persist(tmp_path, completion, output)
    paths["completion_path"].write_bytes(b"broken")
    paths["output_path"].unlink()
    with pytest.raises(ValueError, match="completion SHA-256"):
        read_bound_conversion(bound.binding, prepared.spec(), corpus="technology", **paths)


@pytest.mark.parametrize("change", ["runtime", "source", "proof", "metadata"])
def test_persistent_drift_during_readback_refuses(case, tmp_path, monkeypatch, change):
    bound, prepared, completion, output, runtime = case
    paths = persist(tmp_path, completion, output)
    original = adapter.validate_completed_article

    def validate_then_change(data, **options):
        result = original(data, **options)
        if change == "runtime":
            runtime["fixture"] = "changed-during-readback"
        elif change == "source":
            bound.source.write_text("unapproved replacement")
        else:
            path = (
                prepared.spec().options["text_parse"]["metadata_path"]
                if change == "metadata"
                else bound.binding["quality_path"]
            )
            with open(path, "w") as stream:
                stream.write("{}")
        return result

    monkeypatch.setattr(adapter, "validate_completed_article", validate_then_change)
    with pytest.raises(ValueError):
        read_bound_conversion(bound.binding, prepared.spec(), corpus="technology", **paths)


def test_conversion_failure_returns_no_completion(case, monkeypatch):
    bound, prepared, _, _, _ = case

    def fail(*args):
        raise RuntimeError("not completed")

    monkeypatch.setattr(adapter, "convert", fail)
    with pytest.raises(RuntimeError, match="not completed"):
        complete_bound_conversion(bound.binding, prepared=prepared, corpus="technology")


def test_real_runtime_readback_after_an_intervening_article(tmp_path, monkeypatch):
    bound = bind_article(tmp_path / "article")
    prepared = prepare_bound_conversion(bound.binding, bound.config(), corpus="technology")
    completion, output = complete_bound_conversion(
        bound.binding, prepared=prepared, corpus="technology"
    )
    paths = persist(tmp_path, completion, output)
    other = bind_article(
        tmp_path / "other", text="# Other\n\nWarm dictionaries with other words.\n"
    )
    other_prepared = prepare_bound_conversion(other.binding, other.config(), corpus="technology")
    complete_bound_conversion(other.binding, prepared=other_prepared, corpus="technology")
    monkeypatch.setattr(adapter, "convert", lambda *a: pytest.fail("cached output reconverted"))
    result = read_bound_conversion(bound.binding, bound.config(), corpus="technology", **paths)
    assert result == output
    assert result.jsonl == output.jsonl
