"""Pure conversion identity and verified article descriptor regressions."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import FrozenInstanceError
from functools import partial
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pdf_chunker.adapters import io_joplin_epub as article
from pdf_chunker.conversion_descriptor import compare_conversion, describe_conversion
from pdf_chunker.core_new import convert
from tests.verified_markdown_ingress_test import SOURCE, bind_article


def contract(**changes: object) -> dict:
    return {
        "implementation": "synthetic-converter-revision-1",
        "normalization": "approved-markdown-utf8-v1",
        "chunk_schema": "canonical-v1/contiguous_markdown_v3",
        "serialization": "jsonl-utf8-v1",
        "configuration": {
            "pipeline": ["text_clean", "split_semantic", "emit_jsonl"],
            "options": {"split_semantic": {"chunk_size": 20, "overlap": 0}},
        },
        **changes,
    }


def descriptor(text: str = SOURCE, **changes: object):
    options = {
        "source_path": "/synthetic/article.md",
        "source_format": ".md",
        "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "metadata": {"title": "Example 🧭", "nested": ["café"]},
        "references": {
            "joplin:" + "a" * 32: {"title": "First", "tags": ["example"]},
            "joplin:" + "b" * 32: {"title": "Second", "tags": ["reference"]},
        },
        "corpus": "technology",
        "contract": contract(),
        **changes,
    }
    return describe_conversion(text, **options)


def test_descriptor_round_trip_is_versioned_exact_json_and_detached() -> None:
    metadata = {"title": "🧭", "nested": ["café"]}
    current = descriptor(metadata=metadata)
    before = current.to_json()
    metadata["nested"].append("caller mutation")
    data = json.loads(before)
    assert data["schema"] == 1
    assert json.loads(data["metadata_json"]) == {"title": "🧭", "nested": ["café"]}
    assert current.to_json() == before
    assert current.key == hashlib.sha256(before.encode("utf-8")).hexdigest()
    assert "🧭" in before


def test_new_changed_and_unchanged_are_comparisons_without_deletion_actions() -> None:
    current = descriptor()
    assert compare_conversion(current).status == "new"
    unchanged = compare_conversion(current, previous=descriptor())
    assert unchanged.status == "unchanged"
    assert unchanged.changed_fields == ()
    changed = compare_conversion(descriptor(metadata={"title": "Updated"}), previous=current)
    assert changed.status == "changed"
    assert changed.changed_fields == ("metadata_json",)


@pytest.mark.parametrize(
    "changes,fields",
    [
        ({"source_path": "/moved/article.md"}, ("source_path",)),
        ({"source_format": ".epub"}, ("source_format",)),
        ({"source_sha256": "c" * 64}, ("source_sha256",)),
        ({"corpus": "science"}, ("associations",)),
        ({"contract": contract(implementation="revision-2")}, ("contract_json",)),
        ({"contract": contract(normalization="normalization-2")}, ("contract_json",)),
        ({"contract": contract(chunk_schema="schema-2")}, ("contract_json",)),
        ({"contract": contract(serialization="serialization-2")}, ("contract_json",)),
        ({"contract": contract(configuration={"chunk_size": 40})}, ("contract_json",)),
    ],
)
def test_only_changed_descriptor_components_are_invalidated(changes: dict, fields: tuple) -> None:
    before = descriptor()
    after = descriptor(**changes)
    assert after.key != before.key
    assert compare_conversion(after, previous=before).changed_fields == fields
    assert after.content_sha256 == before.content_sha256
    if "associations" not in fields:
        assert after.associations == before.associations


def test_content_revision_updates_each_association_without_losing_references() -> None:
    before = descriptor()
    after = descriptor(SOURCE + "\nA revision.\n")
    assert after.content_sha256 != before.content_sha256
    assert [a.reference for a in after.associations] == [a.reference for a in before.associations]
    assert all(a.content_sha256 == after.content_sha256 for a in after.associations)
    assert all(a.key != b.key for a, b in zip(after.associations, before.associations, strict=True))


def test_changing_one_note_preserves_the_other_note_association() -> None:
    references = {"joplin:a": {"tags": ["old"]}, "joplin:b": {"tags": ["keep"]}}
    before = descriptor(references=references)
    after = descriptor(references={**references, "joplin:a": {"tags": ["new"]}})
    assert before.associations[0].key != after.associations[0].key
    assert before.associations[1] == after.associations[1]
    assert len(after.associations) == 2


def test_reference_discovery_order_is_irrelevant_but_array_order_is_significant() -> None:
    references = {"joplin:b": {"tags": ["one", "two"]}, "joplin:a": {"title": "A"}}
    first = descriptor(references=references)
    assert first == descriptor(references=dict(reversed(list(references.items()))))
    changed = descriptor(references={**references, "joplin:b": {"tags": ["two", "one"]}})
    assert first.key != changed.key


def test_contract_and_associations_are_detached_immutable_snapshots() -> None:
    configuration = contract()
    references = {"joplin:a": {"tags": ["original"]}}
    before = copy.deepcopy((configuration, references))
    current = descriptor(contract=configuration, references=references)
    serialized = current.to_json()
    assert (configuration, references) == before
    configuration["configuration"]["options"]["split_semantic"]["chunk_size"] = 99
    references["joplin:a"]["tags"].append("changed")
    assert current.to_json() == serialized
    with pytest.raises(FrozenInstanceError):
        current.source_path = "/changed.md"
    with pytest.raises(FrozenInstanceError):
        current.associations[0].corpus = "changed"


@given(st.dictionaries(st.text(alphabet="abc🧭", min_size=1), st.integers(), max_size=8))
def test_mapping_order_invariance_input_preservation_and_partial_application(
    metadata: dict,
) -> None:
    original = copy.deepcopy(metadata)
    describe = partial(
        describe_conversion,
        source_path="/synthetic/article.md",
        source_format=".md",
        source_sha256="a" * 64,
        metadata=metadata,
        references={"joplin:a": {"title": "A"}},
        corpus="technology",
        contract=contract(),
    )
    first = describe(SOURCE)
    descriptor("Other content.\n")
    assert describe(SOURCE) == first
    assert describe(SOURCE, metadata=dict(reversed(list(metadata.items())))) == first
    assert metadata == original


@pytest.mark.parametrize(
    "value",
    [object(), {1: "non-string key"}, {"bad": float("nan")}, float("inf"), {1, 2}, "\ud800"],
)
def test_non_json_or_nonfinite_configuration_is_refused(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        descriptor(contract=contract(configuration={"value": value}))


@pytest.mark.parametrize(
    "field", ["implementation", "normalization", "chunk_schema", "serialization"]
)
def test_missing_or_blank_contract_versions_are_refused(field: str) -> None:
    missing = contract()
    del missing[field]
    with pytest.raises(ValueError, match="contract"):
        descriptor(contract=missing)
    with pytest.raises(ValueError, match="contract"):
        descriptor(contract=contract(**{field: " "}))


@pytest.mark.parametrize("digest", ["", "a" * 63, "A" * 64, True])
def test_source_digest_is_required(digest: object) -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        descriptor(source_sha256=digest)


def test_verified_md_and_epub_share_logical_references_but_have_distinct_physical_keys(
    tmp_path: Path,
) -> None:
    markdown = bind_article(tmp_path / "markdown")
    epub = bind_article(tmp_path / "epub", suffix=".epub")
    describe = partial(article.describe_bound_article, corpus="technology", contract=contract())
    direct = describe(markdown.binding)
    wrapped = describe(epub.binding)
    assert direct.content_sha256 == wrapped.content_sha256
    assert direct.associations == wrapped.associations
    assert direct.key != wrapped.key
    assert direct.source_format == ".md"
    assert wrapped.source_format == ".epub"
    other = bind_article(tmp_path / "other", text="# Other\n\nOther text.\n", note_id="d" * 32)
    describe(other.binding)
    assert describe(markdown.binding).to_json() == direct.to_json()


def test_changed_evidence_cannot_produce_an_unchanged_verified_descriptor(tmp_path: Path) -> None:
    bound = bind_article(tmp_path / "article")
    before = article.describe_bound_article(bound.binding, corpus="technology", contract=contract())
    bound.source.write_bytes(bound.source.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        article.describe_bound_article(bound.binding, corpus="technology", contract=contract())
    assert before.content_sha256 == hashlib.sha256(SOURCE.encode()).hexdigest()


def test_verified_multi_note_metadata_updates_leave_the_other_association_intact(
    tmp_path: Path,
) -> None:
    bound = bind_article(tmp_path / "article")
    config = bound.config()
    sidecar_path = Path(bound.binding["sidecar_path"])
    quality_path = Path(bound.binding["quality_path"])
    first_record_path = Path(bound.binding["record_bindings"][0]["path"])
    sidecar = json.loads(sidecar_path.read_bytes())
    quality = json.loads(quality_path.read_bytes())
    record = json.loads(first_record_path.read_bytes())
    second = copy.deepcopy(record)
    ident = "e" * 32
    second["note"]["id"] = ident
    second["note"]["title"] = "Second note for the same article"
    reference = copy.deepcopy(sidecar["metadata"]["joplin_references"][0])
    reference.update(
        note_id=ident,
        title=second["note"]["title"],
        markdown_link=":/" + ident,
        note_url="joplin://x-callback-url/openNote?id=" + ident,
    )
    sidecar["metadata"]["joplin_references"].append(reference)
    quality["reviewed"].append(
        {"id": ident, "accepted": True, "capture_sha256": second["capture"]["sha256"]}
    )
    second_path = bound.source.parent / "second-record.json"

    def write_evidence(path: Path, value: dict) -> str:
        path.write_text(json.dumps(value), encoding="utf-8")
        return hashlib.sha256(path.read_bytes()).hexdigest()

    binding = {
        **bound.binding,
        "sidecar_sha256": write_evidence(sidecar_path, sidecar),
        "quality_sha256": write_evidence(quality_path, quality),
        "record_bindings": [
            *bound.binding["record_bindings"],
            {"path": str(second_path), "sha256": write_evidence(second_path, second)},
        ],
    }
    describe = partial(article.describe_bound_article, corpus="technology", contract=contract())
    before = describe(binding)
    record["tags"] = ["changed-tag"]
    sidecar["metadata"]["joplin_references"][0]["tags"] = record["tags"]
    changed_binding = {
        **binding,
        "sidecar_sha256": write_evidence(sidecar_path, sidecar),
        "record_bindings": [
            {"path": str(first_record_path), "sha256": write_evidence(first_record_path, record)},
            binding["record_bindings"][1],
        ],
    }
    after = describe(changed_binding)
    assert before.associations[0].key != after.associations[0].key
    assert before.associations[1] == after.associations[1]
    assert after.content_sha256 == before.content_sha256
    assert after.key != before.key
    assert compare_conversion(after, previous=before).changed_fields == (
        "metadata_json",
        "associations",
    )
    native_path = bound.source.parent / "native.json"
    write_evidence(native_path, article.metadata(changed_binding, "technology"))
    rows = convert(str(bound.source), config)
    assert "".join(row["text"] for row in rows) == SOURCE
    assert all(len(row["metadata"]["joplin_references"]) == 2 for row in rows)
