"""Resolved verified-article inputs, using synthetic evidence and no inference."""

from __future__ import annotations

import copy
import json
from dataclasses import FrozenInstanceError, replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pdf_chunker.adapters import article_conversion as adapter
from pdf_chunker.adapters.article_conversion import prepare_bound_conversion
from pdf_chunker.article_conversion import resolve_article_spec
from pdf_chunker.config import PipelineSpec, load_spec
from pdf_chunker.core_new import convert
from pdf_chunker.passes.chunk_options import SplitOptions
from pdf_chunker.passes.emit_jsonl import EmitConfig
from tests.verified_markdown_ingress_test import SOURCE, bind_article


def spec(**options):
    return PipelineSpec(
        pipeline=["pdf_parse", "text_clean", "heading_detect", "split_semantic", "emit_jsonl"],
        options=options,
    )


def resolve(value, **changes):
    return resolve_article_spec(
        value,
        suffix=".md",
        emit_defaults=EmitConfig(max_chars=32768),
        split_defaults=SplitOptions.from_base(400, 50, None),
        **changes,
    )


def test_defaults_overrides_and_parser_resolution():
    original = spec(split_semantic={"chunk_size": 100, "overlap": 0})
    before = copy.deepcopy(original)
    result = resolve(original)
    assert result.pipeline[0] == "text_parse"
    assert result.options["split_semantic"] == {
        "chunk_size": 100,
        "overlap": 0,
        "min_chunk_size": 10,
    }
    assert result.options["emit_jsonl"]["max_chars"] == 32768
    assert result.options["emit_jsonl"]["metadata_key"] == "metadata"
    assert original == before
    assert resolve(result) == result


@given(st.integers(min_value=51, max_value=5000))
def test_explicit_defaults_and_partial_application_agree(chunk_size):
    stage = partial(
        resolve_article_spec,
        suffix=".md",
        emit_defaults=EmitConfig(max_chars=32768),
        split_defaults=SplitOptions.from_base(400, 50, None),
    )
    implicit = stage(spec(split_semantic={"chunk_size": chunk_size}))
    explicit = stage(
        spec(
            split_semantic={
                "chunk_size": chunk_size,
                "overlap": 50,
                "min_chunk_size": max(8, chunk_size // 10),
            },
            emit_jsonl={"max_chars": 32768},
        )
    )
    assert implicit == explicit
    explicit.options["emit_jsonl"]["max_chars"] = 1
    assert implicit.options["emit_jsonl"]["max_chars"] == 32768


@pytest.mark.parametrize(
    "pipeline",
    [
        ["text_clean", "split_semantic", "emit_jsonl"],
        ["pdf_parse", "text_clean", "split_semantic", "ai_enrich", "emit_jsonl"],
        ["pdf_parse", "text_clean", "split_semantic", "split_semantic", "emit_jsonl"],
        ["pdf_parse", "text_clean", "list_detect", "split_semantic", "emit_jsonl"],
        ["pdf_parse", "epub_parse", "text_clean", "split_semantic", "emit_jsonl"],
    ],
)
def test_unqualified_pipelines_refuse(pipeline):
    with pytest.raises(ValueError, match="base article pipeline"):
        resolve(PipelineSpec(pipeline=pipeline))


@pytest.mark.parametrize(
    "options",
    [
        {"ai_enrich": {"enabled": True}},
        {"split_semantic": {"list_continuation_callback": lambda *args: None}},
        {"text_clean": {"interactive_footers": True}},
        {"text_parse": {"exclude_pages": "1"}},
        {"split_semantic": {"min_chunk_size": None}},
        {"ai_enrich": {"enabled": False, "api_key": "synthetic-secret"}},
        {"emit_jsonl": {"metadata_key": "text"}},
        {"emit_jsonl": {"max_chars": float("nan")}},
        {"split_semantic": {"chunk_size": 50}},  # Default overlap is also 50.
        {"split_semantic": {"overlap": -1}},
        {"emit_jsonl": {"unknown_limit": 1}},
    ],
)
def test_effectful_or_unsupported_settings_refuse(options):
    with pytest.raises((TypeError, ValueError)):
        resolve(spec(**options))


def test_preparation_executes_same_rows_and_keeps_snapshot_detached(tmp_path, monkeypatch):
    bound = bind_article(tmp_path / "article")
    original = bound.config()
    before = copy.deepcopy(original)
    monkeypatch.setenv("PDF_CHUNKER_JSONL_META_KEY", "record_metadata")
    prepared = prepare_bound_conversion(bound.binding, original, corpus="technology")
    resolved = prepared.spec()
    assert resolved.options["emit_jsonl"]["metadata_key"] == "record_metadata"
    expected = convert(str(bound.source), original)
    monkeypatch.setenv("PDF_CHUNKER_JSONL_META_KEY", "other_metadata")
    assert convert(str(bound.source), prepared.spec()) == expected
    assert "".join(row["text"] for row in expected) == SOURCE
    assert original == before
    resolved.options["split_semantic"]["chunk_size"] = 1000
    assert prepared.spec().options["split_semantic"]["chunk_size"] == 20
    with pytest.raises(FrozenInstanceError):
        prepared.spec_json = "{}"
    configuration = json.loads(prepared.descriptor.contract_json)["configuration"]
    assert configuration["pipeline"] == prepared.spec().pipeline
    assert configuration["options"] == prepared.spec().options
    assert json.loads(prepared.descriptor.contract_json)["implementation"]


def test_rerun_runtime_and_environment_invalidation(tmp_path, monkeypatch):
    bound = bind_article(tmp_path / "article")
    config = bound.config()
    prepare = partial(prepare_bound_conversion, corpus="technology")
    first = prepare(bound.binding, config)
    other = bind_article(tmp_path / "other", text="# Other\n\nDifferent article.\n")
    convert(str(other.source), other.config())
    assert prepare(bound.binding, config) == first
    monkeypatch.setenv("PDF_CHUNKER_JSONL_META_KEY", "alternate")
    altered = prepare(bound.binding, config)
    assert altered.descriptor.key != first.descriptor.key
    assert altered.descriptor.associations == first.descriptor.associations
    monkeypatch.setenv("PDF_CHUNKER_JSONL_MAX_CHARS", "12345")
    assert prepare(bound.binding, config) == altered  # Explicit max_chars wins.


def test_preparation_checks_metadata_matches_verified_binding(tmp_path):
    bound = bind_article(tmp_path / "article")
    config = bound.config()
    metadata_path = config.options["text_parse"]["metadata_path"]
    with open(metadata_path, encoding="utf-8") as stream:
        native = json.load(stream)
    native["title"] = "Caller replacement"
    with open(metadata_path, "w", encoding="utf-8") as stream:
        json.dump(native, stream)
    with pytest.raises(ValueError, match="native metadata differs"):
        prepare_bound_conversion(bound.binding, config, corpus="technology")


def test_preparation_refuses_changed_source(tmp_path):
    bound = bind_article(tmp_path / "article")
    config = bound.config()
    bound.source.write_text("unapproved replacement", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        prepare_bound_conversion(bound.binding, config, corpus="technology")


def test_default_checked_in_pipeline_resolves(tmp_path):
    bound = bind_article(tmp_path / "article")
    config = bound.config()
    checked_in = load_spec("pipeline.yaml")
    checked_in.options["text_parse"] = config.options["text_parse"]
    prepared = prepare_bound_conversion(bound.binding, checked_in, corpus="technology")
    assert prepared.spec().pipeline[0] == "text_parse"


def test_changed_pass_defaults_are_resolved(tmp_path, monkeypatch):
    from pdf_chunker.passes import split_semantic

    bound = bind_article(tmp_path / "article")
    config = bound.config()
    config.options["split_semantic"] = {}
    first = prepare_bound_conversion(bound.binding, config, corpus="technology")
    monkeypatch.setattr(split_semantic.split_semantic, "chunk_size", 600)
    second = prepare_bound_conversion(bound.binding, config, corpus="technology")
    assert second.spec().options["split_semantic"]["chunk_size"] == 600
    assert second.descriptor.key != first.descriptor.key


def test_emit_default_snapshot_is_explicit():
    baseline = EmitConfig(max_chars=32768)
    resolved = resolve_article_spec(
        spec(),
        suffix=".md",
        emit_defaults=replace(baseline, metadata_key="custom"),
        split_defaults=SplitOptions.from_base(400, 50, None),
    )
    assert resolved.options["emit_jsonl"]["metadata_key"] == "custom"


def test_collector_tracks_source_bytes_and_distribution_versions(tmp_path, monkeypatch):
    root = tmp_path / "package"
    (root / "adapters").mkdir(parents=True)
    (root / "core_new.py").write_text("# runner\n", encoding="utf-8")
    helper = root / "helper.py"
    helper.write_text("# first\n", encoding="utf-8")
    monkeypatch.setattr(adapter, "__file__", str(root / "adapters" / "article_conversion.py"))
    packages = [
        SimpleNamespace(metadata={"Name": "example-dependency"}, version="1.0"),
        SimpleNamespace(metadata={"Name": "another-dependency"}, version="2.0"),
    ]
    monkeypatch.setattr(adapter, "distributions", lambda: iter(packages))
    first = adapter.collect_article_runtime()
    packages.reverse()
    assert adapter.collect_article_runtime() == first
    helper.write_text("# second\n", encoding="utf-8")
    second = adapter.collect_article_runtime()
    assert second["package_sources"] != first["package_sources"]
    assert second["distributions"] == first["distributions"]
    packages[0].version = "2.1"
    third = adapter.collect_article_runtime()
    assert third["distributions"] != second["distributions"]
    assert third["package_sources"] == second["package_sources"]
    (root / "core_new.py").unlink()
    with pytest.raises(ValueError, match="sources required"):
        adapter.collect_article_runtime()


def test_runtime_fingerprint_changes_complete_key_only(tmp_path, monkeypatch):
    bound = bind_article(tmp_path / "article")
    config = bound.config()
    runtime = {"python": "synthetic-3.11", "package_sources": {"helper.py": "a" * 64}}
    monkeypatch.setattr(adapter, "collect_article_runtime", lambda: copy.deepcopy(runtime))
    first = prepare_bound_conversion(bound.binding, config, corpus="technology")
    runtime["package_sources"]["helper.py"] = "b" * 64
    second = prepare_bound_conversion(bound.binding, config, corpus="technology")
    assert second.descriptor.key != first.descriptor.key
    assert second.descriptor.associations == first.descriptor.associations
    assert second.descriptor.content_sha256 == first.descriptor.content_sha256
    assert second.spec_json == first.spec_json


def test_implicit_and_explicit_effective_settings_share_descriptor(tmp_path, monkeypatch):
    bound = bind_article(tmp_path / "article")
    config = bound.config()
    monkeypatch.setenv("PDF_CHUNKER_JSONL_META_KEY", "same_metadata")
    implicit = prepare_bound_conversion(bound.binding, config, corpus="technology")
    config.options["emit_jsonl"]["metadata_key"] = "same_metadata"
    assert prepare_bound_conversion(bound.binding, config, corpus="technology") == implicit
    monkeypatch.setenv("PDF_CHUNKER_JSONL_META_KEY", "ignored_metadata")
    assert prepare_bound_conversion(bound.binding, config, corpus="technology") == implicit


def test_md_and_epub_prepare_for_same_content_without_transcript_reader(tmp_path, monkeypatch):
    from pdf_chunker.adapters import io_text

    def refuse(*args, **kwargs):
        pytest.fail("verified articles must not call the transcript reader")

    monkeypatch.setattr(io_text, "read", refuse)
    markdown = bind_article(tmp_path / "markdown")
    epub = bind_article(tmp_path / "epub", suffix=".epub")
    prepared_md = prepare_bound_conversion(markdown.binding, markdown.config(), corpus="technology")
    prepared_epub = prepare_bound_conversion(epub.binding, epub.config(), corpus="technology")
    assert prepared_md.descriptor.associations == prepared_epub.descriptor.associations
    assert prepared_md.descriptor.content_sha256 == prepared_epub.descriptor.content_sha256
    assert prepared_md.descriptor.key != prepared_epub.descriptor.key
    for bound, prepared in ((markdown, prepared_md), (epub, prepared_epub)):
        rows = convert(str(bound.source), prepared.spec())
        assert "".join(row["text"] for row in rows) == SOURCE
        first_bytes = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
        other = bind_article(tmp_path / (bound.source.suffix[1:] + "-other"), text="# Other\n")
        convert(str(other.source), other.config())
        repeated = convert(str(bound.source), prepared.spec())
        assert (
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in repeated).encode()
            == first_bytes
        )


def test_actual_emit_environment_limit_is_captured(tmp_path, monkeypatch):
    bound = bind_article(tmp_path / "article")
    config = bound.config()
    config.options["emit_jsonl"] = {}
    monkeypatch.setenv("PDF_CHUNKER_JSONL_MAX_CHARS", "32768")
    first = prepare_bound_conversion(bound.binding, config, corpus="technology")
    monkeypatch.setenv("PDF_CHUNKER_JSONL_MAX_CHARS", "1")
    second = prepare_bound_conversion(bound.binding, config, corpus="technology")
    assert first.descriptor.key != second.descriptor.key
    assert first.spec().options["emit_jsonl"]["max_chars"] == 32768
    assert "".join(row["text"] for row in convert(str(bound.source), first.spec())) == SOURCE
    with pytest.raises(ValueError, match="JSONL size limit"):
        convert(str(bound.source), second.spec())


def test_resolved_spec_does_not_write_requested_output(tmp_path):
    bound = bind_article(tmp_path / "article")
    config = bound.config()
    output_path = tmp_path / "should-not-exist.jsonl"
    config.options["emit_jsonl"]["output_path"] = str(output_path)
    prepare_bound_conversion(bound.binding, config, corpus="technology")
    assert not output_path.exists()
    assert set(Path(bound.source.parent).glob("*.jsonl")) == set()


@pytest.mark.parametrize("suffix", [".txt", ".html", ".pdf"])
def test_unqualified_source_formats_refuse(suffix):
    with pytest.raises(ValueError, match=".md or .epub"):
        resolve_article_spec(
            spec(),
            suffix=suffix,
            emit_defaults=EmitConfig(),
            split_defaults=SplitOptions.from_base(400, 50, None),
        )


def test_no_heading_profile_and_explicit_minimum_are_preserved():
    original = spec(split_semantic={"chunk_size": 100, "min_chunk_size": 30})
    original.pipeline.remove("heading_detect")
    resolved = resolve(original)
    assert resolved.pipeline == ["text_parse", "text_clean", "split_semantic", "emit_jsonl"]
    assert resolved.options["split_semantic"]["min_chunk_size"] == 30
