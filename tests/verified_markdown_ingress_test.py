"""Synthetic proof-bound Markdown ingress through the canonical conversion path."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pdf_chunker.adapters import emit_jsonl, io_text
from pdf_chunker.adapters import io_joplin_epub as article
from pdf_chunker.cli import _cli_overrides
from pdf_chunker.config import PipelineSpec
from pdf_chunker.core_new import (
    _enforce_invariants,
    _input_artifact,
    _load_document_metadata,
    _run_passes,
    choose_parser,
    convert,
)

SOURCE = """# Example article 🧭

An approved paragraph with café and [a source](https://example.org/article).

## Code and table

```python
value = "exact spacing"  # keep this newline
print(value)
```

| Item | Value |
| --- | --- |
| café | 🧭 |

    indented_code()
    another_line()

### Transcript syntax is article content

WEBVTT

00:00:01.000 --> 00:00:02.000

HOST:
Preserve this line break.
"""


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class BoundArticle:
    source: Path
    binding: dict
    text: str

    def config(self) -> PipelineSpec:
        metadata_path = self.source.parent / "native.json"
        metadata_path.write_text(
            json.dumps(article.metadata(self.binding, "technology")), encoding="utf-8"
        )
        return PipelineSpec(
            pipeline=["pdf_parse", "text_clean", "heading_detect", "split_semantic", "emit_jsonl"],
            options={
                "epub_parse" if self.source.suffix == ".epub" else "text_parse": {
                    "metadata_path": str(metadata_path)
                },
                "split_semantic": {"chunk_size": 20, "overlap": 0},
                "emit_jsonl": {"max_chars": 32768},
            },
        )


def bind_article(
    root: Path,
    *,
    suffix: str = ".md",
    text: str = SOURCE,
    capture_text: str | None = None,
    note_id: str = "b" * 32,
) -> BoundArticle:
    """Generate portable article evidence, with no real note or document data."""
    root.mkdir(parents=True)
    source = root / ("article" + suffix)
    ident = note_id
    captured = text if capture_text is None else capture_text
    capture = {
        "sha256": digest(captured.encode()),
        "complete": True,
        "url": "https://example.org/article",
        "final_url": "https://example.org/article",
        "archive_url": "https://example.org/snapshot",
        "captured_at": "2026-01-01T00:00:00Z",
        "method": "approved_fixture",
        "snapshot": {"sha256": "c" * 64},
        "linked_article": {"url": "https://example.org/linked"},
        "discussion_items": [{"url": "https://example.org/discussion"}],
    }
    record = {
        "kind": "article",
        "outcome": "verified",
        "note": {"id": ident, "title": "Example article", "parent_id": "example-notebook"},
        "folder_path": "Examples/Articles",
        "tags": ["example"],
        "capture": {**capture, "markdown": captured},
    }
    reference = {
        "note_id": ident,
        "markdown_link": ":/" + ident,
        "note_url": "joplin://x-callback-url/openNote?id=" + ident,
        "title": record["note"]["title"],
        "notebook_id": record["note"]["parent_id"],
        "folder_path": record["folder_path"],
        "tags": record["tags"],
        "capture": capture,
        "urls": ["https://example.org/article", "https://example.org/snapshot"],
    }
    metadata = {
        "schema": "joplin-article-v1",
        "corpus": "technology",
        "source": source.name,
        "source_file": source.name,
        "title": "Example article",
        "content_sha256": digest(text.encode()),
        "content_generation": digest(text.encode()),
        "joplin_references": [reference],
        "urls": reference["urls"],
        "chunk_id": "caller-cannot-own-generated-identity",
        "page": 999,
    }
    quality = {
        "final": True,
        "reviewed": [{"id": ident, "accepted": True, "capture_sha256": capture["sha256"]}],
    }
    evidence = {
        "sidecar_path": {"text": text, "metadata": metadata},
        "quality_path": quality,
        "record": record,
    }
    paths = {key: root / (key + ".json") for key in evidence}
    for key, value in evidence.items():
        paths[key].write_text(json.dumps(value), encoding="utf-8")
    if suffix == ".epub":
        with zipfile.ZipFile(source, "w") as archive:
            archive.writestr("mimetype", "application/epub+zip")
            archive.writestr("OEBPS/article.md", text)
            archive.writestr("OEBPS/provenance.json", json.dumps(metadata))
    else:
        source.write_bytes(text.encode())
    binding = {key: str(paths[key]) for key in ("sidecar_path", "quality_path")} | {
        "path": str(source),
        "source_sha256": digest(source.read_bytes()),
        "sidecar_sha256": digest(paths["sidecar_path"].read_bytes()),
        "quality_sha256": digest(paths["quality_path"].read_bytes()),
        "record_bindings": [
            {"path": str(paths["record"]), "sha256": digest(paths["record"].read_bytes())}
        ],
    }
    return BoundArticle(source, binding, text)


@pytest.mark.parametrize("suffix", [".md", ".MD"])
def test_markdown_selects_shared_page_block_normalization(suffix: str) -> None:
    assert choose_parser(suffix) == "text_parse"


def test_cli_routes_markdown_metadata_to_the_selected_parser() -> None:
    options = _cli_overrides(Path("article.md"), metadata_file=Path("native.json"))
    assert options["text_parse"] == {"metadata_path": "native.json"}


def test_unverified_markdown_is_refused_instead_of_using_transcript_ingress() -> None:
    with pytest.raises(ValueError, match="verified.*Markdown"):
        article.read_bound_article("/unused.md", {})


def test_direct_markdown_matches_verified_epub_text_and_article_provenance(tmp_path: Path) -> None:
    markdown = bind_article(tmp_path / "markdown")
    epub = bind_article(tmp_path / "epub", suffix=".epub")
    direct = convert(str(markdown.source), markdown.config())
    wrapped = convert(str(epub.source), epub.config())
    assert len(direct) > 1
    assert [row["text"] for row in direct] == [row["text"] for row in wrapped]
    assert "".join(row["text"] for row in direct).encode() == markdown.source.read_bytes()
    shared = (
        "joplin_references",
        "urls",
        "content_sha256",
        "content_generation",
        "title",
        "article_title",
        "source_kind",
        "source_system",
        "source_char_start",
        "source_char_end",
    )
    for left, right in zip(direct, wrapped, strict=True):
        assert {key: left["metadata"][key] for key in shared} == {
            key: right["metadata"][key] for key in shared
        }
        meta = left["metadata"]
        assert meta["source_file"] == "article.md"
        assert meta["page"] != 999
        assert meta["chunk_id"] != "caller-cannot-own-generated-identity"
        assert markdown.text[meta["source_char_start"] : meta["source_char_end"]] == left["text"]


def test_unchanged_conversion_bytes_survive_an_intervening_document(tmp_path: Path) -> None:
    first = bind_article(tmp_path / "first")
    other = bind_article(
        tmp_path / "other", text="# Another article\n\nDifferent content.\n", note_id="d" * 32
    )
    config = first.config()
    original = copy.deepcopy(config)
    before = convert(str(first.source), config)
    emit_jsonl.write(before, str(tmp_path / "before.jsonl"))
    convert(str(other.source), other.config())
    after = convert(str(first.source), config)
    emit_jsonl.write(after, str(tmp_path / "after.jsonl"))
    assert before == after
    assert (tmp_path / "before.jsonl").read_bytes() == (tmp_path / "after.jsonl").read_bytes()
    assert config == original


@pytest.mark.parametrize("suffix", [".md", ".MD", ".epub"])
def test_approved_capture_normalization_does_not_rewrite_source(
    tmp_path: Path, suffix: str
) -> None:
    bound = bind_article(
        tmp_path / "article", suffix=suffix, capture_text="\n" + SOURCE.replace("\n", "\r\n") + "\n"
    )
    before = bound.source.read_bytes()
    rows = convert(str(bound.source), bound.config())
    assert "".join(row["text"] for row in rows) == SOURCE
    assert bound.source.read_bytes() == before


@pytest.mark.parametrize("suffix", [".json", ".JSON", ".yaml"])
def test_metadata_decoding_preserves_non_bmp_unicode(tmp_path: Path, suffix: str) -> None:
    metadata = {"title": "Example 🧭", "nested": {"tags": ["café", "🧭"]}}
    path = tmp_path / ("metadata" + suffix)
    path.write_text(json.dumps(metadata, ensure_ascii=suffix.lower() == ".json"), encoding="utf-8")
    assert _load_document_metadata(str(path)) == metadata


@pytest.mark.parametrize("target", ["path", "sidecar_path", "quality_path", "record"])
def test_changed_evidence_fails_before_pipeline_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    bound = bind_article(tmp_path / "article")
    config = bound.config()
    path = Path(
        bound.binding["record_bindings"][0]["path"] if target == "record" else bound.binding[target]
    )
    path.write_bytes(path.read_bytes() + b" ")

    def unexpected_pipeline(*args, **kwargs):
        pytest.fail("changed evidence reached document processing")

    monkeypatch.setattr("pdf_chunker.core_new._run_passes", unexpected_pipeline)
    with pytest.raises(ValueError, match="hash mismatch"):
        convert(str(bound.source), config)


@pytest.mark.parametrize(
    "metadata", [{}, {"verified_joplin_markdown": True}, {"source_kind": "article"}]
)
def test_booleans_or_unrelated_metadata_do_not_authorize_markdown(metadata: dict) -> None:
    with pytest.raises(ValueError, match="verified.*Markdown"):
        article.read_bound_article("/unused.md", metadata)


@pytest.mark.parametrize(
    "raw",
    [
        SOURCE.replace("\n", "\r\n").encode(),
        b"\xef\xbb\xbf" + SOURCE.encode(),
        b"\xff",
        SOURCE[:-1].encode(),
    ],
)
def test_rehashed_nonmatching_source_is_refused_without_normalization(
    tmp_path: Path, raw: bytes
) -> None:
    bound = bind_article(tmp_path / "article")
    bound.source.write_bytes(raw)
    binding = {**bound.binding, "source_sha256": digest(raw)}
    with pytest.raises(ValueError, match="Markdown bytes differ"):
        article.metadata(binding, "technology")
    assert bound.source.read_bytes() == raw


def test_rehashed_unapproved_receipt_cannot_authorize_source(tmp_path: Path) -> None:
    bound = bind_article(tmp_path / "article")
    path = Path(bound.binding["quality_path"])
    quality = json.loads(path.read_bytes())
    quality["final"] = False
    path.write_text(json.dumps(quality), encoding="utf-8")
    binding = {**bound.binding, "quality_sha256": digest(path.read_bytes())}
    with pytest.raises(ValueError, match="final.*approval"):
        article.metadata(binding, "technology")


@pytest.mark.parametrize("step", ["merge_footers", "detect_page_artifacts"])
def test_pdf_artifact_steps_are_rejected_for_markdown(step: str) -> None:
    with pytest.raises(ValueError, match="not supported for .md"):
        _enforce_invariants(PipelineSpec(pipeline=["pdf_parse", step]), input_path="article.md")


@given(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=300))
def test_exact_source_validator_is_pure_and_partially_applicable(text: str) -> None:
    row = {"text": text, "metadata": {"nested": ["untouched"]}}
    original = copy.deepcopy(row)
    verify = partial(article._source_text, suffix=".md", row=row)
    assert verify(text.encode("utf-8")) == text
    with pytest.raises(ValueError, match="Markdown bytes differ"):
        verify(text.encode("utf-8") + b"\x00")
    assert row == original


def test_direct_markdown_never_calls_the_transcript_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bound = bind_article(tmp_path / "article")

    def unexpected_reader(*args, **kwargs):
        pytest.fail("Markdown reached transcript cue stripping")

    monkeypatch.setattr(io_text, "read", unexpected_reader)
    rows = convert(str(bound.source), bound.config())
    assert "".join(row["text"] for row in rows) == SOURCE


def test_cli_writes_exact_markdown_using_the_existing_spec(tmp_path: Path) -> None:
    bound = bind_article(tmp_path / "article")
    bound.config()
    output = tmp_path / "article.jsonl"
    command = [
        sys.executable,
        "-m",
        "pdf_chunker.cli",
        "convert",
        str(bound.source),
        "--spec",
        str(Path(__file__).parents[1] / "pipeline.yaml"),
        "--metadata-file",
        str(bound.source.parent / "native.json"),
        "--out",
        str(output),
        "--chunk-size",
        "20",
        "--overlap",
        "0",
        "--no-enrich",
    ]
    result = subprocess.run(
        command,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1])},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert "".join(row["text"] for row in rows).encode() == bound.source.read_bytes()


@pytest.mark.parametrize("field", ["joplin_references", "content_sha256"])
def test_caller_metadata_cannot_override_verified_provenance(tmp_path: Path, field: str) -> None:
    bound = bind_article(tmp_path / "article")
    config = bound.config()
    path = bound.source.parent / "native.json"
    metadata = json.loads(path.read_bytes())
    metadata[field] = [] if field == "joplin_references" else "0" * 64
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata differs"):
        convert(str(bound.source), config)


class RecordingClient:
    def __init__(self) -> None:
        self.inputs: list[str] = []

    def classify_chunk_utterance(self, text: str, *, tag_configs: dict) -> dict:
        self.inputs.append(text)
        return {"classification": "question", "tags": []}


@pytest.mark.parametrize("limit", [256, 32768])
def test_existing_admission_runs_before_injected_enrichment(tmp_path: Path, limit: int) -> None:
    bound = bind_article(tmp_path / "article")
    client = RecordingClient()
    config = bound.config()
    config.pipeline.insert(-1, "ai_enrich")
    config.options["ai_enrich"] = {"enabled": True, "client": client, "tag_configs": {}}
    config.options["emit_jsonl"]["max_chars"] = limit
    artifact = _input_artifact(str(bound.source), config)
    if limit == 256:
        with pytest.raises(ValueError, match="JSONL"):
            _run_passes(config, artifact)
        assert client.inputs == []
    else:
        result, _ = _run_passes(config, artifact)
        assert "".join(row["text"] for row in result.payload) == SOURCE
        assert "".join(client.inputs) == SOURCE
        assert all(row["metadata"]["enrichment_status"] == "complete" for row in result.payload)
        assert all(len(json.dumps(row, ensure_ascii=False)) <= limit for row in result.payload)
