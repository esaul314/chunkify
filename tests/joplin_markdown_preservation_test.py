import copy
import hashlib
import json
import uuid
import zipfile
from pathlib import Path

import pytest

from pdf_chunker.adapters import io_joplin_epub as article
from pdf_chunker.config import PipelineSpec
from pdf_chunker.core_new import _input_artifact, convert
from pdf_chunker.framework import Artifact
from pdf_chunker.passes.joplin_markdown import slices
from pdf_chunker.passes.split_semantic import make_splitter
from pdf_chunker.passes.text_clean import text_clean


def sha(value):
    return hashlib.sha256(value).hexdigest()


class Fixture:
    def __init__(self, root):
        self.root = Path(root)
        self.text = "# Complete fixture article\n\nA concrete paragraph and [source](https://example.org/).\n"
        self.ident = "a" * 32
        self.record = {
            "kind": "strict_url_only",
            "outcome": "staged_complete",
            "note": {
                "id": self.ident,
                "title": "Fixture article",
                "parent_id": "notebook",
                "created_time": 123,
                "user_created_time": 122,
                "is_todo": 0,
                "is_conflict": 0,
                "deleted_time": 0,
            },
            "folder_path": "Tech/Articles",
            "tags": [{"title": "reviewed"}],
            "capture": {
                "markdown": self.text,
                "sha256": sha(self.text.encode()),
                "complete": True,
                "url": "https://example.org/article",
                "method": "public_html",
            },
        }
        self.ref = {
            "note_id": self.ident,
            "markdown_link": ":/" + self.ident,
            "note_url": "joplin://x-callback-url/openNote?id=" + self.ident,
            "title": "Fixture article",
            "notebook_id": "notebook",
            "folder_path": "Tech/Articles",
            "tags": ["reviewed"],
            "urls": ["https://example.org/article"],
            "created_time": 123,
            "user_created_time": 122,
            "capture": {k: v for k, v in self.record["capture"].items() if k != "markdown"},
        }
        self.meta = {
            "schema": "joplin-article-v1",
            "corpus": "technology",
            "source": "fixture.epub",
            "source_file": "fixture.epub",
            "content_sha256": sha(self.text.encode()),
            "content_generation": sha(self.text.encode()),
            "title": "Fixture article",
            "joplin_references": [self.ref],
            "urls": ["https://example.org/article"],
            "tags": ["reviewed"],
            "enrichment_status": "pending",
            "utterance_type": "article",
        }
        self.quality = {
            "final": True,
            "reviewed": [
                {
                    "id": self.ident,
                    "accepted": True,
                    "capture_sha256": self.record["capture"]["sha256"],
                }
            ],
        }
        self.binding = {}
        self.rewrite()

    def rewrite(self):
        paths = {
            k: self.root / v
            for k, v in [
                ("path", "fixture.epub"),
                ("sidecar_path", "sidecar.jsonl"),
                ("quality_path", "quality.json"),
                ("record", "original.json"),
            ]
        }
        paths["record"].write_text(json.dumps(self.record))
        paths["sidecar_path"].write_text(
            json.dumps({"text": self.text, "metadata": self.meta}) + "\n"
        )
        paths["quality_path"].write_text(json.dumps(self.quality))
        with zipfile.ZipFile(paths["path"], "w") as archive:
            archive.writestr("mimetype", "application/epub+zip")
            archive.writestr("OEBPS/article.md", self.text)
            archive.writestr("OEBPS/provenance.json", json.dumps(self.meta))
        self.binding = {k: str(v) for k, v in paths.items() if k != "record"} | {
            "source_sha256": sha(paths["path"].read_bytes()),
            "sidecar_sha256": sha(paths["sidecar_path"].read_bytes()),
            "quality_sha256": sha(paths["quality_path"].read_bytes()),
            "record_bindings": [
                {"path": str(paths["record"]), "sha256": sha(paths["record"].read_bytes())}
            ],
        }

    def params(self):
        return {
            "corpus": "technology",
            "source_paths": [self.binding["path"]],
            "joplin_article_bindings": [self.binding],
            "workers": 1,
            "source_limit": 1,
            "local_only_enrichment_fallback": True,
            "operation_id": str(uuid.uuid4()),
            "operation_timeout": 30,
            "boot_id": "fixture-boot",
        }

    def native(self):
        return {
            "text": "Native text is untouched.\n    code indentation\n",
            "metadata": article.metadata(self.binding, "technology")
            | {
                "source": "fixture.epub",
                "source_file": "fixture.epub",
                "chunk_id": "native-id",
                "page": 2,
                "enrichment_status": "complete",
                "utterance_type": "explanation",
                "tags": ["native-tag"],
            },
        }


CONTROL = """# HN discussion

Original discussion: https://discussion.example.org/item?id=12345678

2022-08-06T12:34:56+00:00 by user64

> quoted first line
> quoted second line

## SQL example

```sql
CREATE TABLE sample (
    id INTEGER, -- preserve newline-sensitive comment
    name TEXT
);
```

| OS | x86-64 | ARM64 |
| --- | --- | --- |
| Windows | yes | |
| Linux | yes | yes |

[source](https://example.org/article?id=64)
"""


@pytest.fixture
def fixture(tmp_path):
    f = Fixture(tmp_path)
    f.text = CONTROL
    f.record["capture"]["markdown"] = f.text
    f.record["capture"]["sha256"] = sha(f.text.encode())
    f.ref["capture"]["sha256"] = f.record["capture"]["sha256"]
    f.meta["content_sha256"] = f.meta["content_generation"] = sha(f.text.encode())
    f.quality["reviewed"][0]["capture_sha256"] = f.record["capture"]["sha256"]
    f.rewrite()
    f.metadata = article.metadata(f.binding, "technology")
    f.metadata_path = tmp_path / "native-metadata.json"
    f.metadata_path.write_text(json.dumps(f.metadata))
    return f


def spec(f):
    return PipelineSpec(
        pipeline=["pdf_parse", "text_clean", "heading_detect", "split_semantic", "emit_jsonl"],
        options={
            "epub_parse": {"metadata_path": str(f.metadata_path)},
            "split_semantic": {"chunk_size": 20, "overlap": 10},
            "emit_jsonl": {"max_chars": 32768},
        },
    )


def test_end_to_end_verified_markdown_is_exact_and_deterministic(fixture):
    f = fixture
    rows = convert(f.binding["path"], spec(f))
    assert len(rows) > 1
    assert "".join(r["text"] for r in rows) == CONTROL
    assert rows == convert(f.binding["path"], spec(f))
    for r in rows:
        m = r["metadata"]
        assert m["joplin_references"] == [f.ref]
        assert m["source_file"] == "fixture.epub"
        assert m["source_text_strategy"] == "contiguous_markdown_v3"
        assert CONTROL[m["source_char_start"] : m["source_char_end"]] == r["text"]
        assert "chunk_part" not in m
    assert any(
        "id INTEGER, -- preserve newline-sensitive comment\n    name TEXT" in r["text"]
        for r in rows
    )
    assert any("| Windows | yes | |\n| Linux | yes | yes |" in r["text"] for r in rows)


@pytest.mark.parametrize("key", ["source_sha256", "sidecar_sha256", "quality_sha256"])
def test_evidence_tamper_refused_before_native_conversion(fixture, key):
    f = fixture
    f.metadata["joplin_binding"][key] = "0" * 64
    f.metadata_path.write_text(json.dumps(f.metadata))
    with pytest.raises(ValueError, match="hash mismatch"):
        convert(f.binding["path"], spec(f))


def test_forged_metadata_refs_refused(fixture):
    f = fixture
    f.metadata["joplin_references"][0]["tags"] = ["forged"]
    f.metadata_path.write_text(json.dumps(f.metadata))
    with pytest.raises(ValueError, match="differs"):
        convert(f.binding["path"], spec(f))


def test_bound_markdown_mutation_refused_and_clean_is_pure(fixture):
    a = _input_artifact(fixture.binding["path"], spec(fixture))
    original = copy.deepcopy(a.payload)
    b = text_clean(a)
    assert b.payload == original and a.payload == original
    assert b.meta["metrics"]["text_clean"]["strategy"] == "verified_markdown_passthrough"
    payload = copy.deepcopy(a.payload)
    payload["pages"][0]["blocks"][0]["text"] += "forged"
    with pytest.raises(ValueError, match="changed"):
        make_splitter()(Artifact(payload=payload, meta=a.meta))


def test_ordinary_sources_do_not_enter_preservation_route():
    assert article.read_bound_article("/unused.epub", {}) is None
    assert article.read_bound_article("/unused.epub", {"source_kind": "podcast_episode"}) is None
    with pytest.raises(ValueError, match="marker"):
        article.read_bound_article("/unused.epub", {"joplin_binding": {}})


def test_fenced_code_and_tables_remain_complete_atoms():
    parts = [CONTROL[a:b] for a, b in slices(CONTROL, 1)]
    assert "".join(parts) == CONTROL
    assert any("CREATE TABLE sample (" in p and "name TEXT" in p and "```\n" in p for p in parts)
    assert any("| OS | x86-64 | ARM64 |" in p and "| Linux | yes | yes |" in p for p in parts)


class RecordingClient:
    def __init__(self):
        self.inputs = []

    def classify_chunk_utterance(self, text, *, tag_configs):
        self.inputs.append(text)
        return {"classification": "question", "tags": []}


def test_real_native_enrichment_and_serialized_admission_preserve_source(fixture):
    # A deterministic in-process test client exercises the real native pass;
    # this is not a claim of actual AI enrichment or dataset admission.
    from pdf_chunker.core_new import _run_passes

    f = fixture
    client = RecordingClient()
    config = spec(f)
    config.pipeline.insert(-1, "ai_enrich")
    config.options["ai_enrich"] = {"enabled": True, "client": client, "tag_configs": {}}
    result, _ = _run_passes(config, _input_artifact(f.binding["path"], config))
    assert "".join(client.inputs) == CONTROL
    assert "".join(row["text"] for row in result.payload) == CONTROL
    assert all(row["metadata"]["enrichment_status"] == "complete" for row in result.payload)
    assert all(row["metadata"]["joplin_references"] == [f.ref] for row in result.payload)


# Regression context from the independently retained Wiretap source, where a
# chunk beginning with the five-space nested bullet became indented code.
LIST_CONTEXT = """1. Start the local demonstration.
2. Configure the local home app:
   * Host the static files somewhere reachable.
3. Deploy the local home app to your account:
   * Add a fake device:

     ```
     npm run firebase --prefix functions/ -- functions:config:set \
         strand1.leds=16 strand1.channel=1
     npm run deploy --prefix functions/
     ```

     + This tells the cloud fulfillment to include an `otherDeviceIds` field.
   * From the Actions Console, configure the local home SDK.

A separate unindented paragraph ends the list.
"""


def change_text(f, text):
    f.text = text
    digest = sha(text.encode())
    f.record["capture"].update(markdown=text, sha256=digest)
    f.ref["capture"]["sha256"] = digest
    f.meta["content_sha256"] = f.meta["content_generation"] = digest
    f.quality["reviewed"][0]["capture_sha256"] = digest
    f.rewrite()
    f.metadata = article.metadata(f.binding, "technology")
    f.metadata_path.write_text(json.dumps(f.metadata))


def structures(text):
    from markdown_it import MarkdownIt

    return [
        (t.type, t.content) for t in MarkdownIt().parse(text) if t.type in {"fence", "code_block"}
    ]


def test_actual_nested_list_boundary_keeps_rendered_code_context(fixture):
    f = fixture
    change_text(f, LIST_CONTEXT)
    config = spec(f)
    config.options["split_semantic"]["chunk_size"] = 1
    config.options["split_semantic"]["overlap"] = 0
    rows = convert(f.binding["path"], config)
    assert "".join(row["text"] for row in rows) == LIST_CONTEXT
    assert [item for row in rows for item in structures(row["text"])] == structures(LIST_CONTEXT)
    assert structures(LIST_CONTEXT)[0][0] == "fence"
    assert all(not row["text"].startswith("     + This") for row in rows)


def test_document_reference_targets_remain_with_every_use(fixture):
    from markdown_it import MarkdownIt

    f = fixture
    source = (
        "# Reviewed reference context\n\nSee [the report][1] and [the paper][2].\n\n"
        + "Independent intervening prose.\n\n" * 100
        + "[1]:<https://research.example.org/reports/2001/07/REPORT-01-07.pdf>\n\n"
        + "[2]:<https://archive.example.org/download/pdf/123456789.pdf>\n"
    )
    change_text(f, source)
    rows = convert(f.binding["path"], spec(f))
    assert len(rows) == 1
    assert rows[0]["text"] == source

    def links(text):
        return [
            child.attrGet("href")
            for token in MarkdownIt().parse(text)
            for child in (token.children or [])
            if child.type == "link_open"
        ]

    assert links(rows[0]["text"]) == links(source)
    assert len(links(source)) == 2


def test_oversized_context_is_held_before_real_native_enrichment(fixture):
    from pdf_chunker.core_new import _run_passes

    f = fixture
    source = (
        "# Large reference context\n\n[link][1]\n\n"
        + "word " * 10000
        + "\n\n[1]: https://example.org/source\n"
    )
    change_text(f, source)
    client = RecordingClient()
    config = spec(f)
    config.pipeline.insert(-1, "ai_enrich")
    config.options["ai_enrich"] = {"enabled": True, "client": client, "tag_configs": {}}
    with pytest.raises(ValueError, match="context exceeds native JSONL budget"):
        _run_passes(config, _input_artifact(f.binding["path"], config))
    assert client.inputs == []
    assert (
        article.read_bound_article(f.binding["path"], f.metadata)[0]["pages"][0]["blocks"][0][
            "text"
        ]
        == source
    )


@pytest.mark.parametrize(
    "literal",
    [
        (
            "[1]: not *actually* like server rendered react, since "
            "srr only happens the first page load."
        ),
        r"[1]: [https://reference.example.org/wiki/Example\_Author\_(Sample_State\_ex...](https://reference.example.org/wiki/Example_Author_(Sample_State_politician))",
        "```text\n[1]: https://example.org/literal-code\n```",
    ],
)
def test_actual_literal_reference_prefix_does_not_coalesce_whole_document(fixture, literal):
    from markdown_it import MarkdownIt

    source = "# Literal reference control\n\n" + "Independent paragraph.\n\n" * 100 + literal + "\n"
    env = {}
    MarkdownIt("commonmark").parse(source, env)
    assert not env.get("references")
    f = fixture
    change_text(f, source)
    rows = convert(f.binding["path"], spec(f))
    assert len(rows) > 1
    assert "".join(row["text"] for row in rows) == source
    assert [item for row in rows for item in structures(row["text"])] == structures(source)
