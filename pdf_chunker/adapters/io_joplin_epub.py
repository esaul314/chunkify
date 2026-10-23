"""Bounded Joplin sidecars for native conversion; no enrichment or source admission."""

import hashlib
import json
import re
import zipfile
from pathlib import Path
from urllib.parse import urlsplit

CORPORA = {
    "coaching",
    "literature",
    "management",
    "medicine",
    "neuroscience",
    "philosophy",
    "psychology",
    "reference_misc",
    "science",
    "spirituality",
    "technology",
    "writing",
}
MAX_FILE_BYTES = 4 * 1024 * 1024
FIELDS = {
    "path",
    "source_sha256",
    "sidecar_path",
    "sidecar_sha256",
    "quality_path",
    "quality_sha256",
    "record_bindings",
}


def digest(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def path(value):
    if (
        not isinstance(value, str)
        or not Path(value).is_absolute()
        or ".." in Path(value).parts
        or any(c in value for c in "\r\n\x00")
    ):
        raise ValueError("absolute bounded Joplin evidence path required")
    return Path(value)


def file_bytes(name, expected):
    p = path(name)
    if (
        p.is_symlink()
        or not p.is_file()
        or p.stat().st_size > MAX_FILE_BYTES
        or not digest(expected)
    ):
        raise ValueError("regular bounded hash-bound Joplin evidence required")
    raw = p.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Joplin evidence hash mismatch")
    return raw


def metadata(binding, corpus):
    """Read exact EPUB/sidecar/final approval; preserve nested references unaltered."""
    if corpus not in CORPORA:
        raise ValueError("existing topic corpus required")
    epub = file_bytes(binding["path"], binding["source_sha256"])
    sidecar = file_bytes(binding["sidecar_path"], binding["sidecar_sha256"])
    quality = json.loads(file_bytes(binding["quality_path"], binding["quality_sha256"]))
    lines = [json.loads(line) for line in sidecar.decode("utf-8").splitlines() if line.strip()]
    if len(lines) != 1 or not isinstance(lines[0].get("text"), str) or not lines[0]["text"].strip():
        raise ValueError("exact one-article adapter sidecar required")
    row = lines[0]
    meta = row.get("metadata")
    if (
        not isinstance(meta, dict)
        or meta.get("schema") != "joplin-article-v1"
        or meta.get("corpus") != corpus
        or meta.get("source") != Path(binding["path"]).name
        or meta.get("source_file") != Path(binding["path"]).name
        or not digest(meta.get("content_sha256"))
        or meta.get("content_generation") != meta.get("content_sha256")
        or hashlib.sha256(row["text"].encode()).hexdigest() != meta["content_sha256"]
        or not isinstance(meta.get("title"), str)
        or not meta["title"].strip()
    ):
        raise ValueError("adapter article/source/content identity mismatch")
    import io

    with zipfile.ZipFile(io.BytesIO(epub)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(
            i.file_size > MAX_FILE_BYTES for i in archive.infolist()
        ):
            raise ValueError("ambiguous or oversized EPUB member")
        if archive.read("mimetype") != b"application/epub+zip":
            raise ValueError("genuine EPUB packaging required")
        if (
            json.loads(archive.read("OEBPS/provenance.json")) != meta
            or archive.read("OEBPS/article.md") != row["text"].encode()
        ):
            raise ValueError("embedded EPUB provenance/content differs from sidecar")
    refs = meta.get("joplin_references")
    if not isinstance(refs, list) or not refs or quality.get("final") is not True:
        raise ValueError("final hash-bound article quality approval required")
    reviewed = quality.get("reviewed")
    if not isinstance(reviewed, list) or len({r.get("id") for r in reviewed}) != len(reviewed):
        raise ValueError("unambiguous final quality receipt required")
    approved = {r.get("id"): r for r in reviewed}
    originals = {}
    for record_binding in binding["record_bindings"]:
        original = json.loads(file_bytes(record_binding["path"], record_binding["sha256"]))
        ident = original.get("note", {}).get("id")
        if ident in originals:
            raise ValueError("duplicate original capture identity")
        originals[ident] = original
    if len(originals) != len(refs):
        raise ValueError("capture record/reference membership differs")

    def public_url(value):
        if not isinstance(value, str):
            return False
        try:
            parsed = urlsplit(value)
            return (
                parsed.scheme in {"http", "https"}
                and bool(parsed.hostname)
                and not parsed.username
                and not parsed.password
            )
        except ValueError:
            return False

    seen = set()
    for ref in refs:
        if not isinstance(ref, dict):
            raise ValueError("structured Joplin reference required")
        ident = ref.get("note_id")
        capture = ref.get("capture", {})
        proof = approved.get(ident, {})
        if (
            not isinstance(ident, str)
            or re.fullmatch("[0-9a-f]{32}", ident) is None
            or ident in seen
            or ref.get("markdown_link") != ":/" + ident
            or ref.get("note_url") != "joplin://x-callback-url/openNote?id=" + ident
            or capture.get("complete") is not True
            or not digest(capture.get("sha256"))
            or proof.get("accepted") is not True
            or proof.get("capture_sha256") != capture["sha256"]
        ):
            raise ValueError("every Joplin reference needs exact final capture approval")
        original = originals.get(ident, {})
        note = original.get("note", {})
        actual = original.get("capture", {})
        text = actual.get("markdown")
        if (
            original.get("kind", original.get("classification"))
            not in {"strict_url_only", "url_only", "imported_html", "article", "web_clipping"}
            or original.get("outcome")
            not in {
                "captured",
                "staged_complete",
                "updated",
                "verified",
                "existing_clipping_approved",
            }
            or (
                original.get("kind", original.get("classification")) == "web_clipping"
                and original.get("article_approved") is not True
            )
            or not isinstance(text, str)
            or hashlib.sha256(text.encode()).hexdigest() != capture["sha256"]
            or text.replace("\r\n", "\n").replace("\r", "\n").strip("\n") + "\n" != row["text"]
            or any(note.get(k, 0) != 0 for k in ("is_todo", "is_conflict", "deleted_time"))
            or ref.get("title") != note.get("title", "")
            or ref.get("notebook_id") != note.get("parent_id", "")
            or ref.get("folder_path") != original.get("folder_path", "")
            or capture
            != {
                k: actual[k]
                for k in (
                    "sha256",
                    "raw_sha256",
                    "url",
                    "final_url",
                    "archive_url",
                    "archive_timestamp",
                    "method",
                    "captured_at",
                    "complete",
                    "quality",
                    "snapshot",
                    "primary_failure",
                    "linked_article",
                    "discussion_items",
                )
                if k in actual
            }
        ):
            raise ValueError("original capture/note organization differs from sidecar")
        urls = sorted(
            {
                u
                for u in (
                    note.get("source_url"),
                    original.get("url"),
                    actual.get("url"),
                    actual.get("final_url"),
                    actual.get("archive_url"),
                )
                if public_url(u)
            }
        )
        if not urls or ref.get("urls") != urls:
            raise ValueError("original article URLs differ from sidecar")
        tags = sorted(
            {t.get("title") if isinstance(t, dict) else t for t in original.get("tags", [])}
        )
        if ref.get("tags") != tags or any(
            ref.get(k) != note.get(k) for k in ("created_time", "user_created_time") if k in note
        ):
            raise ValueError("original tags/creation dates differ from sidecar")
        seen.add(ident)
    if meta.get("urls") != sorted({u for ref in refs for u in ref["urls"]}):
        raise ValueError("merged article URLs differ from references")
    headings = re.findall(r"^# +(.+)$", row["text"], re.M)
    article_title = next(
        (
            h.strip()
            for h in headings
            if h.strip()
            not in {"Linked article", "Hacker News self-post", "Hacker News discussion"}
        ),
        meta["title"],
    )[:500]
    # Native enrichment owns tags/status/utterance/source/chunk/page fields.
    return {
        k: meta[k]
        for k in ("title", "joplin_references", "urls", "content_generation", "content_sha256")
    } | {
        "source_kind": "joplin_article",
        "source_system": "joplin",
        "article_title": article_title,
        "joplin_binding": {**binding, "schema": 1, "corpus": corpus},
    }


def read_bound_article(source_path, document_metadata):
    """Return verified Markdown only for an explicitly bound Joplin EPUB.

    Ordinary EPUBs use the existing reader. A partial/invalid marker fails closed;
    callers cannot enable preservation merely by passing a pipeline option.
    """
    marked = any(k in document_metadata for k in ("joplin_binding", "source_kind"))
    if not marked or document_metadata.get("source_kind") != "joplin_article":
        if "joplin_binding" in document_metadata:
            raise ValueError("invalid Joplin source marker")
        return None
    binding = document_metadata.get("joplin_binding")
    if (
        not isinstance(binding, dict)
        or binding.get("schema") != 1
        or binding.get("path") != str(Path(source_path).resolve())
        or document_metadata.get("source_system") != "joplin"
    ):
        raise ValueError("exact native Joplin source binding required")
    expected = metadata(binding, binding.get("corpus"))
    if expected != document_metadata:
        raise ValueError("native metadata differs from verified article evidence")
    import io

    with zipfile.ZipFile(
        io.BytesIO(file_bytes(binding["path"], binding["source_sha256"]))
    ) as archive:
        text = archive.read("OEBPS/article.md").decode("utf-8")
    return (
        {
            "type": "page_blocks",
            "source_path": binding["path"],
            "pages": [{"page": 1, "blocks": [{"text": text, "source": {"page": 1}}]}],
            "document_metadata": expected,
        },
        {
            "schema": 1,
            "content_sha256": expected["content_sha256"],
            "source_sha256": binding["source_sha256"],
            "strategy": "contiguous_markdown_v2",
        },
    )
