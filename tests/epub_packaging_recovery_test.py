"""Optional packaging recovery keeps all available declared book bodies."""

import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import pytest

from pdf_chunker.adapters.io_epub import read_epub
from tests.local_source_controls import private_rows, private_value


def make_book(path, items, spine_ids, *, toc="", guide_toc=None):
    package = ElementTree.Element(
        "package", {"xmlns": "http://www.idpf.org/2007/opf", "version": "3.0"}
    )
    ElementTree.SubElement(package, "metadata")
    manifest = ElementTree.SubElement(package, "manifest")
    for item_id, href, media_type, _content, properties in items:
        ElementTree.SubElement(
            manifest,
            "item",
            {"id": item_id, "href": href, "media-type": media_type, "properties": properties},
        )
    spine = ElementTree.SubElement(package, "spine", {"toc": toc})
    for item_id in spine_ids:
        ElementTree.SubElement(spine, "itemref", {"idref": item_id})
    if guide_toc:
        guide = ElementTree.SubElement(package, "guide")
        ElementTree.SubElement(guide, "reference", {"type": "toc", "href": guide_toc})
    with ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr(
            "META-INF/container.xml",
            '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container'
            '"><rootfiles><rootfile full-path="OEBPS/content.opf" media-type="'
            'application/oebps-package+xml"/></rootfiles></container>',
        )
        archive.writestr("OEBPS/content.opf", ElementTree.tostring(package))
        for _, href, _, content, _ in items:
            if content is not None:
                archive.writestr("OEBPS/" + href, content)
    return path


def chapter(item_id, text):
    return (
        item_id,
        item_id + ".xhtml",
        "application/xhtml+xml",
        "<html><body><p>" + text + "</p></body></html>",
        "",
    )


def extracted(doc):
    return "\n".join(block["text"] for page in doc["pages"] for block in page["blocks"])


def test_recovery_is_opt_in_and_missing_nontext_assets_do_not_lose_body(tmp_path, monkeypatch):
    source = make_book(
        tmp_path / "book.epub",
        [chapter("body", "Complete book body."), ("style", "missing.css", "text/css", None, "")],
        ["body"],
    )
    before = source.read_bytes()
    monkeypatch.delenv("PDF_CHUNKER_EPUB_RECOVERY", raising=False)
    with pytest.raises(KeyError):
        read_epub(str(source))
    monkeypatch.setenv("PDF_CHUNKER_EPUB_RECOVERY", "1")
    doc = read_epub(str(source))
    assert extracted(doc) == "Complete book body."
    assert doc["extraction_quality"]["epub_recovery"]["ignored_missing_resources"] == [
        "OEBPS/missing.css"
    ]
    assert source.read_bytes() == before
    assert doc["pages"][0]["blocks"][0]["source"]["filename"] == source.name


def test_broken_navigation_metadata_does_not_block_reading_order(tmp_path, monkeypatch):
    source = make_book(
        tmp_path / "book.epub",
        [
            chapter("last", "Last body."),
            chapter("first", "First body."),
            ("ncx", "broken.ncx", "application/x-dtbncx+xml", "<ncx/>", ""),
        ],
        ["first", "last"],
        toc="ncx",
    )
    monkeypatch.setenv("PDF_CHUNKER_EPUB_RECOVERY", "1")
    doc = read_epub(str(source))
    assert extracted(doc).index("First body.") < extracted(doc).index("Last body.")
    assert doc["extraction_quality"]["epub_recovery"]["documents"] == ["first.xhtml", "last.xhtml"]


def test_absent_toc_metadata_and_declared_navigation_can_be_recovered(tmp_path, monkeypatch):
    source = make_book(
        tmp_path / "book.epub",
        [
            chapter("body", "Complete body."),
            ("toc", "toc.xhtml", "application/xhtml+xml", None, "nav"),
        ],
        ["body"],
        toc="unknown-ncx",
    )
    monkeypatch.setenv("PDF_CHUNKER_EPUB_RECOVERY", "1")
    assert extracted(read_epub(str(source))) == "Complete body."


def test_missing_body_or_unknown_spine_item_remains_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("PDF_CHUNKER_EPUB_RECOVERY", "1")
    missing = make_book(
        tmp_path / "missing.epub",
        [("body", "missing.xhtml", "application/xhtml+xml", None, "")],
        ["body"],
    )
    with pytest.raises(ValueError, match="body document is missing"):
        read_epub(str(missing))
    unknown = make_book(tmp_path / "unknown.epub", [chapter("body", "Body.")], ["missing"])
    with pytest.raises(ValueError, match="spine references an absent"):
        read_epub(str(unknown))


def test_spine_exclusions_and_invalid_option_are_preserved(tmp_path, monkeypatch):
    source = make_book(
        tmp_path / "book.epub",
        [chapter("one", "First."), chapter("two", "Second.")],
        ["one", "two"],
    )
    monkeypatch.setenv("PDF_CHUNKER_EPUB_RECOVERY", "1")
    assert extracted(read_epub(str(source), spine="1")) == "Second."
    monkeypatch.setenv("PDF_CHUNKER_EPUB_RECOVERY", "unknown")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        read_epub(str(source))


@pytest.mark.parametrize("path,raw_sha256", private_rows("epub_packaging", 2))
def test_optional_affected_book_bodies(path, raw_sha256, monkeypatch):
    source = Path(path)
    if not source.is_file():
        pytest.skip("optional local corpus fixture unavailable")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == raw_sha256
    monkeypatch.setenv("PDF_CHUNKER_EPUB_RECOVERY", "1")
    doc = read_epub(path)
    assert len(extracted(doc)) > 5000, "navigation alone is not a recovered book"
    assert all(
        block["source"]["filename"] == source.name
        for page in doc["pages"]
        for block in page["blocks"]
    )
    assert hashlib.sha256(source.read_bytes()).hexdigest() == raw_sha256
    print(
        json.dumps(
            {
                "source_file": source.name,
                "raw_sha256": raw_sha256,
                "extracted_chars": len(extracted(doc)),
                "cpu_only": True,
                "recovery": doc["extraction_quality"],
            }
        )
    )


def test_encrypted_body_remains_an_explicit_error(tmp_path, monkeypatch):
    source = make_book(tmp_path / "encrypted.epub", [chapter("body", "Secret body.")], ["body"])
    with ZipFile(source, "a") as archive:
        archive.writestr(
            "META-INF/encryption.xml",
            '<encryption xmlns="urn:oasis:names:tc:opendocument:xmlns:containe'
            'r"><EncryptedData xmlns="http://www.w3.org/2001/04/xmlenc#"><Ciph'
            'erData><CipherReference URI="OEBPS/body.xhtml"/></CipherData></En'
            "cryptedData></encryption>",
        )
    monkeypatch.setenv("PDF_CHUNKER_EPUB_RECOVERY", "1")
    with pytest.raises(ValueError, match="body document is encrypted"):
        read_epub(str(source))


def test_optional_encrypted_book_stays_unaccepted(monkeypatch):
    path = Path(private_value("control_005", ""))
    if not path.is_file():
        pytest.skip("optional local corpus fixture unavailable")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == private_value("control_006", "")
    monkeypatch.setenv("PDF_CHUNKER_EPUB_RECOVERY", "1")
    with pytest.raises(ValueError, match="body document is encrypted"):
        read_epub(str(path))
