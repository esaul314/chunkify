"""Tests for the EPUB IO adapter."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest


def _create_minimal_epub(path: Path) -> Path:
    container = (
        "<?xml version='1.0' encoding='UTF-8'?>"
        "<container version='1.0' xmlns='urn:oasis:names:tc:opendocument:xmlns:container'>"
        "<rootfiles><rootfile full-path='OEBPS/content.opf'"
        " media-type='application/oebps-package+xml'/></rootfiles></container>"
    )
    content_opf = (
        "<?xml version='1.0' encoding='UTF-8'?>"
        "<package version='3.0' xmlns='http://www.idpf.org/2007/opf'"
        " unique-identifier='uid'>"
        "<metadata xmlns:dc='http://purl.org/dc/elements/1.1/'>"
        "<dc:identifier id='uid'>1</dc:identifier></metadata>"
        "<manifest><item id='chap' href='chapter.xhtml'"
        " media-type='application/xhtml+xml'/></manifest>"
        "<spine><itemref idref='chap'/></spine></package>"
    )
    chapter = (
        "<?xml version='1.0' encoding='UTF-8'?>"
        "<html xmlns='http://www.w3.org/1999/xhtml'><body><p>Hello world</p>"
        "</body></html>"
    )
    with zipfile.ZipFile(path, "w") as epub:
        epub.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        epub.writestr("META-INF/container.xml", container)
        epub.writestr("OEBPS/content.opf", content_opf)
        epub.writestr("OEBPS/chapter.xhtml", chapter)
    return path


pytest.importorskip("ebooklib")


def test_read_epub_adapter(tmp_path: Path) -> None:
    from pdf_chunker.adapters.io_epub import read_epub

    epub_path = _create_minimal_epub(tmp_path / "sample.epub")
    doc = read_epub(str(epub_path))
    assert doc["type"] == "page_blocks"
    pages = doc["pages"]
    assert len(pages) == 1
    assert pages[0]["blocks"][0]["text"] == "Hello world"
