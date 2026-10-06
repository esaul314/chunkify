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


def test_nested_list_item_does_not_duplicate_descendant_text() -> None:
    from bs4 import BeautifulSoup

    from pdf_chunker.epub_parsing import _element_to_block

    soup = BeautifulSoup(
        "<ol><li><a>Chapter 1</a><ol><li><a>Section 1.1</a></li></ol></li></ol>",
        "html.parser",
    )
    outer, inner = soup.find_all("li")

    outer_block = _element_to_block(outer, "book.epub", "nav.xhtml", 1)
    inner_block = _element_to_block(inner, "book.epub", "nav.xhtml", 1)

    assert outer_block is not None
    assert inner_block is not None
    assert outer_block["text"] == "1. Chapter 1"
    assert inner_block["text"] == "1. Section 1.1"


def test_navigation_provenance_is_replaced_when_coalesced() -> None:
    from pdf_chunker.epub_parsing import _coalesce_body_blocks

    blocks = [
        {
            "type": "list_item",
            "text": "1. Chapter 1",
            "language": "en",
            "source": {"filename": "book.epub", "location": "nav.xhtml", "page": 1},
        },
        {
            "type": "paragraph",
            "text": "Chapter body",
            "language": "en",
            "source": {"filename": "book.epub", "location": "chapter1.xhtml", "page": 2},
        },
    ]

    merged = _coalesce_body_blocks(blocks, max_chars=1000, min_anchor=0)

    assert len(merged) == 1
    assert merged[0]["text"] == "1. Chapter 1\n\nChapter body"
    assert merged[0]["source"] == {
        "filename": "book.epub",
        "location": "chapter1.xhtml",
        "page": 1,
    }
