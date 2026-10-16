"""EPUB IO adapter returning canonical PageBlocks."""

from __future__ import annotations

import logging
import os
import posixpath
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote
from xml.etree import ElementTree
from zipfile import ZipFile


@dataclass(frozen=True)
class _BodyItem:
    """Original HTML bytes without EbookLib's navigation or markup rewriting."""

    name: str
    content: bytes

    def get_name(self) -> str:
        return self.name

    def get_content(self) -> bytes:
        return self.content


def _resource_path(directory: str, href: str) -> str:
    name = posixpath.normpath(posixpath.join(directory, unquote(href.split("#", 1)[0])))
    if name.startswith(("/", "../")) or ":" in name or name in ("", ".", ".."):
        raise ValueError(f"EPUB resource path must remain inside its archive: {href}")
    return name


def _read_body_items(path: str) -> tuple[list[_BodyItem], dict[str, Any]]:
    """Read declared bodies in spine order; absent body documents still fail."""
    with ZipFile(path) as archive:
        container = ElementTree.fromstring(archive.read("META-INF/container.xml"))
        rootfile = container.find(".//{*}rootfile[@media-type='application/oebps-package+xml']")
        if rootfile is None or not rootfile.get("full-path"):
            raise ValueError("EPUB has no declared package document")
        package_path = _resource_path("", rootfile.get("full-path", ""))
        package = ElementTree.fromstring(archive.read(package_path))
        manifest, spine = package.find("{*}manifest"), package.find("{*}spine")
        if manifest is None or spine is None:
            raise ValueError("EPUB package requires a manifest and reading spine")
        entries = list(manifest.findall("{*}item"))
        by_id = {item.get("id"): item for item in entries}
        if len(by_id) != len(entries) or None in by_id or "" in by_id:
            raise ValueError("EPUB manifest item IDs must be present and unique")
        spine_ids = [item.get("idref") for item in spine.findall("{*}itemref")]
        if any(item_id not in by_id for item_id in spine_ids):
            raise ValueError("EPUB reading spine references an absent manifest item")
        directory = posixpath.dirname(package_path)
        navigation = {
            _resource_path(directory, item.get("href", ""))
            for item in package.findall("{*}guide/{*}reference")
            if item.get("type") == "toc" and item.get("href")
        }
        names = set(archive.namelist())
        encrypted = set()
        if "META-INF/encryption.xml" in names:
            encryption = ElementTree.fromstring(archive.read("META-INF/encryption.xml"))
            encrypted = {
                _resource_path("", item.get("URI", ""))
                for item in encryption.findall(".//{*}CipherReference")
                if item.get("URI")
            }
        document_types = {"application/xhtml+xml", "text/html"}
        ordered = list(dict.fromkeys(spine_ids + list(by_id)))
        bodies: list[_BodyItem] = []
        ignored: list[str] = []
        seen: set[str] = set()
        for item_id in ordered:
            item = by_id[item_id]
            href = item.get("href", "")
            name = _resource_path(directory, href)
            is_document = item.get("media-type") in document_types
            is_navigation = "nav" in item.get("properties", "").split() or name in navigation
            if name not in names:
                if is_document and not is_navigation:
                    raise ValueError(f"EPUB body document is missing: {name}")
                ignored.append(name)
                continue
            if not is_document or name in seen:
                continue
            if name in encrypted:
                raise ValueError(f"EPUB body document is encrypted: {name}")
            bodies.append(_BodyItem(posixpath.relpath(name, directory or "."), archive.read(name)))
            seen.add(name)
    if not bodies:
        raise ValueError("EPUB package has no available body documents")
    evidence = {
        "mode": "declared-bodies",
        "documents": [item.name for item in bodies],
        "ignored_missing_resources": ignored,
    }
    logging.getLogger(__name__).warning("EPUB packaging recovery: %s", evidence)
    return bodies, evidence


def _page_key(mapping: dict[str, int], block: dict[str, Any]) -> int:
    """Resolve spine index for grouping.

    Unknown locations are appended after known spine items instead of
    collapsing into page ``0``.
    """
    location = block.get("source", {}).get("location")
    return mapping.get(location, len(mapping) + 1)


def _group_blocks(
    blocks: Iterable[dict[str, Any]], mapping: dict[str, int]
) -> list[dict[str, Any]]:
    """Group blocks by computed spine index."""

    def key(blk: dict[str, Any]) -> int:
        return _page_key(mapping, blk)

    sorted_blocks = sorted(blocks, key=key)
    groups = groupby(sorted_blocks, key)
    pages = [{"page": page, "blocks": list(group)} for page, group in groups]
    return pages


def _excluded_spines(spec: str | None, total: int, filename: str) -> set[int]:
    """Parse and validate spine exclusion specification."""

    if not spec:
        return set()
    try:
        from pdf_chunker.page_utils import parse_page_ranges, validate_page_exclusions

        return validate_page_exclusions(parse_page_ranges(spec), total, filename)
    except ValueError:
        return set()


def read_epub(path: str, spine: str | None = None) -> dict[str, Any]:
    """Load an EPUB file and emit grouped PageBlocks."""

    import ebooklib
    from ebooklib import epub

    from pdf_chunker.epub_parsing import (
        EPUB_DOCUMENT_CHAR_LIMIT,
        EPUB_DOCUMENT_MIN_TARGET,
        _coalesce_body_blocks,
        process_epub_item,
    )

    abs_path = str(Path(path).resolve())
    filename = Path(abs_path).name
    recovery_option = os.getenv("PDF_CHUNKER_EPUB_RECOVERY", "0")
    if recovery_option not in ("0", "1"):
        raise ValueError("PDF_CHUNKER_EPUB_RECOVERY must be 0 or 1")
    recovery = None
    if recovery_option == "1":
        spine_items, recovery = _read_body_items(abs_path)
    else:
        book = epub.read_epub(abs_path)
        spine_items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
    mapping = {item.get_name(): idx for idx, item in enumerate(spine_items, 1)}
    excluded = _excluded_spines(spine, len(spine_items), filename)

    blocks = [
        block
        for idx, item in enumerate(spine_items, 1)
        if idx not in excluded
        for block in process_epub_item(item, filename, idx)
    ]
    grouped_blocks = _coalesce_body_blocks(
        blocks,
        max_chars=EPUB_DOCUMENT_CHAR_LIMIT,
        min_anchor=EPUB_DOCUMENT_MIN_TARGET,
    )
    if recovery is not None and not grouped_blocks:
        raise ValueError("EPUB recovery yielded no readable body text")

    result: dict[str, Any] = {
        "type": "page_blocks",
        "source_path": abs_path,
        "pages": _group_blocks(cast(Iterable[dict[str, Any]], grouped_blocks), mapping),
    }
    if recovery is not None:
        result["extraction_quality"] = {"epub_recovery": recovery}
    return result


def read(path: str, exclude_pages: str | None = None) -> dict[str, Any]:
    """Compatibility wrapper exposing a standard ``read`` entrypoint."""

    return read_epub(path, spine=exclude_pages)


def describe_epub(path: str) -> dict[str, str]:
    """Return a lightweight descriptor for an EPUB file."""

    return {"type": "epub_document", "source_path": str(Path(path).resolve())}
