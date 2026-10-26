"""Tests for geometry module zone detection."""

import pytest

pytest.importorskip("fitz")

from pdf_chunker.geometry import (
    DocumentZones,
    detect_document_zones,
    detect_footer_zone,
)


@pytest.fixture
def document_with_repeated_footer():
    import fitz

    with fitz.open() as document:
        for index in range(5):
            page = document.new_page(width=612, height=792)
            page.insert_text((72, 40), "SYNTHETIC REVIEW HEADER", fontsize=9)
            page.insert_text((72, 160), f"Invented body paragraph {index}.", fontsize=12)
            page.insert_text((72, 742), "SYNTHETIC REVIEW FOOTER", fontsize=9)
        yield document


def test_detect_footer_zone_generated_pdf(document_with_repeated_footer):
    margin, confidence = detect_footer_zone(document_with_repeated_footer)
    assert margin is not None
    assert margin > 30
    assert confidence >= 0.8


def test_detect_document_zones(document_with_repeated_footer):
    zones = detect_document_zones(document_with_repeated_footer)
    assert isinstance(zones, DocumentZones)
    assert zones.footer_margin is not None or zones.header_margin is not None
    assert zones.confidence >= 0


def test_document_zones_dataclass():
    """DocumentZones dataclass holds margin values."""
    zones = DocumentZones(
        footer_margin=40.0,
        header_margin=50.0,
        confidence=0.95,
    )
    assert zones.footer_margin == 40.0
    assert zones.header_margin == 50.0
    assert zones.confidence == 0.95


def test_detect_footer_zone_no_footers():
    """Returns None margin when no consistent footer zone detected."""
    import fitz

    # Create a minimal PDF with no footers
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((100, 100), "Content in middle of page")

    margin, confidence = detect_footer_zone(doc)
    doc.close()

    # Should not detect a footer zone
    assert margin is None or confidence < 0.5
