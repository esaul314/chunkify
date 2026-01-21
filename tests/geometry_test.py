"""Tests for geometry module zone detection."""

import pytest

pytest.importorskip("fitz")

from pdf_chunker.geometry import (
    DocumentZones,
    detect_document_zones,
    detect_footer_zone,
)


def test_detect_footer_zone_real_pdf():
    """Detect footer zone in a real PDF with consistent footers."""
    import fitz

    # Use sample PDF if available
    try:
        doc = fitz.open("97things-manager-short.pdf")
    except FileNotFoundError:
        pytest.skip("97things-manager-short.pdf not found")

    margin, confidence = detect_footer_zone(doc)
    doc.close()

    # Footer should be detected with high confidence
    assert margin is not None
    assert margin > 30  # At least 30 points
    assert confidence >= 0.8  # High confidence


def test_detect_document_zones():
    """Detect both header and footer zones."""
    import fitz

    try:
        doc = fitz.open("97things-manager-short.pdf")
    except FileNotFoundError:
        pytest.skip("97things-manager-short.pdf not found")

    zones = detect_document_zones(doc)
    doc.close()

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
