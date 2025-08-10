#!/usr/bin/env python3
"""
Generate Test PDF Script

Creates a minimal test PDF file for validation testing.
This script generates a simple PDF with multiple pages and various text elements
to test PDF processing functionality.
"""

import os
import sys
from pathlib import Path


def create_test_pdf() -> str:
    """Create a test PDF file using reportlab"""
    try:
        # Debug import path and environment
        import sys

        print(f"Python executable: {sys.executable}")
        print(f"Python path: {sys.path[:3]}...")  # Show first 3 entries

        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.units import inch

        print("reportlab imported successfully")
    except ImportError as e:
        print(f"reportlab import failed: {e}")
        print("To install reportlab, run: pip install reportlab")
        print("Creating placeholder PDF info file instead...")
        return create_pdf_placeholder()
    except Exception as e:
        print(f"Unexpected error importing reportlab: {e}")
        print("Creating placeholder PDF info file instead...")
        return create_pdf_placeholder()

    # Create test_data directory if it doesn't exist
    test_data_dir = Path(__file__).parent.parent / "test_data"
    test_data_dir.mkdir(exist_ok=True)

    pdf_path = test_data_dir / "sample_test.pdf"

    # Create PDF
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    width, height = letter

    # Page 1 - Title and introduction
    c.setFont("Helvetica-Bold", 16)
    c.drawString(1 * inch, height - 1 * inch, "Test Document for PDF Processing")

    c.setFont("Helvetica", 12)
    c.drawString(
        1 * inch,
        height - 1.5 * inch,
        "This is a test document created for validating PDF processing functionality.",
    )
    c.drawString(
        1 * inch,
        height - 2 * inch,
        "It contains multiple pages with different types of content to test:",
    )

    c.drawString(1.5 * inch, height - 2.5 * inch, "• Text extraction capabilities")
    c.drawString(1.5 * inch, height - 2.8 * inch, "• Multi-page processing")
    c.drawString(1.5 * inch, height - 3.1 * inch, "• Various text formatting")
    c.drawString(1.5 * inch, height - 3.4 * inch, "• Paragraph and line breaks")

    c.setFont("Helvetica", 10)
    c.drawString(
        1 * inch,
        height - 4 * inch,
        "This document should be processed by the PDF chunking system to extract",
    )
    c.drawString(
        1 * inch,
        height - 4.2 * inch,
        "meaningful text blocks and validate the extraction pipeline.",
    )

    # Page 2 - Sample content with dialogue
    c.showPage()
    c.setFont("Helvetica-Bold", 14)
    c.drawString(1 * inch, height - 1 * inch, "Chapter 1: Sample Content")

    c.setFont("Helvetica", 11)
    c.drawString(
        1 * inch,
        height - 1.5 * inch,
        "This page contains sample text that mimics real document content.",
    )
    c.drawString(
        1 * inch,
        height - 1.8 * inch,
        "It includes paragraphs, dialogue, and various text patterns.",
    )

    c.drawString(
        1 * inch,
        height - 2.3 * inch,
        '"Hello, this is a sample dialogue," said the first speaker.',
    )
    c.drawString(
        1 * inch,
        height - 2.6 * inch,
        '"Yes, this helps test dialogue detection," replied the second.',
    )

    c.drawString(
        1 * inch,
        height - 3.1 * inch,
        "Regular paragraph text continues here. This text should be",
    )
    c.drawString(
        1 * inch,
        height - 3.4 * inch,
        "processed as a normal paragraph block, separate from the",
    )
    c.drawString(1 * inch, height - 3.7 * inch, "dialogue above.")

    c.drawString(
        1 * inch,
        height - 4.2 * inch,
        "Another paragraph with some technical terms like PyMuPDF4LLM",
    )
    c.drawString(
        1 * inch,
        height - 4.5 * inch,
        "and text processing algorithms to test specialized handling.",
    )

    # Page 3 - Lists and structured content
    c.showPage()
    c.setFont("Helvetica-Bold", 14)
    c.drawString(1 * inch, height - 1 * inch, "Chapter 2: Structured Content")

    c.setFont("Helvetica", 11)
    c.drawString(
        1 * inch, height - 1.5 * inch, "This page tests structured content processing:"
    )

    c.drawString(1 * inch, height - 2 * inch, "1. First numbered item")
    c.drawString(1 * inch, height - 2.3 * inch, "2. Second numbered item")
    c.drawString(1 * inch, height - 2.6 * inch, "3. Third numbered item")

    c.drawString(1 * inch, height - 3.1 * inch, "Bullet points:")
    c.drawString(1.2 * inch, height - 3.4 * inch, "• First bullet point")
    c.drawString(1.2 * inch, height - 3.7 * inch, "• Second bullet point")
    c.drawString(1.2 * inch, height - 4 * inch, "• Third bullet point")

    c.drawString(
        1 * inch,
        height - 4.5 * inch,
        "This content tests the system's ability to handle different",
    )
    c.drawString(
        1 * inch, height - 4.8 * inch, "text structures and formatting patterns."
    )

    c.save()
    print(f"Created test PDF: {pdf_path}")
    return str(pdf_path)


def create_pdf_placeholder() -> str:
    """Create a placeholder info file when reportlab is not available"""
    test_data_dir = Path(__file__).parent.parent / "test_data"
    test_data_dir.mkdir(exist_ok=True)

    info_path = test_data_dir / "sample_test_pdf_info.txt"

    with open(info_path, "w") as f:
        f.write("TEST PDF PLACEHOLDER\n")
        f.write("===================\n\n")
        f.write("This file serves as a placeholder for sample_test.pdf\n")
        f.write(
            "The actual PDF could not be created because reportlab is not available.\n\n"
        )
        f.write("To create the test PDF, install reportlab:\n")
        f.write("pip install reportlab\n\n")
        f.write("Then run this script again:\n")
        f.write("python scripts/generate_test_pdf.py\n")

    print(f"Created PDF placeholder info: {info_path}")
    return str(info_path)


if __name__ == "__main__":
    create_test_pdf()
