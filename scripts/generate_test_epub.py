#!/usr/bin/env python3
"""
Generate Test EPUB Script

Creates a minimal test EPUB file for validation testing.
This script generates a simple EPUB with multiple chapters and spine items
to test EPUB processing functionality.
"""

import os
import sys
import zipfile
from pathlib import Path


def create_test_epub() -> str:
    """Create a test EPUB file manually"""
    # Create test_data directory if it doesn't exist
    test_data_dir = Path(__file__).parent.parent / "test_data"
    test_data_dir.mkdir(exist_ok=True)

    epub_path = test_data_dir / "sample_test.epub"

    # EPUB is essentially a ZIP file with specific structure
    with zipfile.ZipFile(epub_path, "w", zipfile.ZIP_DEFLATED) as epub:
        # mimetype file (must be first and uncompressed)
        epub.writestr(
            "mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED
        )

        # META-INF/container.xml
        container_xml = """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
    <rootfiles>
        <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
    </rootfiles>
</container>"""
        epub.writestr("META-INF/container.xml", container_xml)

        # OEBPS/content.opf (package file)
        content_opf = """<?xml version="1.0" encoding="UTF-8"?>
<package version="3.0" xmlns="http://www.idpf.org/2007/opf" unique-identifier="uid">
    <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
        <dc:identifier id="uid">test-epub-001</dc:identifier>
        <dc:title>Test EPUB Document</dc:title>
        <dc:creator>PDF Chunker Test Suite</dc:creator>
        <dc:language>en</dc:language>
        <meta property="dcterms:modified">2024-01-01T00:00:00Z</meta>
    </metadata>
    <manifest>
        <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
        <item id="chapter1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>
        <item id="chapter2" href="chapter2.xhtml" media-type="application/xhtml+xml"/>
        <item id="chapter3" href="chapter3.xhtml" media-type="application/xhtml+xml"/>
    </manifest>
    <spine>
        <itemref idref="nav"/>
        <itemref idref="chapter1"/>
        <itemref idref="chapter2"/>
        <itemref idref="chapter3"/>
    </spine>
</package>"""
        epub.writestr("OEBPS/content.opf", content_opf)

        # OEBPS/nav.xhtml (navigation document)
        nav_xhtml = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
<head>
    <title>Navigation</title>
</head>
<body>
    <nav epub:type="toc">
        <h1>Table of Contents</h1>
        <ol>
            <li><a href="chapter1.xhtml">Chapter 1: Introduction</a></li>
            <li><a href="chapter2.xhtml">Chapter 2: Sample Content</a></li>
            <li><a href="chapter3.xhtml">Chapter 3: Conclusion</a></li>
        </ol>
    </nav>
</body>
</html>"""
        epub.writestr("OEBPS/nav.xhtml", nav_xhtml)

        # OEBPS/chapter1.xhtml
        chapter1_xhtml = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
    <title>Chapter 1: Introduction</title>
</head>
<body>
    <h1>Chapter 1: Introduction</h1>
    <p>This is a test EPUB document created for validating EPUB processing functionality. It contains multiple chapters with different types of content to test the EPUB parsing pipeline.</p>
    
    <p>The document includes:</p>
    <ul>
        <li>Multiple spine items for testing spine discovery</li>
        <li>Various HTML content structures</li>
        <li>Text extraction capabilities</li>
        <li>Chapter navigation and organization</li>
    </ul>
    
    <p>This content should be processed by the EPUB chunking system to extract meaningful text blocks and validate the extraction pipeline.</p>
</body>
</html>"""
        epub.writestr("OEBPS/chapter1.xhtml", chapter1_xhtml)

        # OEBPS/chapter2.xhtml
        chapter2_xhtml = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
    <title>Chapter 2: Sample Content</title>
</head>
<body>
    <h1>Chapter 2: Sample Content</h1>
    <p>This chapter contains sample text that mimics real EPUB content. It includes paragraphs, dialogue, and various text patterns.</p>
    
    <p>"Hello, this is a sample dialogue," said the first speaker.</p>
    <p>"Yes, this helps test dialogue detection in EPUB format," replied the second.</p>
    
    <p>Regular paragraph text continues here. This text should be processed as a normal paragraph block, separate from the dialogue above.</p>
    
    <h2>Subsection with Lists</h2>
    <p>This section tests structured content processing:</p>
    <ol>
        <li>First numbered item</li>
        <li>Second numbered item</li>
        <li>Third numbered item</li>
    </ol>
    
    <p>Another paragraph with some technical terms like <em>PyMuPDF4LLM</em> and <strong>text processing algorithms</strong> to test specialized handling.</p>
</body>
</html>"""
        epub.writestr("OEBPS/chapter2.xhtml", chapter2_xhtml)

        # OEBPS/chapter3.xhtml
        chapter3_xhtml = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
    <title>Chapter 3: Conclusion</title>
</head>
<body>
    <h1>Chapter 3: Conclusion</h1>
    <p>This final chapter concludes the test EPUB document. It provides additional content for testing the complete EPUB processing pipeline.</p>
    
    <blockquote>
        <p>This is a blockquote to test different HTML element handling in the EPUB processing system.</p>
    </blockquote>
    
    <p>The EPUB format allows for rich HTML content, and this test document exercises various elements to ensure comprehensive text extraction and processing.</p>
    
    <h2>Final Notes</h2>
    <p>This test EPUB should validate:</p>
    <ul>
        <li>Spine discovery and item processing</li>
        <li>HTML content extraction</li>
        <li>Multi-chapter document handling</li>
        <li>Text cleaning and normalization</li>
    </ul>
    
    <p>End of test document.</p>
</body>
</html>"""
        epub.writestr("OEBPS/chapter3.xhtml", chapter3_xhtml)

    print(f"Created test EPUB: {epub_path}")
    return str(epub_path)


if __name__ == "__main__":
    create_test_epub()
