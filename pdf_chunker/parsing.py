import os
import sys
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import fitz  # PyMuPDF

from .text_cleaning import _clean_text, _clean_paragraph
from .heading_detection import _detect_heading_fallback
from .page_utils import parse_page_ranges, validate_page_exclusions
from .extraction_fallbacks import (
    _detect_language,
    _assess_text_quality,
    _extract_with_pdftotext,
    _extract_with_pdfminer,
    PDFMINER_AVAILABLE
)

def _extract_text_blocks_from_pdf(filepath: str, exclude_pages: str = None) -> list[dict]:
    """
    Extracts text blocks from a PDF file using PyMuPDF with fallback strategies,
    classifying them as 'heading' or 'paragraph' based on simple heuristics.
    
    Args:
        filepath: Path to the PDF file
        exclude_pages: Page ranges to exclude (e.g., "1,3,5-10,15-20")
    """
    doc = fitz.open(filepath)
    structured_blocks = []
    
    print(f"PDF has {len(doc)} pages", file=sys.stderr)
    
    # Parse and validate page exclusions
    excluded_pages = set()
    if exclude_pages:
        try:
            excluded_pages = parse_page_ranges(exclude_pages)
            excluded_pages = validate_page_exclusions(excluded_pages, len(doc), os.path.basename(filepath))
        except ValueError as e:
            print(f"Error parsing page exclusions: {e}", file=sys.stderr)
            print("Continuing without page exclusions", file=sys.stderr)
            excluded_pages = set()

    # First, try PyMuPDF without TEXT_INHIBIT_SPACES
    all_text = ""

    for page_num, page in enumerate(doc):
        current_page_number = page_num + 1  # Convert to 1-based page numbering

        # Skip excluded pages
        if current_page_number in excluded_pages:
            print(f"Skipping excluded page {current_page_number}", file=sys.stderr)
            continue

        print(f"Processing page {current_page_number}", file=sys.stderr)
        page_blocks = page.get_text("blocks")
    
        for b in page_blocks:
            raw_text = b[4]
            block_text = _clean_text(raw_text)
            all_text += block_text + "\n"
            
            if block_text:
                # To determine if a block is a heading, we need to check its font flags.
                # A simple heuristic is to check if the text is short and bold.
                is_heading = False
                if len(block_text.split()) < 15: # Arbitrary short length for a heading
                    try:
                        block_dict = page.get_text("dict", clip=b[:4])["blocks"]
                        # Defensive checks for block structure
                        if (block_dict and 
                            len(block_dict) > 0 and 
                            isinstance(block_dict[0], dict) and
                            'lines' in block_dict[0] and 
                            block_dict[0]['lines'] and
                            len(block_dict[0]['lines']) > 0 and
                            isinstance(block_dict[0]['lines'][0], dict) and
                            'spans' in block_dict[0]['lines'][0] and
                            block_dict[0]['lines'][0]['spans']):
                            # Check font flags for bold text (flag 2 = bold)
                            is_heading = any(s.get('flags', 0) & 2 for s in block_dict[0]['lines'][0]['spans'])
                    except (KeyError, IndexError, TypeError) as e:
                        print(f"Warning: Unexpected block structure on page {page_num+1}, using fallback heading detection: {e}", file=sys.stderr)
                        # Fallback heading detection heuristics
                        is_heading = _detect_heading_fallback(block_text)
                
                block_type = "heading" if is_heading else "paragraph"
                lang = _detect_language(block_text)
                structured_blocks.append({
                    "type": block_type,
                    "text": block_text,
                    "language": lang,
                    "source": {"filename": os.path.basename(filepath), "page": current_page_number}
                })
    doc.close()
    
    # Assess quality of PyMuPDF extraction
    quality = _assess_text_quality(all_text)
    print(f"PyMuPDF extraction quality: {quality['quality_score']:.2f} (avg_line_length: {quality['avg_line_length']:.1f}, space_density: {quality['space_density']:.3f})", file=sys.stderr)
    
    # If quality is poor, try fallback methods
    if quality['quality_score'] < 0.7:
        print("PyMuPDF quality poor, trying pdftotext fallback...", file=sys.stderr)

        fallback_blocks = _extract_with_pdftotext(filepath, exclude_pages=exclude_pages)
    
        if fallback_blocks:
            return fallback_blocks
        
        if PDFMINER_AVAILABLE:
            print("pdftotext failed, trying pdfminer.six fallback...", file=sys.stderr)

            fallback_blocks = _extract_with_pdfminer(filepath, exclude_pages=exclude_pages)
    
            if fallback_blocks:
                return fallback_blocks
        else:
            print("pdfminer.six not available, skipping fallback", file=sys.stderr)
    
    return structured_blocks

def _get_element_text_content(element) -> str:
    """
    A functional approach to extract text from a BeautifulSoup element,
    correctly handling inline tags without adding extra separators.
    It processes an element's contents and joins them into a single string.
    """
    return ' '.join(
        ' '.join(child.stripped_strings) if hasattr(child, 'stripped_strings') else child
        for child in element.contents
    )

def _extract_text_blocks_from_epub(filepath: str) -> list[dict]:
    """
    Extracts structured text blocks from an EPUB file, using a functional
    approach to gracefully handle inline formatting.
    """
    book = epub.read_epub(filepath)
    
    def process_item(item):
        soup = BeautifulSoup(item.get_content(), 'html.parser')
        body = soup.find('body')
        if not body:
            return []

        # Find all block-level text elements
        elements = body.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        
        # Process each element into a structured block
        blocks = []
        for element in elements:
            raw_text = _get_element_text_content(element)
            block_text = _clean_paragraph(raw_text)
            
            if block_text:
                block_type = "heading" if element.name.startswith('h') else "paragraph"
                lang = _detect_language(block_text)
                blocks.append({
                    "type": block_type,
                    "text": block_text,
                    "language": lang,
                    "source": {"filename": os.path.basename(filepath), "location": item.get_name()}
                })
        return blocks

    # Process all document items and flatten the resulting list of lists
    structured_blocks = [
        block for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT)
        for block in process_item(item)
    ]
    
    return structured_blocks

def extract_structured_text(filepath: str, exclude_pages: str = None) -> list[dict]:
    """
    Extracts a structured representation of text from a file.

    Args:
        filepath: Path to the file to extract text from
        exclude_pages: Page ranges to exclude (e.g., "1,3,5-10,15-20")
    """
    _, extension = os.path.splitext(filepath)
    extension = extension.lower()

    if extension == ".pdf":
        return _extract_text_blocks_from_pdf(filepath, exclude_pages=exclude_pages)
    elif extension == ".epub":
        return _extract_text_blocks_from_epub(filepath)
    else:
        raise ValueError(f"Unsupported file type: '{extension}'.")
