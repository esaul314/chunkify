import os
import re
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import fitz  # PyMuPDF

def _clean_paragraph(paragraph: str) -> str:
    """Replaces all whitespace characters (space, tab, newline, etc.) with a single space."""
    return re.sub(r'\s+', ' ', paragraph).strip()

def _clean_text(text: str) -> str:
    """
    Cleans a block of text by preserving paragraph breaks and cleaning each paragraph.
    This function is designed to be pure and declarative.
    """
    if not text or not text.strip():
        return ""
    
    # Split by paragraph, clean each one, filter out empty ones, and rejoin.
    paragraphs = text.split('\n\n')
    cleaned_paragraphs = (_clean_paragraph(p) for p in paragraphs)
    return '\n\n'.join(p for p in cleaned_paragraphs if p)

def _extract_text_blocks_from_pdf(filepath: str) -> list[dict]:
    """
    Extracts text blocks from a PDF file using PyMuPDF, classifying them
    as 'heading' or 'paragraph' based on simple heuristics.
    """
    doc = fitz.open(filepath)
    structured_blocks = []

    for page_num, page in enumerate(doc):
        page_blocks = page.get_text("blocks", flags=fitz.TEXT_INHIBIT_SPACES)
        for b in page_blocks:
            raw_text = b[4]
            block_text = _clean_text(raw_text)
            
            if block_text:
                # To determine if a block is a heading, we need to check its font flags.
                # A simple heuristic is to check if the text is short and bold.
                is_heading = False
                if len(block_text.split()) < 15: # Arbitrary short length for a heading
                    block_dict = page.get_text("dict", clip=b[:4])["blocks"]
                    if (block_dict and block_dict[0]['lines'] and 
                        any(s['flags'] & 2 for s in block_dict[0]['lines'][0]['spans'])):
                        is_heading = True
                
                block_type = "heading" if is_heading else "paragraph"
                structured_blocks.append({
                    "type": block_type,
                    "text": block_text,
                    "source": {"filename": os.path.basename(filepath), "page": page_num + 1}
                })
    return structured_blocks

def _extract_text_blocks_from_epub(filepath: str) -> list[dict]:
    """
    Extracts structured text blocks from an EPUB file, correctly preserving
    paragraph breaks and structure.
    """
    book = epub.read_epub(filepath)
    structured_blocks = []

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), 'html.parser')
        body = soup.find('body')
        if not body:
            continue
        
        for element in body.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            # Preserve paragraph breaks by using a separator
            raw_text = element.get_text(separator='\n\n', strip=True)
            block_text = _clean_text(raw_text)
            
            if block_text:
                block_type = "heading" if element.name.startswith('h') else "paragraph"
                structured_blocks.append({
                    "type": block_type,
                    "text": block_text,
                    "source": {"filename": os.path.basename(filepath), "location": item.get_name()}
                })
    return structured_blocks

def extract_structured_text(filepath: str) -> list[dict]:
    """
    Extracts a structured representation of text from a file.
    """
    _, extension = os.path.splitext(filepath)
    extension = extension.lower()

    if extension == ".pdf":
        return _extract_text_blocks_from_pdf(filepath)
    elif extension == ".epub":
        return _extract_text_blocks_from_epub(filepath)
    else:
        raise ValueError(f"Unsupported file type: '{extension}'.")
