import os
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import fitz  # PyMuPDF

def _extract_text_blocks_from_pdf(filepath: str) -> list[dict]:
    """
    Extracts text blocks from a PDF file using PyMuPDF, classifying them
    as 'heading' or 'paragraph' based on simple heuristics.
    """
    doc = fitz.open(filepath)
    structured_blocks = []
    
    for page_num, page in enumerate(doc):
        # Extract blocks with detailed information
        blocks = page.get_text("dict", flags=fitz.TEXT_INHIBIT_SPACES)["blocks"]
        
        for block in blocks:
            if block["type"] == 0:  # It's a text block
                for line in block["lines"]:
                    # Basic heuristic: treat bold lines as potential headings
                    # This can be refined with font size checks, all-caps checks, etc.
                    is_heading = any(span["flags"] & 2 for span in line["spans"]) # Check for bold flag
                    
                    block_text = "".join(span["text"] for span in line["spans"]).strip()
                    
                    if block_text:
                        block_type = "heading" if is_heading and len(block_text.split()) < 15 else "paragraph"
                        structured_blocks.append({
                            "type": block_type,
                            "text": block_text,
                            "source": {"filename": os.path.basename(filepath), "page": page_num + 1}
                        })
    return structured_blocks

def _extract_text_blocks_from_epub(filepath: str) -> list[dict]:
    """
    Extracts structured text blocks from an EPUB file, classifying them
    by their HTML tags.
    """
    book = epub.read_epub(filepath)
    structured_blocks = []
    
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), 'html.parser')
        
        # Find all tags that represent distinct blocks of text
        for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
            block_text = element.get_text(strip=True)
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
    Extracts a structured representation of text from a file, dispatching 
    to the correct parser based on the file extension.
    
    Returns a list of dictionaries, where each dictionary represents a
    text block with its type and content.
    """
    _, extension = os.path.splitext(filepath)
    extension = extension.lower()

    if extension == ".pdf":
        return _extract_text_blocks_from_pdf(filepath)
    elif extension == ".epub":
        return _extract_text_blocks_from_epub(filepath)
    else:
        raise ValueError(f"Unsupported file type: '{extension}'. Only .pdf and .epub are supported.")
