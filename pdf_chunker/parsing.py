import os
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup, NavigableString
import pdfplumber

def _extract_text_from_epub(filepath):
    """
    Extracts structured text from an EPUB file, preserving paragraph breaks.
    """
    book = epub.read_epub(filepath)
    full_text = []

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), 'html.parser')
        body = soup.find('body')
        if not body:
            continue
        
        content_parts = []
        # Iterate through all tags to build the text with structure
        for element in body.find_all(True, recursive=True):
            if element.name in ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']:
                # Get text and add paragraph breaks after these tags
                content_parts.append(element.get_text(strip=True).replace('\xa0', ' '))
                content_parts.append('\n\n')
            elif element.name == 'br':
                content_parts.append('\n')

        # Join all parts, then use the cleaning function to normalize
        # This gives a better base for the final text cleaning.
        chapter_text = "".join(content_parts)
        full_text.append(chapter_text)
        
    return "".join(full_text)

def _extract_text_from_pdf(filepath):
    """Extracts text from a PDF file."""
    with pdfplumber.open(filepath) as pdf:
        return "\n\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

def extract_text(filepath):
    """
    Extracts text from a file, dispatching to the correct parser
    based on the file extension.
    """
    _, extension = os.path.splitext(filepath)
    extension = extension.lower()

    if extension == ".pdf":
        return _extract_text_from_pdf(filepath)
    elif extension == ".epub":
        return _extract_text_from_epub(filepath)
    else:
        raise ValueError(f"Unsupported file type: '{extension}'. Only .pdf and .epub are supported.")
