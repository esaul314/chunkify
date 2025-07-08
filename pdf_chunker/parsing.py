import os
import re
import sys
import subprocess
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
from langdetect import detect, LangDetectException

try:
    from pdfminer.high_level import extract_text
    from pdfminer.layout import LAParams
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False

def _detect_language(text: str) -> str:
    """Detects language of a text block, defaults to 'un' (unknown) on failure."""
    try:
        return detect(text)
    except LangDetectException:
        return "un"

def _clean_paragraph(paragraph: str) -> str:
    """
    Replaces all whitespace characters with a single space and removes the BOM character.
    """
    # Remove the BOM character (U+FEFF) which can appear in source files
    cleaned_text = paragraph.replace('\ufeff', '').replace('\u200b', '')
    # Consolidate all other whitespace into single spaces
    return re.sub(r'\s+', ' ', cleaned_text).strip()

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

def _assess_text_quality(text: str) -> dict:
    """
    Assess the quality of extracted text to determine if fallback methods are needed.
    Returns a dict with quality metrics.
    """
    if not text or not text.strip():
        return {"avg_line_length": 0, "space_density": 0, "quality_score": 0}
    
    lines = text.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    
    if not non_empty_lines:
        return {"avg_line_length": 0, "space_density": 0, "quality_score": 0}
    
    # Calculate average line length
    avg_line_length = sum(len(line) for line in non_empty_lines) / len(non_empty_lines)
    
    # Calculate space density (spaces per character)
    total_chars = sum(len(line) for line in non_empty_lines)
    total_spaces = sum(line.count(' ') for line in non_empty_lines)
    space_density = total_spaces / total_chars if total_chars > 0 else 0
    
    # Quality score: penalize very long lines and very low space density
    quality_score = 1.0
    if avg_line_length > 1000:  # Very long lines indicate poor extraction
        quality_score *= 0.3
    if space_density < 0.05:  # Very low space density indicates missing spaces
        quality_score *= 0.2
    
    return {
        "avg_line_length": avg_line_length,
        "space_density": space_density,
        "quality_score": quality_score
    }

def _detect_heading_fallback(text: str) -> bool:
    """
    Fallback heading detection using text characteristics when font analysis fails.
    Uses heuristics like text length, capitalization patterns, and punctuation.
    """
    if not text or not text.strip():
        return False
    
    text = text.strip()
    words = text.split()
    
    # Very short text (1-3 words) is likely a heading
    if len(words) <= 3:
        return True
    
    # Text that's all uppercase might be a heading
    if text.isupper() and len(words) <= 8:
        return True
    
    # Title case text without ending punctuation might be a heading
    if (text.istitle() and 
        len(words) <= 10 and 
        not text.endswith(('.', '!', '?', ';', ':'))):
        return True
    
    # Text that starts with common heading patterns
    heading_starters = ['chapter', 'section', 'part', 'appendix', 'introduction', 'conclusion']
    first_word = words[0].lower() if words else ''
    if first_word in heading_starters and len(words) <= 8:
        return True
    
    # Text that's mostly numbers (like "1.2.3 Some Topic")
    if len(words) >= 2:
        first_part = words[0]
        if re.match(r'^[\d\.\-]+$', first_part) and len(words) <= 8:
            return True
    
    return False

def _extract_text_blocks_from_pdf(filepath: str) -> list[dict]:
    """
    Extracts text blocks from a PDF file using PyMuPDF with fallback strategies,
    classifying them as 'heading' or 'paragraph' based on simple heuristics.
    """
    doc = fitz.open(filepath)
    structured_blocks = []
    
    print(f"PDF has {len(doc)} pages", file=sys.stderr)

    # First, try PyMuPDF without TEXT_INHIBIT_SPACES
    all_text = ""
    for page_num, page in enumerate(doc):
        print(f"Processing page {page_num+1}", file=sys.stderr)
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
                    "source": {"filename": os.path.basename(filepath), "page": page_num + 1}
                })
    doc.close()
    
    # Assess quality of PyMuPDF extraction
    quality = _assess_text_quality(all_text)
    print(f"PyMuPDF extraction quality: {quality['quality_score']:.2f} (avg_line_length: {quality['avg_line_length']:.1f}, space_density: {quality['space_density']:.3f})", file=sys.stderr)
    
    # If quality is poor, try fallback methods
    if quality['quality_score'] < 0.7:
        print("PyMuPDF quality poor, trying pdftotext fallback...", file=sys.stderr)
        fallback_blocks = _extract_with_pdftotext(filepath)
        if fallback_blocks:
            return fallback_blocks
        
        if PDFMINER_AVAILABLE:
            print("pdftotext failed, trying pdfminer.six fallback...", file=sys.stderr)
            fallback_blocks = _extract_with_pdfminer(filepath)
            if fallback_blocks:
                return fallback_blocks
        else:
            print("pdfminer.six not available, skipping fallback", file=sys.stderr)
    
    return structured_blocks

def _extract_with_pdftotext(filepath: str) -> list[dict]:
    """
    Fallback extraction using pdftotext with layout preservation.
    """
    try:
        # Try pdftotext with -layout flag
        result = subprocess.run(
            ['pdftotext', '-layout', filepath, '-'],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            print(f"pdftotext failed with return code {result.returncode}", file=sys.stderr)
            return []
        
        raw_text = result.stdout
        quality = _assess_text_quality(raw_text)
        print(f"pdftotext extraction quality: {quality['quality_score']:.2f}", file=sys.stderr)
        
        if quality['quality_score'] < 0.7:
            return []
        
        # Parse the text into structured blocks
        structured_blocks = []
        paragraphs = raw_text.split('\n\n')
        
        for paragraph in paragraphs:
            block_text = _clean_text(paragraph)
            if block_text:
                # Simple heuristic: short paragraphs with title case might be headings
                is_heading = (len(block_text.split()) < 15 and 
                             block_text.istitle() and 
                             not block_text.endswith('.'))
                
                block_type = "heading" if is_heading else "paragraph"
                lang = _detect_language(block_text)
                structured_blocks.append({
                    "type": block_type,
                    "text": block_text,
                    "language": lang,
                    "source": {"filename": os.path.basename(filepath), "method": "pdftotext"}
                })
        
        return structured_blocks
        
    except subprocess.TimeoutExpired:
        print("pdftotext timed out", file=sys.stderr)
        return []
    except FileNotFoundError:
        print("pdftotext not found - install poppler-utils", file=sys.stderr)
        return []
    except Exception as e:
        print(f"pdftotext extraction failed: {e}", file=sys.stderr)
        return []

def _extract_with_pdfminer(filepath: str) -> list[dict]:
    """
    Final fallback extraction using pdfminer.six with tunable LAParams.
    """
    try:
        # Try different LAParams configurations
        configs = [
            LAParams(char_margin=1.5, word_margin=0.5, line_margin=0.5),
            LAParams(char_margin=2.0, word_margin=0.3, line_margin=0.3),
            LAParams(char_margin=1.0, word_margin=0.8, line_margin=0.8)
        ]
        
        for i, laparams in enumerate(configs):
            print(f"Trying pdfminer config {i+1}/3", file=sys.stderr)
            raw_text = extract_text(filepath, laparams=laparams)
            
            # Apply post-processing to fix missing spaces
            repaired_text = re.sub(r"([a-z])([A-Z])", r"\1 \2", raw_text)
            
            quality = _assess_text_quality(repaired_text)
            print(f"pdfminer config {i+1} quality: {quality['quality_score']:.2f}", file=sys.stderr)
            
            if quality['quality_score'] >= 0.7:
                # Parse the text into structured blocks
                structured_blocks = []
                paragraphs = repaired_text.split('\n\n')
                
                for paragraph in paragraphs:
                    block_text = _clean_text(paragraph)
                    if block_text:
                        # Simple heuristic for headings
                        is_heading = (len(block_text.split()) < 15 and 
                                     block_text.istitle() and 
                                     not block_text.endswith('.'))
                        
                        block_type = "heading" if is_heading else "paragraph"
                        lang = _detect_language(block_text)
                        structured_blocks.append({
                            "type": block_type,
                            "text": block_text,
                            "language": lang,
                            "source": {"filename": os.path.basename(filepath), "method": "pdfminer"}
                        })
                
                return structured_blocks
        
        print("All pdfminer configurations failed quality check", file=sys.stderr)
        return []
        
    except Exception as e:
        print(f"pdfminer extraction failed: {e}", file=sys.stderr)
        return []

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
