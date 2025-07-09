import os
import re
import sys
import subprocess
from langdetect import detect, LangDetectException

try:
    from pdfminer.high_level import extract_text
    from pdfminer.layout import LAParams
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False

from .text_cleaning import _clean_text
from .heading_detection import _detect_heading_fallback

def _detect_language(text: str) -> str:
    """Detects language of a text block, defaults to 'un' (unknown) on failure."""
    try:
        return detect(text)
    except LangDetectException:
        return "un"

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
