import os


def use_pymupdf4llm() -> bool:
    """Return True if PyMuPDF4LLM should be enabled based on env var."""
    val = os.getenv("PDF_CHUNKER_USE_PYMUPDF4LLM", "")
    return val.lower() not in ("false", "0", "no", "off")
