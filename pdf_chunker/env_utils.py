import os


def use_pymupdf4llm() -> bool:
    """Return True if PyMuPDF4LLM should be enabled based on env var."""
    val = os.getenv("PDF_CHUNKER_USE_PYMUPDF4LLM")
    if val is None:
        return True
    return val.lower() in {"true", "1", "yes", "on"}
