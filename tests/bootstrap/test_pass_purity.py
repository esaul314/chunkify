import re
from pathlib import Path

DISALLOWED = re.compile(r"\b(fitz|subprocess|requests|litellm|PyPDF)\b")


def test_pass_modules_have_no_io_imports():
    for p in Path("pdf_chunker/passes").glob("*.py"):
        text = p.read_text(encoding="utf-8")
        assert not DISALLOWED.search(text), f"Disallowed import in {p}"
