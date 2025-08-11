import sys

sys.path.insert(0, ".")

from pdf_chunker.text_cleaning import clean_text


def test_empty_bullet_lines_removed():
    raw = "Intro:\n-\n- Item two\n•\n• Item one"
    cleaned = clean_text(raw)
    lines = [
        line.strip()
        for line in cleaned.splitlines()
        if line.lstrip().startswith(("•", "-"))
    ]
    assert len(lines) == 2
    assert all(line not in {"•", "-"} for line in lines)
