from pdf_chunker.text_cleaning import clean_text


def test_numbered_list_preservation():
    text = (
        "1. First numbered item continues to the next line without a period\n"
        "extra words\n\n"
        "2. Second numbered item also wraps onto a new line\n"
        "more words\n\n"
        "3. Third numbered item keeps going on a new line\n"
        "still more"
    )
    cleaned = clean_text(text)
    lines = cleaned.splitlines()
    assert len(lines) == 3
    assert all(lines[i].startswith(f"{i+1}.") for i in range(3))
    assert all(not line.rstrip().endswith(".") for line in lines)
    assert "\n\n" not in cleaned
