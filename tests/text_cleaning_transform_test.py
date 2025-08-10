def test_text_cleaning_transform(pdf_case):
    raw, func, expected = pdf_case
    assert func(raw).rstrip() == expected
