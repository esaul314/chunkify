from pdf_chunker.framework import registry


def test_registry_is_mapping():
    reg = registry()
    assert isinstance(reg, dict)


def test_registry_expected_keys_subset():
    # Update this set as passes are registered
    expected = {
        "pdf_parse",
        "text_clean",
        "heading_detect",
        "extraction_fallback",
        "split_semantic",
    }
    assert expected.issubset(set(registry().keys()))
