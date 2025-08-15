from pdf_chunker.framework import registry


def test_registry_is_mapping():
    reg = registry()
    assert isinstance(reg, dict)


def test_registry_expected_keys_subset():
    # Update this set as passes are registered (e.g., {"pdf_parse", "text_clean", ...})
    expected = set()
    assert expected.issubset(set(registry().keys()))
