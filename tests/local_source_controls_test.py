"""Optional controls must be explicit and malformed configurations must fail."""

import json

import pytest

from tests.local_source_controls import private_rows, private_value


def test_absent_optional_controls_supply_no_local_sources(monkeypatch):
    monkeypatch.delenv("PDF_CHUNKER_TEST_CONTROLS", raising=False)
    assert private_value("source", "absent") == "absent"
    assert len(private_rows("source_rows", 2)) == 1
    assert private_rows("source_rows", 2)[0].marks[0].name == "skip"


def test_explicit_controls_reject_missing_keys_and_bad_rows(tmp_path, monkeypatch):
    source = tmp_path / "controls.json"
    monkeypatch.setenv("PDF_CHUNKER_TEST_CONTROLS", str(source))
    source.write_text(json.dumps({"version": 1, "values": {"rows": [["example.pdf", "a" * 64]]}}))
    assert private_rows("rows", 2) == [["example.pdf", "a" * 64]]
    with pytest.raises(ValueError, match="Incomplete"):
        private_value("absent")
    source.write_text(json.dumps({"version": 1, "values": {"rows": ["invalid"]}}))
    with pytest.raises(ValueError, match="Invalid"):
        private_rows("rows", 2)
    source.write_text("[]")
    with pytest.raises(ValueError, match="Invalid"):
        private_value("rows")
