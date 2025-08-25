import json

from pdf_chunker.adapters import emit_jsonl


def test_write(tmp_path):
    rows = [{"a": 1}, {"b": 2}]
    out = tmp_path / "rows.jsonl"
    emit_jsonl.write(rows, str(out))
    with out.open() as f:
        assert [json.loads(line) for line in f.read().splitlines()] == rows


def test_write_creates_parent_directory(tmp_path):
    rows = [{"a": 1}, {"b": 2}]
    out = tmp_path / "nested" / "rows.jsonl"
    emit_jsonl.write(rows, str(out))
    assert out.parent.is_dir()
    with out.open() as f:
        assert [json.loads(line) for line in f.read().splitlines()] == rows
