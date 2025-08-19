import json

from pdf_chunker.adapters import emit_jsonl


def test_write(tmp_path):
    rows = [{"a": 1}, {"b": 2}]
    out = tmp_path / "rows.jsonl"
    emit_jsonl.write(rows, str(out))
    with out.open() as f:
        assert [json.loads(line) for line in f.read().splitlines()] == rows
