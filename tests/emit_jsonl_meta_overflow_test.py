from pdf_chunker.passes.emit_jsonl import _rows_from_item


def test_rows_from_item_handles_oversize_meta() -> None:
    item = {"text": "ok", "meta": {"huge": "x" * 9000}}
    assert _rows_from_item(item) == []
