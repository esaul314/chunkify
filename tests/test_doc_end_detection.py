from pdf_chunker.framework import Artifact, run_step


def _page(texts):
    return {"blocks": [{"text": t} for t in texts]}


def test_ignores_table_of_contents_dot_leaders():
    doc = {
        "type": "page_blocks",
        "pages": [
            _page(["Table of Contents"]),
            _page(["Chapter 1 . . . . 5"]),
            _page(["Real text here."]),
        ],
    }
    out = run_step("detect_doc_end", Artifact(doc))
    assert len(out.payload["pages"]) == 3
    assert out.meta["metrics"]["detect_doc_end"]["truncated_pages"] == 0


def test_ignores_toc_entry_named_end():
    doc = {
        "type": "page_blocks",
        "pages": [
            _page(["Table of Contents"]),
            _page(["Intro", ". . . .", "1", "END", ". . . .", "2"]),
            _page(["Real text here."]),
        ],
    }
    out = run_step("detect_doc_end", Artifact(doc))
    assert len(out.payload["pages"]) == 3
    assert out.meta["metrics"]["detect_doc_end"]["truncated_pages"] == 0


def test_truncates_after_explicit_end_marker():
    doc = {
        "type": "page_blocks",
        "pages": [
            _page(["Some text"]),
            _page(["THE END"]),
            _page(["Extra stuff"]),
        ],
    }
    out = run_step("detect_doc_end", Artifact(doc))
    assert len(out.payload["pages"]) == 2
    assert out.meta["metrics"]["detect_doc_end"]["truncated_pages"] == 1
