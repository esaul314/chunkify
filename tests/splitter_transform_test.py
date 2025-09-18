from pdf_chunker.splitter import _split_text_into_chunks
from pdf_chunker.passes.split_semantic import (
    _SplitSemanticPass,
    _chunk_items,
    _get_split_fn,
)


def test_splitter_respects_cleaning(pdf_case):
    raw, func, expected = pdf_case
    chunks = _split_text_into_chunks(func(raw).rstrip(), chunk_size=100, overlap=0)
    assert chunks == [expected]


def test_splitter_size_and_overlap():
    text = " ".join(f"w{i}" for i in range(20))
    chunks = _split_text_into_chunks(text, chunk_size=10, overlap=2)
    assert [len(c.split()) for c in chunks] == [10, 10]
    assert chunks[1].split()[0] == "w8"


def test_split_semantic_merges_bullet_heading_continuation():
    doc = {
        "type": "page_blocks",
        "source_path": "platform-eng-excerpt.pdf",
        "pages": [
            {
                "page": 1,
                "blocks": [
                    {
                        "text": "• With platform as a service (PaaS), the vendor takes full ownership of operating",
                        "type": "heading",
                    },
                    {
                        "text": "The application's infrastructure, which means rather than offering primitives, they offer higher-level abstractions so that the application runs in a scalable sandbox.",
                        "type": "paragraph",
                    },
                ],
            }
        ],
    }

    split_pass = _SplitSemanticPass()
    split_fn, _ = _get_split_fn(
        split_pass.chunk_size, split_pass.overlap, split_pass.min_chunk_size
    )
    items = list(_chunk_items(doc, split_fn, split_pass.generate_metadata))
    text = items[0]["text"]

    assert "operating the application's infrastructure" in text
    assert "operating\nThe" not in text
