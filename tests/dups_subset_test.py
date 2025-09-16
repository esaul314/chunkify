from pdf_chunker.diagnostics.dups import find_dups_pageblocks, find_dups_chunks


def test_pageblock_overlap_detected():
    blocks = [
        {"text": "zero one two three four five six", "page": 1},
        {"text": "alpha zero one two three four five", "page": 2},
    ]
    dups = find_dups_pageblocks(blocks)
    pages = {d["first"]["page"] for d in dups} | {d["second"]["page"] for d in dups}
    assert pages == {1, 2}


def test_chunk_overlap_detected():
    chunks = [
        {"text": "a b c d e f g", "metadata": {"chunk_id": 1}},
        {"text": "x a b c d e y", "metadata": {"chunk_id": 2}},
    ]
    dups = find_dups_chunks(chunks)
    ids = {d["first"].get("chunk_id") for d in dups} | {d["second"].get("chunk_id") for d in dups}
    assert ids == {1, 2}
