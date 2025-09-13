from pdf_chunker.diagnostics.dups import find_dups_pageblocks, find_dups_chunks


def test_pageblock_subset_detected():
    blocks = [
        {"text": "alpha beta gamma", "page": 1},
        {"text": "beta gamma", "page": 2},
    ]
    dups = find_dups_pageblocks(blocks)
    assert dups and dups[0]["first"]["page"] == 1 and dups[0]["second"]["page"] == 2


def test_chunk_subset_detected():
    chunks = [
        {"text": "hello world from chunk", "metadata": {"chunk_id": 1}},
        {"text": "world from", "metadata": {"chunk_id": 2}},
    ]
    dups = find_dups_chunks(chunks)
    ids = {d["first"].get("chunk_id") for d in dups} | {d["second"].get("chunk_id") for d in dups}
    assert ids == {1, 2}
