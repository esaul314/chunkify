from haystack.dataclasses import Document

from pdf_chunker import utils
from pdf_chunker import source_matchers as sm


def test_substring_match():
    block = {"text": "hello world"}
    assert sm.substring_match("hello", block, [])
    assert not sm.substring_match("bye", block, [])


def test_find_source_block_uses_injected_matcher():
    chunk = Document(content="greetings")
    blocks = [{"text": "irrelevant", "source": {"filename": "f", "page": 1}}]

    def always_match(chunk_start, block, blocks):
        return True

    result = utils._find_source_block(chunk, {}, blocks, matchers=[("custom", always_match)])
    assert result == blocks[0]
