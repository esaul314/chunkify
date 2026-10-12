"""Body extraction must not depend on an EPUB choosing paragraph tags."""
from tests.local_source_controls import private_rows
import pytest
from bs4 import BeautifulSoup
from ebooklib.epub import EpubBook, EpubHtml
from pdf_chunker.epub_parsing import process_epub_item

def body_text(markup: str) -> str:
    item = EpubHtml(uid='chapter', file_name='chapter.xhtml')
    EpubBook().add_item(item)
    item.set_content('<html><body>' + markup + '</body></html>')
    blocks = process_epub_item(item, 'book.epub', 3)
    assert all((block['source']['location'] == 'chapter.xhtml' for block in blocks))
    assert all((block['source']['page'] == 3 for block in blocks))
    return '\n'.join((block['text'] for block in blocks))

def test_div_span_body_is_extracted_in_order() -> None:
    text = body_text('<div><span>First sentence.</span></div><div>Second sentence.</div>')
    assert text.count('First sentence.') == text.count('Second sentence.') == 1
    assert text.index('First sentence.') < text.index('Second sentence.')

def test_mixed_semantic_tags_and_wrapper_text_are_not_duplicated() -> None:
    markup = '<div>Before.<h1>Chapter</h1><p>Middle sentence.</p><div>After.</div>End.</div>'
    text = body_text(markup)
    expected = ['Before.', 'Chapter', 'Middle sentence.', 'After.', 'End.']
    assert all((text.count(part) == 1 for part in expected))
    assert [text.index(part) for part in expected] == sorted((text.index(part) for part in expected))

def test_nested_wrappers_and_list_items_have_single_coverage() -> None:
    text = body_text('<div><section><p>Paragraph.</p><ul><li><p>Outer item.</p><ul><li>Inner item.</li></ul></li></ul></section></div>')
    assert all((text.count(part) == 1 for part in ['Paragraph.', 'Outer item.', 'Inner item.']))

def test_inline_fragments_and_br_preserve_word_boundaries() -> None:
    text = body_text('<div>A re<span>stored</span> book.<br/>Another <em>sentence</em>.</div>')
    assert 'restored' in text
    assert 'book.Another' not in text
    assert 'Another sentence.' in text

def test_nonreading_tags_are_excluded_from_fallback_text() -> None:
    text = body_text('<div>Readable.<script>script junk</script><style>style junk</style><nav><div>navigation junk</div></nav><div>Final.</div></div>')
    assert 'Readable.' in text and 'Final.' in text
    assert 'junk' not in text

def test_content_walk_does_not_mutate_dom() -> None:
    from pdf_chunker.epub_parsing import _body_elements
    soup = BeautifulSoup('<body><div>Text <span>continued</span>.</div><p>End.</p></body>', 'html.parser')
    original = str(soup)
    assert list(_body_elements(soup.body))
    assert str(soup) == original

def test_existing_semantic_navigation_list_is_preserved() -> None:
    text = body_text('<nav><ol><li>Chapter entry.</li></ol><div>Unstructured navigation.</div></nav><div>Body sentence.</div>')
    assert '1. Chapter entry.' in text
    assert 'Unstructured navigation.' not in text
    assert 'Body sentence.' in text

@pytest.mark.parametrize('path,expected_hash', private_rows('epub_body', 2))
def test_optional_real_books_recover_body_without_network(path, expected_hash, monkeypatch):
    """Existing local books only; no download, enrichment, or canonical output."""
    import hashlib
    import json
    import socket
    from pathlib import Path
    from pdf_chunker import PipelineSpec, convert
    from pdf_chunker.adapters.io_epub import read_epub
    from pdf_chunker.config import load_spec
    source = Path(path)
    if not source.is_file():
        pytest.skip('optional local corpus fixture unavailable')
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected_hash

    def refuse_network(*args, **kwargs):
        raise AssertionError('CPU fixture must not open a network connection')
    monkeypatch.setattr(socket.socket, 'connect', refuse_network)
    monkeypatch.setenv('PDF_CHUNKER_JSONL_MAX_CHARS', '16384')
    doc = read_epub(path)
    blocks = [block for page in doc['pages'] for block in page['blocks']]
    chars = sum((len(block['text']) for block in blocks))
    assert chars > 50000, 'headings or navigation alone are not a recovered book'
    assert all((block['source']['filename'] == source.name for block in blocks))
    active = load_spec('pipeline_rag.yaml')
    spec = PipelineSpec(pipeline=[name for name in active.pipeline if name != 'ai_enrich'], options={name: value for name, value in active.options.items() if name != 'ai_enrich'})
    rows = convert(path, spec)
    assert len(rows) > 20
    assert all((row['metadata']['source_file'] == source.name for row in rows))
    assert all((len(json.dumps(row, ensure_ascii=False)) <= 16384 for row in rows))
    print(json.dumps({'source_file': source.name, 'raw_sha256': expected_hash, 'extracted_chars': chars, 'rows': len(rows), 'cpu_only': True}))
