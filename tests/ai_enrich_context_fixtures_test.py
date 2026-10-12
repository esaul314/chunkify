"""Optional, offline real-PDF regression; no inference or fixture downloads.

Source paths, hashes and expected controls come from a private JSON manifest.
Missing optional files/dependencies skip; present but changed files fail.
Only tokenizer data is read, never model weights. Run with ``-s`` for evidence.
"""
from tests.local_source_controls import private_value
import json
import os
import socket
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import pytest
REVISION = '4da05a8edb55c6046cce958586c33b61da07bb79'
TOKENIZER_HASHES = {'tokenizer.json': 'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4', 'tokenizer_config.json': 'd5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101'}
PREFIX_HASH = private_value('control_001', '')
CANONICAL_CHUNK_HASHES = private_value('context_chunk_hashes', {})
REPO = Path(__file__).resolve().parents[1]
PREFIX_PASSES = ['pdf_parse', 'text_clean', 'heading_detect', 'split_semantic']

@dataclass(frozen=True)
class SourceFixture:
    name: str
    path: str
    source_hash: str
    chunks: int
    index: int
    text_hash: str
    page: int
    page_range: str
    list_kind: str
    old_input: int
    revised_input: int
SOURCES = tuple((SourceFixture(**row) for row in private_value('context_sources', ())))

def _hash(path):
    return sha256(path.read_bytes()).hexdigest()

def _required_file(path, expected):
    if not path.is_file():
        pytest.skip(f'optional offline context fixture absent: {path}')
    assert _hash(path) == expected, f'fixture hash changed: {path}'
    return path

@pytest.fixture(autouse=True)
def _nltk_data(monkeypatch):
    """Override the shared downloader: this module must stay offline in CI."""

    def forbidden(*args, **kwargs):
        raise AssertionError('offline fixture regression attempted network/download')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(socket.socket, 'connect_ex', forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    nltk = pytest.importorskip('nltk')
    monkeypatch.setattr(nltk, 'download', forbidden)

@pytest.fixture(scope='module')
def rendered_counter():
    hf_home = Path(os.environ.get('HF_HOME', str(Path.home() / '.cache/huggingface')))
    cache = Path(os.environ.get('HF_HUB_CACHE', str(hf_home / 'hub')))
    snapshot = Path(os.environ.get('PDF_CHUNKER_CONTEXT_TOKENIZER_DIR', str(cache / 'models--Qwen--Qwen3-8B-AWQ' / 'snapshots' / REVISION)))
    for filename, expected in TOKENIZER_HASHES.items():
        _required_file(snapshot / filename, expected)
    tokenizers = pytest.importorskip('tokenizers')
    jinja2 = pytest.importorskip('jinja2')
    tokenizer = tokenizers.Tokenizer.from_file(str(snapshot / 'tokenizer.json'))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    config = json.loads((snapshot / 'tokenizer_config.json').read_text())
    template = jinja2.Environment().from_string(config['chat_template'])

    def count(prompt):
        rendered = template.render(messages=[{'role': 'user', 'content': '/no_think\n' + prompt}], tools=None, add_generation_prompt=True, enable_thinking=False)
        assert '<|im_start|>user\n/no_think\n' in rendered
        assert rendered.endswith('<|im_start|>assistant\n<think>\n\n</think>\n\n')
        return len(tokenizer.encode(rendered, add_special_tokens=False).ids)
    return (count, snapshot)

def _without_enrichment(chunk):
    result = deepcopy(chunk)
    for key in ('utterance_type', 'tags'):
        result.pop(key, None)
    for key in ('meta', 'metadata'):
        if key in result:
            for added in ('utterance_type', 'tags', 'enrichment_status'):
                result[key].pop(added, None)
    if 'meta' in result and 'metadata' not in result:
        result['metadata'] = deepcopy(result['meta'])
    return result

@pytest.mark.parametrize('fixture', SOURCES, ids=lambda source: source.name)
def test_real_source_rendered_requests_and_provenance(fixture, rendered_counter, monkeypatch):
    pytest.importorskip('fitz')
    source = _required_file(Path(os.environ.get(f'PDF_CHUNKER_CONTEXT_{fixture.name.upper()}_PDF', fixture.path)), fixture.source_hash)
    spec_path = Path(os.environ.get('PDF_CHUNKER_CONTEXT_ACTIVE_SPEC', str(REPO / 'pipeline_rag.yaml')))
    import yaml
    from pdf_chunker.adapters.ai_enrich import _tag_configs_for_options
    from pdf_chunker.adapters.request_budget import prepare_request_counter
    from pdf_chunker.config import PipelineSpec
    from pdf_chunker.core_new import convert
    from pdf_chunker.framework import Artifact
    from pdf_chunker.passes.ai_enrich import _classify_chunk, _effective_tag_configs, _grounded_tags, classify_chunk_utterance
    from pdf_chunker.passes.emit_jsonl import emit_jsonl
    data = yaml.safe_load(spec_path.read_text())
    assert data['pipeline'][:4] == PREFIX_PASSES
    prefix = {'pipeline': data['pipeline'][:4], 'options': {name: deepcopy(data['options'][name]) for name in PREFIX_PASSES if name in data['options']}}
    assert sha256(json.dumps(prefix, sort_keys=True).encode()).hexdigest() == PREFIX_HASH
    spec = PipelineSpec(**prefix)
    assert 'ai_enrich' not in spec.options and 'emit_jsonl' not in spec.options
    chunks = convert(str(source), spec)
    original = deepcopy(chunks)
    assert len(chunks) == fixture.chunks
    canonical_hash = sha256(json.dumps(original, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    assert canonical_hash == CANONICAL_CHUNK_HASHES[fixture.name]
    target = chunks[fixture.index]
    assert sha256(target['text'].encode()).hexdigest() == fixture.text_hash
    metadata = target.get('metadata', target.get('meta'))
    assert metadata['chunk_id'] == f'{source.name}_p{fixture.page - 1}_c{fixture.index}'
    assert metadata['page'] == fixture.page
    assert metadata['page_range'] == fixture.page_range
    assert metadata['list_kind'] == fixture.list_kind
    assert metadata['block_type'] == 'list_item'
    if fixture.name == 'source_two':
        assert target['text'].startswith('REFERENCES\n')
        assert target['text'].endswith(private_value('control_002', ''))
    tag_options = deepcopy(data['options']['ai_enrich'])
    tag_options['tags_dir'] = str(REPO / 'config/tags')
    tags = _tag_configs_for_options(tag_options)
    original_tags = deepcopy(tags)
    counts = []
    independent_count, snapshot = rendered_counter
    assert snapshot.name == REVISION
    assert snapshot.parent.name == 'snapshots'
    assert snapshot.parent.parent.name == 'models--Qwen--Qwen3-8B-AWQ'
    monkeypatch.setenv('HF_HUB_CACHE', str(snapshot.parents[2]))
    candidate_options = yaml.safe_load((REPO / 'pipeline_rag.yaml').read_text())['options']['ai_enrich']
    contract = candidate_options['request_budget']
    assert contract['context_tokens'] == 4096 and contract['margin_tokens'] == 64
    assert contract['enable_thinking'] is False
    assert contract['revision'] == REVISION
    adapter_count = prepare_request_counter(contract, model=candidate_options['model'])

    class CapturingClient:

        def classify_chunk_utterance(self, text, *, tag_configs):

            def completion(prompt):
                assert f'Text: "{text}"' in prompt
                count = independent_count(prompt)
                assert adapter_count([{'role': 'user', 'content': '/no_think\n' + prompt}]) == count
                counts.append(count)
                assert count + 256 + 64 <= 4096, f'{fixture.name} request {len(counts) - 1}: {count}+256+64 exceeds4096'
                effective = _effective_tag_configs(tag_configs)
                candidates = {category: _grounded_tags(values, text) for category, values in effective.items()}
                lines = [f"- {category}: {', '.join(values)}" for category, values in candidates.items() if values]
                actual = prompt.split('Allowed topical tags by category:\n', 1)[1].split('\nChoose 1-4 tags', 1)[0].strip().splitlines() if 'Allowed topical tags by category:\n' in prompt else []
                assert actual == lines
                return 'Classification: explanation\nTags: []'
            return classify_chunk_utterance(text, tag_configs=tag_configs, completion_fn=completion)
    client = CapturingClient()
    enriched = [_classify_chunk(chunk, client=client, tag_configs=tags) for chunk in chunks]
    assert len(counts) == fixture.chunks
    assert counts[fixture.index] == fixture.revised_input
    assert tags == original_tags and chunks == original
    for before, after in zip(original, enriched, strict=True):
        assert before['text'].encode() == after['text'].encode()
        assert _without_enrichment(before) == _without_enrichment(after)
        assert after['utterance_type'] == 'explanation'
        assert after['meta']['enrichment_status'] == 'complete'
        assert after['metadata']['enrichment_status'] == 'complete'
    monkeypatch.setenv('PDF_CHUNKER_JSONL_MAX_CHARS', '16384')
    monkeypatch.setenv('PDF_CHUNKER_JSONL_META_KEY', 'metadata')

    def serialize(items):
        return emit_jsonl(Artifact(payload={'type': 'chunks', 'items': items}, meta={'semantic_pipeline': 'canonical-v1'})).payload
    before_rows, rows = (serialize(original), serialize(enriched))
    encoded = [json.dumps(row, ensure_ascii=False) for row in rows]
    assert len(rows) == len(chunks)
    assert all((len(line) <= 16384 for line in encoded))
    assert [_without_enrichment(row) for row in rows] == [_without_enrichment(row) for row in before_rows]
    assert [row['text'] for row in rows] == [chunk['text'].strip() for chunk in original]
    assert enriched[fixture.index]['meta']['chunk_id'] == metadata['chunk_id']
    print('CONTEXT_FIXTURE ' + json.dumps({'source': fixture.name, 'adapter_counter_matches_every_prompt': True, 'prefix_sha256': PREFIX_HASH, 'source_sha256': fixture.source_hash, 'canonical_chunks_sha256': sha256(json.dumps(original, sort_keys=True, ensure_ascii=False).encode()).hexdigest(), 'chunks': len(chunks), 'requests': len(counts), 'all_rendered_inputs': counts, 'target_index': fixture.index, 'target_text_sha256': fixture.text_hash, 'old_input_from_separate_active_capture': fixture.old_input, 'revised_target_input': counts[fixture.index], 'largest_input_plus_output_plus_margin': max(counts) + 256 + 64, 'parent_provenance_unchanged': True, 'emitted_rows': len(rows), 'largest_serialized_row_chars': max(map(len, encoded)), 'emitter_child_id': rows[fixture.index]['metadata']['chunk_id']}, sort_keys=True))
