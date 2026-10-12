"""Exact admission and grounding regressions; no model/provider requests."""

import copy
import sys
from types import SimpleNamespace

import pytest

from pdf_chunker.adapters import ai_enrich as adapter
from pdf_chunker.adapters.request_budget import prepare_request_counter
from pdf_chunker.passes.ai_enrich import _grounded_tags, classify_chunk_utterance


def _budget(context=4096, margin=64):
    return {
        "model": "Qwen/Qwen3-8B-AWQ",
        "revision": "a" * 40,
        "context_tokens": context,
        "margin_tokens": margin,
        "enable_thinking": False,
        "tokenizer_sha256": "b" * 64,
        "tokenizer_config_sha256": "c" * 64,
    }


@pytest.mark.parametrize(
    "input_tokens,output,margin,allowed",
    [
        (3840, 256, 0, True),
        (3841, 256, 0, False),
        (3776, 256, 64, True),
        (3777, 256, 64, False),
        (3904, 128, 64, True),
        (3905, 128, 64, False),
        (3732, 300, 64, True),
        (3733, 300, 64, False),
    ],
)
def test_full_wire_request_and_reservation(monkeypatch, input_tokens, output, margin, allowed):
    calls, counted = [], []

    def counter(messages):
        counted.append(copy.deepcopy(messages))
        return input_tokens

    monkeypatch.setattr(adapter, "prepare_request_counter", lambda *_a, **_kw: counter)

    def provider(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Classification: explanation\nTags: []")
                )
            ]
        )

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=provider))
    contract = _budget(margin=margin)
    completion = adapter.init_llm(
        api_key="test", model="qwen3-8b-local", max_tokens=output, request_budget=contract
    )
    # Mutating the caller's mapping cannot change the admitted serving contract.
    contract["context_tokens"] = 99999
    if allowed:
        completion("full text and instructions")
        assert calls[0]["messages"] == counted[0]
        assert calls[0]["max_tokens"] == output
    else:
        with pytest.raises(adapter.CompletionError, match="verified context budget"):
            completion("full text and instructions")
        assert calls == []
    assert counted == [[{"role": "user", "content": "/no_think\nfull text and instructions"}]]


def test_unknown_or_missing_contract_fails_before_transport(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    with pytest.raises(ValueError, match="model does not match"):
        prepare_request_counter(_budget(), model="another-provider")
    with pytest.raises(FileNotFoundError):
        prepare_request_counter(_budget(), model="qwen3-8b-local")
    invalid = _budget()
    invalid["enable_thinking"] = True
    with pytest.raises(ValueError, match="disabled-thinking"):
        prepare_request_counter(invalid, model="qwen3-8b-local")


@pytest.mark.parametrize(
    "field,value",
    [
        ("context_tokens", True),
        ("context_tokens", 0),
        ("margin_tokens", -1),
        ("margin_tokens", 4096),
        ("revision", "main"),
        ("tokenizer_sha256", "bad"),
    ],
)
def test_malformed_contract_is_not_a_guess(monkeypatch, tmp_path, field, value):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    budget = _budget()
    budget[field] = value
    with pytest.raises(ValueError):
        prepare_request_counter(budget, model="qwen3-8b-local")


def test_cached_hash_mismatch_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    snapshot = tmp_path / "models--Qwen--Qwen3-8B-AWQ" / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "tokenizer.json").write_bytes(b"untrusted")
    with pytest.raises(ValueError, match="hash mismatch"):
        prepare_request_counter(_budget(), model="qwen3-8b-local")


def test_prompt_grounding_preserves_text_vocabulary_and_result_validation():
    text = "Mixed CASE software architecture with design patterns and Python."
    tags = {
        "concepts": ["software_architecture", "design_patterns", "testing", "python", "tool"],
        "content_type": ["explanation"],
        "duplicate": ["python"],
    }
    original = copy.deepcopy(tags)
    captured = []

    def completion(prompt):
        captured.append(prompt)
        return "Classification: explanation\nTags: [software_architecture, testing, python]"

    result = classify_chunk_utterance(text, tag_configs=tags, completion_fn=completion)
    assert result == {"classification": "explanation", "tags": ["software_architecture", "python"]}
    assert tags == original
    assert f'Text: "{text}"' in captured[0]
    assert "- concepts: software_architecture, design_patterns, python" in captured[0]
    assert "- duplicate: python" in captured[0]
    assert "- content_type:" not in captured[0]
    assert _grounded_tags(tags["concepts"], text) == [
        "software_architecture",
        "design_patterns",
        "python",
    ]


def test_budget_failure_cannot_mark_a_chunk_complete():
    from pdf_chunker.passes.ai_enrich import _classify_chunk

    client = adapter.Client(
        completion_fn=lambda _p: (_ for _ in ()).throw(adapter.CompletionError("budget")),
        tag_configs={"concepts": ["python"]},
    )
    chunk = {"text": "Python example", "metadata": {"source_file": "fixture.pdf", "page": 3}}
    before = copy.deepcopy(chunk)
    with pytest.raises(RuntimeError, match="completion failed"):
        _classify_chunk(chunk, client=client, tag_configs={"concepts": ["python"]})
    assert chunk == before
