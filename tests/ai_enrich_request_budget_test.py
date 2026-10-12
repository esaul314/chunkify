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


@pytest.mark.parametrize(
    "text", ["a" * 999, "🦎日本語é" * 300, "\n \t" * 800, 'x"\nResponse:y' * 100]
)
def test_windows_are_lossless_deterministic_and_guard_both_prompts(text):
    from pdf_chunker.passes.ai_enrich import _enrichment_windows

    seen = []

    def fits(prompt):
        seen.append(prompt)
        # Deliberately nonmonotone: a longer candidate can fit where a shorter
        # one does not. The partition must not rely on binary-search token math.
        source = prompt.rsplit('Text: "', 1)[1].rsplit('"\n\nResponse:', 1)[0]
        return len(source) <= 90 and len(source) % 7 != 2

    options = {"tag_configs": {"topics": ["python"]}, "request_fits": fits}
    windows = _enrichment_windows(text, **options)
    assert "".join(windows) == text
    assert windows == _enrichment_windows(text, **options)
    assert all(len(window) <= 90 for window in windows)
    assert all("Classification:" in prompt for prompt in seen)


def test_impossible_instruction_budget_is_not_false_completion():
    from pdf_chunker.passes.ai_enrich import _enrichment_windows

    with pytest.raises(ValueError, match="single character"):
        _enrichment_windows("text", tag_configs={}, request_fits=lambda _p: False)


def test_window_failure_and_repair_failure_cannot_be_accepted():
    from pdf_chunker.passes.ai_enrich import _classify_chunk

    calls = []

    def completion(prompt):
        calls.append(prompt)
        if len(calls) == 2:
            raise adapter.CompletionError("failed window")
        return "Classification: explanation\nTags: []"

    completion.request_fits = lambda p: len(p) < 1800
    chunk = {"text": "x" * 3000, "metadata": {"chunk_id": "keep", "source_file": "x.pdf"}}
    before = copy.deepcopy(chunk)
    with pytest.raises(RuntimeError, match="completion failed"):
        _classify_chunk(chunk, client=adapter.Client(completion_fn=completion), tag_configs={})
    assert chunk == before

    def repair_failure(prompt):
        if "single best utterance" in prompt:
            raise adapter.CompletionError("failed repair")
        return "Classification: unclassified\nTags: []"

    repair_failure.request_fits = completion.request_fits
    with pytest.raises(RuntimeError, match="completion failed"):
        _classify_chunk(chunk, client=adapter.Client(completion_fn=repair_failure), tag_configs={})
    assert chunk == before


def test_window_aggregation_is_stable_and_keeps_full_validation_vocabulary():
    def completion(prompt):
        text = prompt.rsplit('Text: "', 1)[1].rsplit('"\n\nResponse:', 1)[0]
        label = "explanation" if "Python" in text else "instruction"
        return f"Classification: {label}\nTags: [python, configuration, invented]"

    completion.request_fits = lambda p: len(p) <= 1550
    text = "Python " * 150 + "configuration " * 50
    tags = {"topics": ["python", "configuration", "testing"]}
    original = copy.deepcopy(tags)
    result = classify_chunk_utterance(text, tag_configs=tags, completion_fn=completion)
    assert result == {"classification": "explanation", "tags": ["python", "configuration"]}
    assert tags == original


def test_fitting_and_unbudgeted_calls_remain_single_requests():
    calls = []

    def completion(prompt):
        calls.append(prompt)
        return "Classification: explanation\nTags: [python]"

    expected = classify_chunk_utterance(
        "Python", tag_configs={"x": ["python"]}, completion_fn=completion
    )
    first = calls.pop()
    completion.request_fits = lambda _p: True
    assert (
        classify_chunk_utterance("Python", tag_configs={"x": ["python"]}, completion_fn=completion)
        == expected
    )
    assert calls == [first]


def test_windowed_fallback_failure_fails_source_without_changing_legacy_behavior():
    from pdf_chunker.passes.ai_enrich import _classify_chunk

    def completion(_prompt):
        return "Classification: explanation\nTags: []"

    def failed_fallback(_prompt):
        raise TimeoutError("fallback unavailable")

    text = "Python " * 500
    tags = {"topics": ["python"]}
    legacy = classify_chunk_utterance(
        text, tag_configs=tags, completion_fn=completion, fallback_completion_fn=failed_fallback
    )
    assert legacy["classification"] == "explanation"
    completion.request_fits = lambda prompt: len(prompt) < 1800
    client = adapter.Client(
        completion_fn=completion, tag_configs=tags, fallback_completion_fn=failed_fallback
    )
    original = {"text": text, "metadata": {"chunk_id": "keep", "source_file": "x.pdf"}}
    chunk = copy.deepcopy(original)
    with pytest.raises(RuntimeError, match="completion failed"):
        _classify_chunk(chunk, client=client, tag_configs=tags)
    assert chunk == original


def test_oversized_chunk_is_windowed_without_changing_parent(monkeypatch):
    from pdf_chunker.passes.ai_enrich import _classify_chunk

    calls = []
    monkeypatch.setattr(
        adapter, "prepare_request_counter", lambda *_a, **_kw: lambda m: len(m[0]["content"])
    )

    def provider(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Classification: explanation\nTags: [python]")
                )
            ]
        )

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=provider))
    completion = adapter.init_llm(
        api_key="test", model="qwen3-8b-local", request_budget=_budget(context=2300)
    )
    client = adapter.Client(completion_fn=completion, tag_configs={"topics": ["python"]})
    text = ("Python Unicode café 日本語 🦎\n" * 300) + "TAIL SENTINEL"
    chunk = {"text": text, "metadata": {"chunk_id": "unchanged", "source_file": "x.pdf"}}
    before = copy.deepcopy(chunk)
    result = _classify_chunk(chunk, client=client, tag_configs={"topics": ["python"]})
    assert len(calls) > 1
    windows = [
        c["messages"][0]["content"].rsplit('Text: "', 1)[1].rsplit('"\n\nResponse:', 1)[0]
        for c in calls
    ]
    assert "".join(windows) == text
    assert all(len(c["messages"][0]["content"]) + 256 + 64 <= 2300 for c in calls)
    assert result["text"] == text
    assert result["metadata"]["chunk_id"] == "unchanged"
    assert result["metadata"]["source_file"] == "x.pdf"
    assert result["metadata"]["enrichment_status"] == "complete"
    assert result["tags"] == ["python"]
    assert chunk == before
