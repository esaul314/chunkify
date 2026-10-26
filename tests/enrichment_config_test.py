"""Pure serving settings and exact existing wire requests; no provider access."""

import copy
import sys
from dataclasses import FrozenInstanceError, asdict
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pdf_chunker.adapters import ai_enrich as adapter
from pdf_chunker.enrichment_config import (
    completion_request,
    resolve_fallback_settings,
    resolve_primary_settings,
)


def test_defaults_are_explicit_frozen_and_credential_free():
    environment = {"OPENAI_API_KEY": "synthetic-secret"}
    primary = resolve_primary_settings({}, environment=environment)
    fallback = resolve_fallback_settings({}, environment=environment)
    assert asdict(primary) == {
        "model": "qwen3-8b-local",
        "api_base": "http://localhost:8100/v1",
        "max_tokens": 256,
        "no_think": True,
    }
    assert asdict(fallback) == {
        "model": "gpt-4.1-nano",
        "api_base": None,
        "max_tokens": 128,
        "no_think": False,
    }
    assert "synthetic-secret" not in repr((primary, fallback))
    with pytest.raises(FrozenInstanceError):
        primary.model = "changed"


@pytest.mark.parametrize("resolver", [resolve_primary_settings, resolve_fallback_settings])
def test_explicit_options_override_environment_without_mutation(resolver):
    options = {"model": "explicit", "api_base": "http://explicit/v1", "max_tokens": 17}
    environment = {
        "PDF_CHUNKER_AI_ENRICH_MODEL": "primary-env",
        "PDF_CHUNKER_AI_ENRICH_FALLBACK_MODEL": "fallback-env",
        "PDF_CHUNKER_AI_ENRICH_MAX_TOKENS": "99",
        "PDF_CHUNKER_AI_ENRICH_FALLBACK_MAX_TOKENS": "88",
    }
    before = copy.deepcopy((options, environment))
    stage = partial(resolver, environment=environment)
    resolved = stage(options)
    assert (resolved.model, resolved.api_base, resolved.max_tokens) == (
        "explicit",
        "http://explicit/v1",
        17,
    )
    assert stage(options) == resolved
    assert (options, environment) == before
    options["model"] = "later"
    environment["PDF_CHUNKER_AI_ENRICH_MAX_TOKENS"] = "999"
    assert resolved.model == "explicit"
    assert resolved.max_tokens == 17


def test_primary_environment_alias_precedence():
    environment = {
        "OPENAI_MODEL": "openai",
        "OPENAI_API_BASE_URL": "http://openai/v1",
        "PDF_CHUNKER_AI_ENRICH_BASE_URL": "http://alias/v1",
        "PDF_CHUNKER_AI_ENRICH_API_BASE": "http://primary/v1",
        "PDF_CHUNKER_AI_ENRICH_MODEL": "primary",
        "PDF_CHUNKER_AI_ENRICH_MAX_TOKENS": "41",
    }
    result = resolve_primary_settings({}, environment=environment)
    assert (result.model, result.api_base, result.max_tokens) == (
        "primary",
        "http://primary/v1",
        41,
    )
    assert (
        resolve_primary_settings({"base_url": "http://option/v1"}, environment=environment).api_base
        == "http://option/v1"
    )
    del environment["PDF_CHUNKER_AI_ENRICH_API_BASE"]
    assert resolve_primary_settings({}, environment=environment).api_base == "http://alias/v1"
    del environment["PDF_CHUNKER_AI_ENRICH_BASE_URL"]
    del environment["PDF_CHUNKER_AI_ENRICH_MODEL"]
    assert resolve_primary_settings({}, environment=environment).api_base == "http://openai/v1"
    assert resolve_primary_settings({}, environment=environment).model == "openai"


def test_fallback_uses_its_own_environment_and_never_primary_aliases():
    environment = {
        "OPENAI_MODEL": "ignored",
        "OPENAI_API_BASE_URL": "http://ignored/v1",
        "PDF_CHUNKER_AI_ENRICH_MODEL": "ignored",
        "PDF_CHUNKER_AI_ENRICH_FALLBACK_MODEL": "fallback",
        "PDF_CHUNKER_AI_ENRICH_FALLBACK_API_BASE": "http://fallback/v1",
        "PDF_CHUNKER_AI_ENRICH_FALLBACK_MAX_TOKENS": "43",
    }
    result = resolve_fallback_settings({}, environment=environment)
    assert (result.model, result.api_base, result.max_tokens, result.no_think) == (
        "fallback",
        "http://fallback/v1",
        43,
        False,
    )


@pytest.mark.parametrize(
    "raw, expected",
    [(None, True), (" YES ", True), ("1", True), ("off", False), ("false", False), ("", False)],
)
def test_no_think_environment_policy_and_explicit_override(raw, expected):
    environment = {} if raw is None else {"PDF_CHUNKER_AI_ENRICH_NO_THINK": raw}
    assert resolve_primary_settings({}, environment=environment).no_think is expected
    assert resolve_primary_settings({"no_think": False}, environment=environment).no_think is False


@pytest.mark.parametrize("resolver", [resolve_primary_settings, resolve_fallback_settings])
@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "32"])
def test_explicit_invalid_token_limits_refuse_instead_of_falling_back(resolver, value):
    with pytest.raises(ValueError, match="max_tokens"):
        resolver({"max_tokens": value}, environment={})


@pytest.mark.parametrize("resolver", [resolve_primary_settings, resolve_fallback_settings])
@pytest.mark.parametrize("value", ["0", "-1", "no", "1.5"])
def test_invalid_environment_token_limits_refuse(resolver, value):
    environment = {
        "PDF_CHUNKER_AI_ENRICH_MAX_TOKENS": value,
        "PDF_CHUNKER_AI_ENRICH_FALLBACK_MAX_TOKENS": value,
    }
    with pytest.raises(ValueError, match="max_tokens"):
        resolver({}, environment=environment)


@pytest.mark.parametrize(
    "options",
    [{"model": 1}, {"model": " "}, {"api_base": 1}, {"api_base": " "}, {"no_think": "false"}],
)
def test_invalid_primary_labels_and_boolean_refuse(options):
    with pytest.raises(ValueError):
        resolve_primary_settings(options, environment={})


@given(st.text(alphabet=st.characters(blacklist_categories=("Cs",))))
def test_wire_composition_preserves_full_unicode_prompt_and_is_detached(prompt):
    settings = resolve_primary_settings({}, environment={})
    stage = partial(completion_request, settings=settings)
    request = stage(prompt)
    assert request == {
        "model": settings.model,
        "messages": [{"role": "user", "content": "/no_think\n" + prompt}],
        "temperature": 0.0,
        "max_tokens": settings.max_tokens,
        "api_base": settings.api_base,
        "custom_llm_provider": "openai",
    }
    request["messages"][0]["content"] = "changed"
    assert stage(prompt)["messages"][0]["content"] == "/no_think\n" + prompt


@pytest.mark.parametrize(
    "model, no_think, prefix",
    [
        ("qwen3-8b-local", True, True),
        ("Qwen/Qwen3-8B-AWQ", True, True),
        ("qwen3-14b-local", True, True),
        ("Qwen/Qwen3-14B-AWQ", True, True),
        ("qwen3-8b-local", False, False),
        ("other", True, False),
    ],
)
def test_no_think_wire_prefix_applies_only_to_existing_primary_aliases(model, no_think, prefix):
    settings = resolve_primary_settings({"model": model, "no_think": no_think}, environment={})
    assert completion_request("text", settings=settings)["messages"][0]["content"] == (
        "/no_think\ntext" if prefix else "text"
    )


@pytest.mark.parametrize("fallback", [False, True])
def test_existing_adapter_uses_recorded_settings_and_keeps_credential_at_boundary(
    monkeypatch, fallback
):
    monkeypatch.setattr("dotenv.load_dotenv", lambda: None)
    monkeypatch.setattr(adapter.os, "environ", {})
    calls = []

    def provider(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))])

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=provider))
    options = {"model": "qwen3-8b-local", "max_tokens": 37}
    resolve = resolve_fallback_settings if fallback else resolve_primary_settings
    init = adapter.init_fallback_llm if fallback else adapter.init_llm
    settings = resolve(options, environment={})
    callback = init(api_key="synthetic-secret", **options)
    assert callback is not None
    monkeypatch.setenv("PDF_CHUNKER_AI_ENRICH_MODEL", "later")
    monkeypatch.setenv("PDF_CHUNKER_AI_ENRICH_FALLBACK_MODEL", "later")
    assert callback("exact 界🧭\ntext") == "answer"
    assert calls == [
        {**completion_request("exact 界🧭\ntext", settings=settings), "api_key": "synthetic-secret"}
    ]
    assert "synthetic-secret" not in repr(settings)


@pytest.mark.parametrize("init", [adapter.init_llm, adapter.init_fallback_llm])
def test_adapter_rejects_invalid_settings_before_provider_import(monkeypatch, init):
    monkeypatch.setattr("dotenv.load_dotenv", lambda: None)
    monkeypatch.setattr(adapter.os, "environ", {})
    monkeypatch.setitem(sys.modules, "litellm", None)
    with pytest.raises(ValueError, match="max_tokens"):
        init(api_key="synthetic-secret", max_tokens=0)


def test_pure_resolution_does_not_discover_environment_files_or_clients(monkeypatch):
    monkeypatch.setattr(adapter.os, "environ", {"PDF_CHUNKER_AI_ENRICH_MODEL": "ambient"})
    monkeypatch.setattr(Path, "open", lambda *_a, **_kw: pytest.fail("unexpected file access"))
    monkeypatch.setattr("dotenv.load_dotenv", lambda: pytest.fail("unexpected dotenv access"))
    monkeypatch.setitem(sys.modules, "litellm", None)
    assert resolve_primary_settings({}, environment={}).model == "qwen3-8b-local"
    assert resolve_fallback_settings({}, environment={}).model == "gpt-4.1-nano"


def test_unconfigured_or_explicitly_disabled_fallback_does_not_resolve_or_import(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda: None)
    monkeypatch.setattr(adapter.os, "environ", {})
    monkeypatch.setitem(sys.modules, "litellm", None)
    # Preserve the existing lazy no-key path, including unused invalid options.
    assert adapter.init_fallback_llm(max_tokens=0) is None
    monkeypatch.setattr(
        adapter, "init_fallback_llm", lambda **_kw: pytest.fail("disabled fallback")
    )
    assert adapter.resolve_fallback_completion({"fallback_enabled": False}) == (
        None,
        "fallback_disabled",
    )


def test_adapter_resolves_environment_before_capturing_request_settings(monkeypatch):
    monkeypatch.setattr(adapter.os, "environ", {})

    def load_environment():
        monkeypatch.setenv("PDF_CHUNKER_AI_ENRICH_MODEL", "dotenv-model")
        monkeypatch.setenv("PDF_CHUNKER_AI_ENRICH_MAX_TOKENS", "13")

    calls = []

    def provider(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))])

    monkeypatch.setattr("dotenv.load_dotenv", load_environment)
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=provider))
    callback = adapter.init_llm(api_key="synthetic-secret")
    callback("text")
    assert (calls[0]["model"], calls[0]["max_tokens"]) == ("dotenv-model", 13)
    assert calls[0]["messages"] == [{"role": "user", "content": "text"}]
