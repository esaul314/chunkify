"""Article identity follows dictionary state, not incidental cache warming."""

from __future__ import annotations

import copy
import hashlib
import io
import json
from types import SimpleNamespace

import pyphen
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pdf_chunker import convert, utils
from pdf_chunker.adapters import article_conversion as articles
from pdf_chunker.adapters import readability_runtime as runtime
from tests.verified_markdown_ingress_test import bind_article


@pytest.fixture
def dictionaries(tmp_path, monkeypatch):
    """Use real Pyphen parsing with tiny, isolated synthetic resources/caches."""
    hyphen_path = tmp_path / "hyphen.dic"
    hyphen_path.write_text("UTF-8\na1b\n", encoding="utf-8")
    cmu_path = tmp_path / "cmu.dict"
    cmu_path.write_bytes(b"EXAMPLE IH0 G Z AE1 M P AH0 L\n")
    monkeypatch.setattr(pyphen, "hdcache", {})
    monkeypatch.setattr(pyphen, "LANGUAGES", {"en_US": hyphen_path})
    hyphenator = pyphen.Pyphen(lang="en_US")
    pronunciations = {"example": [["IH0", "G", "Z", "AE1", "M", "P", "AH0", "L"]]}
    monkeypatch.setattr(utils, "_get_pyphen", lambda lang: hyphenator)
    monkeypatch.setattr(utils, "_get_cmudict", lambda lang: pronunciations)
    monkeypatch.setattr(utils, "_load_cmudict", lambda: pronunciations)
    package = SimpleNamespace(dict_stream=lambda: io.BytesIO(cmu_path.read_bytes()))
    monkeypatch.setattr(runtime, "import_module", lambda name: package)
    return SimpleNamespace(
        hyphen_path=hyphen_path,
        cmu_path=cmu_path,
        hyphenator=hyphenator,
        pronunciations=pronunciations,
    )


def test_resource_bytes_and_loaded_tables_are_pinned_without_disclosing_them(dictionaries):
    state = runtime.collect_readability_runtime()
    assert state["language"] == "en_US"
    assert (
        state["cmudict"]["resource_sha256"]
        == hashlib.sha256(dictionaries.cmu_path.read_bytes()).hexdigest()
    )
    assert (
        state["pyphen"]["resource_sha256"]
        == hashlib.sha256(dictionaries.hyphen_path.read_bytes()).hexdigest()
    )
    assert state["cmudict"]["available"] is True
    assert len(state["cmudict"]["loaded_sha256"]) == 64
    assert len(state["pyphen"]["loaded_sha256"]) == 64
    encoded = json.dumps(state)
    assert str(dictionaries.cmu_path.parent) not in encoded
    assert "IH0" not in encoded
    assert "a1b" not in encoded


def test_cache_warming_is_identity_neutral_and_collection_preserves_state(dictionaries):
    first = runtime.collect_readability_runtime()
    dictionaries.hyphenator.positions("zzabzz")
    dictionaries.hyphenator.positions("unrelated")
    before = copy.deepcopy(vars(dictionaries.hyphenator.hd))
    pronunciations = copy.deepcopy(dictionaries.pronunciations)
    assert runtime.collect_readability_runtime() == first
    assert vars(dictionaries.hyphenator.hd) == before
    assert dictionaries.pronunciations == pronunciations


@pytest.mark.parametrize("resource", ["cmu_path", "hyphen_path"])
def test_changed_resource_bytes_invalidate_even_when_loaded_table_is_unchanged(
    dictionaries, resource
):
    first = runtime.collect_readability_runtime()
    path = getattr(dictionaries, resource)
    path.write_bytes(path.read_bytes() + b"# revision\n")
    second = runtime.collect_readability_runtime()
    name = "cmudict" if resource == "cmu_path" else "pyphen"
    assert first[name]["resource_sha256"] != second[name]["resource_sha256"]
    assert first[name]["loaded_sha256"] == second[name]["loaded_sha256"]


def test_loaded_pronunciation_mutation_changes_identity_without_a_version_change(dictionaries):
    first = runtime.collect_readability_runtime()
    dictionaries.pronunciations["example"][0].append("AH1")
    second = runtime.collect_readability_runtime()
    assert first["cmudict"]["resource_sha256"] == second["cmudict"]["resource_sha256"]
    assert first["cmudict"]["loaded_sha256"] != second["cmudict"]["loaded_sha256"]


@given(st.dictionaries(st.text(alphabet="abcdef", min_size=1, max_size=10), st.integers(0, 9)))
def test_loaded_state_hash_ignores_mapping_order_and_detaches_values(values):
    original = copy.deepcopy(values)
    first = runtime._state_hash(values)
    assert runtime._state_hash(dict(reversed(tuple(values.items())))) == first
    assert values == original
    values["new-word"] = 1
    assert runtime._state_hash(values) != first


@pytest.mark.parametrize("field", ["patterns", "left", "right", "maxlen"])
def test_loaded_hyphenation_settings_and_patterns_affect_identity(dictionaries, field):
    first = runtime.collect_readability_runtime()
    hyphenator = dictionaries.hyphenator
    if field == "patterns":
        hyphenator.hd.patterns["cd"] = (1, (1,))
    elif field == "maxlen":
        hyphenator.hd.maxlen += 1
    else:
        setattr(hyphenator, field, getattr(hyphenator, field) + 1)
    second = runtime.collect_readability_runtime()
    assert first["pyphen"]["resource_sha256"] == second["pyphen"]["resource_sha256"]
    assert first["pyphen"]["loaded_sha256"] != second["pyphen"]["loaded_sha256"]


def test_poisoned_cache_refuses_without_clearing_or_repairing_it(dictionaries):
    hyphenator = dictionaries.hyphenator
    assert hyphenator.positions("zzabzz") == [3]
    hyphenator.hd.cache["zzabzz"] = []
    before = copy.deepcopy(vars(hyphenator.hd))
    with pytest.raises(ValueError, match="hyphenation cache"):
        runtime.collect_readability_runtime()
    assert vars(hyphenator.hd) == before


def test_absent_optional_cmu_records_fallback_without_importing_it(dictionaries, monkeypatch):
    monkeypatch.setattr(utils, "_load_cmudict", None)
    monkeypatch.setattr(utils, "_get_cmudict", lambda lang: None)

    def refuse(name):
        pytest.fail("absent optional CMU must not be imported")

    monkeypatch.setattr(runtime, "import_module", refuse)
    state = runtime.collect_readability_runtime()
    assert state["cmudict"]["available"] is False
    assert state["cmudict"]["resource_sha256"] is None


def test_cmu_loader_fallback_is_distinct_from_loaded_dictionary(dictionaries, monkeypatch):
    first = runtime.collect_readability_runtime()
    monkeypatch.setattr(utils, "_get_cmudict", lambda lang: None)
    fallback = runtime.collect_readability_runtime()
    assert fallback["cmudict"]["available"] is False
    assert fallback["cmudict"]["resource_sha256"] == first["cmudict"]["resource_sha256"]
    assert fallback["cmudict"]["loaded_sha256"] != first["cmudict"]["loaded_sha256"]


def test_unsupported_hyphenator_refuses_instead_of_omitting_state(dictionaries, monkeypatch):
    monkeypatch.setattr(utils, "_get_pyphen", lambda lang: SimpleNamespace())
    with pytest.raises(ValueError, match="canonical readability hyphenator"):
        runtime.collect_readability_runtime()


@pytest.mark.parametrize("resource", ["cmu_path", "hyphen_path"])
def test_unreadable_declared_resource_refuses(dictionaries, resource):
    getattr(dictionaries, resource).unlink()
    with pytest.raises(FileNotFoundError):
        runtime.collect_readability_runtime()


def test_prepared_identity_follows_resources_and_preserves_note_associations(
    dictionaries, tmp_path
):
    bound = bind_article(tmp_path / "article", text="# Example\n\nAn example zzabzz.\n")
    first = articles.prepare_bound_conversion(bound.binding, bound.config(), corpus="technology")
    rows = convert(str(bound.source), first.spec())
    other = bind_article(tmp_path / "other", text="# Other\n\nUnrelated words.\n")
    convert(str(other.source), other.config())
    assert (
        articles.prepare_bound_conversion(bound.binding, bound.config(), corpus="technology")
        == first
    )
    assert convert(str(bound.source), first.spec()) == rows
    dictionaries.pronunciations["example"][0].append("AH1")
    changed = articles.prepare_bound_conversion(bound.binding, bound.config(), corpus="technology")
    assert first.descriptor.key != changed.descriptor.key
    assert first.descriptor.associations == changed.descriptor.associations
    changed_rows = convert(str(bound.source), changed.spec())
    assert changed_rows[0]["metadata"]["readability"] != rows[0]["metadata"]["readability"]
    assert [row["text"] for row in changed_rows] == [row["text"] for row in rows]


def test_real_runtime_is_stable_across_conversion_and_contains_readability(tmp_path):
    bound = bind_article(tmp_path / "article")
    first = articles.prepare_bound_conversion(bound.binding, bound.config(), corpus="technology")
    contract = json.loads(first.descriptor.contract_json)
    assert contract["configuration"]["runtime"]["readability"]["language"] == "en_US"
    rows = convert(str(bound.source), first.spec())
    other = bind_article(tmp_path / "other", text="# Other\n\nNonrepresentational zzyyxx.\n")
    convert(str(other.source), other.config())
    assert (
        articles.prepare_bound_conversion(bound.binding, bound.config(), corpus="technology")
        == first
    )
    repeated = convert(str(bound.source), first.spec())
    original_bytes = b"".join((json.dumps(row, ensure_ascii=False) + "\n").encode() for row in rows)
    assert original_bytes == b"".join(
        (json.dumps(row, ensure_ascii=False) + "\n").encode() for row in repeated
    )


def test_preparation_refuses_poisoned_cache_before_creating_any_artifact(dictionaries, tmp_path):
    bound = bind_article(tmp_path / "article")
    dictionaries.hyphenator.hd.cache["zzabzz"] = []
    with pytest.raises(ValueError, match="hyphenation cache"):
        articles.prepare_bound_conversion(bound.binding, bound.config(), corpus="technology")
    assert dictionaries.hyphenator.hd.cache["zzabzz"] == []
    assert not list(tmp_path.rglob("*.jsonl"))
