from types import MappingProxyType

import pytest

import pdf_chunker.framework as framework
from pdf_chunker import convert as public_convert
from pdf_chunker.config import PipelineSpec
from pdf_chunker.core_new import (
    PipelineContractError,
    UnsupportedFormatError,
    choose_parser,
    convert,
    run_convert,
)
from pdf_chunker.framework import Artifact, register, registry


class _StubPass:
    name = "stub"
    input_type = object
    output_type = object

    def __call__(self, artifact: Artifact[object]) -> Artifact[object]:
        return artifact


def test_public_convert_is_canonical_runner() -> None:
    assert public_convert is convert


def test_artifact_detaches_nested_metadata() -> None:
    metadata = {"metrics": {"nested": {"value": 1}}}
    artifact = Artifact(payload={}, meta=metadata)

    metadata["metrics"]["nested"]["value"] = 2

    assert artifact.meta["metrics"]["nested"]["value"] == 1


def test_register_rejects_accidental_name_collision(monkeypatch: pytest.MonkeyPatch) -> None:
    first = _StubPass()
    second = _StubPass()
    monkeypatch.setattr(framework, "_REGISTRY", MappingProxyType({"stub": first}))

    with pytest.raises(ValueError, match="pass name collision"):
        register(second)


def test_register_remains_idempotent_for_same_object(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubPass()
    monkeypatch.setattr(framework, "_REGISTRY", MappingProxyType({"stub": stub}))

    assert register(stub) is stub
    assert registry()["stub"] is stub


def test_choose_parser_reports_unsupported_format() -> None:
    with pytest.raises(UnsupportedFormatError, match="supported formats"):
        choose_parser(".docx")


def test_runner_rejects_wrong_payload_shape_early() -> None:
    spec = PipelineSpec(pipeline=["text_clean", "split_semantic"])
    artifact = Artifact(
        payload={"type": "chunks", "items": []},
        meta={"input": "document.pdf"},
    )

    with pytest.raises(PipelineContractError, match="text_clean input requires 'page_blocks'"):
        run_convert(artifact, spec)


def test_runner_rejects_invalid_split_options_early() -> None:
    spec = PipelineSpec(
        pipeline=["text_clean", "split_semantic"],
        options={"split_semantic": {"chunk_size": 10, "overlap": 10}},
    )
    artifact = Artifact(
        payload={"type": "page_blocks", "pages": []},
        meta={"input": "document.pdf"},
    )

    with pytest.raises(ValueError, match="overlap must be smaller"):
        run_convert(artifact, spec)
