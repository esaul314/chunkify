import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import pytest

from pdf_chunker.config import PipelineSpec
from pdf_chunker.core_new import _input_artifact, run_convert
from pdf_chunker.framework import Artifact, register, registry


def _report(path: Path) -> dict:
    return json.loads(path.read_text())


def test_run_report_written_on_success(tmp_path):
    pdf = Path("sample_book0-1.pdf")
    spec = PipelineSpec(
        pipeline=["pdf_parse"], options={"run_report": {"output_path": str(tmp_path / "r.json")}}
    )
    a = _input_artifact(str(pdf), spec)
    run_convert(a, spec)
    report = _report(tmp_path / "r.json")
    assert "pdf_parse" in report["timings"]
    assert report["metrics"]["page_count"] >= 1
    env = report["metrics"]["env"]
    assert {"sys_version", "platform", "passes"} <= env.keys()
    digest = env["passes"].get("pdf_parse", "")
    assert re.fullmatch(r"[0-9a-f]{32}", digest)
    assert isinstance(report["warnings"], list)


def test_run_report_written_on_failure(tmp_path, monkeypatch):
    @dataclass(frozen=True)
    class Boom:
        name: str = "boom"
        input_type: type = Artifact
        output_type: type = Artifact

        def __call__(self, a: Artifact) -> Artifact:  # pragma: no cover - exercised in test
            raise RuntimeError("boom")

    monkeypatch.setattr("pdf_chunker.framework._REGISTRY", MappingProxyType(registry()))
    register(Boom())
    pdf = Path("sample_book0-1.pdf")
    spec = PipelineSpec(
        pipeline=["pdf_parse", "boom"],
        options={"run_report": {"output_path": str(tmp_path / "r.json")}},
    )
    a = _input_artifact(str(pdf), spec)
    with pytest.raises(RuntimeError):
        run_convert(a, spec)
    report = _report(tmp_path / "r.json")
    assert set(report["timings"]) >= {"pdf_parse", "boom"}
    assert report["warnings"] == []
    env = report["metrics"]["env"]
    assert "pdf_parse" in env.get("passes", {})
