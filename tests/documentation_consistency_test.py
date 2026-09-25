"""Guard the active documentation and dependency declarations against drift."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from pdf_chunker.config import load_spec
from pdf_chunker.framework import registry

ROOT = Path(__file__).parents[1]


def _runtime_names(values: list[str]) -> set[str]:
    return {re.split(r"[<>=!~\[]", value, maxsplit=1)[0].lower() for value in values}


def test_active_roadmap_is_the_single_current_plan() -> None:
    roadmap = (ROOT / "docs/IMPLEMENTATION_ROADMAP.md").read_text(encoding="utf-8")
    assert "P2-C — Documentation and dependency cleanup" in roadmap

    for name in ("REFACTORING_ROADMAP.md", "STRATEGIC_REFACTORING_PLAN.md"):
        text = (ROOT / "docs" / name).read_text(encoding="utf-8")
        assert "Historical record" in text
        assert "IMPLEMENTATION_ROADMAP.md" in text

    stabilization = (ROOT / "docs/test_stabilization_plan.md").read_text(encoding="utf-8")
    assert "Historical record" in stabilization
    assert "IMPLEMENTATION_ROADMAP.md" in stabilization


def test_declared_dependency_profiles_match_supported_boundaries() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = _runtime_names(project["project"]["dependencies"])
    extras = project["project"]["optional-dependencies"]

    assert {"ebooklib", "beautifulsoup4"} <= dependencies
    assert "litellm" in _runtime_names(extras["enrichment"])
    assert "pdfminer.six" in _runtime_names(extras["pdf-fallback"])
    assert "pymupdf4llm" in _runtime_names(extras["pdf-enhancement"])
    assert "-e ." in (ROOT / "requirements.txt").read_text(encoding="utf-8")


def test_pipeline_specs_use_registered_passes() -> None:
    registered = set(registry())
    for filename in ("pipeline.yaml", "pipeline_rag.yaml"):
        spec = load_spec(ROOT / filename)
        assert set(spec.pipeline) <= registered
        assert spec.pipeline[0] == "pdf_parse"


def test_agents_pass_table_lists_the_live_registry() -> None:
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    table = agents.split("<!-- BEGIN AUTO-PASSES -->", 1)[1].split("<!-- END AUTO-PASSES -->", 1)[0]
    assert all(f"| `{name}` |" in table for name in registry())
