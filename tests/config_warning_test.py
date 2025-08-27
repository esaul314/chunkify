import pytest
from pdf_chunker.config import load_spec


def _write_yaml(path, content):
    path.write_text(content, encoding="utf-8")


def test_warns_on_unknown_pipeline_option(tmp_path):
    yaml_text = """
pipeline:
- pdf_parse
options:
  ghost_pass:
    foo: 1
"""
    spec_path = tmp_path / "pipeline.yaml"
    _write_yaml(spec_path, yaml_text)
    with pytest.warns(UserWarning, match="ghost_pass"):
        load_spec(spec_path)
