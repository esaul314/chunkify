import textwrap
import warnings

import pytest

from pdf_chunker.config import load_spec


def test_disabled_pass_option_suppresses_warning(tmp_path):
    cfg = tmp_path / "pipeline.yaml"
    cfg.write_text(textwrap.dedent(
        """
        pipeline: []
        options:
          ai_enrich:
            enabled: false
          "":
            foo: bar
        """
    ))
    with warnings.catch_warnings(record=True) as w:
        load_spec(cfg)
    assert not w


def test_enabled_unknown_option_emits_warning(tmp_path):
    cfg = tmp_path / "pipeline.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
            pipeline: [pdf_parse]
            options:
              pdf_parse:
                engine: native
              extra_pass:
                foo: 1
            """
        )
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_spec(cfg)

    assert [w.message.args[0] for w in caught] == ["Unknown pipeline options: extra_pass"]


def test_load_spec_merges_env_and_cli_overrides(tmp_path, monkeypatch):
    cfg = tmp_path / "pipeline.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
            pipeline: [pdf_parse, split_semantic]
            options:
              pdf_parse:
                engine: native
                retries: 2
              split_semantic:
                target_tokens: 800
            """
        )
    )
    monkeypatch.setenv("PDF_PARSE__ENGINE", "pymupdf4llm")
    overrides = {"split_semantic": {"target_tokens": 900, "overlap": 25}}

    spec = load_spec(cfg, overrides=overrides)

    assert spec.pipeline == ["pdf_parse", "split_semantic"]
    assert spec.options["pdf_parse"] == {"engine": "pymupdf4llm", "retries": 2}
    assert spec.options["split_semantic"] == {"target_tokens": 900, "overlap": 25}


def test_non_mapping_yaml_raises(tmp_path, monkeypatch):
    cfg = tmp_path / "pipeline.yaml"
    cfg.write_text("- not-a-mapping\n- still-not-a-mapping\n")

    with pytest.raises(TypeError, match="top-level mapping"):
        load_spec(cfg)
