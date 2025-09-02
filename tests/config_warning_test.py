import textwrap
import warnings

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
