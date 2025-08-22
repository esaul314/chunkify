import sys
from pathlib import Path

from typer.testing import CliRunner

from pdf_chunker.cli import app


def test_convert_no_enrich_skips_ai_import(monkeypatch, tmp_path):
    class Blocker:
        def find_spec(self, fullname, path, target=None):
            if fullname == "pdf_chunker.adapters.ai_enrich":
                raise ImportError("ai_enrich should not be imported")
            return None

    monkeypatch.setattr(sys, "meta_path", [Blocker(), *sys.meta_path])
    out = tmp_path / "out.jsonl"
    sample = Path(__file__).resolve().parents[1] / "golden" / "samples" / "tiny.pdf"
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["convert", str(sample), "--no-enrich", "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
