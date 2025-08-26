from pathlib import Path

from tests.scripts_cli_test import _run_cli
from tests.utils.materialize import materialize_base64


def test_verbose_outputs_pass_names(tmp_path: Path) -> None:
    pdf_path = materialize_base64(
        Path("tests/golden/samples/sample.pdf.b64"), tmp_path, "sample.pdf"
    )
    out_file = tmp_path / "out.jsonl"
    result = _run_cli(
        "convert",
        str(pdf_path),
        "--chunk-size",
        "1000",
        "--overlap",
        "0",
        "--out",
        str(out_file),
        "--verbose",
        cwd=tmp_path,
    )
    assert result.returncode == 0
    assert "pdf_parse:" in result.stdout
