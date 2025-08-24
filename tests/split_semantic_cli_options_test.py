import base64
import json
from pathlib import Path

from typer.testing import CliRunner

from pdf_chunker.cli import app
from pdf_chunker.passes import split_semantic as split_mod


def _materialize(b64_path: Path, tmp_path: Path) -> Path:
    data = base64.b64decode(b64_path.read_text())
    out = tmp_path / "sample.pdf"
    out.write_bytes(data)
    return out


def test_cli_flags_propagate_to_split_semantic(tmp_path, monkeypatch):
    pdf_path = _materialize(Path("tests/golden/samples/sample.pdf.b64"), tmp_path)
    out_path = tmp_path / "out.jsonl"
    seen: dict[str, object] = {}

    orig = split_mod._SplitSemanticPass.__call__

    def spy(self, a):  # type: ignore[override]
        seen.update(
            {
                "chunk_size": self.chunk_size,
                "overlap": self.overlap,
                "generate_metadata": self.generate_metadata,
            }
        )
        return orig(self, a)

    monkeypatch.setattr(split_mod._SplitSemanticPass, "__call__", spy)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "convert",
            str(pdf_path),
            "--chunk-size",
            "100",
            "--overlap",
            "10",
            "--no-metadata",
            "--out",
            str(out_path),
        ],
    )
    assert result.exit_code == 0
    assert seen == {"chunk_size": 100, "overlap": 10, "generate_metadata": False}
    rows = [
        json.loads(line)
        for line in out_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows and all("meta" not in row for row in rows)
