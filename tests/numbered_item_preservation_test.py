import os
import subprocess
from pathlib import Path


TARGET = "Most engineers don't want to learn a whole new toolset for infrequent tasks."


def test_numbered_item_preserved(tmp_path: Path) -> None:
    pdf = Path("platform-eng-excerpt.pdf").resolve()
    spec = Path("pipeline.yaml").resolve()
    out = tmp_path / "out.jsonl"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    subprocess.run(
        [
            "python",
            "-m",
            "pdf_chunker.cli",
            "convert",
            str(pdf),
            "--spec",
            str(spec),
            "--out",
            str(out),
            "--no-enrich",
        ],
        check=True,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    text = out.read_text()
    assert text.count(TARGET) == 1
    assert TARGET + "\n\nInfrastructure" not in text
