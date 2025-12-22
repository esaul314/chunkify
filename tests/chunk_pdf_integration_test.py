import json
import os
import subprocess
from pathlib import Path


def test_chunk_pdf_generates_jsonl(tmp_path: Path) -> None:
    pdf = Path("test_data/sample_test.pdf").resolve()
    script = Path("scripts/chunk_pdf.py").resolve()
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    result = subprocess.run(
        ["python", str(script), str(pdf)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
        check=True,
    )
    lines = [
        json.loads(line) for line in result.stdout.splitlines() if line.strip().startswith("{")
    ]
    assert lines and all(
        {"text", "metadata"} <= line.keys() and {"chunk_id", "source"} <= line["metadata"].keys()
        for line in lines
    )
    assert not any(tmp_path.iterdir())
