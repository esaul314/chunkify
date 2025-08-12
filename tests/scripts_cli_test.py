import os
import subprocess
from pathlib import Path

import pytest


SCRIPTS = [
    ("chunk_pdf.py", ["--help"], 0, "usage"),
    ("detect_duplicates.py", [], 1, "usage"),
]


@pytest.mark.parametrize("script,args,code,keyword", SCRIPTS)
def test_cli_invocation(script, args, code, keyword):
    result = subprocess.run(
        ["python", str(Path("scripts") / script), *args],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "."},
    )
    assert result.returncode == code
    output = (result.stdout + result.stderr).lower()
    assert keyword in output


def test_validate_chunks_sh_errors_on_empty(tmp_path):
    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("")
    result = subprocess.run(
        ["bash", "scripts/validate_chunks.sh", str(empty_file)],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "."},
    )
    assert result.returncode != 0
    assert "empty" in result.stderr.lower()
