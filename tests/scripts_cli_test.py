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
