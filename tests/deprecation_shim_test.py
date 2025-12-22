from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_chunk_pdf_delegates() -> None:
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root)}
    result = subprocess.run(
        [sys.executable, "-m", "scripts.chunk_pdf", "--help"],
        capture_output=True,
        text=True,
        env=env,
        cwd=root,
    )
    assert result.returncode == 0
    assert "deprecated" in result.stderr.lower()
    assert all(k in result.stdout for k in ("convert", "inspect"))
