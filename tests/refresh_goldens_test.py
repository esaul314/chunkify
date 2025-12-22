from __future__ import annotations

import os
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
TARGET = BASE / "tests" / "golden" / "expected" / "tiny.jsonl"


def test_refresh_goldens_dry_run() -> None:
    before = TARGET.read_text(encoding="utf-8")
    cmd = ["python", "scripts/refresh_goldens.py"]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=BASE,
        env={**os.environ, "PYTHONPATH": str(BASE)},
    )
    after = TARGET.read_text(encoding="utf-8")
    assert result.returncode == 0
    assert result.stdout  # script emits diff or status
    assert before == after
