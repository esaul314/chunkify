from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

_RUN_ID = uuid4().hex


def _path(step: str) -> Path:
    base = Path("artifacts") / "trace" / _RUN_ID
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{step}.json"


def write_snapshot(step: str, data: Any) -> None:
    """Persist ``data`` for ``step`` under a unique run directory."""
    _path(step).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
