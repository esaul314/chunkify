from __future__ import annotations

import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from subprocess import run

import pytest

from scripts import parity as sp


def pytest_sessionstart(session: pytest.Session) -> None:
    os.environ.update({"PDF_CHUNKER_ENRICH": "0", "AI_ENRICH_ENABLED": "0"})


@pytest.fixture(scope="session")
def classify_stub() -> Callable[[str], dict[str, object]]:
    return lambda text, *, tag_configs=None, completion_fn=None: {
        "classification": "none",
        "tags": [],
    }


@pytest.fixture(autouse=True)
def _patch_legacy_llm(
    monkeypatch: pytest.MonkeyPatch, classify_stub: Callable[[str], dict[str, object]]
) -> None:
    try:  # pragma: no cover - legacy optional
        import pdf_chunker.ai_enrichment as legacy
    except Exception:
        return
    monkeypatch.setattr(legacy, "init_llm", lambda: classify_stub)
    monkeypatch.setattr(legacy, "classify_chunk_utterance", classify_stub)


@pytest.fixture(autouse=True)
def _patch_run_new(monkeypatch: pytest.MonkeyPatch) -> None:
    def _run(pdf: Path, out: Path, flags: Sequence[str] = ()) -> Path:
        run(
            [
                sys.executable,
                "-m",
                "pdf_chunker.cli",
                "convert",
                str(pdf),
                *flags,
                "--no-enrich",
                "--out",
                str(out),
            ],
            check=True,
        )
        return out

    monkeypatch.setattr(sp, "_run_new", _run)
