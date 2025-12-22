from importlib import import_module
from typing import Any

from . import ai_enrich, emit_jsonl, io_pdf


def _load_epub() -> Any:
    try:  # pragma: no cover - optional dependency
        return import_module(".io_epub", __name__)
    except ModuleNotFoundError:  # pragma: no cover
        return None


io_epub = _load_epub()


def _missing(*_: Any, **__: Any) -> None:
    raise ModuleNotFoundError("ebooklib is required for EPUB support")


describe_epub, read_epub = (
    (io_epub.describe_epub, io_epub.read_epub) if io_epub else (_missing, _missing)
)

__all__ = ["ai_enrich", "emit_jsonl", "io_epub", "io_pdf", "describe_epub", "read_epub"]
