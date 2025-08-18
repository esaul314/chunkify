from . import ai_enrich, emit_jsonl, io_pdf

try:  # pragma: no cover - optional dependency
    from . import io_epub  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    io_epub = None  # type: ignore

from .io_epub import describe_epub, read_epub  # noqa: F401

__all__ = ["ai_enrich", "emit_jsonl", "io_epub", "io_pdf"]
__all__ += ["describe_epub", "read_epub"]
