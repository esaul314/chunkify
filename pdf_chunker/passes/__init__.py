"""Lazy pass accessors that avoid shadowing submodules."""

from importlib import import_module
from typing import Any

_PASS_MODULES = [
    "ai_enrich",
    "emit_jsonl",
    "extraction_fallback",
    "heading_detect",
    "list_detect",
    "detect_doc_end",
    "merge_footers",
    "detect_page_artifacts",
    "pdf_parse",
    "epub_parse",
    "split_semantic",
    "text_clean",
]

# Import submodules for registration side effects without polluting the package
# namespace. This keeps ``pdf_chunker.passes.<module>`` importable while ensuring
# each pass registers itself with the framework.
for _mod in _PASS_MODULES:  # pragma: no cover - import side effects only
    import_module(f".{_mod}", __name__)

__all__ = list(_PASS_MODULES)


def __getattr__(name: str) -> Any:  # pragma: no cover - simple delegation
    if name in __all__:
        module = import_module(f".{name}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
