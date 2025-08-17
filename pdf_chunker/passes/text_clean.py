from __future__ import annotations

from pdf_chunker.framework import Artifact, register


def clean_text(text: str) -> str:
    """Pure text normalization wrapper.

    Uses the heavy text_cleaning implementation only when invoked to keep
    import times lean.
    """
    from pdf_chunker.text_cleaning import _clean_text_impl

    return _clean_text_impl(text)


class _TextCleanPass:
    name = "text_clean"
    input_type = str
    output_type = str

    def __call__(self, a: Artifact) -> Artifact:
        result = clean_text(a.payload)
        meta = dict(a.meta or {})
        meta.setdefault("metrics", {}).setdefault("text_clean", {})["normalized"] = True
        return Artifact(payload=result, meta=meta)


text_clean = register(_TextCleanPass())
