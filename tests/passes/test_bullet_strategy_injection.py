from __future__ import annotations

from pdf_chunker.framework import Artifact
from pdf_chunker.passes.emit_jsonl import _EmitJsonlPass
from pdf_chunker.passes.split_semantic import make_splitter
from pdf_chunker.strategies.bullets import BulletHeuristicStrategy


class RecordingStrategy(BulletHeuristicStrategy):
    """Record bullet heuristic lookups while preserving base behaviour."""

    def __init__(self) -> None:
        super().__init__(bullet_chars="~")
        object.__setattr__(self, "calls", [])

    def starts_with_bullet(self, text: str) -> bool:  # type: ignore[override]
        calls = object.__getattribute__(self, "calls")
        calls.append(("bullet", text))
        return super().starts_with_bullet(text)

    def starts_with_number(self, text: str) -> bool:  # type: ignore[override]
        calls = object.__getattribute__(self, "calls")
        calls.append(("number", text))
        return super().starts_with_number(text)


def _page_blocks_doc() -> dict[str, object]:
    block = {
        "text": "~ custom choice\n~ final selection",
        "source": {"page": 1, "filename": "doc.pdf"},
    }
    intro = {"text": "Options:", "source": {"page": 1, "filename": "doc.pdf"}}
    closing = {
        "text": "Detailed explanation continues.",
        "source": {"page": 1, "filename": "doc.pdf"},
    }
    return {
        "type": "page_blocks",
        "source_path": "doc.pdf",
        "pages": [
            {"page": 1, "blocks": [intro, block, closing]},
        ],
    }


def test_strategy_reused_across_split_and_emit_passes() -> None:
    strategy = RecordingStrategy()
    splitter = make_splitter(bullet_strategy=strategy)

    split_artifact = splitter(Artifact(payload=_page_blocks_doc()))
    chunks = split_artifact.payload["items"]
    assert chunks

    calls_after_split = list(strategy.calls)
    assert any("custom choice" in text for _, text in calls_after_split)

    emitter = _EmitJsonlPass()
    emitter.bullet_strategy = strategy

    emitted = emitter(Artifact(payload=split_artifact.payload)).payload
    assert emitted
    assert len(strategy.calls) > len(calls_after_split)
