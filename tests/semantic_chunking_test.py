from pdf_chunker.framework import Artifact
from pdf_chunker.passes.split_semantic import _SplitSemanticPass
import re


def _doc(text: str) -> dict:
    return {
        "type": "page_blocks",
        "pages": [{"page": 1, "blocks": [{"text": text}]}],
    }


def test_limits_and_metrics() -> None:
    """Chunks obey soft/hard limits and expose metrics."""
    text = "x" * 26_000
    splitter = _SplitSemanticPass(chunk_size=100_000, overlap=0)
    art = splitter(Artifact(payload=_doc(text)))
    chunks = [c["text"] for c in art.payload["items"]]
    metrics = art.meta["metrics"]["split_semantic"]
    assert len(chunks) > 1
    assert all(len(c) <= 8_000 for c in chunks)
    assert metrics["soft_limit_hits"] == 1


def test_parameter_propagation() -> None:
    """Custom chunk sizing parameters propagate to the splitter."""
    words = " ".join(f"w{i}" for i in range(20))
    opts = {
        "options": {
            "split_semantic": {
                "chunk_size": 5,
                "overlap": 1,
                "min_chunk_size": 2,
            }
        }
    }
    art = _SplitSemanticPass()(Artifact(payload=_doc(words), meta=opts))
    texts = [c["text"] for c in art.payload["items"]]
    counts = [len(t.split()) for t in texts]
    assert counts == [5, 5, 5, 5, 4]
    assert texts[1].split()[0] == "w4"


def test_no_chunk_starts_mid_sentence() -> None:
    """Chunks begin at sentence boundaries and never start mid-sentence."""
    end_re = re.compile(r"[.?!][\"')\]]*$")
    long_sentence = " ".join(f"w{i}" for i in range(120)) + "."
    text = f"{long_sentence} Next one."
    art = _SplitSemanticPass(chunk_size=10, overlap=0)(Artifact(payload=_doc(text)))
    chunks = [c["text"] for c in art.payload["items"]]
    assert all(end_re.search(prev.rstrip()) for prev in chunks[:-1])


def test_blocks_merge_into_sentence() -> None:
    """Adjacent blocks merge so chunks don't start mid-sentence."""
    doc = {
        "type": "page_blocks",
        "pages": [
            {
                "page": 1,
                "blocks": [
                    {"text": "Cloud"},
                    {"text": "development envs are new."},
                ],
            }
        ],
    }
    art = _SplitSemanticPass()(Artifact(payload=doc))
    texts = [c["text"] for c in art.payload["items"]]
    assert texts == ["Cloud development envs are new."]
