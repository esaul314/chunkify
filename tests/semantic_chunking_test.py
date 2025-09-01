from pdf_chunker.framework import Artifact
from pdf_chunker.passes.split_semantic import _SplitSemanticPass


def _doc(text: str) -> dict:
    return {
        "type": "page_blocks",
        "pages": [{"page": 1, "blocks": [{"text": text}]}],
    }


def test_limits_and_metrics() -> None:
    """Chunks obey soft/hard limits and expose metrics."""
    text = "x" * 26_000  # exceeds hard limit
    art = _SplitSemanticPass(chunk_size=100_000, overlap=0)(Artifact(payload=_doc(text)))
    chunk = art.payload["items"][0]["text"]
    metrics = art.meta["metrics"]["split_semantic"]
    assert len(chunk) == 8_000
    assert metrics["hard_limit_hit"] and metrics["soft_limit_hits"] == 1


def test_parameter_propagation() -> None:
    """Custom chunk sizing parameters propagate to the splitter."""
    words = " ".join(f"w{i}" for i in range(20))
    art = _SplitSemanticPass(chunk_size=5, overlap=1, min_chunk_size=2)(
        Artifact(payload=_doc(words))
    )
    texts = [c["text"] for c in art.payload["items"]]
    counts = [len(t.split()) for t in texts]
    assert counts == [5, 5, 5, 5, 4]
    assert texts[1].split()[0] == "w4"
