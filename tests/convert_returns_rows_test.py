from pdf_chunker.config import PipelineSpec
from pdf_chunker.core_new import convert


def test_convert_returns_rows():
    spec = PipelineSpec(pipeline=["pdf_parse", "text_clean", "split_semantic"])
    rows = convert("sample-local-pdf.pdf", spec)
    assert rows, "convert should yield chunk rows"
