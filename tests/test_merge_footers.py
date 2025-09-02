from pdf_chunker.framework import Artifact
from pdf_chunker.passes.merge_footers import merge_footers


def _build_doc():
    return {
        "type": "page_blocks",
        "pages": [
            {
                "page": 1,
                "blocks": [
                    {
                        "text": "This intro paragraph is long enough to prevent footer merging.",  # noqa: E501
                    },
                    {"text": "Footer line one"},
                    {"text": "Footer line two"},
                ],
            }
        ],
    }


def test_footer_lines_merged():
    doc = _build_doc()
    result = merge_footers(Artifact(payload=doc))
    blocks = result.payload["pages"][0]["blocks"]
    assert len(blocks) == 2
    assert blocks[-1]["text"] == "Footer line one Footer line two"
    assert result.meta["metrics"]["merge_footers"]["merged_lines"] == 1
