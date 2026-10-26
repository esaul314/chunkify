"""Original generated crosspage failure: physical source identity precedes roles."""

import hashlib

import pytest

from pdf_chunker.config import PipelineSpec
from pdf_chunker.core_new import _input_artifact

BODY = "The following page contains substantive explanatory text. " * 8
CONTENTS = "Contents\nChapter One  1\nChapter Two  2\nChapter Three  3"


def original_crosspage_source(tmp_path, first=CONTENTS):
    fitz = pytest.importorskip("fitz")
    path = tmp_path / "synthetic-crosspage.pdf"
    with fitz.open() as document:
        for text in (first, "Introduction\n" + BODY):
            page = document.new_page()
            page.insert_textbox((72, 72, 523, 770), text, fontsize=11)
        document.save(path)
    return path


@pytest.mark.parametrize(
    "first", [CONTENTS, "Acknowledgments\nWe thank the reviewers and remain grateful."]
)
def test_structure_ingress_keeps_physical_pages_before_classification(tmp_path, first):
    path = original_crosspage_source(tmp_path, first)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    artifact = _input_artifact(
        str(path),
        PipelineSpec(options={"section_detect": {"enabled": True}}),
    )
    pages = artifact.payload["pages"]
    assert pages[1]["blocks"], "following-page body must not live in a preceding section"
    assert "substantive explanatory text" not in "\n".join(b["text"] for b in pages[0]["blocks"])
    assert "substantive explanatory text" in "\n".join(b["text"] for b in pages[1]["blocks"])
    assert all(b["source"]["page"] == p["page"] for p in pages for b in p["blocks"])
    assert all(b["source"]["raw_source"] for p in pages for b in p["blocks"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
