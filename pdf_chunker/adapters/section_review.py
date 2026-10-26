"""Source-structure I/O and terminal review over the shared pure decision schema."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pdf_chunker.interactive import Decision
from pdf_chunker.sections import answer_from_decision


def source_structure(path: str, doc: dict[str, Any]) -> dict[str, Any]:
    """Read hash-verified original layout without changing extraction defaults."""
    raw = Path(path).read_bytes()
    proof: dict[str, Any] = {
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "outline": [],
        "pages": [],
    }
    if Path(path).suffix.lower() == ".pdf":
        import fitz

        with fitz.open(stream=raw, filetype="pdf") as pdf:
            proof["outline"] = [
                {"level": level, "title": title, "page": page}
                for level, title, page in pdf.get_toc()
                if page > 0
            ]
            for number, page in enumerate(pdf, 1):
                layout = page.get_text("dict")
                physical = {block[5]: block for block in page.get_text("blocks") if block[6] == 0}
                blocks = [
                    {
                        "index": block.get("number", index),
                        "raw_text": physical.get(block.get("number", index), [None] * 5)[4],
                        "bbox": list(block["bbox"]),
                        "text": "\n".join(
                            "".join(s["text"] for s in line.get("spans", ()))
                            for line in block.get("lines", ())
                        ),
                        "fonts": sorted(
                            {
                                (s.get("font"), s.get("size"))
                                for line in block.get("lines", ())
                                for s in line.get("spans", ())
                            }
                        ),
                    }
                    for index, block in enumerate(layout.get("blocks", ()))
                    if block.get("type") == 0
                ]
                annotations = [
                    {
                        "xref": annotation.xref,
                        "type": annotation.type[1],
                        "bbox": list(annotation.rect),
                        "text": str(annotation.info.get("content") or ""),
                    }
                    for annotation in page.annots() or ()
                ]
                proof["pages"].append(
                    {
                        "page": number,
                        "height": page.rect.height,
                        "text": page.get_text(),
                        "blocks": blocks,
                        "annotations": annotations,
                        "image_count": len(page.get_images()),
                    }
                )
    else:
        # ponytail: existing EPUB navigation is already carried by extraction;
        # do not introduce a second EPUB reader or text normalization.
        for page in doc.get("pages", ()):
            proof["pages"].append(
                {
                    "page": page.get("page"),
                    "text": "\n".join(str(b.get("text", "")) for b in page.get("blocks", ())),
                }
            )
            for block in page.get("blocks", ()):
                source = block.get("source") or {}
                title = source.get("section_title") or source.get("heading")
                if title:
                    proof["outline"].append({"page": page.get("page"), "title": title})
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != proof["source_sha256"]:
        raise ValueError("source changed while reading section structure")
    return proof


def terminal_review(context: Any) -> Decision:
    """Terminal callback; unanswered questions remain pending and no global learning."""
    q = context.extra or {}
    print(json.dumps(q, indent=2, ensure_ascii=False))
    if q.get("kind") == "section_role":
        role = input(
            "Role (body/contents/acknowledgments/permissions/references/index/unknown), "
            "blank=pending: "
        ).strip()
        if not role:
            return Decision(action="skip")
        reason = input("Evidence/reason: ").strip()
        result = Decision(action="classify", section_role=role, origin="user", reason=reason)
    else:
        answer = input("Boundary starts here? y/n, blank=pending: ").strip().lower()
        if not answer:
            return Decision(action="skip")
        if answer not in {"y", "n"}:
            raise ValueError("boundary answer must be y or n")
        result = Decision(
            action="classify",
            section_boundary=answer == "y",
            origin="user",
            reason=input("Evidence/reason: ").strip(),
        )
    answer_from_decision(q, result)
    return result


def read_answers(path: str) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list) or any(not isinstance(a, dict) for a in data):
        raise ValueError("section answers must be a JSON array of complete bound answers")
    return data


def write_questions(path: str, questions: list[dict[str, Any]]) -> None:
    """Exclusive private machine-readable exchange; no silent overwrite."""
    import os

    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(questions, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())


def write_preservation(path: str, report: dict[str, Any]) -> None:
    """Exclusive private accounting sidecar, never an implicit approval."""
    import os

    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(report, stream, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
