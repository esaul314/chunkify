"""Generate invented PDF controls without external tools or document excerpts.

The generated content is dedicated to the public domain under CC0-1.0.
Only explicit output paths are written; run from the project root:
python scripts/generate_structure_fixtures.py --output-directory NEW_DIRECTORY
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _literal(text: str) -> bytes:
    return text.encode("cp1252").replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")


def pdf_bytes(pages: list[list[tuple[str, int, bool]]]) -> bytes:
    """Produce deterministic text PDFs with standard fonts and no attachments."""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        (
            "<< /Type /Pages /Kids ["
            + " ".join(f"{5 + 2*i} 0 R" for i in range(len(pages)))
            + f"] /Count {len(pages)} >>"
        ).encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>",
    ]
    for index, lines in enumerate(pages):
        page_id = 5 + 2 * index
        objects.append(
            (
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                "/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> "
                f"/Contents {page_id+1} 0 R >>"
            ).encode()
        )
        content = []
        y = 720
        for text, size, bold in lines:
            font = "F2" if bold else "F1"
            content.append(
                f"BT /{font} {size} Tf 72 {y} Td (".encode() + _literal(text) + b") Tj ET"
            )
            y -= size + 10
        stream = b"\n".join(content)
        objects.append(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream")
    raw = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, value in enumerate(objects, 1):
        offsets.append(len(raw))
        raw.extend(f"{number} 0 obj\n".encode() + value + b"\nendobj\n")
    start = len(raw)
    raw.extend(f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        raw.extend(f"{offset:010d} 00000 n \n".encode())
    raw.extend(
        f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n".encode()
    )
    return bytes(raw)


def structure_pages() -> list[list[tuple[str, int, bool]]]:
    pages = []
    for page in range(1, 56):
        marker = "marker" + chr(65 + page // 26) + chr(65 + page % 26)
        pages.append(
            [
                (f"Example section {marker}", 16, True),
                (f"This invented paragraph belongs to example {marker}.", 11, False),
                (
                    f"Readers compare {marker} records and retain complete source locations.",
                    11,
                    False,
                ),
                (
                    f"Each {marker} example explains a useful boundary using invented text.",
                    11,
                    False,
                ),
                (
                    f"The {marker} review checks content, lists, captions and source page counts.",
                    11,
                    False,
                ),
            ]
        )
    pages[0] = [
        ("CHAPTER 1: Invented structure controls", 16, True),
        ("Why Example Systems Need Structure", 15, True),
        ("A complete generated example begins with a clear source description.", 11, False),
        ("The layout is seen in Figure 1-1.", 11, False),
        ("Figure 1-1. A synthetic diagram showing ordered records", 10, False),
        ("The explanation continues after the caption without losing its label.", 11, False),
    ]
    pages[1].extend(
        [
            ("Table 1-1. Example review decisions", 11, True),
            ("Record       Status       Action", 11, False),
            ("First        Complete     Retain", 11, False),
            ("Second       Pending      Review", 11, False),
        ]
    )
    pages[2] = [
        ("Review choices", 16, True),
        ("The workshop records these choices:", 11, False),
        ("1. Most participants prefer familiar tools for occasional workshop tasks.", 11, False),
        ("Infrastructure examples remain separate from numbered instructions.", 11, False),
        ("2. Reviewers retain useful explanations and exact original locations.", 11, False),
    ]
    pages[3] = [
        ("How Example Systems Keep Records Clear", 15, True),
        ("A useful description provides enough context to explain each step.", 11, False),
        ("The reader follows the same order from source pages to emitted records.", 11, False),
        ("Every record preserves complete words and the labels attached to them.", 11, False),
    ]
    pages[4] = [
        ("Reviewing an example", 16, True),
        ("A patient reader checks the full example before drawing conclusions.", 11, False),
        ("But building examples takes careful review and consistent source records.", 11, False),
        ("The preceding context must remain available with this continuation.", 11, False),
    ]
    pages[5] = [
        ("Context and boundaries", 16, True),
        ("The example follows a clear sequence:", 11, False),
        ("It only works because every step retains context.", 11, False),
        ("A reviewer compares records and confirms their original order.", 11, False),
    ]
    pages[6] = [
        ("Reuse", 15, True),
        ("Core to the example is a stable sequence of independently useful records.", 11, False),
        ("• Retain complete source text", 11, False),
        ("• Review uncertain boundaries", 11, False),
        ("• Keep captions beside their callouts", 11, False),
    ]
    pages[7] = [
        ("Final Review", 15, True),
        ("The team records ownership of maintaining the example's documents.", 11, False),
        ("Independent checks compare the complete ordered output with its source.", 11, False),
        ("A short useful passage remains eligible even when its section is uncertain.", 11, False),
    ]
    pages[8] = [
        ("Record", 15, True),
        ("We use an invented definition of an ordered record.", 11, False),
        ("Each record retains its original source and useful surrounding context.", 11, False),
    ]
    pages[9] = [
        ("Composability", 16, True),
        ("Treat records as independently useful.", 11, False),
        ("Records stay composable.", 11, False),
    ]
    pages[-1].append(
        ("Synthetic cobalt markers remain visible throughout this generated document.", 11, False)
    )
    return pages


def fixtures() -> dict[str, bytes]:
    return {
        "synthetic-structure.pdf": pdf_bytes(structure_pages()),
        "synthetic-bullets.pdf": pdf_bytes(
            [
                [
                    ("Invented review list", 15, True),
                    ("• Preserve complete words", 11, False),
                    ("• Retain useful context", 11, False),
                    ("• Review uncertain sections", 11, False),
                ]
            ]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    directory = parser.parse_args().output_directory
    directory.mkdir(parents=True, exist_ok=False)
    for name, value in fixtures().items():
        (directory / name).write_bytes(value)


if __name__ == "__main__":
    main()
