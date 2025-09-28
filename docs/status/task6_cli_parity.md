# Task 6 — Golden Conversion & CLI Parity

## Summary
- ✅ Regenerated `platform-eng-excerpt.pdf` via the Typer CLI with enrichment disabled and compared the JSONL rows against the HEAD^ baseline.
- ✅ Output text content remained identical across all 65 chunks; only list-related metadata changed, reflecting intentional detection improvements.
- ⚠️ EPUB smoke test exposed stale expectations: the CLI now emits two list-aware chunks for `sample.epub`, while the golden fixture still contains a single paragraph entry.

## Evidence
- `python -m pdf_chunker.cli convert platform-eng-excerpt.pdf --spec pipeline.yaml --out /tmp/platform-eng-new.jsonl --no-enrich` (current branch) — CLI completed with warnings only.【863a4e†L1-L19】
- `python -m pdf_chunker.cli convert platform-eng-excerpt.pdf --spec pipeline.yaml --out /tmp/platform-eng-old.jsonl --no-enrich` (HEAD^ via temporary worktree) — established the comparison baseline.【01b952†L1-L2】
- Canonical diff script confirmed four metadata-only deltas (rows 2, 8, 15, 61).【c79967†L23-L61】
- `pytest tests/golden/test_conversion_epub_cli.py::test_conversion_epub_cli` failed because the expected EPUB golden file has not been updated for the second list chunk.【291842†L1-L47】

## Diff Details

| Row | Chunk ID | Page | Metadata Change |
| --- | -------- | ---- | ---------------- |
| 2 | `platform-eng-excerpt.pdf_p3_c9` | 4 | `block_type` flipped from `paragraph` to `list_item`; `list_kind="styled"` now present for the table-of-contents style list.【c79967†L23-L37】
| 8 | `platform-eng-excerpt.pdf_p15_c31` | 16 | Role list now tagged as `list_item` with `list_kind="styled"`, matching the enumerated responsibilities.【c79967†L38-L44】
| 15 | `platform-eng-excerpt.pdf_p21_c47` | 22 | Section bullet headings carry the `list_item` classification and styled list metadata.【c79967†L45-L51】
| 61 | `platform-eng-excerpt.pdf_p52_c134` | 53 | Strategy bullet points receive `list_item` and `list_kind="styled"` metadata while preserving text.【c79967†L52-L61】

No text bodies differed, and all updated chunks retained their original pagination, readability scores, and chunk identifiers.

## Completion Report
- **Outcome:** CLI parity holds for canonical PDF fixtures; list-aware metadata enrichments are the only differences versus the HEAD^ baseline.
- **Ancillary Findings:** EPUB CLI regression uses a single-chunk golden file, but the converter now emits two list-aware chunks, suggesting the golden expectation should be refreshed alongside list metadata work.【291842†L23-L47】
- **Regression Guidance:** When altering list detection, rerun the PDF conversion above and `pytest tests/golden/test_conversion_epub_cli.py::test_conversion_epub_cli` after updating the EPUB golden fixture to ensure both PDF and EPUB pipelines stay aligned.
