# Migration Map – Pass Integration

| Existing module/function | Target pass | Current I/O | Target Artifact I/O | Side effects → adapters | Gaps/unknowns |
|---|---|---|---|---|---|
| `pdf_parsing.extract_text_blocks_from_pdf` | `pdf_parse` | in: `filepath str`, `exclude_pages?` → out: `list[block dict]` | `Artifact(PDFPath)` → `Artifact(list[Block])` | reads PDF via `fitz`, env `PDF_CHUNKER_USE_PYMUPDF4LLM` | none |
| `text_cleaning.clean_text` | `text_clean` | in/out: `str` | `Artifact(str)` → `Artifact(str)` | env check for PyMuPDF4LLM | preview/logging boundaries |
| `heading_detection.enhance_blocks_with_heading_metadata` | `heading_detect` | in: `list[block]` → out: `list[block w/heading]` | `Artifact(list[Block])` → same | none | config for heading levels |
| `splitter._split_text_into_chunks` | `split_semantic` | in: `text str` → out: `list[str]` chunks | `Artifact(str)` → `Artifact(list[str])` | optional LangChain import | chunk metadata shape |
| `extraction_fallbacks.execute_fallback_extraction` | `extraction_fallbacks` | in: `filepath str`, `exclude_pages?` → out: `list[block]` | `Artifact(PDFPath)` → `Artifact(list[Block])` | pdftotext/pdfminer subprocess, file reads | metrics contract |
| `ai_enrichment.classify_chunk_utterance` | `ai_enrich` | in: `chunk str` + LLM → out: `tags dict` | `Artifact(str)` → `Artifact(dict)` | external LLM call | token usage accounting |
| `list_detection.is_bullet_list_pair` et al. | `list_detect` | in: `str` pairs → `bool` | `Artifact(list[str])` → `Artifact(list[str]/bool)` | none | final list metadata API |

## Adapter Boundaries

- `pdf_chunker/adapters/io_pdf.py`: open PDFs, apply exclusions, run pdftotext/pdfminer as fallbacks.
- `pdf_chunker/adapters/io_epub.py`: read EPUB contents and spine, honor page/section exclusions.
- `pdf_chunker/adapters/emit_jsonl.py`: write chunk artifacts to JSONL.
