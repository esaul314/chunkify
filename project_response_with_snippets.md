1. To run the entire pipeline:
- Here is an example of how the legacy script could be run: `python -m scripts.chunk_pdf --no-metadata --exclude-pages "1-4,31-35" document.pdf > document.jsonl`

2. Canonical data shapes actually used today
- The TypedDict `TextBlock` models EPUB segments with `type`, `text`, `language`, and `source` fields
```python
class TextBlock(TypedDict):
    """Structured representation of extracted text blocks."""

    type: str
    text: str
    language: str
    source: Dict[str, str]
```

- PDF parsing emits dictionaries shaped like `{ "type": ..., "text": ..., "language": ..., "source": { "filename": ..., "page": ..., "location": ... }, "bbox": ... }`
```python
structured.append(
    {
        "type": block_type,
        "text": block_text,
        "language": default_language(),
        "source": {"filename": filename, "page": page_num, "location": None},
        "bbox": b[:4],
    }
)
```

- Character mapping relies on the frozen dataclass `CharSpan(start, end, original_index)`
```python
@dataclass(frozen=True)
class CharSpan:
    start: int
    end: int
    original_index: int
```

- Final chunk rows are `{ "text": ..., "metadata": { "source": ..., "chunk_id": ..., "page": ..., "location": ..., "block_type": ..., "language": ..., "readability": ..., "utterance_type": ..., "importance": ..., "list_kind": ... } }`
```python
result = {"text": final_text, "metadata": metadata}
```

- Validation results use `ValidationReport(total_chunks, empty_text, mid_sentence_starts, overlong, duplications, boundary_overlaps)`
```python
@dataclass(frozen=True)
class ValidationReport:
    total_chunks: int
    empty_text: int
    mid_sentence_starts: int
    overlong: int
    duplications: List[Dict[str, Any]]
    boundary_overlaps: List[Dict[str, Any]]
```


3. Exact module paths + functions for each stage
- Parser (PDF): `pdf_chunker/pdf_parsing.py::extract_text_blocks_from_pdf`

- Parser (EPUB): `pdf_chunker/epub_parsing.py::extract_text_blocks_from_epub`

- Text cleaning: `pdf_chunker/text_cleaning.py::clean_paragraph`, `clean_text`

- Heading detection: `pdf_chunker/heading_detection.py::_detect_heading_fallback`

- List detection: `pdf_chunker/list_detection.py::starts_with_bullet`, `is_bullet_list_pair`, `is_numbered_continuation`, etc.

- Splitter/semantic chunker: `pdf_chunker/splitter.py::semantic_chunker`

- Extraction fallbacks: `_extract_with_pdftotext`, `_extract_with_pdfminer` in `pdf_chunker/extraction_fallbacks.py`

- AI enrichment/tagging: `_load_tag_configs`, `init_llm`, `classify_chunk_utterance` in `pdf_chunker/ai_enrichment.py`

- JSONL writer: `scripts/chunk_pdf.py::main` prints JSON lines to stdout


4. Side effects / IO boundaries
- `extract_text_blocks_from_pdf` opens files via `fitz.open`

- `extract_text_blocks_from_epub` uses `epub.read_epub` to read archives

- `_extract_with_pdftotext` shells out using `subprocess.run`

- `litellm.completion` performs network calls for enrichment

- `chunk_pdf.main` writes JSONL by printing serialized chunks


5. Config sources
- Env vars: `PDF_CHUNKER_USE_PYMUPDF4LLM` for PyMuPDF4LLM features; `OPENAI_API_KEY` for enrichment

- YAML: tag vocabularies under `config/tags` loaded by `_load_tag_configs`

- CLI flags: `--chunk_size` (default 400), `--overlap` (50), `--exclude-pages`, `--no-metadata`, `--list-spines` in `chunk_pdf.py`

- Defaults: `min_chunk_size` defaults to `max(8, chunk_size // 10)`; `_truncate_chunk` uses an 8 kB soft limit


6. Known edge cases you care about
- Footnotes stripped to avoid mid‑sentence splits

- Header/footer artifacts removed, including trailing “|” fragments

- Hyphenation and bullet list fixes ensure proper joins and list integrity

- Underscore emphasis removal and list metadata propagation from PyMuPDF4LLM

- Cross‑page paragraph merges and comma continuation fixes prevent orphaned fragments


7. Representative small PDFs/EPUBs
- Root samples: `sample_book-bullets.pdf`, `sample_book-quote.pdf`, `sample_book-footer.pdf`, `sample_local-pdf.pdf`

- Unit-test fixture: `test_data/sample_test.pdf`

- `generate_test_pdf.py` / `generate_test_epub.py` scripts produce additional synthetic files


8. Current tests that must stay green
- `pdf_extraction_test.py`, `ai_enrichment_test.py`, `semantic_chunking_test.py`, `page_exclusion_test.py`, `epub_spine_test.py`, `process_document_override_test.py`, `env_utils_test.py`, `list_detection_edge_case_test.py`, `page_artifact_detection_test.py`, `page_artifacts_edge_case_test.py`, `footer_artifact_test.py`, `artifact_block_test.py`, `scripts_cli_test.py`, `splitter_transform_test.py`, `text_cleaning_transform_test.py`


9. Performance constraints
- Semantic pass enforces an 8 k character soft limit with 25 k hard truncation

- `_truncate_chunk` trims text beyond 8 k characters and validation flags overlong chunks at that threshold

- Default chunk targets: 400 chars with 50-char overlap via CLI; minimum chunk size defaults to `max(8, chunk_size // 10)`

- `pdftotext` fallback times out after 60 s per call

- LLM enrichment requests cap at 100 tokens per completion
