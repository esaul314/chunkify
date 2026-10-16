# Opt-in native PDF OCR

Set PDF_CHUNKER_OCR_LANGUAGE to installed Tesseract language names, for example
eng or eng+fra. Unset or empty preserves ordinary extraction. The existing native
and fallback extraction runs first; only uncovered text-empty pages are recovered
with PyMuPDF at 200dpi. Page ranges prevent duplicate recovery. Blank pages stay
empty and engine failures remain explicit; no source PDF or replacement is written.

Blocks retain original path, filename, one-based page, bbox and pymupdf_ocr method.
Strict configured margins apply before adding blocks. Existing extraction metrics
remain separate from OCR attempted/recovered page, block and character counts.

Where supported, PDF_CHUNKER_OCR_CPUS accepts two or three distinct allowed CPU IDs
for bounded spawn workers; absent or blank retains serial processing. Select IDs
from the environment's allowed CPU set. Each worker opens its own PDF, uses one
OpenMP thread, retains the process group and yields results in source order.
Failures do not return partial success or silently fall back to serial extraction.

Use generated scan and blank-page tests in tests/io_pdf_ocr_test.py and ordered
payload/failure/affinity tests in tests/io_pdf_parallel_ocr_test.py. Optional source
checks require private controls and are not public fixture defaults. Installed OCR
language data is required; no download or installation occurs automatically.

OCR is limited to text-empty pages, adds no OCR-specific timeout, and does not
prove semantic completeness or downstream retrieval quality. Review original-page
coverage separately before publishing any derived artifact.
