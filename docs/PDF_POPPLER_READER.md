# Explicit native-text Poppler PDF reader (0.1.8 candidate)

Set `PDF_CHUNKER_PDF_READER=pdftotext` only for an explicitly selected PDF.
Unset it for the existing MuPDF reader, fallbacks and optional serial/three-core
OCR. An empty, whitespace, misspelled or other reader value is an error.

The adapter uses `pdfinfo` with `LC_ALL=C`, `LANG=C`, a positive integer timeout
(default60 seconds), and one positive page count (at most the PDF integer limit,
2147483647). It then calls the existing
`fallbacks._extract_with_pdftotext` / `run_pdftotext` with the existing
`pdftotext -layout ABSOLUTE_ORIGINAL_PATH -` arguments and timeout. The absolute
argument prevents leading-dash option confusion; returned provenance retains the
adapter's original `str(Path(path))`, original filename, one-based page and method
`pdftotext`. There is no repaired copy, new dependency or automatic format change.
The source suffix must be `.pdf` (case-insensitive); pdfinfo must accept it as PDF.

No MuPDF API, primary extractor, enhancement, OCR engine, rendering or worker pool
is called in this mode. Module imports are allowed. Nonempty OCR language or CPU
settings, enhancement, interactive extraction, or any specified geometric margin
(including zero) refuse before subprocess execution. Poppler text has no bounding
boxes; it cannot honestly provide those features. Do not enable this mode globally
or combine it with the ongoing ordinary OCR workload.

The existing fallback's raw-text quality threshold0.7, selected-page filtering and
artifact-line cleanup remain in use. The new mode also requires exactly the
pdfinfo number of raw page slots, each terminated by Poppler's form feed, BEFORE
exclusions/cleanup. Missing/extra slots or a missing final terminator fail; data
is never discarded to force acceptance. Nonempty selected body is required.
Every selected slot is returned, including genuinely blank pages. The normal
quality report names the explicit reader and reports text-bearing page coverage
and missing-body pages; blank slots are not invented text. Returned blocks with
invalid/out-of-range/excluded page numbers or wrong source method/filename refuse.
Existing exclusions remain one-based with ranges/sequences; out-of-range exclusion
numbers have no effect, as before. Selecting no pages returns an empty page list
after pdfinfo without invoking pdftotext. Selecting only blank pages fails.

Without `expected_page_count`, the shared fallback's previous behavior is retained
byte for byte. No OCR implementation, CPU configuration, pipeline, caller,
enrichment, indexing or default-reader behavior is changed by this candidate.

## Verification

Run tests/io_pdf_poppler_test.py with the existing supported environment. Generated
fixtures cover page slots, reading order, exclusions, provenance and explicit errors.
The optional source adapter test requires all three PDF_CHUNKER_POPPLER_TEST_PATH,
PDF_CHUNKER_POPPLER_TEST_SHA256 and PDF_CHUNKER_POPPLER_TEST_PAGES pins. Absent pins
skip; partial, invalid or changed pins fail. No real document path is searched by
default. Optional source checks do not authorize downstream publication.
