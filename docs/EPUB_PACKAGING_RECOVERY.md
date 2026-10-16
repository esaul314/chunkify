# Reading books with broken optional EPUB resources

Set `PDF_CHUNKER_EPUB_RECOVERY=1` for a deliberate retry when EbookLib cannot
load an otherwise readable book because of missing CSS/images or malformed
navigation metadata. The default reader remains unchanged.

The recovery adapter reads the package's declared HTML bodies directly from
the original archive, first in spine order and then any remaining declared
HTML documents. Original body bytes, filenames and document locations remain
the inputs to the existing semantic extractor. No source archive is rewritten,
no external resource is fetched, and normal chunking/enrichment still apply.

Missing nontext resources and explicitly declared navigation are recorded in
the adapter's extraction-quality evidence. Missing body documents and unknown
reading-spine IDs fail explicitly. Recovery does not make a corrupt ZIP or a
book with absent body content complete.
Encrypted body resources also fail explicitly; the adapter does not decrypt
them or accept metadata/navigation as the book's body.

CPU regressions: `pytest tests/epub_packaging_recovery_test.py
tests/io_epub_adapter_test.py tests/epub_body_markup_test.py`. The first module
also uses two hash-pinned affected local books when they are present; it does
not enrich, publish or write a canonical dataset.
