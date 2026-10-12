# EPUB body extraction

Container markup is traversed in reading order. Direct body text, div/span
containers and semantic paragraphs retain their content without duplicate wrapper
text. Lists and navigation keep their semantic metadata; script/style/noscript
content is excluded. Input DOM objects remain unchanged.

Run tests/epub_body_markup_test.py for generated regressions. Optional local-source
checks use explicitly supplied private controls, remain offline, verify source
hashes and preserve provenance and serialized-record limits. No actual source
names, document text or acceptance receipts belong in public source.
