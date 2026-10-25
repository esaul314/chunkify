# Direct verified Markdown articles

The canonical converter accepts `.md` (case insensitive) for an article carrying
the existing Joplin evidence contract. This is the first P1-E increment. It removes
the EPUB packaging requirement for those articles while reusing the existing
verifier, contiguous Markdown splitter, enrichment admission and JSONL emitter.
Ordinary PDF, EPUB and transcript conversion retain their existing behavior.

## Evidence and usage

Prepare the same `joplin-article-v1` sidecar, final quality receipt and original
capture records used for a verified EPUB article. The sidecar's `source` and
`source_file` must name the Markdown file. The binding contains:

- `path` and `source_sha256`: the original Markdown file and its raw SHA-256.
- `sidecar_path` and `sidecar_sha256`: exactly one article row with `text` and
  `metadata`, including its content hash and complete note references.
- `quality_path` and `quality_sha256`: the final receipt approving every capture.
- `record_bindings`: original capture records, each with `path` and `sha256`.

Evidence paths must be absolute, regular bounded files under the existing verifier
rules. Each file is limited to 4 MiB. The upstream coordinator owns capture,
approval and corpus eligibility; conversion does not create an approval receipt.

Create the native metadata file with the existing adapter API, supplying the
prepared binding and its declared corpus:

```python
import json
from pathlib import Path

from pdf_chunker.adapters.io_joplin_epub import metadata

# binding is the complete evidence mapping described above, supplied upstream.
native = metadata(binding, "technology")
Path("native-metadata.json").write_text(
    json.dumps(native, ensure_ascii=False), encoding="utf-8"
)
```

Then convert through the existing CLI and pipeline spec:

```bash
pdf_chunker convert ./article.md --spec pipeline.yaml \
  --metadata-file ./native-metadata.json --out ./article.jsonl --no-enrich
```

Library callers use `pdf_chunker.convert(path, spec)` with
`options.text_parse.metadata_path` pointing to the same native metadata file.
The `pdf_parse` slot is replaced with `text_parse`, which only normalizes the
canonical page-block envelope. The transcript reader is never invoked for `.md`.
Missing proof, changed hashes, caller-altered metadata or incomplete approval
raises an error before semantic processing or enrichment.

## Preservation and identity

Direct source bytes must equal the UTF-8 encoding of the verified sidecar text.
The reader does not remove a BOM, change newlines, join paragraphs or strip cue
timings. Inputs must already match the approved canonical article; the existing
capture verifier still compares captured Markdown after CR/CRLF normalization and
exactly one final newline. A differing physical source is rejected, never rewritten.
JSON metadata files are parsed as JSON so escaped Unicode surrogate pairs decode
correctly; YAML metadata retains the existing YAML/date normalization policy.

The source hash and content hash are verified separately. Text and metadata are
returned from the same verified read, without reopening the EPUB to recover text.
EPUB input still requires its embedded article and provenance to match the sidecar.

Pure passes keep exact contiguous source slices, including headings, code, tables,
links, Unicode, indentation and meaningful line breaks. Slice offsets refer to
Python string characters; concatenating emitted `text` values and encoding as
UTF-8 reconstructs the admitted Markdown bytes. Existing list/reference context
rules remain in force. There is no overlap on this preservation path. PDF artifact
passes (`merge_footers`, `detect_page_artifacts`) are rejected for `.md` input.

Article title, URLs, content revision and all note references (including notebook,
tags and capture/archive/discussion provenance) pass through the existing verifier.
Generated location and chunk identity fields remain converter-owned. Logical note
references survive a format change; current chunk IDs still include the physical
source filename and are not promised to match between EPUB and Markdown. Exact
reruns produce identical rows and JSONL bytes even after another conversion.

The existing complete serialized-record admission still runs before enrichment.
Metadata that cannot fit is refused before calling the injected client. Existing
Markdown context safeguards can also hold oversized articles rather than split a
reference/list context incorrectly. The opt-in
`preflight_article_output(data, descriptor=..., expected_sha256=...,
max_row_bytes=..., enrichment_headroom_bytes=...)` now measures complete UTF-8
base rows against a declared downstream byte budget. It returns an immutable
report with all oversized rows for review, preserving context, text and provenance.
Callers must check `fits` before enrichment and validate actual enriched bytes
afterwards. This remains separate from the existing `max_chars` admission contract;
see [conversion descriptors](CONVERSION_DESCRIPTORS.md#utf-8-byte-preflight-before-enrichment).

## Verification and remaining work

```bash
pytest tests/verified_markdown_ingress_test.py \
  tests/joplin_markdown_preservation_test.py tests/text_adapter_test.py
nox -s lint typecheck tests
```

Tests generate synthetic evidence and exercise actual CLI output, EPUB parity,
exact reconstruction, unchanged reruns after another document, Unicode metadata,
tampering and refusal before enrichment. They use an injected in-process client;
no model service or upstream source admission is exercised.

The opt-in [conversion descriptors](CONVERSION_DESCRIPTORS.md) add
independent note associations, with fresh evidence verification and pure
new/changed/unchanged comparison. Base article preparation now resolves the
bounded canonical runner settings and fingerprints package sources/runtime versions
plus raw and loaded readability dictionaries. Inconsistent hyphenation caches
refuse preparation; normal warming preserves identity. The opt-in
`execute_bound_conversion(binding, prepared=prepared, corpus="technology")`
checks matching snapshots before/after synchronous conversion and validates base
JSONL hash, schema, provenance and complete reconstruction without writing files.
It returns only after normal completion. `complete_bound_conversion` also returns
canonical completion facts. Read-only `read_bound_conversion` re-verifies proof,
settings/runtime and persisted record/output bytes against an independently
retained completion hash; it never falls back to conversion or repairs files.
Atomic receipt writing and interrupted-batch recovery remain P2-D work.
Separate enrichment reuse and integrated byte admission remain open, alongside
document-text/HTML/supplied-text input and downstream integration qualification. P2-D receipts
and cleanup follow those contracts. Existing EPUB sources, releases and vector-store
contents are unchanged.
