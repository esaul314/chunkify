# Article conversion descriptors

P1-E now exposes an opt-in library contract for comparing article conversion
inputs. It separates logical note references, content revisions, physical sources
and conversion settings. It does not change `convert()`, emitted chunk IDs or
JSONL, and it does not execute a cache, write receipts or update a vector store.

## Verified adapter usage

Use the existing approved binding described in [VERIFIED_MARKDOWN.md](VERIFIED_MARKDOWN.md).
The adapter re-verifies source, sidecar, final approval and original captures on
every call, then passes that snapshot to the pure descriptor builder:

```python
from functools import partial

from pdf_chunker.adapters.io_joplin_epub import describe_bound_article
from pdf_chunker.conversion_descriptor import compare_conversion

contract = {
    "implementation": pinned_converter_revision,
    "normalization": "approved-markdown-utf8-v1",
    "chunk_schema": "canonical-v1/contiguous_markdown_v3",
    "serialization": pinned_jsonl_serialization_version,
    "configuration": resolved_conversion_configuration,
}
describe = partial(describe_bound_article, corpus="technology", contract=contract)
current = describe(binding)
change = compare_conversion(current, previous=previous_descriptor)
print(change.status, change.changed_fields)
```

`binding`, `previous_descriptor`, version pins and resolved configuration are
supplied by the caller. For a first observation, omit `previous` or pass `None`.
The coordinator owns matching a physical source across runs; a path is not the
logical article identity.

The contract requires exactly the four nonempty version strings shown above and
a configuration mapping. The caller must resolve the effective ordered pipeline,
all pass defaults/options, relevant environment policies, dependency versions and
implementation revisions that can change base rows. A raw `PipelineSpec` with
omitted defaults is insufficient for safe artifact reuse. Version names are
declared pins, not automatically collected or verified implementation hashes.
The current API checks structure and JSON values, not completeness of that claim.

Configuration, metadata and reference provenance are copied into canonical JSON
strings in frozen records. There is no implicit `str()`/`repr()` fallback for
clients, callbacks or other opaque values. String mapping keys are mandatory;
nonfinite numbers and unpaired Unicode surrogates are refused. Mapping insertion
order does not affect identity. Sequence order remains significant, and Unicode
and article text are never normalized by the descriptor builder.

## Identity and comparison

| Component | Meaning and invalidation |
| --- | --- |
| `content_sha256` | SHA-256 of exact semantic text encoded as UTF-8; independent of an EPUB envelope. |
| `source_path`, `source_format`, `source_sha256` | Exact physical source identity and raw-byte hash, retained separately from logical identity. |
| `contract_json` | Snapshot of supplied implementation, normalization, chunk schema, serialization and resolved settings. |
| `metadata_json` | Complete supplied native metadata, including evidence bindings that appear in rows. |
| `associations` | One sorted `NoteAssociation` per logical reference, with content revision, corpus and full per-note provenance. |
| `key` | SHA-256 of the UTF-8 descriptor JSON, with descriptor schema version 1. |

`to_json()` returns compact JSON with sorted keys and literal Unicode. Nested
configuration/metadata/provenance snapshots are JSON strings, explicitly named
`*_json`. Association keys also include schema version 1. A notebook, tag, URL,
capture or corpus change affects that note's association. Another note linked to
the same content keeps its own association; content equality does not merge away
references. A content revision updates every association for that content.

Verified Markdown and EPUB for the same article share logical references,
content revision and association keys. Their complete conversion keys differ:
the current converter includes physical filenames and bindings in chunk IDs and
payloads. Moving a source or repackaging an EPUB therefore conservatively changes
the conversion descriptor without creating a new logical note. This contract
does not promise cross-format chunk IDs or reuse of physical payloads.

`compare_conversion(current, previous=...)` is pure and data-first. It returns
`new`, `changed` or `unchanged`, plus changed descriptor fields in declaration
order. It issues no deletion or reference-removal instructions, including when a
reference is absent from the next snapshot. No earlier snapshot or input mapping
is modified. Pairing descriptors from unrelated sources remains a caller error.

An unchanged descriptor identifies equal declared inputs; it is not proof that a
cached artifact exists, reconstructs the source, fits a budget or finished
successfully. Evidence must still be checked, and any future reuse must validate
the output's own hash, schema, reconstruction and terminal completion. Enrichment
needs its own model/prompt/vocabulary/serving contract and output validation.

These snapshots retain full provenance and local evidence paths. Keep generated
descriptors in the private caller-owned artifact scope. The publication privacy
gate applies before sharing them; do not put credentials into configuration.

## Verification and remaining work

```bash
pytest tests/conversion_descriptor_test.py tests/verified_markdown_ingress_test.py \
  tests/joplin_markdown_preservation_test.py tests/text_adapter_test.py
nox -s lint typecheck tests
```

Synthetic tests cover Markdown/EPUB identity, a verified two-note metadata update,
actual conversion retaining both references, tampered evidence, exact Unicode,
mapping-order laws, detached immutable snapshots, partial application and an
unchanged rerun after another article. No source admission or model call occurs.

Next work must connect effective runner settings/version resolution to descriptors
and qualify validated base/enrichment artifact reuse. A persisted descriptor reader,
batch execution, receipts, UTF-8 row preflight, document-text/HTML inputs and the
downstream idempotency pilot remain open. P0-A and P2-D remain open as recorded in
the [implementation roadmap](IMPLEMENTATION_ROADMAP.md).
