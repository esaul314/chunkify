# Article conversion descriptors

P1-E now exposes an opt-in library contract for comparing article conversion
inputs. It separates logical note references, content revisions, physical sources
and conversion settings. Base article preparation also resolves runner defaults
and fingerprints the installed implementation/runtime. It does not change
`convert()`, emitted chunk IDs or JSONL, and it does not execute a cache, write
receipts or update a vector store.

## Resolved base article preparation

Use this path when the existing canonical base article pipeline is sufficient:

```python
from functools import partial

from pdf_chunker import convert
from pdf_chunker.adapters.article_conversion import prepare_bound_conversion
from pdf_chunker.config import load_spec
from pdf_chunker.conversion_descriptor import compare_conversion

spec = load_spec("pipeline.yaml", overrides={
    "text_parse": {"metadata_path": "/private/run/native-metadata.json"},
})
prepare = partial(prepare_bound_conversion, corpus="technology")
prepared = prepare(binding, spec)
change = compare_conversion(prepared.descriptor, previous=previous_descriptor)
rows = convert(binding["path"], prepared.spec())
```

Prepare the approved binding and native metadata file as described in
[VERIFIED_MARKDOWN.md](VERIFIED_MARKDOWN.md). Use `epub_parse.metadata_path` for
an EPUB. Preparation re-verifies the existing evidence and requires that the
active parser's metadata file matches the binding's complete native metadata.
It reads inputs without writing outputs, receipts or metadata files. `convert`
performs fresh verification again; preparation is not transferable proof.

The qualified ordered pipeline is a parser slot, `text_clean`, optional
`heading_detect`, `split_semantic`, and `emit_jsonl`. Preparation refuses other
compositions, enrichment, interaction, extraction exclusions/margins, opaque
collaborators and supplied API keys. A disabled `ai_enrich` option group is allowed
for the checked-in default spec, but an AI pass is not part of this profile.
Ordinary PDF, transcript and unverified article conversion keep their current APIs.

`load_spec` resolves YAML/environment/explicit overrides first. The adapter then
reads the actual registered splitter's sizing defaults and `EmitConfig.from_env()`.
Pure, data-first `resolve_article_spec(spec, suffix=..., emit_defaults=...,
split_defaults=...)` remaps the parser, derives the effective minimum chunk size
using the existing `SplitOptions`, and fills all emission settings. Explicit
options win over defaults. Inert supplied options are retained conservatively;
the profile's verified-text passthrough and contiguous Markdown implementation
are pinned by code hashes. Explicit null `min_chunk_size` is refused because the
runner's runtime override cannot execute it. This API adds no new default values.

`PreparedArticleConversion` holds an immutable descriptor and `spec_json`.
`prepared.spec()` returns a fresh, detached `PipelineSpec`. Execute that spec
unchanged to use the described settings; editing its copy requires preparation
again. Later valid emission environment changes cannot replace its explicit
settings. Execution still reads environment defaults before applying overrides,
so a newly malformed environment value can cause a failure. No settings are
written to the process environment or registered passes.

The automatic implementation pin hashes the runtime snapshot: SHA-256 of every
package Python source (including shared helpers), Python implementation/version
and Unicode version, and sorted installed distribution names/versions. Source
reads fail explicitly if unavailable. No Git command, network request, dependency
installation or model loading occurs. Including all installed distributions is
conservative: even an unrelated dependency upgrade invalidates the full key.
Defaults, source edits or dependency upgrades affect conversion identity while
independent logical note associations remain unchanged. Equivalent omitted and
explicit effective settings produce the same snapshot when other supplied
options agree.

This is an input snapshot, not a hermetic runtime attestation: source changes
after import, in-memory monkeypatches, and modified external resource bytes are
not certified. Readability currently uses optional CMU/pyphen dictionary data and
process caches. Before enabling artifact reuse, qualify those resource bytes and
runtime-state assumptions as well as output integrity and completion. Retain the
existing output validation requirement even for an unchanged prepared descriptor.
There is no cache executor or enrichment reuse in this increment.

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

For this lower-level declared-contract API, `binding`, `previous_descriptor`,
version pins and resolved configuration are supplied by the caller. For a first
observation, omit `previous` or pass `None`.
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
pytest tests/article_conversion_test.py tests/conversion_descriptor_test.py \
  tests/verified_markdown_ingress_test.py \
  tests/joplin_markdown_preservation_test.py tests/text_adapter_test.py
nox -s lint typecheck tests
```

Synthetic tests cover effective defaults/environment precedence, implementation
and dependency invalidation, actual prepared runner/JSONL parity, no writes,
Markdown/EPUB identity, a verified two-note metadata update,
actual conversion retaining both references, tampered evidence, exact Unicode,
mapping-order laws, detached immutable snapshots, partial application and an
unchanged rerun after another article. No source admission or model call occurs.

Base article preparation connects effective runner settings/version resolution
for the qualified composition above. Next qualify resource/runtime assumptions and
validated base artifacts, then resolve and validate the separate enrichment contract.
A persisted descriptor reader, batch execution, receipts, UTF-8 row preflight,
document-text/HTML inputs and the
downstream idempotency pilot remain open. P0-A and P2-D remain open as recorded in
the [implementation roadmap](IMPLEMENTATION_ROADMAP.md).
