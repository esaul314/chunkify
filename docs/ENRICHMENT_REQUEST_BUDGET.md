# Enrichment request budget

RAG enrichment retains every canonical source-text byte, chunk ID and provenance
field. Its prompt advertises only tags eligible under the existing grounding rule.
Parsing, post-response grounding and lexical/optional-provider fallback still use
the full effective vocabulary. Prompt changes may change model choices; identical
classification outputs are not promised.

## Resolved serving settings

`pdf_chunker.enrichment_config` is the shared pure boundary used by both existing
completion adapters. `resolve_primary_settings(options, environment=...)` and
`resolve_fallback_settings(options, environment=...)` return frozen
`CompletionSettings(model, api_base, max_tokens, no_think)` values. They read only
the supplied mappings, preserve inputs, and support partial application:

```python
from dataclasses import asdict
from functools import partial

from pdf_chunker.enrichment_config import completion_request, resolve_primary_settings

resolve = partial(resolve_primary_settings, environment={})
settings = resolve({"model": "qwen3-8b-local", "max_tokens": 256})
request = completion_request("Full source text: 界🧭", settings=settings)
assert request["messages"][0]["content"] == "/no_think\nFull source text: 界🧭"
assert asdict(settings)["max_tokens"] == request["max_tokens"]
```

Primary precedence is explicit options, `PDF_CHUNKER_AI_ENRICH_*`, supported
OpenAI aliases, then existing local defaults. `api_base` precedes the `base_url`
option alias; environment base aliases retain their existing order: `API_BASE`,
`BASE_URL`, then `OPENAI_API_BASE_URL`. The fallback resolver accepts its own
adapter-local `model`, `api_base` and `max_tokens` options, uses only fallback
environment keys, and retains the provider's default route when no base is set.
It never adds the primary Qwen directive. Both request builders retain temperature
0.0 and exact prompt text, apart from that existing primary directive.

Explicit token limits must be positive integers; zero, booleans, floats and
strings raise `ValueError`. Environment token strings are parsed as integers and
must also be positive. Previously zero/False could silently select a default,
and fallback invalid limits could reach transport. Explicit primary `no_think`
must be a boolean or `None`; strings such as `"false"` now refuse instead of
becoming truthy. Environment spelling is unchanged: missing means true;
`1/true/yes/on` (ignoring case/outer whitespace) mean true and other strings mean
false. Supplied model/base labels must resolve to nonblank strings; existing
empty-value fallback precedence is retained.

`completion_request(prompt, settings=...)` returns fresh nested request containers
without API-key fields. The adapter adds the key only at transport. The adapter
still loads dotenv first, resolves credentials, loads an optional pinned tokenizer,
and captures settings once per callback; later environment changes do not alter
that callback. Without credentials the optional fallback still returns `None`
before validating unused settings or importing a provider. Explicitly disabled
fallback remains disabled.

These settings describe outgoing requests. They do not establish model revision,
running-server configuration, prompt/vocabulary identity, successful completion
or permission to reuse enriched output. Binding those inputs to validated base
bytes and validating enriched results against that identity remain open. The
existing [article byte preflight](CONVERSION_DESCRIPTORS.md#utf-8-byte-preflight-before-enrichment)
remains opt-in. No new client, runner, cache schema or dependency is introduced.

## UTF-8 row admission

Opt into a downstream complete-row byte limit through the existing pipeline spec:

```yaml
options:
  emit_jsonl:
    max_chars: 8000
    max_row_bytes: 8192
```

`max_row_bytes` must be a positive integer or `null` (disabled, the default).
Booleans, strings, floats and nonpositive limits refuse. There is no new dedicated
environment variable or CLI flag; the existing spec and `EMIT_JSONL__MAX_ROW_BYTES`
override mechanism apply. Both limits are independent and must fit. Measure the
entire literal-Unicode JSON object, including keys, escaped text and metadata,
encoded as UTF-8; exclude its LF delimiter. Equality fits. File sizes include LF.

When `ai_enrich` precedes `emit_jsonl` in the canonical pipeline, the runner passes
both limits and the active metadata key to existing pre-classification admission.
Projection includes the full eligible tag vocabulary, longest classification and
completion status. The same monotone prefix search splits ordinary oversized
chunks losslessly with stable `chunk_part`, preserving ordered text and provenance.
Impossible metadata and existing part identities that would need resplitting
refuse before any classification. All chunks are admitted before the first call.
Verified Markdown requiring repartitioning refuses for explicit context review;
code, tables and links are not cut apart by this feature.

Actual enriched rows are checked again because projection cannot bound arbitrary
injected client results. Every final emitted row is also byte-checked, including
conversion without AI and legacy emission. Oversized results raise `ValueError`
without truncation or automatic retry. The final emitter validates rather than
introducing a new splitting path. Direct writer use remains a serialization-only
API; these checks belong to the pipeline and validators.

Prepared base article settings capture this option, so changing it changes
conversion identity even when the existing bytes still fit. Execution and pure
`validate_article_output` readback enforce the pinned byte limit. The separate
`preflight_article_output` report still measures caller-declared downstream bytes
and reserve without changing the descriptor or running enrichment. This feature
does not establish model/prompt/vocabulary identity or enriched-output reuse.

## Offline request admission

`pipeline_rag.yaml` sets an opt-in `ai_enrich.request_budget` mapping:

```yaml
request_budget:
  model: "Qwen/Qwen3-8B-AWQ"
  revision: "4da05a8edb55c6046cce958586c33b61da07bb79"
  context_tokens: 4096
  margin_tokens: 64
  enable_thinking: false
  tokenizer_sha256: "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
  tokenizer_config_sha256: "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101"
```

The adapter reads only this exact cached snapshot under `HF_HUB_CACHE`, or
`HF_HOME/hub`, or the default Hugging Face cache. It verifies both file hashes
before constructing an in-memory tokenizer/template. No network, downloads,
weights or Transformers are needed. Missing or mismatched files fail explicitly.
Serving model/revision/context/disabled-thinking contract must be checked by the
operator; file hashes alone do not prove a running server's configuration.

Admission counts the actual user message after any `/no_think` prefix, the pinned
chat template with generation prefix and disabled-thinking block, plus actual
`max_tokens` and margin. The tokenizer's model_max_length and nominal400word
splitter target are not serving budgets. With context4096/output256/margin64,
complete input3776 fits and3777 fails before provider transport. Classification
repair uses the same guarded callback. Other callers without request_budget
retain their prior adapter behavior; unrelated optional fallback providers do
not inherit the local Qwen context contract.

Version0.1.5 adds deterministic lossless enrichment windows for this opted-in
callback. A fitting primary/possible-repair pair retains its original single
classification path. Oversized text is halved into contiguous character slices,
preferring whitespace near the midpoint, until BOTH complete request forms fit.
This makes no assumption that tokenizer counts are monotone. Windows concatenate
exactly to the parent; canonical stored text, IDs and source provenance do not
change. No model weights, downloads, token truncation or context bypass are used.

Every window follows the existing full-vocabulary parsing, grounding, repair and
optional fallback path. Parent classification uses the largest sum of stripped
character lengths among classified windows, ties resolved by first document
appearance; unclassified windows win only when no meaningful classification
exists. Tags are a stable deduplicated union, validated against the full effective
vocabulary and grounded against the complete parent. Identical model choices are
not promised. A failed window or repair fails the source immediately; impossible
instruction/single-character budgets fail without provider transport. Neither
partial windows nor conversion failures can be published as successful artifacts.

The completion callback exposes an optional `request_fits(prompt)` capability,
using exactly the same immutable tokenizer/wire/output/margin contract as actual
transport. Custom or unbudgeted callbacks retain their original behavior. Window
planning is in-memory; tokenizer cache loading remains adapter-owned. Every actual
primary or repair request still repeats admission before provider transport.

Tests:
```bash
pytest tests/enrichment_config_test.py tests/utf8_admission_test.py tests/serialized_admission_test.py tests/ai_enrich_request_budget_test.py tests/ai_enrichment_test.py tests/ai_enrich_pass_test.py
pytest -s tests/ai_enrich_context_fixtures_test.py
nox -s lint typecheck tests
```

Preserve one-worker managed-GPU restoration, request-drain/native/ownership gates,
full source/chunk readback and final normal-model acceptance. This converter
repair does not diagnose or authorize recovery of a kernel/runtime fault.
