# Enrichment request budget

RAG enrichment retains every canonical source-text byte, chunk ID and provenance
field. Its prompt advertises only tags eligible under the existing grounding rule.
Parsing, post-response grounding and lexical/optional-provider fallback still use
the full effective vocabulary. Prompt changes may change model choices; identical
classification outputs are not promised.

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
pytest tests/ai_enrich_request_budget_test.py tests/ai_enrichment_test.py tests/ai_enrich_pass_test.py
pytest -s tests/ai_enrich_context_fixtures_test.py
nox -s lint typecheck tests
```

Preserve one-worker managed-GPU restoration, request-drain/native/ownership gates,
full source/chunk readback and final normal-model acceptance. This converter
repair does not diagnose or authorize recovery of a kernel/runtime fault.
