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

An overflow does not truncate, skip or mark a source complete. The existing
conversion failure remains retryable and cannot be published as a successful
artifact. Lossless enrichment-window splitting is not implemented by this guard;
any source still oversized after candidate reduction needs explicit reviewed
handling before whole-corpus acceptance.

Tests:
```bash
pytest tests/ai_enrich_request_budget_test.py tests/ai_enrichment_test.py tests/ai_enrich_pass_test.py
nox -s lint typecheck tests
```

Preserve one-worker managed-GPU restoration, request-drain/native/ownership gates,
full source/chunk readback and final normal-model acceptance. This converter
repair does not diagnose or authorize recovery of a kernel/runtime fault.
