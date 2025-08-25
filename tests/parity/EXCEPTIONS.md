# Parity Exceptions

This allowlist records expected divergences between the legacy and new
pipelines. Each entry is keyed by the tuple `(test_name, flags, fixture)`.
The value can specify fields to ignore during comparison and an optional
normalized expectation for the full row set.

```yaml
(test_name, flags, fixture):
  ignore: [field, nested.field]
  expect:
    - {text: "..."}
```

Current candidates:

- `test_e2e_parity_flags[--chunk-size 200 --overlap 10]` on `tiny.pdf` –
  new pipeline uses `meta` instead of `metadata`; ignore this key name
  difference.
- `test_e2e_parity_flags[--no-metadata]` on `tiny_a.pdf` – trailing space
  trimmed by the new pipeline; accept whitespace cleanup.
- `pipeline_parity_test.test_no_metadata_rows_contain_only_text` on
  `tiny_b.pdf` – stray newline removed during normalization; accept the
  sanitized output.
```
