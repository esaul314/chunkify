# Task Stub 1 — Footer Suffix Diagnostics

## Objective
Reproduce the conditions that keep footer bullet markers alive so we can pinpoint which helper fails to strip them.

## Scenario
We modeled a semantic-split record that ends with footer bullets:

```
A real paragraph that ends properly.
• Footnote A
• Footnote B
```

This reflects the post-merge state in `test_footer_newlines_joined`, where the narrative paragraph absorbs the footer bullets before `_strip_footer_suffix` gets a chance to prune them.

## Findings
* `_record_trailing_footer_lines` refuses to flag the suffix because `_starts_list_like` only inspects the leading line, so narrative text at the top of the block short-circuits footer detection.【F:pdf_chunker/passes/split_semantic.py†L772-L803】
* `_trim_footer_suffix` therefore returns the original text, and `_strip_footer_suffix` leaves the block untouched; we confirmed this by invoking the helpers directly in an interactive session.【ea8ff9†L1-L7】
* Even when the block contains only footer bullets, `_drop_trailing_bullet_footers` declines to prune them—the surrounding context (`context_allows`) fails, so suffix detection never activates.【ade042†L1-L2】【F:pdf_chunker/page_artifacts.py†L167-L197】

## Conclusion
The diagnostics show that Task Stub 1’s goal is met: we captured the precise helper outputs and identified why footer bullets survive—both the list gate (`_starts_list_like`) and the context gate (`_drop_trailing_bullet_footers`) filter out these suffixes. Subsequent work needs to relax these predicates when bullets appear solely at the tail of an otherwise narrative block.
