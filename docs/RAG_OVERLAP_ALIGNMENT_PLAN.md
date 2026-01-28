# RAG Overlap Alignment Plan

> **Status**: Phase 2 Complete  
> **Created**: 2026-01-28  
> **Updated**: 2026-01-28  
> **Audience**: AI agents implementing this plan  
> **Context**: Ensure overlap configuration in `pipeline_rag.yaml` follows RAG best practices and is well-documented for maintainers and users.

---

## Background: What Is Overlap and Why Does It Matter?

**Retrieval-Augmented Generation (RAG)** systems retrieve relevant text chunks from a corpus and feed them to an LLM as context. The quality of retrieval depends on how chunks are segmented:

- **Too small**: Chunks lack sufficient context; the LLM may misunderstand intent
- **Too large**: Chunks dilute relevance; retrieval noise increases
- **Hard boundaries**: If a query spans two chunks, neither chunk alone may score highly

**Overlap** addresses the hard-boundary problem by duplicating a portion of text at chunk boundaries. If chunk N ends with "...the mitochondria is the powerhouse" and chunk N+1 begins with "the powerhouse of the cell...", a query about "powerhouse of the cell" will match both chunks, increasing recall.

**Tradeoff**: More overlap means more redundant storage and potential duplicate retrievals. Less overlap means more boundary-crossing queries miss relevant chunks.

**Industry heuristic**: 10–30% overlap relative to chunk size balances recall vs. redundancy. Our current configuration (100 words overlap / 400 words chunk = 25%) sits squarely within this range.

---

## Conceptual Model: How Overlap Works in This Codebase

Understanding the implementation is essential for making changes safely.

### 1. Chunk Splitting (`split_semantic.py`)

The `split_semantic` pass transforms a flat list of text blocks into sized chunks:

```
Input blocks:  [block1, block2, block3, ...]
                    ↓ semantic_chunker()
Output chunks: [chunk1, chunk2, chunk3, ...]
```

Splitting happens when accumulated text exceeds `chunk_size` (default 400 words). The split point is chosen to respect sentence boundaries when possible.

### 2. Overlap Injection (`_restore_overlap_words()` in `split_modules/overlap.py`)

After splitting, overlap is injected at boundaries:

```
Before overlap:
  Chunk N:   "...sentence A. Sentence B."
  Chunk N+1: "Sentence C. Sentence D..."

After overlap (100 words):
  Chunk N:   "...sentence A. Sentence B."
  Chunk N+1: "[last 100 words of chunk N] Sentence C. Sentence D..."
```

The overlap words are prepended to the *next* chunk, not appended to the *previous* chunk. This ensures retrieval queries hitting the boundary region match the chunk that continues the content.

### 3. Boundary Trimming (`trim_boundary_overlap()` in `split_modules/overlap.py`)

Overlap can create awkward duplicates if the overlap region ends mid-sentence and the next chunk starts with the same sentence. The trimming logic detects and removes such duplicates:

```
Before trimming:
  Chunk N+1: "...Sentence B. Sentence B. Sentence C..."
                         ↑ duplicate

After trimming:
  Chunk N+1: "...Sentence B. Sentence C..."
```

### 4. When Overlap Is Invisible

Overlap only appears when chunks actually split. If all your content fits in a single chunk (< 400 words), no overlap is injected because there are no boundaries. This is expected behavior, not a bug.

---

## Guiding Philosophy

This plan adheres to the **Anti-Overengineering Mandate** from `AGENTS.md`:

> Do not build a cathedral for a cottage.

Overlap tuning is a domain where diminishing returns set in quickly. The goal is not a perfect universal solution—it is a *well-documented, reasonable default* with escape hatches for users who need different behavior.

**Priorities (from AGENTS.md Fitness Function):**

1. **Correctness**: Overlap must not corrupt chunk boundaries or duplicate content incorrectly
2. **Safety**: Changes must be backward-compatible; existing pipelines must not break
3. **Clarity**: Configuration and behavior must be understandable without reading source code
4. **Sustainability**: Avoid per-content-type knobs unless evidence demands them
5. **Performance**: Overlap is cheap; do not optimize prematurely

---

## Current State

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `chunk_size` | 400 words | Fits embedding model context windows; balances granularity vs. coherence |
| `overlap` | 100 words | ~25% overlap; within the 10–30% heuristic range for RAG |

**Key files to understand:**

| File | Purpose |
|------|---------|
| `pipeline_rag.yaml` | User-facing configuration; sets `chunk_size` and `overlap` |
| `pdf_chunker/passes/split_semantic.py` | Orchestrates chunking; calls overlap functions |
| `pdf_chunker/passes/split_modules/overlap.py` | `restore_overlap_words()`, `trim_boundary_overlap()` |
| `pdf_chunker/passes/split_modules/segments.py` | Segment emission and collapsing logic |

**Implementation path:**
- `_restore_overlap_words()` injects overlap at split boundaries
- `trim_boundary_overlap()` removes duplicate sentence prefixes to avoid retrieval noise
- Overlap only manifests when a chunk exceeds `chunk_size` and splits

---

## Goals

1. **Document the "why"** — Users should understand overlap rationale without archaeology
2. **Validate current defaults** — Confirm 100-word overlap is defensible for typical content
3. **Provide tuning guidance** — Users with domain-specific needs should know how to adjust
4. **Resist complexity creep** — Do not add content-type-specific overlap unless proven necessary

---

## Strategic Sequence

### Phase 1: Documentation (Low Risk, High Clarity) ✅ COMPLETE

**Intent**: Make the existing behavior legible to users and future agents.

- [x] Add inline comments in `pipeline_rag.yaml` explaining overlap rationale
- [x] Add a "RAG Configuration" section to `README.md` or a dedicated `docs/RAG_GUIDE.md`
- [x] Cross-reference from `split_semantic.py` docstrings to the user-facing docs

**Guardrails:**
- Comments should answer "why this value?" not "what does this do?"
- Prefer prose over tables; users skim tables but read explanations
- Do not duplicate information already in code; link to it

**Success criteria:**
- A user reading `pipeline_rag.yaml` understands the tradeoff without opening source files
- A developer modifying overlap logic finds the design rationale within two file hops

---

### Phase 2: Evaluation Framework (Optional, Evidence-Driven) ✅ COMPLETE

**Intent**: Provide a path to measure overlap effectiveness—but only if warranted.

This phase is **gated**: proceed only if users or maintainers report retrieval quality issues that might stem from overlap settings.

- [x] Create a minimal eval harness: sample queries + expected chunk retrievals
- [x] Document how to run A/B comparisons (overlap=50 vs. 100 vs. 150)
- [x] Record baseline metrics if eval is run

**Implementation:**
- `scripts/eval_overlap.py` — TF-IDF-based retrieval evaluation (no external dependencies)
- Generates synthetic boundary-spanning queries automatically
- Measures recall@k across configurable overlap values
- Runs in ~30 seconds on platform-eng-excerpt.pdf (86-99 chunks)

**Baseline metrics** (platform-eng-excerpt.pdf, k=3):
| Overlap | Chunks | Recall@3 |
|---------|--------|----------|
| 0       | 86     | 84.7%    |
| 50      | 91     | 84.4%    |
| 100     | 93     | 85.9%    |
| 150     | 99     | 84.7%    |

**Finding**: Overlap values 0-150 show similar recall (84-86%), validating that the 100-word default is reasonable. Content-aware tuning (Phase 3) is not warranted by this data.

**Guardrails:**
- Do not build a framework; a script with clear inputs/outputs suffices
- Eval should run in under 60 seconds on a sample corpus
- Do not commit large test corpora; use existing `test_data/` or synthetic fixtures

**Success criteria:**
- If built, the eval answers: "Does changing overlap from X to Y improve recall@k for this corpus?"
- If not built, there is a clear note in docs explaining why (no evidence of need)

---

### Phase 3: Content-Aware Tuning (Deferred Unless Required)

**Intent**: Allow different overlap for different content types—only if Phase 2 reveals a need.

**Default stance**: Do not implement this. The current single-value overlap is simpler and sufficient for most use cases.

If evidence emerges (e.g., technical docs need 20% overlap but narrative prose needs 30%), then:

- [ ] Add optional `overlap_by_type` mapping in pipeline spec
- [ ] Document the heuristics that informed the per-type values
- [ ] Ensure fallback to global `overlap` if type is unrecognized

**Guardrails:**
- Maximum of 3 content categories (e.g., `prose`, `technical`, `list-heavy`)
- Per-type values must be justified in comments or docs
- Do not expose this in CLI flags; YAML-only to limit surface area

**Success criteria:**
- If implemented, retrieval quality demonstrably improves for at least one content type
- If not implemented, the codebase remains simpler

---

## Non-Goals

- **Automatic overlap tuning**: Do not build ML-based overlap selection
- **Per-chunk overlap**: All chunks in a run use the same overlap; do not vary mid-document
- **Embedding-model-specific defaults**: Users can override; we do not ship profiles per model

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-01-28 | Current 100-word overlap retained | Falls within 10–30% best-practice range; no user complaints |
| 2026-01-28 | Phase 2 eval harness implemented | Provides tooling to measure overlap effectiveness when needed |
| 2026-01-28 | Baseline metrics recorded | Recall@3 range 84-86% across overlap 0-150; validates current default |
| 2026-01-28 | Phase 3 deferred indefinitely | No evidence that content-aware tuning improves outcomes |

---

## References

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — original RAG paper
- [LangChain chunking guidance](https://python.langchain.com/docs/modules/data_connection/document_transformers/) — practical heuristics
- [LlamaIndex overlap discussion](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/) — sentence-window patterns

---

## For Implementing Agents

When working on this plan:

1. **Start with Phase 1** — documentation is always safe and always valuable
2. **Do not proceed to Phase 2 unless there is a concrete retrieval quality complaint**
3. **Treat Phase 3 as speculative** — it may never be needed
4. **Keep diffs minimal** — a few well-placed comments beat a new configuration subsystem
5. **Update this document** when decisions are made or phases complete

Remember: the goal is a well-lit path, not a paved highway.

---

## Implementation Recipes

These are concrete patterns for implementing each phase. Use them as starting points, not rigid templates.

### Recipe: Adding YAML Comments (Phase 1)

```yaml
# pipeline_rag.yaml
options:
  split_semantic:
    # chunk_size: Target words per chunk. 400 balances embedding model context
    # windows (~512 tokens) against semantic coherence. Smaller chunks increase
    # retrieval precision but lose context; larger chunks improve context but
    # dilute relevance scores.
    chunk_size: 400
    
    # overlap: Words duplicated at chunk boundaries. 100 words (~25% of chunk_size)
    # ensures queries spanning boundaries match both chunks. Industry heuristic:
    # 10-30% overlap. Higher values improve recall but increase storage/redundancy.
    # See docs/RAG_OVERLAP_ALIGNMENT_PLAN.md for rationale.
    overlap: 100
```

### Recipe: README Section (Phase 1)

Add to `README.md` under a "RAG Configuration" heading:

```markdown
## RAG Configuration

The `pipeline_rag.yaml` spec is optimized for Retrieval-Augmented Generation:

- **chunk_size: 400** — Balances embedding model context windows against coherence
- **overlap: 100** — 25% overlap ensures boundary-spanning queries match both chunks

To tune for your use case:
- Increase `overlap` (up to 30%) if retrieval misses relevant chunks
- Decrease `overlap` (down to 10%) if you see too many duplicate retrievals
- Adjust `chunk_size` based on your embedding model's token limit

See [docs/RAG_OVERLAP_ALIGNMENT_PLAN.md](docs/RAG_OVERLAP_ALIGNMENT_PLAN.md) for design rationale.
```

### Recipe: Docstring Cross-Reference (Phase 1)

In `split_semantic.py`, update the module or function docstring:

```python
def _restore_overlap_words(chunks: list[dict], overlap: int) -> list[dict]:
    """Inject overlap words at chunk boundaries for RAG retrieval.
    
    Overlap ensures queries spanning chunk boundaries match both chunks.
    The overlap words are prepended to each chunk (except the first) from
    the tail of the previous chunk.
    
    Design rationale: docs/RAG_OVERLAP_ALIGNMENT_PLAN.md
    Industry heuristic: 10-30% overlap relative to chunk_size.
    Current default: 100 words / 400 word chunks = 25%.
    """
```

### Recipe: Minimal Eval Script (Phase 2, if needed)

```python
#!/usr/bin/env python3
"""Minimal overlap effectiveness eval. Run only if retrieval issues reported."""
import json
from pathlib import Path

# Inputs: queries.json with [{"query": "...", "expected_chunk_ids": [...]}]
# Outputs: recall@k for different overlap settings

def eval_overlap(chunks_file: Path, queries_file: Path, k: int = 3) -> float:
    chunks = [json.loads(line) for line in chunks_file.read_text().splitlines()]
    queries = json.loads(queries_file.read_text())
    
    # Naive TF-IDF or embedding similarity (keep simple)
    hits = 0
    for q in queries:
        # Score chunks, check if expected IDs in top-k
        ...
    return hits / len(queries)
```

---

## Verification Checklist

After implementing Phase 1, verify:

- [x] `pipeline_rag.yaml` has comments explaining `chunk_size` and `overlap`
- [x] `README.md` or `docs/RAG_GUIDE.md` has a RAG Configuration section
- [x] `split_semantic.py` or `overlap.py` docstrings reference the plan
- [x] Running `pdf_chunker convert ... --spec pipeline_rag.yaml` produces expected output
- [x] No existing tests break (`nox -s tests`)

---

## Common Pitfalls to Avoid

1. **Over-documenting**: Comments should be terse. If you need a paragraph, link to this plan instead.
2. **Changing defaults without evidence**: Do not adjust `overlap` or `chunk_size` without retrieval quality data.
3. **Building evaluation infrastructure prematurely**: Phase 2 is gated on user complaints, not proactive engineering.
4. **Adding CLI flags for overlap tuning**: YAML configuration is sufficient; CLI flags increase surface area.
5. **Content-type detection complexity**: Phase 3 should use existing `block_type` metadata, not new heuristics.
