## scripts/AGENTS.md

`````markdown
# AGENTS

CLI entry points and automation scripts.
Please, keep this file up-to-date with the latest code structure. If you notice that the code structure has changed, please update this file accordingly.

## Responsibilities
- Maintained operational scripts include `chunk_pdf.py`, `evaluate_corpus.py`,
  `replay_from_snapshot.py`, `bisect_dups.py`, `detect_duplicates.py`,
  `validate_chunks.sh`, and the small fixture/documentation generators used by
  the test and CI workflows.
- Diagnostic and experimental scripts include `benchmark_extraction.py`,
  `compare_text_quality.py`, `epoche_platform_eng.py`, `eval_overlap.py`,
  `experiment_pymupdf4llm.py`, `llm_correction.py`, and the ad-hoc repair or
  investigation helpers. They are not part of the canonical conversion path;
  changes to them must not be mistaken for pipeline behavior changes.
- `evaluate_corpus.py`: Runs the tracked corpus manifest and writes a category-based
  quality report; it must not tune overlap or collapse categories into one score.
- ```chunk_pdf.py`: Deprecated compatibility frontend delegating parsing and execution to
  `pdf_chunker convert`; it must not grow a second conversion implementation.
- `validate_chunks.sh`: Semantic and size checks.
- `detect_duplicates.py`: Overlap/duplicate analysis.
- `bisect_dups.py`: Locate first pass introducing duplicates via snapshot replay (checks `pdf_parse`→`text_clean`→`heading_detect`→`list_detect`→`split_semantic`).
- `_apply.sh`: Batch orchestration.
- `parity.py`: Run legacy and new pipelines for comparison.
- `epoche_platform_eng.py`: Trace harness verifying duplicate and pass execution counts.

## AI Agent Guidance
- Delegate core logic to library modules.
- CLI wrappers must delegate option parsing and execution to the canonical `pdf_chunker.cli`
  runner so modern and compatibility invocations stay behaviorally identical.
- Output logs and JSON forming for downstream tools.

## Usage
- Convert a PDF via CLI:
  ```bash
  pdf_chunker convert ./synthetic-structure.pdf --spec pipeline.yaml --out ./data/platform-eng.jsonl --no-enrich
  ```
- The deprecated compatibility frontend accepts the same conversion flags:
  ```bash
  python -m scripts.chunk_pdf ./synthetic-structure.pdf --out ./data/platform-eng.jsonl --no-metadata
  ```
- Equivalent script invocation:
  ```bash
  python -m scripts.chunk_pdf --no-metadata ./synthetic-structure.pdf > data/platform-eng.jsonl
  ```
- Replay remaining passes from a snapshot and optionally check for duplicates:
  ```bash
  python scripts/replay_from_snapshot.py \
    --snapshot artifacts/trace/<run_id>/pdf_parse.json \
    --from pdf_parse --spec pipeline.yaml \
    --out /tmp/replay.jsonl --check-dups
  ```
  Flags:
  - `--snapshot PATH`: snapshot JSON file to seed replay.
  - `--from STEP`: pass name matching the snapshot.
  - `--spec FILE`: pipeline specification (default `pipeline.yaml`).
  - `--out PATH`: destination for the replayed JSONL.
  - `--check-dups`: print top duplicates and write `<final_pass>_dups.json`.
  - 
- Locate the first pass that introduces duplicates using a trace bundle:
  ```bash
  python scripts/bisect_dups.py --dir artifacts/trace/<run_id> --spec pipeline.yaml
  ```

  Flags:
  - `--dir PATH`: directory containing per-pass snapshots.
  - `--spec FILE`: pipeline specification (default `pipeline.yaml`).

## Corpus evaluation

- Evaluate representative PDF, EPUB, and transcript fixtures:
  ```bash
  python scripts/evaluate_corpus.py \
    --manifest docs/corpus_manifest.yaml \
    --output /tmp/pdf-chunker-quality.json
  ```
  The report separates preservation checks, page coverage, extraction engine and
  fallback use, artifact removals, chunk sizes, overlap, duplicate rate, and
  metadata completeness. Use retrieval evaluation separately before changing overlap.

- Convert a transcript through the same canonical pipeline as other supported
  inputs:
  ```bash
  pdf_chunker convert ./episode.txt --out ./data/episode.jsonl --metadata-file ./episode.metadata.yaml --no-enrich
  ```
  WebVTT cue timing is retained as `timestamp_start`/`timestamp_end` metadata;
  PDF footer/artifact passes are not valid for `.txt` input.

## Current boundary

The canonical conversion implementation lives in `pdf_chunker/core_new.py` and
`pdf_chunker/cli.py`. Scripts may inspect, replay, validate, or benchmark that
path, but they must not introduce a second parser or chunking implementation.
```

---

## Bounded CPU section pilots
prepare_section_pilots.py --manifest INPUT.json --output NEW_DIRECTORY binds
source/artifact hashes and delegates to canonical extraction, classification,
splitting and admission. It writes private exclusive CPU artifacts/sidecars and
acceptance; no enrichment, GPU, inference, indexing or service operations.

verify_section_pilots.py independently validates all sidecar/source/artifact hashes,
exact serialized-part reconstruction, bound block identities and ordered per-block
token coverage; reports are exclusive private0600 and inspect no model/index state.
