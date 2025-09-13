## scripts/AGENTS.md

`````markdown
# AGENTS

CLI entry points and automation scripts.
Please, keep this file up-to-date with the latest code structure. If you notice that the code structure has changed, please update this file accordingly.

## Responsibilities
- ```chunk_pdf.py`: Full pipeline runner.
- `validate_chunks.sh`: Semantic and size checks.
- `detect_duplicates.py`: Overlap/duplicate analysis.
- `bisect_dups.py`: Locate first pass introducing duplicates via snapshot replay.
- `_apply.sh`: Batch orchestration.
- `parity.py`: Run legacy and new pipelines for comparison.
- `epoche_platform_eng.py`: Trace harness verifying duplicate and pass execution counts.

## AI Agent Guidance
- Delegate core logic to library modules.
- CLI must use `argparse` or POSIX `getopts`.
- Output logs and JSON forming for downstream tools.

## Usage
- Convert a PDF via CLI:
  ```bash
  pdf_chunker convert ./platform-eng-excerpt.pdf --spec pipeline.yaml --out ./data/platform-eng.jsonl --no-enrich
  ```
- Equivalent script invocation:
  ```bash
  python -m scripts.chunk_pdf --no-metadata ./platform-eng-excerpt.pdf > data/platform-eng.jsonl
  ```
- Replay remaining passes from a snapshot and optionally check for duplicates:
  ```bash
  python scripts/replay_from_snapshot.py \
    --snapshot artifacts/trace/<run_id>/pdf_parse.json \
    --from pdf_parse --spec pipeline.yaml \
    --out /tmp/replay.jsonl --check-dups
  ```
- Locate the first pass that introduces duplicates using a trace bundle:
  ```bash
  python scripts/bisect_dups.py --dir artifacts/trace/<run_id> --spec pipeline.yaml
  ```

## Known Issues
- Command-line help may be outdated.
- `_apply.sh` exit codes not consistently handled.
```

---

