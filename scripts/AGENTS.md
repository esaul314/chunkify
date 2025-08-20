## scripts/AGENTS.md

`````markdown
# AGENTS

CLI entry points and automation scripts.
Please, keep this file up-to-date with the latest code structure. If you notice that the code structure has changed, please update this file accordingly.

## Responsibilities
- ```chunk_pdf.py`: Full pipeline runner.
- `validate_chunks.sh`: Semantic and size checks.
- `detect_duplicates.py`: Overlap/duplicate analysis.
- `_apply.sh`: Batch orchestration.
- `parity.py`: Run legacy and new pipelines for comparison.

## AI Agent Guidance
- Delegate core logic to library modules.
- CLI must use `argparse` or POSIX `getopts`.
- Output logs and JSON forming for downstream tools.

## Known Issues
- Command-line help may be outdated.
- `_apply.sh` exit codes not consistently handled.
```

---

