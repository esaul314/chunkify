## scripts/AGENTS.md

```markdown
# AGENTS

CLI entry points and automation utilities for triggering pipeline behavior and validating outputs.

## Responsibilities
- `chunk_pdf.py`: Main CLI for processing documents end-to-end
- `validate_chunks.sh`: Structural integrity checker
- `detect_duplicates.py`: Duplicate/overlap detection in output

## AI Agent Guidance
- Do not duplicate core logic — import from main modules
- All scripts must use `argparse` or shell `getopts`
- Scripts must fail loudly and emit actionable logs

## Known Issues
- Some script usage modes are undocumented or outdated
```

---

