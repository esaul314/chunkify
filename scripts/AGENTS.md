## scripts/AGENTS.md

```markdown
# AGENTS

CLI scripts for triggering pipeline components and validating results.

## Responsibilities
- `chunk_pdf.py`: CLI interface to run the 3-pass pipeline
- `validate_chunks.sh`: Ensures semantic and size boundary correctness
- `detect_duplicates.py`: Scans for repeated chunk content

## AI Agent Guidance
- Scripts must use CLI args and never hardcode paths
- Avoid business logic — delegate to core modules
- Output logs should be human-readable or JSON where applicable
- There are a couple of standalone scripts that are nevertheless potentially very useful.
- Notably, find_glued_words.py, but really, this folder contains additional scripts that it may be useful to integrate some of the logic or to run manually.

```

---

