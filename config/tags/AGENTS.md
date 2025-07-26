## config/tags/AGENTS.md

```markdown
# AGENTS

YAML-based domain vocabularies that guide semantic enrichment during the AI pass.

## Purpose
- External configuration of structured tags
- Allows modular tagging by domain (e.g., philosophy, technical, PM)

## AI Agent Guidance
- Never modify existing tag files without intent review
- Add new vocabularies using lowercase, underscore filenames
- Tags should be layered, non-destructive, and human-legible
- Ensure YAML validity and UTF-8 safety

## Known Issues
- Some tag vocabularies may be under-used due to misalignment between classifiers and YAML scopes.
```

---

