## tests/utils/AGENTS.md

```markdown
# AGENTS

Shared logic for test orchestration and validation.

## AI Agent Guidance
- Keep shell POSIX-compatible
- Allow override of paths via env vars
- Formatters must not modify logic
- Prefer logging functions to inline `echo` or `print`
- Use environment variables for path overrides.
- Provide clear logging for failures.

## Known Issues
- Logging consistency may vary across utilities
```

