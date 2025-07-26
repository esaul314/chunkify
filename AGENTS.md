# AGENTS.md — Project Guidance for OpenAI Codex

This Agents.md suite provides structured guidance to OpenAI Codex and other AI agents working across this modular Python library for chunking PDF and EPUB documents in preparation for use with local LLM pipelines, particularly for RAG workflows.

## Project Structure for OpenAI Codex Navigation

The main directory `pdf_chunker/` is structured as follows:

* `/pdf_chunker/pdf_chunker`: Core logic organized into single-responsibility modules
* `/config/tags`: YAML configuration for domain-specific enrichment
* `/scripts`: CLI interfaces for human operators and automation
* `/tests`: Modular test infrastructure with validation tools
* `/tests/utils`: Shared test orchestration and logging helpers

## Coding Conventions for OpenAI Codex

* Use Python 3.10+
* Follow declarative and functional programming paradigms
* Each module should do one thing well, with minimal dependencies
* Avoid circular imports and tightly coupled orchestration
* Ensure clarity, testability, and composability

## Testing Requirements for OpenAI Codex

Tests are run using `pytest` and shell scripts:

```bash
pytest tests/  # Run the full test suite
bash tests/run_all_tests.sh  # Run orchestrated shell test suite
```

Tests must:

* Be idempotent and stateless
* Validate transformation behavior over implementation detail
* Use reusable fixtures or functions when possible

## Pull Request Guidelines for OpenAI Codex

When OpenAI Codex helps create a PR:

1. Include meaningful commit messages and diffs
2. Validate all changes using both Python and shell tests
3. Focus each PR on a single responsibility
4. Avoid formatting-only changes unless required for consistency
5. Clearly annotate new logic with docstrings or inline comments

## Programmatic Checks for OpenAI Codex

Before submitting:

```bash
# Python formatting and linting
black pdf_chunker/ tests/
flake8 pdf_chunker/

# Type checking (if using type hints)
mypy pdf_chunker/

# Run all validations
bash scripts/validate_chunks.sh
```

All tests must pass and generated chunks must remain structurally sound.

---

