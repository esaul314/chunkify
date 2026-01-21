# pdf-chunker

CLI pipeline to convert PDFs and EPUBs into JSONL chunks for RAG workflows.

## Quickstart

### Installation
```bash
pip install pdf-chunker
```

### Basic usage
```bash
pdf_chunker convert "sample.pdf" --out out.jsonl --chunk-size 400 --overlap 50
```

### RAG-optimized spec
```bash
pdf_chunker convert "sample.pdf" --spec pipeline_rag.yaml --out out.rag.jsonl
```

### Enrichment troubleshooting
If tags are unexpectedly empty, enable AI-enrichment diagnostics:
```bash
PDF_CHUNKER_AI_ENRICH_DEBUG=1 pdf_chunker convert "sample.pdf" --spec pipeline_rag.yaml --out out.rag.jsonl --verbose
```
To log a small sample of model responses (for parsing issues), set a sample limit:
```bash
PDF_CHUNKER_AI_ENRICH_DEBUG=1 PDF_CHUNKER_AI_ENRICH_DEBUG_SAMPLES=1 pdf_chunker convert "sample.pdf" --spec pipeline_rag.yaml --out out.rag.jsonl --verbose
```
If AI enrichment is enabled but no completion client can be created (e.g., `litellm` missing),
the CLI now raises an error instead of silently producing empty tags.

### TTS Preparation (Strict Character Limits)
To generate chunks suitable for Text-to-Speech engines (strict character limit, no overlap, no metadata):
```bash
pdf_chunker convert "book.pdf" --out tts.jsonl --max-chars 1000 --no-metadata
```

## pipeline.yaml options

| YAML key | CLI flag | Description | Default |
| --- | --- | --- | --- |
| `ai_enrich.enabled` | `--enrich/--no-enrich` | Toggle AI enrichment pass | `false` |
| `pdf_parse.exclude_pages` | `--exclude-pages` | Comma-separated pages to skip | – |
| `split_semantic.chunk_size` | `--chunk-size` | Target tokens per chunk | `400` |
| `split_semantic.overlap` | `--overlap` | Tokens to overlap neighboring chunks | `50` |
| `split_semantic.generate_metadata` | `--no-metadata` (negates) | Include per-chunk metadata | `true` |
| `emit_jsonl.output_path` | — | Path for resulting JSONL | `output.jsonl` |
| `text_clean.footer_patterns` | `--footer-pattern` | Regex patterns to strip as footers | – |
| `text_clean.interactive_footers` | `--interactive` | Prompt for footer confirmation | `false` |

> Supply a custom spec via `--spec pipeline.yaml` to override defaults.

## Footer Detection

Many books include chapter titles or book titles as running footers on each page. When PDFs are extracted, these footers often appear merged into the text body, causing mid-sentence interruptions like:

```
"There's conflicting scientific literature on the subject.

Scale Communication Through Writing 202 Aside from that, we can all likely agree..."
```

### Stripping Footers with Patterns

Use `--footer-pattern` to specify regex patterns matching footer text. The pattern should match the footer title portion—the pipeline automatically handles the page number and inline structure.

```bash
# Strip footers like "Scale Communication Through Writing 123"
pdf_chunker convert book.pdf --out out.jsonl --footer-pattern "Scale Communication.*"

# Strip generic chapter footers like "Chapter 5" or "Part II"
pdf_chunker convert book.pdf --out out.jsonl --footer-pattern "Chapter \d+" --footer-pattern "Part [IVX]+"

# Multiple patterns can be combined (repeatable flag)
pdf_chunker convert book.pdf --out out.jsonl \
  --footer-pattern "Book Title.*" \
  --footer-pattern "Chapter \d+"
```

### Interactive Mode

When you're unsure what footer patterns exist in a document, use `--interactive` to have the CLI automatically detect potential footers and prompt you for confirmation:

```bash
pdf_chunker convert book.pdf --out out.jsonl --interactive
```

**Heuristic detection**: Without `--footer-pattern`, the CLI uses a heuristic to find inline footers matching the pattern `\n\n{TitleCase Words} {PageNumber}` (e.g., "Scale Communication Through Writing 202").

The CLI will display candidate footer text and ask for confirmation:
```
--- Footer candidate (page 202, confidence 70%) ---
  Scale Communication Through Writing 202
Treat as footer? [Y/n] 
```

Your decisions are cached by title, so similar footers on subsequent pages are handled automatically.

**Combined mode**: You can also use `--interactive` with `--footer-pattern` to get prompts for pattern matches:

```bash
pdf_chunker convert book.pdf --out out.jsonl --interactive --footer-pattern "Chapter.*"
```

### YAML Configuration

For repeated use, add footer patterns to your pipeline spec:

```yaml
# pipeline.yaml
options:
  text_clean:
    footer_patterns:
      - "Scale Communication.*"
      - "Chapter \\d+"
      - "Part [IVX]+"
```

### How Inline Footers Work

The pipeline handles two footer scenarios:

1. **Block-level footers**: Entire text blocks that are just footers (e.g., a standalone "Chapter 5" block) are removed entirely.

2. **Inline footers**: Footers merged mid-text with a `\n\n` prefix and page number suffix are surgically removed while preserving surrounding content.

The inline pattern structure is: `\n\n{title_pattern}\s+{page_number}`

For example, with pattern `Scale Communication.*`:
- **Before**: `"...scientific literature.\n\nScale Communication Through Writing 202 Aside from that..."`
- **After**: `"...scientific literature.\n\n Aside from that..."`

## Development

See [AGENTS.md](AGENTS.md) for contributor guidelines. The passes table in that file is auto-generated between the fenced markers; run `python scripts/update_agents_md.py` to refresh it.
Before introducing or altering architecture-level components or dependencies, consult the project maintainers to ensure alignment with overall design goals.
