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
| `pdf_parse.footer_margin` | `--footer-margin` | Points from bottom to exclude as footer zone | – |
| `pdf_parse.header_margin` | `--header-margin` | Points from top to exclude as header zone | – |
| – | `--auto-detect-zones` | Auto-detect footer/header zones from page geometry | `false` |
| `split_semantic.chunk_size` | `--chunk-size` | Target tokens per chunk | `400` |
| `split_semantic.overlap` | `--overlap` | Tokens to overlap neighboring chunks | `50` |
| `split_semantic.generate_metadata` | `--no-metadata` (negates) | Include per-chunk metadata | `true` |
| `split_semantic.interactive_lists` | `--interactive-lists` | Prompt for list continuation confirmation | `false` |
| `emit_jsonl.output_path` | — | Path for resulting JSONL | `output.jsonl` |
| `text_clean.footer_patterns` | `--footer-pattern` | Regex patterns to strip as footers | – |
| `text_clean.interactive_footers` | `--interactive-footers` | Prompt for footer confirmation | `false` |
| – | `--interactive` | Enable all interactive prompts (footers + lists) | `false` |
| – | `--teach` | Learn from interactive decisions and persist for future runs | `false` |

> Supply a custom spec via `--spec pipeline.yaml` to override defaults.

## Footer Detection

Many books include chapter titles or book titles as running footers on each page. When PDFs are extracted, these footers often appear merged into the text body, causing mid-sentence interruptions like:

```
"There's conflicting scientific literature on the subject.

Scale Communication Through Writing 202 Aside from that, we can all likely agree..."
```

### Geometric Zone Detection (Recommended)

The most reliable footer removal uses **positional data** to exclude footer zones during extraction—before text blocks are merged. This approach eliminates context bleeding and works with any footer content.

```bash
# Auto-detect footer zones based on page geometry
pdf_chunker convert book.pdf --out out.jsonl --auto-detect-zones

# Interactive mode: review detected zones before processing
pdf_chunker convert book.pdf --out out.jsonl --interactive --auto-detect-zones

# Manually specify footer margin (points from page bottom)
pdf_chunker convert book.pdf --out out.jsonl --footer-margin 40

# Combine with page exclusions (skip cover, TOC, etc.)
pdf_chunker convert book.pdf --out out.jsonl --auto-detect-zones --exclude-pages 1-5
```

**How it works:**
- `--auto-detect-zones`: Analyzes Y coordinates of bottom blocks across sampled pages to find consistent footer positions
- `--footer-margin N`: Excludes all content within N points of the page bottom
- `--header-margin N`: Excludes all content within N points of the page top
- `--exclude-pages`: Skips specified pages during zone detection (useful for excluding cover pages, TOC, or appendices that have different layouts)

**Interactive zone discovery:**

When using `--interactive` with `--auto-detect-zones`, the CLI shows all detected footer candidates and lets you choose:

```
Analyzing 20 pages for zone detection...
(Excluding pages: [1, 2, 3])

Found 2 potential footer zone(s):

[1] Footer Zone Candidate
    Position: 621.0pt from top (45.5pt margin from bottom)
    Confidence: 93% (18/20 pages)
    Sample content:
      • Scale Communication Through Writing 202
      • Chapter 5: Advanced Topics 203

[2] Footer Zone Candidate
    Position: 640.0pt from top (26.5pt margin from bottom)
    Confidence: 45% (9/20 pages)
    Sample content:
      • 202
      • 203

Enter the number of the zone to exclude, or 0 to skip footer exclusion.
Your choice [1]: 
```

**YAML Configuration:**

```yaml
# pipeline.yaml
options:
  pdf_parse:
    footer_margin: 40.0  # Points from bottom
    header_margin: null  # Points from top (optional)
```

### Stripping Footers with Patterns (Legacy)

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

When you're unsure what footer patterns exist in a document, use `--interactive` to enable all interactive prompts (footers and list continuations), or use the granular flags for specific features:

```bash
# Enable all interactive prompts
pdf_chunker convert book.pdf --out out.jsonl --interactive

# Enable only footer prompts
pdf_chunker convert book.pdf --out out.jsonl --interactive-footers

# Enable only list continuation prompts  
pdf_chunker convert book.pdf --out out.jsonl --interactive-lists
```

**Footer heuristic detection**: Without `--footer-pattern`, the CLI uses a heuristic to find inline footers matching the pattern `\n\n{TitleCase Words} {PageNumber}` (e.g., "Scale Communication Through Writing 202").

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

### Learning Mode (--teach)

When processing multiple similar documents, you can use `--teach` mode to persist your interactive decisions. Once learned, patterns are automatically applied in future runs:

```bash
# First run: interactively teach the pipeline about your document's patterns
pdf_chunker convert book1.pdf --out out1.jsonl --interactive --teach

# Subsequent runs: learned patterns are applied automatically
pdf_chunker convert book2.pdf --out out2.jsonl
```

**How it works:**
- When you confirm or reject a footer/list continuation in `--teach` mode, the pattern is saved
- Patterns are stored in `~/.config/pdf_chunker/learned_patterns.yaml`
- On subsequent runs, matching patterns trigger the learned decision automatically
- Confidence thresholds determine when to apply learned patterns vs. prompt again

**Managing learned patterns:**
```bash
# View learned patterns
cat ~/.config/pdf_chunker/learned_patterns.yaml

# Reset learned patterns
rm ~/.config/pdf_chunker/learned_patterns.yaml
```

**Use cases:**
- Processing a book series with consistent footer formatting
- Batch converting documents from the same publisher
- Building organization-specific pattern libraries

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

## List Continuation Detection

PDF extraction often splits multi-line list items into separate text blocks. For example, a bullet point that wraps across lines may be extracted as:

```
Block 1: "• Reduce wordiness."
Block 2: "For every word ask: what information is it conveying?"
```

When this happens, naïve chunking produces semantically broken output where list item text is separated from its continuation.

### Automatic Merging

The `split_semantic` pass automatically detects and merges list continuations using heuristics:

- **Incomplete list items**: Short items (≤5 words), items ending with continuation punctuation (`,;:`), or items with unbalanced delimiters (parentheses, brackets, quotes)
- **Continuation signals**: Text that starts with lowercase letters or continuation words ("and", "or", "which", etc.)

### Interactive List Confirmation

For uncertain cases, enable interactive mode to manually confirm list continuations:

```bash
# Enable list continuation prompts only
pdf_chunker convert book.pdf --out out.jsonl --interactive-lists

# Enable all interactive prompts (footers + lists)
pdf_chunker convert book.pdf --out out.jsonl --interactive
```

The CLI will display the list item and candidate continuation:
```
--- List continuation candidate (page 15, confidence 75%) ---
  List item: • Reduce wordiness.
  Candidate: For every word ask: what information is it conveying?
  Heuristic: item_looks_incomplete+continuation_word
Merge into list item? [Y/n]
```

**Confidence-based decisions:**
- **High confidence (≥85%)**: Decision is applied automatically
- **Medium confidence (30-85%)**: Interactive prompt shown
- **Low confidence (<30%)**: Default behavior applied

Special pattern detection includes:
- **Q&A sequences**: Automatically detects "Q:" / "A:" patterns and applies appropriate merging
- **Colon-prefixed lists**: Recognizes items like "Key points:" followed by bullet lists

### YAML Configuration

```yaml
# pipeline.yaml
options:
  split_semantic:
    interactive_lists: true  # Enable list continuation prompts
```

For example, with pattern `Scale Communication.*`:
- **Before**: `"...scientific literature.\n\nScale Communication Through Writing 202 Aside from that..."`
- **After**: `"...scientific literature.\n\n Aside from that..."`

## Development

See [AGENTS.md](AGENTS.md) for contributor guidelines. The passes table in that file is auto-generated between the fenced markers; run `python scripts/update_agents_md.py` to refresh it.
Before introducing or altering architecture-level components or dependencies, consult the project maintainers to ensure alignment with overall design goals.
