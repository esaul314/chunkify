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

> Supply a custom spec via `--spec pipeline.yaml` to override defaults.

## Development

See [AGENTS.md](AGENTS.md) for contributor guidelines. The passes table in that file is auto-generated between the fenced markers; run `python scripts/update_agents_md.py` to refresh it.
Before introducing or altering architecture-level components or dependencies, consult the project maintainers to ensure alignment with overall design goals.
