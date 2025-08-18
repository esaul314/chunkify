"""Shim module preserving public API for AI enrichment utilities.

The original logic now lives in :mod:`pdf_chunker.passes.ai_enrich`
(for pure classification) and :mod:`pdf_chunker.adapters.ai_enrich`
(for LLM and file I/O). This module re-exports those functions to
maintain compatibility.
"""

import os
import sys

from pdf_chunker.adapters.ai_enrich import (
    _load_tag_configs,
    init_llm,
    _process_chunk_for_file,
    _process_jsonl_file,
)
from pdf_chunker.passes.ai_enrich import classify_chunk_utterance

__all__ = [
    "classify_chunk_utterance",
    "_load_tag_configs",
    "init_llm",
    "_process_chunk_for_file",
    "_process_jsonl_file",
]


def main() -> None:
    """Run the AI enrichment script from the command line."""
    if len(sys.argv) != 3:
        print(
            "Usage: python -m pdf_chunker.ai_enrichment <input_file.jsonl> <output_file.jsonl>",
            file=sys.stderr,
        )
        sys.exit(1)

    input_file, output_file = sys.argv[1], sys.argv[2]
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at '{input_file}'", file=sys.stderr)
        sys.exit(1)

    print(f"Starting AI enrichment for '{input_file}'...")
    completion_fn = init_llm()
    _process_jsonl_file(input_file, output_file, completion_fn)
    print(f"Enrichment complete. Output saved to '{output_file}'.")


if __name__ == "__main__":
    main()
