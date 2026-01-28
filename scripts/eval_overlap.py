#!/usr/bin/env python3
"""Minimal overlap effectiveness evaluation for RAG chunking.

This script measures how well different overlap settings support retrieval
of boundary-spanning queries. It uses simple TF-IDF scoring to avoid
external dependencies.

Usage:
    # Generate chunks with different overlap values and compare recall
    python scripts/eval_overlap.py --corpus test_data/sample_test.pdf

    # Compare specific overlap values
    python scripts/eval_overlap.py --corpus test_data/sample_test.pdf --overlaps 0,50,100,150

    # Use custom queries file
    python scripts/eval_overlap.py --corpus test_data/sample_test.pdf --queries queries.json

Design rationale: docs/RAG_OVERLAP_ALIGNMENT_PLAN.md
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

# ---------------------------------------------------------------------------
# TF-IDF implementation (no external dependencies)
# ---------------------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer with lowercasing."""
    return re.findall(r"\b\w+\b", text.lower())


def compute_tf(tokens: list[str]) -> dict[str, float]:
    """Compute term frequency for a document."""
    counts = Counter(tokens)
    total = len(tokens)
    return {term: count / total for term, count in counts.items()} if total else {}


def compute_idf(documents: list[list[str]]) -> dict[str, float]:
    """Compute inverse document frequency across corpus."""
    n_docs = len(documents)
    if n_docs == 0:
        return {}

    doc_freq: Counter[str] = Counter()
    for doc_tokens in documents:
        doc_freq.update(set(doc_tokens))

    return {term: math.log((n_docs + 1) / (freq + 1)) + 1 for term, freq in doc_freq.items()}


def tfidf_score(query_tokens: list[str], doc_tf: dict[str, float], idf: dict[str, float]) -> float:
    """Compute TF-IDF similarity score between query and document."""
    score = 0.0
    for token in query_tokens:
        if token in doc_tf:
            score += doc_tf[token] * idf.get(token, 1.0)
    return score


# ---------------------------------------------------------------------------
# Chunk generation
# ---------------------------------------------------------------------------


def generate_chunks(corpus_path: Path, overlap: int, chunk_size: int = 400) -> list[dict]:
    """Generate chunks from corpus with specified overlap."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        output_path = Path(f.name)

    try:
        cmd = [
            sys.executable,
            "-m",
            "pdf_chunker.cli",
            "convert",
            str(corpus_path),
            "--out",
            str(output_path),
            "--chunk-size",
            str(chunk_size),
            "--overlap",
            str(overlap),
            "--no-enrich",
        ]
        subprocess.run(cmd, capture_output=True, check=True)

        chunks = []
        for line in output_path.read_text().splitlines():
            if line.strip():
                chunks.append(json.loads(line))
        return chunks
    finally:
        output_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@dataclass
class QueryResult:
    """Result of evaluating a single query."""

    query: str
    expected_chunks: list[int]
    retrieved_chunks: list[int]
    hit: bool


@dataclass
class EvalResult:
    """Result of evaluating overlap effectiveness."""

    overlap: int
    recall_at_k: float
    query_results: list[QueryResult]
    num_chunks: int


def create_boundary_queries(chunks: list[dict]) -> list[dict]:
    """Create synthetic queries that span chunk boundaries.

    These queries specifically test whether overlap helps retrieval
    by targeting text that appears at the end of one chunk and
    the beginning of the next.
    """
    queries = []

    for i in range(len(chunks) - 1):
        curr_text = chunks[i].get("text", "")
        next_text = chunks[i + 1].get("text", "")

        # Extract last sentence fragment from current chunk
        curr_tokens = tokenize(curr_text)
        next_tokens = tokenize(next_text)

        if len(curr_tokens) < 10 or len(next_tokens) < 10:
            continue

        # Create a query spanning the boundary (last 5 words + first 5 words)
        boundary_tokens = curr_tokens[-5:] + next_tokens[:5]
        query_text = " ".join(boundary_tokens)

        queries.append(
            {
                "query": query_text,
                "expected_chunk_indices": [i, i + 1],
                "description": f"Boundary query spanning chunks {i} and {i + 1}",
            }
        )

    return queries


def evaluate_overlap(
    chunks: list[dict],
    queries: list[dict],
    k: int = 3,
    overlap: int = 0,
) -> EvalResult:
    """Evaluate retrieval quality for given chunks and queries.

    Parameters
    ----------
    chunks : list[dict]
        Chunked documents with 'text' field.
    queries : list[dict]
        Queries with 'query' and 'expected_chunk_indices' fields.
    k : int
        Number of top results to consider for recall calculation.
    overlap : int
        Overlap value used (for reporting).

    Returns
    -------
    EvalResult
        Recall@k and per-query results.
    """
    if not chunks or not queries:
        return EvalResult(
            overlap=overlap, recall_at_k=0.0, query_results=[], num_chunks=len(chunks)
        )

    # Tokenize all chunks
    chunk_tokens = [tokenize(c.get("text", "")) for c in chunks]
    chunk_tfs = [compute_tf(tokens) for tokens in chunk_tokens]
    idf = compute_idf(chunk_tokens)

    results = []
    hits = 0

    for q in queries:
        query_tokens = tokenize(q["query"])
        expected = set(q.get("expected_chunk_indices", []))

        # Score all chunks
        scores = [(i, tfidf_score(query_tokens, tf, idf)) for i, tf in enumerate(chunk_tfs)]
        scores.sort(key=lambda x: -x[1])

        # Get top-k chunk indices
        top_k = [idx for idx, _ in scores[:k]]

        # Check if any expected chunk is in top-k
        hit = bool(expected & set(top_k))
        if hit:
            hits += 1

        results.append(
            QueryResult(
                query=q["query"],
                expected_chunks=list(expected),
                retrieved_chunks=top_k,
                hit=hit,
            )
        )

    recall = hits / len(queries) if queries else 0.0
    return EvalResult(
        overlap=overlap,
        recall_at_k=recall,
        query_results=results,
        num_chunks=len(chunks),
    )


def run_ab_comparison(
    corpus_path: Path,
    overlaps: Sequence[int],
    k: int = 3,
    chunk_size: int = 400,
    queries_file: Path | None = None,
) -> list[EvalResult]:
    """Run A/B comparison across different overlap values.

    Parameters
    ----------
    corpus_path : Path
        Path to PDF or EPUB file.
    overlaps : Sequence[int]
        List of overlap values to compare.
    k : int
        Top-k for recall calculation.
    chunk_size : int
        Target chunk size in words.
    queries_file : Path | None
        Optional custom queries file. If None, synthetic boundary queries are generated.

    Returns
    -------
    list[EvalResult]
        Results for each overlap value.
    """
    results = []

    for overlap in overlaps:
        print(f"  Generating chunks with overlap={overlap}...", file=sys.stderr)
        chunks = generate_chunks(corpus_path, overlap, chunk_size)

        if queries_file and queries_file.exists():
            queries = json.loads(queries_file.read_text())
        else:
            queries = create_boundary_queries(chunks)

        print(
            f"  Evaluating {len(queries)} queries against {len(chunks)} chunks...",
            file=sys.stderr,
        )
        result = evaluate_overlap(chunks, queries, k=k, overlap=overlap)
        results.append(result)

    return results


def format_results(results: list[EvalResult], verbose: bool = False) -> str:
    """Format evaluation results as a readable report."""
    lines = [
        "=" * 60,
        "RAG Overlap Evaluation Results",
        "=" * 60,
        "",
        "Summary:",
        "-" * 40,
        f"{'Overlap':>10} | {'Chunks':>8} | {'Recall@3':>10}",
        "-" * 40,
    ]

    for r in results:
        lines.append(f"{r.overlap:>10} | {r.num_chunks:>8} | {r.recall_at_k:>10.2%}")

    lines.append("-" * 40)

    if verbose and results:
        lines.append("")
        lines.append(f"Per-Query Results (overlap={results[-1].overlap}):")
        lines.append("-" * 40)
        for qr in results[-1].query_results[:5]:  # Show first 5
            status = "✓" if qr.hit else "✗"
            lines.append(f"  {status} Query: {qr.query[:50]}...")
            lines.append(f"    Expected: {qr.expected_chunks}, Got: {qr.retrieved_chunks}")

    lines.append("")
    lines.append("Interpretation:")
    lines.append("  - Higher recall@k means better retrieval of boundary-spanning content")
    lines.append("  - If recall improves with overlap, boundaries are being crossed")
    lines.append("  - If recall is similar across values, overlap tuning has diminishing returns")
    lines.append("")
    lines.append("See docs/RAG_OVERLAP_ALIGNMENT_PLAN.md for tuning guidance.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    """Run overlap evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate RAG overlap effectiveness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="Path to PDF or EPUB file to chunk",
    )
    parser.add_argument(
        "--overlaps",
        type=str,
        default="0,50,100,150",
        help="Comma-separated overlap values to compare (default: 0,50,100,150)",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=None,
        help="Optional JSON file with custom queries",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=400,
        help="Target chunk size in words (default: 400)",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=3,
        help="Top-k for recall calculation (default: 3)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show per-query results",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    args = parser.parse_args()

    if not args.corpus.exists():
        print(f"Error: Corpus file not found: {args.corpus}", file=sys.stderr)
        return 1

    overlaps = [int(x.strip()) for x in args.overlaps.split(",")]

    print(f"Evaluating overlap effectiveness for: {args.corpus}", file=sys.stderr)
    print(f"Overlap values: {overlaps}", file=sys.stderr)
    print(f"Chunk size: {args.chunk_size}, k={args.k}", file=sys.stderr)
    print("", file=sys.stderr)

    results = run_ab_comparison(
        corpus_path=args.corpus,
        overlaps=overlaps,
        k=args.k,
        chunk_size=args.chunk_size,
        queries_file=args.queries,
    )

    if args.json:
        output = [
            {
                "overlap": r.overlap,
                "recall_at_k": r.recall_at_k,
                "num_chunks": r.num_chunks,
                "k": args.k,
            }
            for r in results
        ]
        print(json.dumps(output, indent=2))
    else:
        print(format_results(results, verbose=args.verbose))

    return 0


if __name__ == "__main__":
    sys.exit(main())
