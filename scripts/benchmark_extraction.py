#!/usr/bin/env python3
"""
Performance Benchmarking Script for PDF Extraction Methods

This script compares the performance of the hybrid PyMuPDF4LLM extraction approach
against the traditional three-tier fallback system (PyMuPDF → pdftotext → pdfminer.six).

Usage:
    python scripts/benchmark_extraction.py [pdf_files...] --output benchmark_report.json
    python scripts/benchmark_extraction.py sample_book.pdf --iterations 5 --output results.json
"""

import argparse
import json
import time
import tracemalloc
import statistics
import sys
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pdf_chunker.core import process_document
from pdf_chunker.pdf_parsing import extract_text_blocks_from_pdf
from pdf_chunker.pymupdf4llm_integration import (
    is_pymupdf4llm_available,
    extract_with_pymupdf4llm,
    assess_pymupdf4llm_quality,
    PyMuPDF4LLMExtractionError,
)
from pdf_chunker.extraction_fallbacks import (
    _assess_text_quality,
    execute_fallback_extraction,
)


@dataclass
class ExtractionMetrics:
    """Metrics for a single extraction run"""

    method: str
    pdf_file: str
    success: bool
    extraction_time: float
    memory_peak_mb: float
    text_length: int
    chunk_count: int
    heading_count: int
    avg_chunk_size: float
    quality_score: float
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkResults:
    """Complete benchmark results for comparison"""

    hybrid_metrics: List[ExtractionMetrics]
    traditional_metrics: List[ExtractionMetrics]
    comparison_summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hybrid_metrics": [m.to_dict() for m in self.hybrid_metrics],
            "traditional_metrics": [m.to_dict() for m in self.traditional_metrics],
            "comparison_summary": self.comparison_summary,
        }


def measure_extraction_performance(
    extraction_func, pdf_file: str, method_name: str, **kwargs
) -> ExtractionMetrics:
    """
    Measure performance metrics for a single extraction method.

    Args:
        extraction_func: Function to call for extraction
        pdf_file: Path to PDF file
        method_name: Name of the extraction method
        **kwargs: Additional arguments for extraction function

    Returns:
        ExtractionMetrics with performance data
    """
    # Start memory tracking
    tracemalloc.start()
    start_time = time.time()

    try:
        # Execute extraction
        result = extraction_func(pdf_file, **kwargs)

        # Calculate metrics
        extraction_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        memory_peak_mb = peak / 1024 / 1024  # Convert to MB

        # Analyze results
        if isinstance(result, list):
            # Traditional extraction returns list of blocks
            blocks = result
            chunks = result  # For traditional, blocks are essentially chunks
        else:
            # Core process_document returns structured result
            chunks = result if isinstance(result, list) else []
            blocks = chunks

        # Calculate text metrics
        text_length = sum(len(block.get("text", "")) for block in blocks)
        chunk_count = len(chunks)
        heading_count = sum(
            1
            for block in blocks
            if block.get("is_heading", False) or block.get("type") == "heading"
        )
        avg_chunk_size = text_length / chunk_count if chunk_count > 0 else 0

        # Assess quality
        combined_text = "\n".join(block.get("text", "") for block in blocks)
        quality_assessment = _assess_text_quality(combined_text)
        quality_score = quality_assessment.get("quality_score", 0.0)

        return ExtractionMetrics(
            method=method_name,
            pdf_file=os.path.basename(pdf_file),
            success=True,
            extraction_time=extraction_time,
            memory_peak_mb=memory_peak_mb,
            text_length=text_length,
            chunk_count=chunk_count,
            heading_count=heading_count,
            avg_chunk_size=avg_chunk_size,
            quality_score=quality_score,
        )

    except Exception as e:
        extraction_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        memory_peak_mb = peak / 1024 / 1024

        return ExtractionMetrics(
            method=method_name,
            pdf_file=os.path.basename(pdf_file),
            success=False,
            extraction_time=extraction_time,
            memory_peak_mb=memory_peak_mb,
            text_length=0,
            chunk_count=0,
            heading_count=0,
            avg_chunk_size=0,
            quality_score=0.0,
            error_message=str(e),
        )

    finally:
        tracemalloc.stop()


def extract_with_hybrid_approach(pdf_file: str, **kwargs) -> List[Dict[str, Any]]:
    """
    Extract using the hybrid PyMuPDF4LLM approach with fallback.

    Args:
        pdf_file: Path to PDF file
        **kwargs: Additional arguments (chunk_size, overlap, etc.)

    Returns:
        List of extracted chunks/blocks
    """
    # Use the core process_document function which implements the hybrid approach
    result = process_document(
        pdf_file,
        chunk_size=kwargs.get("chunk_size", 8000),
        overlap=kwargs.get("overlap", 200),
        exclude_pages=kwargs.get("exclude_pages"),
        generate_metadata=True,
        ai_enrichment=False,  # Skip AI enrichment for performance testing
    )

    return result


def extract_with_traditional_approach(pdf_file: str, **kwargs) -> List[Dict[str, Any]]:
    """
    Extract using only the traditional three-tier fallback approach.

    Args:
        pdf_file: Path to PDF file
        **kwargs: Additional arguments (exclude_pages, etc.)

    Returns:
        List of extracted blocks
    """
    # Force traditional extraction by bypassing PyMuPDF4LLM
    # We'll temporarily disable PyMuPDF4LLM for this test
    original_available = None

    try:
        # Temporarily disable PyMuPDF4LLM to force traditional extraction
        import pdf_chunker.pymupdf4llm_integration as pymupdf4llm_module

        original_available = pymupdf4llm_module.PYMUPDF4LLM_AVAILABLE
        pymupdf4llm_module.PYMUPDF4LLM_AVAILABLE = False

        # Use the PDF parsing function which will now use traditional methods
        blocks = extract_text_blocks_from_pdf(
            pdf_file, exclude_pages=kwargs.get("exclude_pages")
        )

        return blocks

    finally:
        # Restore original PyMuPDF4LLM availability
        if original_available is not None:
            pymupdf4llm_module.PYMUPDF4LLM_AVAILABLE = original_available


def run_benchmark_iteration(
    pdf_file: str, iteration: int, total_iterations: int, **kwargs
) -> tuple[ExtractionMetrics, ExtractionMetrics]:
    """
    Run a single benchmark iteration comparing both methods.

    Args:
        pdf_file: Path to PDF file
        iteration: Current iteration number
        total_iterations: Total number of iterations
        **kwargs: Additional arguments for extraction

    Returns:
        Tuple of (hybrid_metrics, traditional_metrics)
    """
    print(f"  Iteration {iteration + 1}/{total_iterations}...")

    # Test hybrid approach
    print(f"    Testing hybrid approach...")
    hybrid_metrics = measure_extraction_performance(
        extract_with_hybrid_approach, pdf_file, "hybrid_pymupdf4llm", **kwargs
    )

    # Test traditional approach
    print(f"    Testing traditional approach...")
    traditional_metrics = measure_extraction_performance(
        extract_with_traditional_approach, pdf_file, "traditional_fallback", **kwargs
    )

    return hybrid_metrics, traditional_metrics


def calculate_statistics(metrics_list: List[ExtractionMetrics]) -> Dict[str, Any]:
    """
    Calculate statistical summary for a list of metrics.

    Args:
        metrics_list: List of ExtractionMetrics

    Returns:
        Dictionary with statistical summary
    """
    if not metrics_list:
        return {}

    successful_metrics = [m for m in metrics_list if m.success]

    if not successful_metrics:
        return {
            "success_rate": 0.0,
            "total_runs": len(metrics_list),
            "successful_runs": 0,
            "failed_runs": len(metrics_list),
        }

    # Extract numeric values for statistics
    extraction_times = [m.extraction_time for m in successful_metrics]
    memory_peaks = [m.memory_peak_mb for m in successful_metrics]
    text_lengths = [m.text_length for m in successful_metrics]
    chunk_counts = [m.chunk_count for m in successful_metrics]
    heading_counts = [m.heading_count for m in successful_metrics]
    quality_scores = [m.quality_score for m in successful_metrics]

    def safe_stats(values: List[float]) -> Dict[str, float]:
        """Calculate statistics safely handling edge cases"""
        if not values:
            return {"mean": 0, "median": 0, "std_dev": 0, "min": 0, "max": 0}

        return {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values),
        }

    return {
        "success_rate": len(successful_metrics) / len(metrics_list),
        "total_runs": len(metrics_list),
        "successful_runs": len(successful_metrics),
        "failed_runs": len(metrics_list) - len(successful_metrics),
        "extraction_time": safe_stats(extraction_times),
        "memory_peak_mb": safe_stats(memory_peaks),
        "text_length": safe_stats(text_lengths),
        "chunk_count": safe_stats(chunk_counts),
        "heading_count": safe_stats(heading_counts),
        "quality_score": safe_stats(quality_scores),
    }


def compare_methods(
    hybrid_stats: Dict[str, Any], traditional_stats: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare performance between hybrid and traditional methods.

    Args:
        hybrid_stats: Statistics for hybrid method
        traditional_stats: Statistics for traditional method

    Returns:
        Comparison analysis
    """
    comparison = {
        "performance_comparison": {},
        "quality_comparison": {},
        "reliability_comparison": {},
        "recommendations": [],
    }

    # Performance comparison
    if (
        hybrid_stats.get("extraction_time", {}).get("mean", 0) > 0
        and traditional_stats.get("extraction_time", {}).get("mean", 0) > 0
    ):

        hybrid_time = hybrid_stats["extraction_time"]["mean"]
        traditional_time = traditional_stats["extraction_time"]["mean"]
        speed_ratio = traditional_time / hybrid_time

        comparison["performance_comparison"] = {
            "hybrid_avg_time": hybrid_time,
            "traditional_avg_time": traditional_time,
            "speed_ratio": speed_ratio,
            "faster_method": "hybrid" if speed_ratio > 1.0 else "traditional",
            "speed_improvement": abs(speed_ratio - 1.0) * 100,
        }

        # Memory comparison
        hybrid_memory = hybrid_stats.get("memory_peak_mb", {}).get("mean", 0)
        traditional_memory = traditional_stats.get("memory_peak_mb", {}).get("mean", 0)

        if hybrid_memory > 0 and traditional_memory > 0:
            memory_ratio = hybrid_memory / traditional_memory
            comparison["performance_comparison"]["memory_comparison"] = {
                "hybrid_avg_memory_mb": hybrid_memory,
                "traditional_avg_memory_mb": traditional_memory,
                "memory_ratio": memory_ratio,
                "lower_memory_method": (
                    "hybrid" if memory_ratio < 1.0 else "traditional"
                ),
            }

    # Quality comparison
    hybrid_quality = hybrid_stats.get("quality_score", {}).get("mean", 0)
    traditional_quality = traditional_stats.get("quality_score", {}).get("mean", 0)

    if hybrid_quality > 0 or traditional_quality > 0:
        comparison["quality_comparison"] = {
            "hybrid_avg_quality": hybrid_quality,
            "traditional_avg_quality": traditional_quality,
            "quality_difference": hybrid_quality - traditional_quality,
            "better_quality_method": (
                "hybrid" if hybrid_quality > traditional_quality else "traditional"
            ),
        }

        # Heading detection comparison
        hybrid_headings = hybrid_stats.get("heading_count", {}).get("mean", 0)
        traditional_headings = traditional_stats.get("heading_count", {}).get("mean", 0)

        comparison["quality_comparison"]["heading_detection"] = {
            "hybrid_avg_headings": hybrid_headings,
            "traditional_avg_headings": traditional_headings,
            "heading_difference": hybrid_headings - traditional_headings,
            "better_heading_detection": (
                "hybrid" if hybrid_headings > traditional_headings else "traditional"
            ),
        }

    # Reliability comparison
    hybrid_success_rate = hybrid_stats.get("success_rate", 0)
    traditional_success_rate = traditional_stats.get("success_rate", 0)

    comparison["reliability_comparison"] = {
        "hybrid_success_rate": hybrid_success_rate,
        "traditional_success_rate": traditional_success_rate,
        "reliability_difference": hybrid_success_rate - traditional_success_rate,
        "more_reliable_method": (
            "hybrid"
            if hybrid_success_rate > traditional_success_rate
            else "traditional"
        ),
    }

    # Generate recommendations
    recommendations = []

    # Performance recommendations
    perf_comp = comparison.get("performance_comparison", {})
    if perf_comp.get("speed_improvement", 0) > 10:
        faster = perf_comp.get("faster_method", "unknown")
        improvement = perf_comp.get("speed_improvement", 0)
        recommendations.append(f"{faster.title()} method is {improvement:.1f}% faster")

    # Quality recommendations
    qual_comp = comparison.get("quality_comparison", {})
    if abs(qual_comp.get("quality_difference", 0)) > 0.1:
        better = qual_comp.get("better_quality_method", "unknown")
        diff = abs(qual_comp.get("quality_difference", 0))
        recommendations.append(
            f"{better.title()} method has {diff:.2f} higher quality score"
        )

    # Heading detection recommendations
    heading_comp = qual_comp.get("heading_detection", {})
    if abs(heading_comp.get("heading_difference", 0)) > 0.5:
        better = heading_comp.get("better_heading_detection", "unknown")
        diff = abs(heading_comp.get("heading_difference", 0))
        recommendations.append(
            f"{better.title()} method detects {diff:.1f} more headings on average"
        )

    # Reliability recommendations
    rel_comp = comparison.get("reliability_comparison", {})
    if abs(rel_comp.get("reliability_difference", 0)) > 0.05:
        better = rel_comp.get("more_reliable_method", "unknown")
        diff = abs(rel_comp.get("reliability_difference", 0)) * 100
        recommendations.append(f"{better.title()} method is {diff:.1f}% more reliable")

    # Overall recommendation
    if not recommendations:
        recommendations.append("Both methods perform similarly across all metrics")

    comparison["recommendations"] = recommendations

    return comparison


def run_benchmark(
    pdf_files: List[str],
    iterations: int = 3,
    chunk_size: int = 8000,
    overlap: int = 200,
    exclude_pages: Optional[str] = None,
) -> BenchmarkResults:
    """
    Run complete benchmark comparing hybrid vs traditional extraction.

    Args:
        pdf_files: List of PDF files to test
        iterations: Number of iterations per file
        chunk_size: Chunk size for processing
        overlap: Overlap size for chunking
        exclude_pages: Pages to exclude from processing

    Returns:
        Complete benchmark results
    """
    print(
        f"Starting benchmark with {len(pdf_files)} PDF files, {iterations} iterations each"
    )
    print(f"PyMuPDF4LLM available: {is_pymupdf4llm_available()}")

    all_hybrid_metrics = []
    all_traditional_metrics = []

    extraction_kwargs = {
        "chunk_size": chunk_size,
        "overlap": overlap,
        "exclude_pages": exclude_pages,
    }

    for i, pdf_file in enumerate(pdf_files):
        print(
            f"\nProcessing file {i + 1}/{len(pdf_files)}: {os.path.basename(pdf_file)}"
        )

        if not os.path.exists(pdf_file):
            print(f"  WARNING: File not found: {pdf_file}")
            continue

        for iteration in range(iterations):
            hybrid_metrics, traditional_metrics = run_benchmark_iteration(
                pdf_file, iteration, iterations, **extraction_kwargs
            )

            all_hybrid_metrics.append(hybrid_metrics)
            all_traditional_metrics.append(traditional_metrics)

            # Print iteration summary
            print(
                f"    Hybrid: {hybrid_metrics.extraction_time:.2f}s, "
                f"{hybrid_metrics.text_length} chars, "
                f"{hybrid_metrics.chunk_count} chunks, "
                f"quality: {hybrid_metrics.quality_score:.2f}"
            )
            print(
                f"    Traditional: {traditional_metrics.extraction_time:.2f}s, "
                f"{traditional_metrics.text_length} chars, "
                f"{traditional_metrics.chunk_count} chunks, "
                f"quality: {traditional_metrics.quality_score:.2f}"
            )

    # Calculate statistics
    print("\nCalculating statistics...")
    hybrid_stats = calculate_statistics(all_hybrid_metrics)
    traditional_stats = calculate_statistics(all_traditional_metrics)

    # Compare methods
    comparison = compare_methods(hybrid_stats, traditional_stats)

    # Create summary
    comparison_summary = {
        "benchmark_config": {
            "pdf_files": [os.path.basename(f) for f in pdf_files],
            "iterations_per_file": iterations,
            "total_iterations": len(all_hybrid_metrics),
            "chunk_size": chunk_size,
            "overlap": overlap,
            "exclude_pages": exclude_pages,
            "pymupdf4llm_available": is_pymupdf4llm_available(),
        },
        "hybrid_statistics": hybrid_stats,
        "traditional_statistics": traditional_stats,
        "comparison_analysis": comparison,
    }

    return BenchmarkResults(
        hybrid_metrics=all_hybrid_metrics,
        traditional_metrics=all_traditional_metrics,
        comparison_summary=comparison_summary,
    )


def print_benchmark_summary(results: BenchmarkResults):
    """Print a human-readable summary of benchmark results"""
    summary = results.comparison_summary
    config = summary["benchmark_config"]
    comparison = summary["comparison_analysis"]

    print("\n" + "=" * 80)
    print("PDF EXTRACTION PERFORMANCE BENCHMARK RESULTS")
    print("=" * 80)

    print(f"\nBenchmark Configuration:")
    print(f"  Files tested: {', '.join(config['pdf_files'])}")
    print(f"  Iterations per file: {config['iterations_per_file']}")
    print(f"  Total test runs: {config['total_iterations']}")
    print(f"  Chunk size: {config['chunk_size']}")
    print(f"  Overlap: {config['overlap']}")
    print(f"  PyMuPDF4LLM available: {config['pymupdf4llm_available']}")

    # Performance comparison
    perf = comparison.get("performance_comparison", {})
    if perf:
        print(f"\nPerformance Comparison:")
        print(f"  Hybrid average time: {perf.get('hybrid_avg_time', 0):.2f}s")
        print(f"  Traditional average time: {perf.get('traditional_avg_time', 0):.2f}s")
        print(f"  Faster method: {perf.get('faster_method', 'unknown').title()}")
        print(f"  Speed improvement: {perf.get('speed_improvement', 0):.1f}%")

        mem_comp = perf.get("memory_comparison", {})
        if mem_comp:
            print(
                f"  Hybrid average memory: {mem_comp.get('hybrid_avg_memory_mb', 0):.1f} MB"
            )
            print(
                f"  Traditional average memory: {mem_comp.get('traditional_avg_memory_mb', 0):.1f} MB"
            )
            print(
                f"  Lower memory method: {mem_comp.get('lower_memory_method', 'unknown').title()}"
            )

    # Quality comparison
    qual = comparison.get("quality_comparison", {})
    if qual:
        print(f"\nQuality Comparison:")
        print(f"  Hybrid average quality: {qual.get('hybrid_avg_quality', 0):.2f}")
        print(
            f"  Traditional average quality: {qual.get('traditional_avg_quality', 0):.2f}"
        )
        print(
            f"  Better quality method: {qual.get('better_quality_method', 'unknown').title()}"
        )

        heading = qual.get("heading_detection", {})
        if heading:
            print(
                f"  Hybrid average headings: {heading.get('hybrid_avg_headings', 0):.1f}"
            )
            print(
                f"  Traditional average headings: {heading.get('traditional_avg_headings', 0):.1f}"
            )
            print(
                f"  Better heading detection: {heading.get('better_heading_detection', 'unknown').title()}"
            )

    # Reliability comparison
    rel = comparison.get("reliability_comparison", {})
    if rel:
        print(f"\nReliability Comparison:")
        print(f"  Hybrid success rate: {rel.get('hybrid_success_rate', 0):.1%}")
        print(
            f"  Traditional success rate: {rel.get('traditional_success_rate', 0):.1%}"
        )
        print(
            f"  More reliable method: {rel.get('more_reliable_method', 'unknown').title()}"
        )

    # Recommendations
    recommendations = comparison.get("recommendations", [])
    if recommendations:
        print(f"\nRecommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")

    print("\n" + "=" * 80)


def main():
    """Main function to run the benchmark"""
    parser = argparse.ArgumentParser(
        description="Benchmark PDF extraction performance: hybrid vs traditional methods"
    )
    parser.add_argument("pdf_files", nargs="+", help="PDF files to benchmark")
    parser.add_argument(
        "--output",
        "-o",
        default="benchmark_report.json",
        help="Output file for detailed results (default: benchmark_report.json)",
    )
    parser.add_argument(
        "--iterations",
        "-i",
        type=int,
        default=3,
        help="Number of iterations per file (default: 3)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8000,
        help="Chunk size for processing (default: 8000)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=200,
        help="Overlap size for chunking (default: 200)",
    )
    parser.add_argument("--exclude-pages", help='Pages to exclude (e.g., "1,3,5-10")')
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress detailed output"
    )

    args = parser.parse_args()

    # Validate PDF files
    valid_files = []
    for pdf_file in args.pdf_files:
        if os.path.exists(pdf_file):
            valid_files.append(pdf_file)
        else:
            print(f"WARNING: File not found: {pdf_file}")

    if not valid_files:
        print("ERROR: No valid PDF files found")
        sys.exit(1)

    # Run benchmark
    try:
        results = run_benchmark(
            pdf_files=valid_files,
            iterations=args.iterations,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            exclude_pages=args.exclude_pages,
        )

        # Save detailed results
        with open(args.output, "w") as f:
            json.dump(results.to_dict(), f, indent=2)

        if not args.quiet:
            print_benchmark_summary(results)

        print(f"\nDetailed results saved to: {args.output}")

        # Exit with appropriate code based on results
        hybrid_success = results.comparison_summary["hybrid_statistics"].get(
            "success_rate", 0
        )
        traditional_success = results.comparison_summary["traditional_statistics"].get(
            "success_rate", 0
        )

        if hybrid_success < 0.8 or traditional_success < 0.8:
            print("WARNING: Low success rate detected in benchmark results")
            sys.exit(1)

    except Exception as e:
        print(f"ERROR: Benchmark failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
