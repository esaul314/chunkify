#!/usr/bin/env python3
"""
PyMuPDF4LLM Experiment Script

Compares the current PDF extraction pipeline with PyMuPDF4LLM to evaluate:
- Text extraction quality
- Heading detection accuracy
- Chunk boundary consistency
- Page exclusion capabilities
- Metadata richness

Usage:
    python scripts/experiment_pymupdf4llm.py [pdf_file] [--pages-to-exclude 1,2,3]
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import time

# Add the project root to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pymupdf4llm
except ImportError:
    print("ERROR: pymupdf4llm not installed. Install with: pip install pymupdf4llm")
    sys.exit(1)

from pdf_chunker.core import process_document
from pdf_chunker.pdf_parsing import extract_text_blocks_from_pdf
from pdf_chunker.text_cleaning import clean_text


@dataclass
class ExtractionResult:
    """Results from a single extraction method"""
    method: str
    text_length: int
    chunk_count: int
    heading_count: int
    processing_time: float
    chunks: List[Dict[str, Any]]
    headings: List[str]
    metadata: Dict[str, Any]
    errors: List[str]


@dataclass
class ComparisonReport:
    """Comparison report between extraction methods"""
    pdf_file: str
    current_pipeline: ExtractionResult
    pymupdf4llm: ExtractionResult
    comparison_metrics: Dict[str, Any]
    recommendations: List[str]


def extract_with_current_pipeline(pdf_path: str, pages_to_exclude: Optional[List[int]] = None) -> ExtractionResult:
    """Extract using the current pipeline"""
    start_time = time.time()
    errors = []
    
    try:
        # Use the full pipeline
        result = process_document(
            pdf_path, 
            pages_to_exclude=pages_to_exclude or [],
            domain_tags_file=None  # Skip AI enrichment for comparison
        )
        
        chunks = result.get('chunks', [])
        text_length = sum(len(chunk.get('text', '')) for chunk in chunks)
        
        # Extract headings from chunks
        headings = []
        for chunk in chunks:
            if chunk.get('metadata', {}).get('is_heading', False):
                headings.append(chunk.get('text', '').strip())
        
        processing_time = time.time() - start_time
        
        return ExtractionResult(
            method="Current Pipeline",
            text_length=text_length,
            chunk_count=len(chunks),
            heading_count=len(headings),
            processing_time=processing_time,
            chunks=chunks,
            headings=headings,
            metadata=result.get('metadata', {}),
            errors=errors
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        errors.append(f"Pipeline error: {str(e)}")
        
        return ExtractionResult(
            method="Current Pipeline",
            text_length=0,
            chunk_count=0,
            heading_count=0,
            processing_time=processing_time,
            chunks=[],
            headings=[],
            metadata={},
            errors=errors
        )


def extract_with_pymupdf4llm(pdf_path: str, pages_to_exclude: Optional[List[int]] = None) -> ExtractionResult:
    """Extract using PyMuPDF4LLM"""
    start_time = time.time()
    errors = []
    
    try:
        # PyMuPDF4LLM extraction
        md_text = pymupdf4llm.to_markdown(
            pdf_path,
            pages=None if not pages_to_exclude else [i for i in range(1, 1000) if i not in pages_to_exclude],
            write_images=False,
            image_path=None,
            image_format="png",
            dpi=150
        )
        
        # Parse the markdown to extract headings and create chunks
        lines = md_text.split('\n')
        headings = []
        chunks = []
        current_chunk = []
        current_heading = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Detect markdown headings
            if line.startswith('#'):
                # Save previous chunk if it exists
                if current_chunk:
                    chunk_text = '\n'.join(current_chunk)
                    chunks.append({
                        'text': chunk_text,
                        'metadata': {
                            'is_heading': False,
                            'heading': current_heading,
                            'source': 'pymupdf4llm'
                        }
                    })
                    current_chunk = []
                
                # Extract heading text
                heading_text = line.lstrip('#').strip()
                headings.append(heading_text)
                current_heading = heading_text
                
                # Add heading as a chunk
                chunks.append({
                    'text': heading_text,
                    'metadata': {
                        'is_heading': True,
                        'heading_level': len(line) - len(line.lstrip('#')),
                        'source': 'pymupdf4llm'
                    }
                })
            else:
                current_chunk.append(line)
        
        # Add final chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'metadata': {
                    'is_heading': False,
                    'heading': current_heading,
                    'source': 'pymupdf4llm'
                }
            })
        
        text_length = len(md_text)
        processing_time = time.time() - start_time
        
        return ExtractionResult(
            method="PyMuPDF4LLM",
            text_length=text_length,
            chunk_count=len(chunks),
            heading_count=len(headings),
            processing_time=processing_time,
            chunks=chunks,
            headings=headings,
            metadata={'markdown_length': len(md_text)},
            errors=errors
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        errors.append(f"PyMuPDF4LLM error: {str(e)}")
        
        return ExtractionResult(
            method="PyMuPDF4LLM",
            text_length=0,
            chunk_count=0,
            heading_count=0,
            processing_time=processing_time,
            chunks=[],
            headings=[],
            metadata={},
            errors=errors
        )


def calculate_comparison_metrics(current: ExtractionResult, pymupdf4llm: ExtractionResult) -> Dict[str, Any]:
    """Calculate comparison metrics between the two extraction methods"""
    
    metrics = {}
    
    # Text length comparison
    if current.text_length > 0 and pymupdf4llm.text_length > 0:
        length_ratio = pymupdf4llm.text_length / current.text_length
        metrics['text_length_ratio'] = length_ratio
        metrics['text_length_difference'] = pymupdf4llm.text_length - current.text_length
    else:
        metrics['text_length_ratio'] = 0
        metrics['text_length_difference'] = 0
    
    # Chunk count comparison
    metrics['chunk_count_difference'] = pymupdf4llm.chunk_count - current.chunk_count
    
    # Heading detection comparison
    metrics['heading_count_difference'] = pymupdf4llm.heading_count - current.heading_count
    
    # Performance comparison
    if current.processing_time > 0:
        speed_ratio = current.processing_time / pymupdf4llm.processing_time
        metrics['speed_ratio'] = speed_ratio
    else:
        metrics['speed_ratio'] = 0
    
    # Error comparison
    metrics['current_errors'] = len(current.errors)
    metrics['pymupdf4llm_errors'] = len(pymupdf4llm.errors)
    
    # Heading overlap analysis
    current_headings_lower = [h.lower().strip() for h in current.headings]
    pymupdf4llm_headings_lower = [h.lower().strip() for h in pymupdf4llm.headings]
    
    common_headings = set(current_headings_lower) & set(pymupdf4llm_headings_lower)
    metrics['heading_overlap_count'] = len(common_headings)
    
    if current.heading_count > 0:
        metrics['heading_overlap_ratio'] = len(common_headings) / current.heading_count
    else:
        metrics['heading_overlap_ratio'] = 0
    
    return metrics


def generate_recommendations(comparison: ComparisonReport) -> List[str]:
    """Generate recommendations based on comparison results"""
    recommendations = []
    
    current = comparison.current_pipeline
    pymupdf4llm = comparison.pymupdf4llm
    metrics = comparison.comparison_metrics
    
    # Error analysis
    if current.errors and not pymupdf4llm.errors:
        recommendations.append("PyMuPDF4LLM shows better error handling - consider migration")
    elif pymupdf4llm.errors and not current.errors:
        recommendations.append("Current pipeline shows better error handling - keep current approach")
    
    # Text extraction quality
    if metrics.get('text_length_ratio', 0) > 1.1:
        recommendations.append("PyMuPDF4LLM extracts significantly more text - may indicate better extraction")
    elif metrics.get('text_length_ratio', 0) < 0.9:
        recommendations.append("Current pipeline extracts more text - PyMuPDF4LLM may be missing content")
    
    # Heading detection
    if metrics.get('heading_count_difference', 0) > 5:
        recommendations.append("PyMuPDF4LLM detects significantly more headings - may improve document structure")
    elif metrics.get('heading_count_difference', 0) < -5:
        recommendations.append("Current pipeline detects more headings - PyMuPDF4LLM may be missing structure")
    
    # Performance
    if metrics.get('speed_ratio', 0) > 2:
        recommendations.append("PyMuPDF4LLM is significantly faster - performance benefit")
    elif metrics.get('speed_ratio', 0) < 0.5:
        recommendations.append("Current pipeline is faster - PyMuPDF4LLM has performance cost")
    
    # Heading overlap
    if metrics.get('heading_overlap_ratio', 0) < 0.5:
        recommendations.append("Low heading overlap - methods detect different document structures")
    elif metrics.get('heading_overlap_ratio', 0) > 0.8:
        recommendations.append("High heading overlap - methods agree on document structure")
    
    if not recommendations:
        recommendations.append("Results are comparable - choice depends on specific requirements")
    
    return recommendations


def run_comparison(pdf_path: str, pages_to_exclude: Optional[List[int]] = None) -> ComparisonReport:
    """Run comparison between current pipeline and PyMuPDF4LLM"""
    
    print(f"Running comparison on: {pdf_path}")
    if pages_to_exclude:
        print(f"Excluding pages: {pages_to_exclude}")
    
    print("\n1. Extracting with current pipeline...")
    current_result = extract_with_current_pipeline(pdf_path, pages_to_exclude)
    
    print("2. Extracting with PyMuPDF4LLM...")
    pymupdf4llm_result = extract_with_pymupdf4llm(pdf_path, pages_to_exclude)
    
    print("3. Calculating comparison metrics...")
    metrics = calculate_comparison_metrics(current_result, pymupdf4llm_result)
    
    comparison = ComparisonReport(
        pdf_file=pdf_path,
        current_pipeline=current_result,
        pymupdf4llm=pymupdf4llm_result,
        comparison_metrics=metrics,
        recommendations=[]
    )
    
    comparison.recommendations = generate_recommendations(comparison)
    
    return comparison


def print_comparison_report(comparison: ComparisonReport):
    """Print a formatted comparison report"""
    
    print(f"\n{'='*80}")
    print(f"PDF EXTRACTION COMPARISON REPORT")
    print(f"{'='*80}")
    print(f"File: {comparison.pdf_file}")
    print(f"{'='*80}")
    
    current = comparison.current_pipeline
    pymupdf4llm = comparison.pymupdf4llm
    metrics = comparison.comparison_metrics
    
    # Summary table
    print(f"\n{'EXTRACTION SUMMARY':<30} {'Current':<15} {'PyMuPDF4LLM':<15} {'Difference':<15}")
    print(f"{'-'*75}")
    print(f"{'Text Length':<30} {current.text_length:<15} {pymupdf4llm.text_length:<15} {metrics.get('text_length_difference', 0):<15}")
    print(f"{'Chunk Count':<30} {current.chunk_count:<15} {pymupdf4llm.chunk_count:<15} {metrics.get('chunk_count_difference', 0):<15}")
    print(f"{'Heading Count':<30} {current.heading_count:<15} {pymupdf4llm.heading_count:<15} {metrics.get('heading_count_difference', 0):<15}")

    print(f"{'Processing Time (s)':<30} {current.processing_time:<15.2f} {pymupdf4llm.processing_time:<15.2f} {(pymupdf4llm.processing_time - current.processing_time):<15.2f}")
    
    print(f"{'Errors':<30} {len(current.errors):<15} {len(pymupdf4llm.errors):<15} {len(pymupdf4llm.errors) - len(current.errors):<15}")

    # Detailed metrics
    print(f"\n{'DETAILED METRICS'}")
    print(f"{'-'*40}")
    print(f"Text Length Ratio: {metrics.get('text_length_ratio', 0):.2f}")
    print(f"Speed Ratio: {metrics.get('speed_ratio', 0):.2f}")
    print(f"Heading Overlap: {metrics.get('heading_overlap_count', 0)} / {current.heading_count} ({metrics.get('heading_overlap_ratio', 0):.1%})")
    
    # Errors
    if current.errors or pymupdf4llm.errors:
        print(f"\n{'ERRORS'}")
        print(f"{'-'*40}")
        if current.errors:
            print("Current Pipeline Errors:")
            for error in current.errors:
                print(f"  - {error}")
        if pymupdf4llm.errors:
            print("PyMuPDF4LLM Errors:")
            for error in pymupdf4llm.errors:
                print(f"  - {error}")
    
    # Recommendations
    print(f"\n{'RECOMMENDATIONS'}")
    print(f"{'-'*40}")
    for i, rec in enumerate(comparison.recommendations, 1):
        print(f"{i}. {rec}")
    
    print(f"\n{'='*80}")


def save_detailed_report(comparison: ComparisonReport, output_file: str):
    """Save detailed comparison report to JSON file"""
    
    # Convert dataclasses to dictionaries for JSON serialization
    report_data = {
        'pdf_file': comparison.pdf_file,
        'current_pipeline': asdict(comparison.current_pipeline),
        'pymupdf4llm': asdict(comparison.pymupdf4llm),
        'comparison_metrics': comparison.comparison_metrics,
        'recommendations': comparison.recommendations,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Compare PDF extraction methods')
    parser.add_argument('pdf_file', help='Path to PDF file to analyze')
    parser.add_argument('--pages-to-exclude', help='Comma-separated list of page numbers to exclude (e.g., 1,2,3)')
    parser.add_argument('--output', help='Output file for detailed JSON report')
    
    args = parser.parse_args()
    
    # Validate PDF file
    if not os.path.exists(args.pdf_file):
        print(f"ERROR: PDF file not found: {args.pdf_file}")
        sys.exit(1)
    
    # Parse pages to exclude
    pages_to_exclude = None
    if args.pages_to_exclude:
        try:
            pages_to_exclude = [int(p.strip()) for p in args.pages_to_exclude.split(',')]
        except ValueError:
            print("ERROR: Invalid pages format. Use comma-separated integers (e.g., 1,2,3)")
            sys.exit(1)
    
    # Run comparison
    try:
        comparison = run_comparison(args.pdf_file, pages_to_exclude)
        print_comparison_report(comparison)
        
        # Save detailed report if requested
        if args.output:
            save_detailed_report(comparison, args.output)
        
    except Exception as e:
        print(f"ERROR: Comparison failed: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
