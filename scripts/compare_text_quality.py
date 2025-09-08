#!/usr/bin/env python3
"""
Text Quality Comparison Script for PDF Extraction Methods

This script compares the text quality of JSONL outputs from the hybrid PyMuPDF4LLM
extraction approach against the traditional three-tier fallback system. It focuses
on text integrity metrics like sentence boundaries, word joining, whitespace handling,
ligature translation, and detection of misplaced headers/footers.

Usage:
    python scripts/compare_text_quality.py [pdf_files...] --output quality_comparison.json
    python scripts/compare_text_quality.py sample_book.pdf --detailed --output results.json
"""

import argparse
import json
import re
import sys
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import statistics

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pdf_chunker.core import process_document
from pdf_chunker.pdf_parsing import extract_text_blocks_from_pdf
from pdf_chunker.text_cleaning import clean_text, clean_paragraph


@dataclass
class TextQualityMetrics:
    """Metrics for text quality assessment"""

    method: str
    pdf_file: str
    total_chunks: int
    total_text_length: int
    avg_chunk_length: float
    sentence_integrity_score: float
    word_joining_score: float
    whitespace_quality_score: float
    ligature_translation_score: float
    header_footer_contamination_score: float
    overall_quality_score: float
    issues_detected: List[str]
    sample_issues: List[Dict[str, str]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QualityComparison:
    """Complete quality comparison results"""

    hybrid_metrics: TextQualityMetrics
    traditional_metrics: TextQualityMetrics
    comparison_analysis: Dict[str, Any]
    detailed_examples: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hybrid_metrics": self.hybrid_metrics.to_dict(),
            "traditional_metrics": self.traditional_metrics.to_dict(),
            "comparison_analysis": self.comparison_analysis,
            "detailed_examples": self.detailed_examples,
        }


def extract_with_hybrid_method(pdf_file: str) -> List[Dict[str, Any]]:
    """
    Extract text using the hybrid PyMuPDF4LLM approach.

    Args:
        pdf_file: Path to PDF file

    Returns:
        List of chunks with text and metadata
    """
    try:
        result = process_document(
            pdf_file,
            chunk_size=8000,
            overlap=200,
            generate_metadata=True,
            ai_enrichment=False,  # Skip AI enrichment for quality testing
        )
        return result
    except Exception as e:
        print(f"ERROR: Hybrid extraction failed for {pdf_file}: {e}")
        return []


def extract_with_traditional_method(pdf_file: str) -> List[Dict[str, Any]]:
    """
    Extract text using only the traditional three-tier fallback approach.

    Args:
        pdf_file: Path to PDF file

    Returns:
        List of chunks with text and metadata
    """
    try:
        # Temporarily disable PyMuPDF4LLM to force traditional extraction
        import pdf_chunker.pymupdf4llm_integration as pymupdf4llm_module

        original_available = pymupdf4llm_module.PYMUPDF4LLM_AVAILABLE
        pymupdf4llm_module.PYMUPDF4LLM_AVAILABLE = False

        blocks = [asdict(b) for b in extract_text_blocks_from_pdf(pdf_file)]

        chunks = [
            {
                "text": block.get("text", ""),
                "metadata": {
                    "chunk_id": f"traditional_{i}",
                    "page": block.get("source", {}).get("page", 1),
                    "language": block.get("language", "unknown"),
                    "type": block.get("type", "paragraph"),
                    "extraction_method": "traditional",
                },
            }
            for i, block in enumerate(blocks)
        ]

        return chunks

    except Exception as e:
        print(f"ERROR: Traditional extraction failed for {pdf_file}: {e}")
        return []
    finally:
        # Restore original PyMuPDF4LLM availability
        if "original_available" in locals():
            pymupdf4llm_module.PYMUPDF4LLM_AVAILABLE = original_available


def assess_sentence_integrity(text: str) -> Tuple[float, List[str]]:
    """
    Assess sentence integrity by detecting broken sentences and improper boundaries.

    Args:
        text: Text to analyze

    Returns:
        Tuple of (score, issues_list) where score is 0-1
    """
    issues = []

    # Split into sentences using basic punctuation
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return 0.0, ["No sentences detected"]

    broken_sentences = 0
    total_sentences = len(sentences)

    for sentence in sentences:
        # Check for sentences that are too short (likely broken)
        if len(sentence.split()) < 3:
            broken_sentences += 1
            issues.append(f"Very short sentence: '{sentence[:50]}...'")

        # Check for sentences starting with lowercase (likely continuation)
        if sentence and sentence[0].islower():
            broken_sentences += 1
            issues.append(f"Sentence starts with lowercase: '{sentence[:50]}...'")

        # Check for sentences ending mid-word (hyphenation issues)
        if sentence.endswith("-"):
            broken_sentences += 1
            issues.append(f"Sentence ends with hyphen: '{sentence[-20:]}'")

    # Calculate score (higher is better)
    score = max(0.0, 1.0 - (broken_sentences / total_sentences))

    return score, issues[:5]  # Limit to 5 examples


def assess_word_joining(text: str) -> Tuple[float, List[str]]:
    """
    Assess word joining quality by detecting improperly joined or split words.

    Args:
        text: Text to analyze

    Returns:
        Tuple of (score, issues_list) where score is 0-1
    """
    issues = []

    # Common patterns indicating word joining problems
    patterns = [
        (r"\b[a-z]+[A-Z][a-z]+\b", "CamelCase words (likely joined)"),
        (r"\b\w{1,2}\s+\w{1,2}\s+\w{1,2}\b", "Excessive single/double letter words"),
        (r"\b[a-z]+-\s*\n\s*[a-z]+\b", "Hyphenated words across lines"),
        (r"\b\w+\s+\w+(?=\w)", "Missing spaces between words"),
        (r"\b[a-z]+[0-9]+[a-z]+\b", "Words with embedded numbers"),
    ]

    total_words = len(text.split())
    if total_words == 0:
        return 0.0, ["No words detected"]

    problem_count = 0

    for pattern, description in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            problem_count += len(matches)
            # Add sample issues
            for match in matches[:3]:  # Limit to 3 examples per pattern
                issues.append(f"{description}: '{match}'")

    # Calculate score (higher is better)
    score = max(0.0, 1.0 - (problem_count / total_words))

    return score, issues[:10]  # Limit to 10 examples


def assess_whitespace_quality(text: str) -> Tuple[float, List[str]]:
    """
    Assess whitespace handling quality by detecting excessive or missing whitespace.

    Args:
        text: Text to analyze

    Returns:
        Tuple of (score, issues_list) where score is 0-1
    """
    issues = []

    # Check for excessive newlines
    excessive_newlines = len(re.findall(r"\n{3,}", text))
    if excessive_newlines > 0:
        issues.append(f"Found {excessive_newlines} instances of 3+ consecutive newlines")

    # Check for missing spaces after punctuation
    missing_spaces = len(re.findall(r"[.!?][a-zA-Z]", text))
    if missing_spaces > 0:
        issues.append(f"Found {missing_spaces} instances of missing spaces after punctuation")

    # Check for excessive spaces
    excessive_spaces = len(re.findall(r" {3,}", text))
    if excessive_spaces > 0:
        issues.append(f"Found {excessive_spaces} instances of 3+ consecutive spaces")

    # Check for tabs mixed with spaces (inconsistent indentation)
    mixed_whitespace = "\t" in text and "  " in text
    if mixed_whitespace:
        issues.append("Mixed tabs and spaces detected")

    # Check for trailing whitespace on lines
    lines = text.split("\n")
    trailing_whitespace = sum(1 for line in lines if line.endswith(" ") or line.endswith("\t"))
    if trailing_whitespace > len(lines) * 0.1:  # More than 10% of lines
        issues.append(f"Excessive trailing whitespace on {trailing_whitespace} lines")

    # Calculate score based on issues found
    total_issues = excessive_newlines + missing_spaces + excessive_spaces + trailing_whitespace
    if mixed_whitespace:
        total_issues += 1

    # Normalize by text length
    text_length = len(text)
    if text_length == 0:
        return 0.0, ["No text to analyze"]

    issue_density = total_issues / (text_length / 1000)  # Issues per 1000 characters
    score = max(0.0, 1.0 - min(issue_density, 1.0))

    return score, issues[:8]  # Limit to 8 examples


def assess_ligature_translation(text: str) -> Tuple[float, List[str]]:
    """
    Assess ligature translation quality by detecting untranslated ligatures.

    Args:
        text: Text to analyze

    Returns:
        Tuple of (score, issues_list) where score is 0-1
    """
    issues = []

    # Common ligatures that should be translated
    ligatures = {
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬀ": "ff",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "ﬆ": "st",
        "ﬅ": "ft",
        "﬈": "ft",
        "ﬗ": "et",
        "ﬞ": "at",
    }

    total_chars = len(text)
    if total_chars == 0:
        return 0.0, ["No text to analyze"]

    untranslated_count = 0

    for ligature, replacement in ligatures.items():
        count = text.count(ligature)
        if count > 0:
            untranslated_count += count
            issues.append(
                f"Untranslated ligature '{ligature}' found {count} times (should be '{replacement}')"
            )

    # Also check for other Unicode ligature characters
    unicode_ligatures = re.findall(r"[\uFB00-\uFB4F]", text)
    if unicode_ligatures:
        unique_ligatures = set(unicode_ligatures)
        untranslated_count += len(unicode_ligatures)
        issues.append(f"Other Unicode ligatures found: {list(unique_ligatures)}")

    # Calculate score (higher is better)
    ligature_density = untranslated_count / total_chars
    score = max(
        0.0, 1.0 - min(ligature_density * 1000, 1.0)
    )  # Scale by 1000 for reasonable scoring

    return score, issues[:5]  # Limit to 5 examples


def assess_header_footer_contamination(text: str) -> Tuple[float, List[str]]:
    """
    Assess header/footer contamination by detecting repeated patterns and page artifacts.

    Args:
        text: Text to analyze

    Returns:
        Tuple of (score, issues_list) where score is 0-1
    """
    issues = []

    lines = text.split("\n")
    if len(lines) < 3:
        return 1.0, []  # Too short to have header/footer issues

    # Check for repeated short lines (likely headers/footers)
    line_counts = {}
    for line in lines:
        stripped = line.strip()
        if stripped and len(stripped) < 50:  # Short lines more likely to be headers/footers
            line_counts[stripped] = line_counts.get(stripped, 0) + 1

    repeated_lines = [(line, count) for line, count in line_counts.items() if count > 1]
    contamination_score = 0

    for line, count in repeated_lines:
        if count > 2:  # Repeated more than twice
            contamination_score += count
            issues.append(f"Repeated line '{line}' appears {count} times (likely header/footer)")

    # Check for page numbers in text
    page_numbers = re.findall(r"\b(?:page\s+)?\d+\b", text, re.IGNORECASE)
    isolated_numbers = [num for num in page_numbers if re.match(r"^\d+$", num.strip())]
    if len(isolated_numbers) > 3:
        contamination_score += len(isolated_numbers)
        issues.append(f"Found {len(isolated_numbers)} isolated numbers (possible page numbers)")

    # Check for common header/footer patterns
    header_footer_patterns = [
        r"\b(?:chapter|section)\s+\d+\b",
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",  # Dates
        r"\b\d{1,2}:\d{2}\s*(?:AM|PM)?\b",  # Times
        r"\b(?:copyright|©)\b",
        r"\b(?:confidential|draft|preliminary)\b",
    ]

    for pattern in header_footer_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            contamination_score += len(matches)
            issues.append(f"Header/footer pattern found: {matches[:3]}")  # Show first 3 matches

    # Calculate score (higher is better, lower contamination)
    total_lines = len(lines)
    contamination_ratio = contamination_score / total_lines
    score = max(0.0, 1.0 - min(contamination_ratio, 1.0))

    return score, issues[:8]  # Limit to 8 examples


def analyze_text_quality(
    chunks: List[Dict[str, Any]], method_name: str, pdf_file: str
) -> TextQualityMetrics:
    """
    Analyze text quality for a list of chunks.

    Args:
        chunks: List of text chunks with metadata
        method_name: Name of extraction method
        pdf_file: PDF file name

    Returns:
        TextQualityMetrics with quality assessment
    """
    if not chunks:
        return TextQualityMetrics(
            method=method_name,
            pdf_file=pdf_file,
            total_chunks=0,
            total_text_length=0,
            avg_chunk_length=0,
            sentence_integrity_score=0.0,
            word_joining_score=0.0,
            whitespace_quality_score=0.0,
            ligature_translation_score=0.0,
            header_footer_contamination_score=0.0,
            overall_quality_score=0.0,
            issues_detected=["No chunks extracted"],
            sample_issues=[],
        )

    # Combine all text for analysis
    all_text = "\n".join(chunk.get("text", "") for chunk in chunks)
    total_text_length = len(all_text)
    avg_chunk_length = total_text_length / len(chunks) if chunks else 0

    # Assess different quality dimensions
    sentence_score, sentence_issues = assess_sentence_integrity(all_text)
    word_score, word_issues = assess_word_joining(all_text)
    whitespace_score, whitespace_issues = assess_whitespace_quality(all_text)
    ligature_score, ligature_issues = assess_ligature_translation(all_text)
    contamination_score, contamination_issues = assess_header_footer_contamination(all_text)

    # Calculate overall quality score (weighted average)
    weights = {
        "sentence": 0.3,
        "word": 0.25,
        "whitespace": 0.2,
        "ligature": 0.15,
        "contamination": 0.1,
    }

    overall_score = (
        sentence_score * weights["sentence"]
        + word_score * weights["word"]
        + whitespace_score * weights["whitespace"]
        + ligature_score * weights["ligature"]
        + contamination_score * weights["contamination"]
    )

    # Collect all issues
    all_issues = (
        sentence_issues + word_issues + whitespace_issues + ligature_issues + contamination_issues
    )

    # Create sample issues with categories
    sample_issues = []
    issue_categories = [
        ("Sentence Integrity", sentence_issues),
        ("Word Joining", word_issues),
        ("Whitespace Quality", whitespace_issues),
        ("Ligature Translation", ligature_issues),
        ("Header/Footer Contamination", contamination_issues),
    ]

    for category, issues in issue_categories:
        for issue in issues[:2]:  # Limit to 2 examples per category
            sample_issues.append(
                {
                    "category": category,
                    "issue": issue,
                    "severity": (
                        "high" if category in ["Sentence Integrity", "Word Joining"] else "medium"
                    ),
                }
            )

    return TextQualityMetrics(
        method=method_name,
        pdf_file=pdf_file,
        total_chunks=len(chunks),
        total_text_length=total_text_length,
        avg_chunk_length=avg_chunk_length,
        sentence_integrity_score=sentence_score,
        word_joining_score=word_score,
        whitespace_quality_score=whitespace_score,
        ligature_translation_score=ligature_score,
        header_footer_contamination_score=contamination_score,
        overall_quality_score=overall_score,
        issues_detected=all_issues[:15],  # Limit to 15 total issues
        sample_issues=sample_issues[:10],  # Limit to 10 sample issues
    )


def compare_text_quality(
    hybrid_metrics: TextQualityMetrics, traditional_metrics: TextQualityMetrics
) -> Dict[str, Any]:
    """
    Compare text quality between hybrid and traditional methods.

    Args:
        hybrid_metrics: Quality metrics for hybrid method
        traditional_metrics: Quality metrics for traditional method

    Returns:
        Comparison analysis
    """
    comparison = {
        "overall_comparison": {},
        "dimension_comparison": {},
        "recommendations": [],
        "quality_summary": {},
    }

    # Overall comparison
    hybrid_overall = hybrid_metrics.overall_quality_score
    traditional_overall = traditional_metrics.overall_quality_score

    comparison["overall_comparison"] = {
        "hybrid_score": hybrid_overall,
        "traditional_score": traditional_overall,
        "difference": hybrid_overall - traditional_overall,
        "better_method": ("hybrid" if hybrid_overall > traditional_overall else "traditional"),
        "improvement_percentage": abs(hybrid_overall - traditional_overall) * 100,
    }

    # Dimension-by-dimension comparison
    dimensions = [
        ("sentence_integrity", "Sentence Integrity"),
        ("word_joining", "Word Joining"),
        ("whitespace_quality", "Whitespace Quality"),
        ("ligature_translation", "Ligature Translation"),
        ("header_footer_contamination", "Header/Footer Contamination"),
    ]

    dimension_results = {}
    for dim_key, dim_name in dimensions:
        hybrid_score = getattr(hybrid_metrics, f"{dim_key}_score")
        traditional_score = getattr(traditional_metrics, f"{dim_key}_score")

        dimension_results[dim_key] = {
            "name": dim_name,
            "hybrid_score": hybrid_score,
            "traditional_score": traditional_score,
            "difference": hybrid_score - traditional_score,
            "better_method": ("hybrid" if hybrid_score > traditional_score else "traditional"),
        }

    comparison["dimension_comparison"] = dimension_results

    # Generate recommendations
    recommendations = []

    # Overall recommendation
    if hybrid_overall > traditional_overall + 0.05:  # 5% threshold
        recommendations.append(
            f"Hybrid method shows {(hybrid_overall - traditional_overall) * 100:.1f}% better overall text quality"
        )
    elif traditional_overall > hybrid_overall + 0.05:
        recommendations.append(
            f"Traditional method shows {(traditional_overall - hybrid_overall) * 100:.1f}% better overall text quality"
        )
    else:
        recommendations.append("Both methods show similar overall text quality")

    # Dimension-specific recommendations
    for dim_key, dim_data in dimension_results.items():
        if abs(dim_data["difference"]) > 0.1:  # 10% threshold
            better = dim_data["better_method"]
            improvement = abs(dim_data["difference"]) * 100
            recommendations.append(
                f"{better.title()} method is {improvement:.1f}% better at {dim_data['name'].lower()}"
            )

    # Issue-based recommendations
    hybrid_issue_count = len(hybrid_metrics.issues_detected)
    traditional_issue_count = len(traditional_metrics.issues_detected)

    if hybrid_issue_count < traditional_issue_count:
        recommendations.append(
            f"Hybrid method has {traditional_issue_count - hybrid_issue_count} fewer quality issues"
        )
    elif traditional_issue_count < hybrid_issue_count:
        recommendations.append(
            f"Traditional method has {hybrid_issue_count - traditional_issue_count} fewer quality issues"
        )

    comparison["recommendations"] = recommendations

    # Quality summary
    comparison["quality_summary"] = {
        "hybrid_chunks": hybrid_metrics.total_chunks,
        "traditional_chunks": traditional_metrics.total_chunks,
        "hybrid_text_length": hybrid_metrics.total_text_length,
        "traditional_text_length": traditional_metrics.total_text_length,
        "hybrid_avg_chunk_length": hybrid_metrics.avg_chunk_length,
        "traditional_avg_chunk_length": traditional_metrics.avg_chunk_length,
        "hybrid_issue_count": len(hybrid_metrics.issues_detected),
        "traditional_issue_count": len(traditional_metrics.issues_detected),
    }

    return comparison


def generate_detailed_examples(
    hybrid_chunks: List[Dict[str, Any]],
    traditional_chunks: List[Dict[str, Any]],
    pdf_file: str,
) -> List[Dict[str, Any]]:
    """
    Generate detailed examples showing text quality differences.

    Args:
        hybrid_chunks: Chunks from hybrid method
        traditional_chunks: Chunks from traditional method
        pdf_file: PDF file name

    Returns:
        List of detailed comparison examples
    """
    examples = []

    # Compare first few chunks for detailed analysis
    max_examples = min(3, len(hybrid_chunks), len(traditional_chunks))

    for i in range(max_examples):
        hybrid_text = hybrid_chunks[i].get("text", "")
        traditional_text = traditional_chunks[i].get("text", "")

        # Analyze differences
        example = {
            "chunk_index": i,
            "pdf_file": pdf_file,
            "hybrid_text_preview": (
                hybrid_text[:200] + "..." if len(hybrid_text) > 200 else hybrid_text
            ),
            "traditional_text_preview": (
                traditional_text[:200] + "..." if len(traditional_text) > 200 else traditional_text
            ),
            "length_comparison": {
                "hybrid_length": len(hybrid_text),
                "traditional_length": len(traditional_text),
                "difference": len(hybrid_text) - len(traditional_text),
            },
            "quality_differences": [],
        }

        # Check for specific quality differences

        # Sentence count comparison
        hybrid_sentences = len(re.split(r"[.!?]+", hybrid_text))
        traditional_sentences = len(re.split(r"[.!?]+", traditional_text))
        if abs(hybrid_sentences - traditional_sentences) > 1:
            example["quality_differences"].append(
                {
                    "type": "sentence_count",
                    "hybrid_value": hybrid_sentences,
                    "traditional_value": traditional_sentences,
                    "description": f"Sentence count differs: hybrid={hybrid_sentences}, traditional={traditional_sentences}",
                }
            )

        # Word count comparison
        hybrid_words = len(hybrid_text.split())
        traditional_words = len(traditional_text.split())
        if abs(hybrid_words - traditional_words) > 5:
            example["quality_differences"].append(
                {
                    "type": "word_count",
                    "hybrid_value": hybrid_words,
                    "traditional_value": traditional_words,
                    "description": f"Word count differs: hybrid={hybrid_words}, traditional={traditional_words}",
                }
            )

        # Newline count comparison
        hybrid_newlines = hybrid_text.count("\n")
        traditional_newlines = traditional_text.count("\n")
        if abs(hybrid_newlines - traditional_newlines) > 2:
            example["quality_differences"].append(
                {
                    "type": "newline_count",
                    "hybrid_value": hybrid_newlines,
                    "traditional_value": traditional_newlines,
                    "description": f"Newline count differs: hybrid={hybrid_newlines}, traditional={traditional_newlines}",
                }
            )

        examples.append(example)

    return examples


def run_text_quality_comparison(pdf_files: List[str]) -> List[QualityComparison]:
    """
    Run complete text quality comparison for multiple PDF files.

    Args:
        pdf_files: List of PDF files to analyze

    Returns:
        List of quality comparison results
    """
    results = []

    for pdf_file in pdf_files:
        print(f"Analyzing text quality for: {os.path.basename(pdf_file)}")

        if not os.path.exists(pdf_file):
            print(f"  WARNING: File not found: {pdf_file}")
            continue

        # Extract with both methods
        print("  Extracting with hybrid method...")
        hybrid_chunks = extract_with_hybrid_method(pdf_file)

        print("  Extracting with traditional method...")
        traditional_chunks = extract_with_traditional_method(pdf_file)

        if not hybrid_chunks and not traditional_chunks:
            print(f"  WARNING: Both extraction methods failed for {pdf_file}")
            continue

        # Analyze text quality
        print("  Analyzing hybrid text quality...")
        hybrid_metrics = analyze_text_quality(
            hybrid_chunks, "hybrid_pymupdf4llm", os.path.basename(pdf_file)
        )

        print("  Analyzing traditional text quality...")
        traditional_metrics = analyze_text_quality(
            traditional_chunks, "traditional_fallback", os.path.basename(pdf_file)
        )

        # Compare methods
        print("  Comparing text quality...")
        comparison_analysis = compare_text_quality(hybrid_metrics, traditional_metrics)

        # Generate detailed examples
        detailed_examples = generate_detailed_examples(
            hybrid_chunks, traditional_chunks, os.path.basename(pdf_file)
        )

        # Create comparison result
        result = QualityComparison(
            hybrid_metrics=hybrid_metrics,
            traditional_metrics=traditional_metrics,
            comparison_analysis=comparison_analysis,
            detailed_examples=detailed_examples,
        )

        results.append(result)

        # Print summary
        print(f"  Hybrid overall quality: {hybrid_metrics.overall_quality_score:.2f}")
        print(f"  Traditional overall quality: {traditional_metrics.overall_quality_score:.2f}")
        print(f"  Better method: {comparison_analysis['overall_comparison']['better_method']}")

    return results


def print_quality_summary(results: List[QualityComparison]) -> None:
    """Print a human-readable summary of text quality comparison results"""
    if not results:
        print("No results to display")
        return

    print("\n" + "=" * 80)
    print("TEXT QUALITY COMPARISON RESULTS")
    print("=" * 80)

    for i, result in enumerate(results):
        hybrid = result.hybrid_metrics
        traditional = result.traditional_metrics
        comparison = result.comparison_analysis

        print(f"\nFile {i+1}: {hybrid.pdf_file}")
        print("-" * 40)

        # Overall scores
        print(f"Overall Quality Scores:")
        print(f"  Hybrid (PyMuPDF4LLM): {hybrid.overall_quality_score:.3f}")
        print(f"  Traditional:          {traditional.overall_quality_score:.3f}")
        print(
            f"  Better method:        {comparison['overall_comparison']['better_method'].title()}"
        )
        print(
            f"  Improvement:          {comparison['overall_comparison']['improvement_percentage']:.1f}%"
        )

        # Dimension scores
        print(f"\nDimension Scores:")
        dimensions = comparison["dimension_comparison"]
        for dim_key, dim_data in dimensions.items():
            print(f"  {dim_data['name']}:")
            print(
                f"    Hybrid: {dim_data['hybrid_score']:.3f}, Traditional: {dim_data['traditional_score']:.3f}"
            )
            print(f"    Better: {dim_data['better_method'].title()}")

        # Content statistics
        print(f"\nContent Statistics:")
        summary = comparison["quality_summary"]
        print(
            f"  Chunks:     Hybrid={summary['hybrid_chunks']}, Traditional={summary['traditional_chunks']}"
        )
        print(
            f"  Text length: Hybrid={summary['hybrid_text_length']}, Traditional={summary['traditional_text_length']}"
        )
        print(
            f"  Avg chunk:  Hybrid={summary['hybrid_avg_chunk_length']:.0f}, Traditional={summary['traditional_avg_chunk_length']:.0f}"
        )
        print(
            f"  Issues:     Hybrid={summary['hybrid_issue_count']}, Traditional={summary['traditional_issue_count']}"
        )

        # Key recommendations
        recommendations = comparison["recommendations"]
        if recommendations:
            print(f"\nKey Findings:")
            for j, rec in enumerate(recommendations[:3], 1):  # Show top 3
                print(f"  {j}. {rec}")

    # Overall summary across all files
    if len(results) > 1:
        print(f"\n" + "=" * 80)
        print("OVERALL SUMMARY")
        print("=" * 80)

        hybrid_scores = [r.hybrid_metrics.overall_quality_score for r in results]
        traditional_scores = [r.traditional_metrics.overall_quality_score for r in results]

        avg_hybrid = statistics.mean(hybrid_scores)
        avg_traditional = statistics.mean(traditional_scores)

        print(f"Average Quality Scores:")
        print(f"  Hybrid:     {avg_hybrid:.3f}")
        print(f"  Traditional: {avg_traditional:.3f}")
        print(f"  Difference: {avg_hybrid - avg_traditional:+.3f}")

        hybrid_wins = sum(
            1
            for r in results
            if r.comparison_analysis["overall_comparison"]["better_method"] == "hybrid"
        )
        traditional_wins = len(results) - hybrid_wins

        print(f"\nMethod Performance:")
        print(f"  Hybrid wins:     {hybrid_wins}/{len(results)} files")
        print(f"  Traditional wins: {traditional_wins}/{len(results)} files")

    print("\n" + "=" * 80)


def main() -> None:
    """Main function to run the text quality comparison"""
    parser = argparse.ArgumentParser(
        description="Compare text quality between hybrid PyMuPDF4LLM and traditional PDF extraction methods"
    )
    parser.add_argument("pdf_files", nargs="+", help="PDF files to analyze for text quality")
    parser.add_argument(
        "--output",
        "-o",
        default="text_quality_comparison.json",
        help="Output file for detailed results (default: text_quality_comparison.json)",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Include detailed text examples in output",
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress detailed output")

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

    # Run text quality comparison
    try:
        results = run_text_quality_comparison(valid_files)

        if not results:
            print("ERROR: No comparison results generated")
            sys.exit(1)

        # Prepare output data
        output_data = {
            "comparison_date": "2025-07-20T15:47:07Z",
            "files_analyzed": [os.path.basename(f) for f in valid_files],
            "total_comparisons": len(results),
            "results": [result.to_dict() for result in results],
        }

        # Add summary statistics
        if results:
            hybrid_scores = [r.hybrid_metrics.overall_quality_score for r in results]
            traditional_scores = [r.traditional_metrics.overall_quality_score for r in results]

            output_data["summary_statistics"] = {
                "average_hybrid_quality": statistics.mean(hybrid_scores),
                "average_traditional_quality": statistics.mean(traditional_scores),
                "hybrid_wins": sum(
                    1
                    for r in results
                    if r.comparison_analysis["overall_comparison"]["better_method"] == "hybrid"
                ),
                "traditional_wins": sum(
                    1
                    for r in results
                    if r.comparison_analysis["overall_comparison"]["better_method"] == "traditional"
                ),
                "quality_improvement": statistics.mean(hybrid_scores)
                - statistics.mean(traditional_scores),
            }

        # Save detailed results
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)

        if not args.quiet:
            print_quality_summary(results)

        print(f"\nDetailed results saved to: {args.output}")

        # Exit with appropriate code based on results
        if results:
            avg_hybrid = statistics.mean([r.hybrid_metrics.overall_quality_score for r in results])
            avg_traditional = statistics.mean(
                [r.traditional_metrics.overall_quality_score for r in results]
            )

            if avg_hybrid < 0.5 or avg_traditional < 0.5:
                print("WARNING: Low text quality scores detected")
                sys.exit(1)

    except Exception as e:
        print(f"ERROR: Text quality comparison failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
