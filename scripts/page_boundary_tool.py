#!/usr/bin/env python3
"""
Page Boundary Handling Test and Comparison Script

This script tests page boundary handling with both traditional and PyMuPDF4LLM approaches.
It includes specific tests for:
- Sentences spanning pages
- Header/footer filtering
- Footnote handling
- Text flow reconstruction
- Side-by-side comparison of extraction quality

Usage:
    python scripts/test_page_boundaries.py <pdf_file>
"""

import sys
import os
import json
import re
from typing import Dict, List, Any, Tuple
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pdf_chunker.pdf_parsing import extract_text_blocks_from_pdf
from pdf_chunker.text_cleaning import clean_text
from pdf_chunker.pymupdf4llm_integration import (
    is_pymupdf4llm_available,
    clean_text_with_pymupdf4llm,
)


def extract_with_traditional_approach(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract text using traditional approach only."""
    # Temporarily disable PyMuPDF4LLM for traditional extraction
    os.environ["PDF_CHUNKER_USE_PYMUPDF4LLM"] = "false"
    try:
        blocks = extract_text_blocks_from_pdf(pdf_path)
        return blocks
    finally:
        # Restore environment
        if "PDF_CHUNKER_USE_PYMUPDF4LLM" in os.environ:
            del os.environ["PDF_CHUNKER_USE_PYMUPDF4LLM"]


def extract_with_pymupdf4llm_approach(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract text using PyMuPDF4LLM enhanced approach."""
    # Enable PyMuPDF4LLM for enhanced extraction
    os.environ["PDF_CHUNKER_USE_PYMUPDF4LLM"] = "true"
    try:
        blocks = extract_text_blocks_from_pdf(pdf_path)
        return blocks
    finally:
        # Restore environment
        if "PDF_CHUNKER_USE_PYMUPDF4LLM" in os.environ:
            del os.environ["PDF_CHUNKER_USE_PYMUPDF4LLM"]


def analyze_sentence_boundaries(text: str) -> Dict[str, Any]:
    """Analyze sentence boundary quality in extracted text."""
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Look for incomplete sentences (very short fragments)
    short_fragments = [s for s in sentences if len(s.split()) <= 4]

    # Look for sentences that start with lowercase (potential continuation issues)
    lowercase_starts = [s for s in sentences if s and s[0].islower()]

    # Look for abrupt endings (sentences ending mid-word or with unusual patterns)
    abrupt_endings = []
    for sentence in sentences:
        if sentence:
            # Check for sentences ending with single letters or very short words
            words = sentence.split()
            if words and len(words[-1]) <= 2 and not words[-1].isdigit():
                abrupt_endings.append(sentence)

    return {
        "total_sentences": len(sentences),
        "short_fragments": len(short_fragments),
        "lowercase_starts": len(lowercase_starts),
        "abrupt_endings": len(abrupt_endings),
        "short_fragment_examples": short_fragments[:3],
        "lowercase_start_examples": lowercase_starts[:3],
        "abrupt_ending_examples": abrupt_endings[:3],
    }


def detect_page_artifacts(blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Detect headers, footers, and page artifacts in extracted blocks."""
    artifacts = {
        "page_numbers": [],
        "headers": [],
        "footers": [],
        "footnotes": [],
        "other_artifacts": [],
    }

    for i, block in enumerate(blocks):
        text = block.get("text", "").strip()
        page = block.get("source", {}).get("page", 0)

        if not text:
            continue

        # Page numbers (standalone digits)
        if re.match(r"^\d+$", text):
            artifacts["page_numbers"].append(
                {"block_index": i, "page": page, "text": text}
            )
            continue

        # Headers (common patterns at top of pages)
        if re.match(r"^(chapter\s+\d+|page\s+\d+|table\s+of\s+contents)", text.lower()):
            artifacts["headers"].append(
                {
                    "block_index": i,
                    "page": page,
                    "text": text[:50] + "..." if len(text) > 50 else text,
                }
            )
            continue

        # Footnotes (text starting with numbers or footnote markers)
        if re.match(r"^\d+\s+[a-z]", text.lower()) or "footnote" in text.lower():
            artifacts["footnotes"].append(
                {
                    "block_index": i,
                    "page": page,
                    "text": text[:50] + "..." if len(text) > 50 else text,
                }
            )
            continue

        # Other potential artifacts (very short text with specific patterns)
        if len(text.split()) <= 3 and (
            any(char.isdigit() for char in text)
            or text.isupper()
            or re.match(r"^[A-Z\s]+$", text)
        ):
            artifacts["other_artifacts"].append(
                {"block_index": i, "page": page, "text": text}
            )

    return artifacts


def analyze_text_flow_quality(blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze the quality of text flow across page boundaries."""
    flow_issues = []
    page_transitions = []

    for i in range(len(blocks) - 1):
        curr_block = blocks[i]
        next_block = blocks[i + 1]

        curr_text = curr_block.get("text", "").strip()
        next_text = next_block.get("text", "").strip()

        curr_page = curr_block.get("source", {}).get("page", 0)
        next_page = next_block.get("source", {}).get("page", 0)

        # Check for page transitions
        if curr_page != next_page:
            page_transitions.append(
                {
                    "from_page": curr_page,
                    "to_page": next_page,
                    "curr_text_end": curr_text[-50:] if curr_text else "",
                    "next_text_start": next_text[:50] if next_text else "",
                    "block_indices": [i, i + 1],
                }
            )

            # Check for potential flow issues at page boundaries
            if curr_text and next_text:
                # Sentence cut off (no punctuation at end, lowercase start)
                if (
                    not curr_text.endswith((".", "!", "?", ":", ";"))
                    and next_text
                    and next_text[0].islower()
                ):
                    flow_issues.append(
                        {
                            "type": "sentence_cut_off",
                            "from_page": curr_page,
                            "to_page": next_page,
                            "description": f"Sentence appears cut off between pages {curr_page}-{next_page}",
                            "curr_text_end": curr_text[-30:],
                            "next_text_start": next_text[:30],
                        }
                    )

                # Word split (ends with partial word, starts with continuation)
                curr_words = curr_text.split()
                next_words = next_text.split()
                if (
                    curr_words
                    and next_words
                    and len(curr_words[-1]) < 4
                    and next_words[0][0].islower()
                ):
                    flow_issues.append(
                        {
                            "type": "word_split",
                            "from_page": curr_page,
                            "to_page": next_page,
                            "description": f"Word appears split between pages {curr_page}-{next_page}",
                            "split_word": f"{curr_words[-1]}|{next_words[0]}",
                        }
                    )

    return {
        "page_transitions": page_transitions,
        "flow_issues": flow_issues,
        "total_transitions": len(page_transitions),
        "total_flow_issues": len(flow_issues),
    }


def compare_text_quality(
    traditional_blocks: List[Dict[str, Any]], pymupdf4llm_blocks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Compare text quality between traditional and PyMuPDF4LLM approaches."""

    # Combine text from all blocks
    traditional_text = "\n\n".join(
        block.get("text", "") for block in traditional_blocks
    )
    pymupdf4llm_text = "\n\n".join(
        block.get("text", "") for block in pymupdf4llm_blocks
    )

    # Analyze sentence boundaries
    traditional_sentences = analyze_sentence_boundaries(traditional_text)
    pymupdf4llm_sentences = analyze_sentence_boundaries(pymupdf4llm_text)

    # Detect page artifacts
    traditional_artifacts = detect_page_artifacts(traditional_blocks)
    pymupdf4llm_artifacts = detect_page_artifacts(pymupdf4llm_blocks)

    # Analyze text flow quality
    traditional_flow = analyze_text_flow_quality(traditional_blocks)
    pymupdf4llm_flow = analyze_text_flow_quality(pymupdf4llm_blocks)

    # Calculate quality scores
    def calculate_quality_score(sentences, artifacts, flow):
        score = 1.0

        # Penalize for sentence issues
        if sentences["total_sentences"] > 0:
            fragment_ratio = sentences["short_fragments"] / sentences["total_sentences"]
            lowercase_ratio = (
                sentences["lowercase_starts"] / sentences["total_sentences"]
            )
            abrupt_ratio = sentences["abrupt_endings"] / sentences["total_sentences"]

            score -= fragment_ratio * 0.3
            score -= lowercase_ratio * 0.2
            score -= abrupt_ratio * 0.2

        # Penalize for artifacts
        total_artifacts = (
            len(artifacts["page_numbers"])
            + len(artifacts["headers"])
            + len(artifacts["footers"])
            + len(artifacts["other_artifacts"])
        )
        if total_artifacts > 0:
            score -= min(total_artifacts * 0.05, 0.3)

        # Penalize for flow issues
        if flow["total_transitions"] > 0:
            flow_issue_ratio = flow["total_flow_issues"] / flow["total_transitions"]
            score -= flow_issue_ratio * 0.4

        return max(score, 0.0)

    traditional_score = calculate_quality_score(
        traditional_sentences, traditional_artifacts, traditional_flow
    )
    pymupdf4llm_score = calculate_quality_score(
        pymupdf4llm_sentences, pymupdf4llm_artifacts, pymupdf4llm_flow
    )

    return {
        "traditional": {
            "quality_score": traditional_score,
            "total_blocks": len(traditional_blocks),
            "total_characters": len(traditional_text),
            "sentences": traditional_sentences,
            "artifacts": traditional_artifacts,
            "flow": traditional_flow,
        },
        "pymupdf4llm": {
            "quality_score": pymupdf4llm_score,
            "total_blocks": len(pymupdf4llm_blocks),
            "total_characters": len(pymupdf4llm_text),
            "sentences": pymupdf4llm_sentences,
            "artifacts": pymupdf4llm_artifacts,
            "flow": pymupdf4llm_flow,
        },
        "comparison": {
            "quality_improvement": pymupdf4llm_score - traditional_score,
            "block_count_change": len(pymupdf4llm_blocks) - len(traditional_blocks),
            "character_count_change": len(pymupdf4llm_text) - len(traditional_text),
            "better_approach": (
                "PyMuPDF4LLM"
                if pymupdf4llm_score > traditional_score
                else "Traditional"
            ),
        },
    }


def generate_comparison_report(
    comparison_results: Dict[str, Any], pdf_path: str
) -> str:
    """Generate a detailed comparison report."""
    report = []
    report.append("=" * 80)
    report.append("PAGE BOUNDARY HANDLING COMPARISON REPORT")
    report.append("=" * 80)
    report.append(f"PDF File: {pdf_path}")
    report.append(f"PyMuPDF4LLM Available: {is_pymupdf4llm_available()}")
    report.append("")

    # Overall quality comparison
    trad = comparison_results["traditional"]
    pymu = comparison_results["pymupdf4llm"]
    comp = comparison_results["comparison"]

    report.append("OVERALL QUALITY SCORES")
    report.append("-" * 40)
    report.append(f"Traditional Approach:  {trad['quality_score']:.3f}")
    report.append(f"PyMuPDF4LLM Approach:  {pymu['quality_score']:.3f}")
    report.append(f"Quality Improvement:   {comp['quality_improvement']:+.3f}")
    report.append(f"Better Approach:       {comp['better_approach']}")
    report.append("")

    # Block and character counts
    report.append("EXTRACTION STATISTICS")
    report.append("-" * 40)
    report.append(f"Traditional Blocks:    {trad['total_blocks']}")
    report.append(f"PyMuPDF4LLM Blocks:    {pymu['total_blocks']}")
    report.append(f"Block Count Change:    {comp['block_count_change']:+d}")
    report.append("")
    report.append(f"Traditional Characters: {trad['total_characters']}")
    report.append(f"PyMuPDF4LLM Characters: {pymu['total_characters']}")
    report.append(f"Character Count Change: {comp['character_count_change']:+d}")
    report.append("")

    # Sentence analysis
    report.append("SENTENCE BOUNDARY ANALYSIS")
    report.append("-" * 40)
    report.append(f"                        Traditional  PyMuPDF4LLM")
    report.append(
        f"Total Sentences:        {trad['sentences']['total_sentences']:>11d}  {pymu['sentences']['total_sentences']:>11d}"
    )
    report.append(
        f"Short Fragments:        {trad['sentences']['short_fragments']:>11d}  {pymu['sentences']['short_fragments']:>11d}"
    )
    report.append(
        f"Lowercase Starts:       {trad['sentences']['lowercase_starts']:>11d}  {pymu['sentences']['lowercase_starts']:>11d}"
    )
    report.append(
        f"Abrupt Endings:         {trad['sentences']['abrupt_endings']:>11d}  {pymu['sentences']['abrupt_endings']:>11d}"
    )
    report.append("")

    # Page artifacts
    report.append("PAGE ARTIFACT DETECTION")
    report.append("-" * 40)
    report.append(f"                        Traditional  PyMuPDF4LLM")
    report.append(
        f"Page Numbers:           {len(trad['artifacts']['page_numbers']):>11d}  {len(pymu['artifacts']['page_numbers']):>11d}"
    )
    report.append(
        f"Headers:                {len(trad['artifacts']['headers']):>11d}  {len(pymu['artifacts']['headers']):>11d}"
    )
    report.append(
        f"Footers:                {len(trad['artifacts']['footers']):>11d}  {len(pymu['artifacts']['footers']):>11d}"
    )
    report.append(
        f"Footnotes:              {len(trad['artifacts']['footnotes']):>11d}  {len(pymu['artifacts']['footnotes']):>11d}"
    )
    report.append(
        f"Other Artifacts:        {len(trad['artifacts']['other_artifacts']):>11d}  {len(pymu['artifacts']['other_artifacts']):>11d}"
    )
    report.append("")

    # Text flow analysis
    report.append("TEXT FLOW ANALYSIS")
    report.append("-" * 40)
    report.append(f"                        Traditional  PyMuPDF4LLM")
    report.append(
        f"Page Transitions:       {trad['flow']['total_transitions']:>11d}  {pymu['flow']['total_transitions']:>11d}"
    )
    report.append(
        f"Flow Issues:            {trad['flow']['total_flow_issues']:>11d}  {pymu['flow']['total_flow_issues']:>11d}"
    )

    # Print Flow Issue Ratio for both columns in a single line
    if trad["flow"]["total_transitions"] > 0:
        trad_flow_ratio = (
            trad["flow"]["total_flow_issues"] / trad["flow"]["total_transitions"]
        )
        trad_ratio_str = f"{trad_flow_ratio:>11.3f}"
    else:
        trad_ratio_str = f"{'N/A':>11s}"

    if pymu["flow"]["total_transitions"] > 0:
        pymu_flow_ratio = (
            pymu["flow"]["total_flow_issues"] / pymu["flow"]["total_transitions"]
        )
        pymu_ratio_str = f"{pymu_flow_ratio:>11.3f}"
    else:
        pymu_ratio_str = f"{'N/A':>11s}"

    report.append(f"Flow Issue Ratio:       {trad_ratio_str}  {pymu_ratio_str}")

    report.append("")

    # Examples of issues
    if trad["flow"]["flow_issues"] or pymu["flow"]["flow_issues"]:
        report.append("FLOW ISSUE EXAMPLES")
        report.append("-" * 40)

        if trad["flow"]["flow_issues"]:
            report.append("Traditional Approach Issues:")
            for issue in trad["flow"]["flow_issues"][:3]:
                report.append(f"  {issue['type']}: {issue['description']}")
                if "curr_text_end" in issue:
                    report.append(f"    End: ...{issue['curr_text_end']}")
                    report.append(f"    Start: {issue['next_text_start']}...")
            report.append("")

        if pymu["flow"]["flow_issues"]:
            report.append("PyMuPDF4LLM Approach Issues:")
            for issue in pymu["flow"]["flow_issues"][:3]:
                report.append(f"  {issue['type']}: {issue['description']}")
                if "curr_text_end" in issue:
                    report.append(f"    End: ...{issue['curr_text_end']}")
                    report.append(f"    Start: {issue['next_text_start']}...")
            report.append("")

    # Recommendations
    report.append("RECOMMENDATIONS")
    report.append("-" * 40)

    if comp["quality_improvement"] > 0.1:
        report.append("✓ PyMuPDF4LLM approach shows significant improvement")
        report.append("  Recommendation: Enable PyMuPDF4LLM for this document type")
    elif comp["quality_improvement"] > 0.05:
        report.append("✓ PyMuPDF4LLM approach shows moderate improvement")
        report.append("  Recommendation: Consider enabling PyMuPDF4LLM")
    elif comp["quality_improvement"] > -0.05:
        report.append("≈ Both approaches perform similarly")
        report.append("  Recommendation: Use traditional approach for consistency")
    else:
        report.append("✗ PyMuPDF4LLM approach shows degradation")
        report.append("  Recommendation: Use traditional approach")

    report.append("")
    report.append("=" * 80)

    return "\n".join(report)


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/test_page_boundaries.py <pdf_file>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    if not os.path.exists(pdf_path):
        print(f"Error: PDF file '{pdf_path}' not found")
        sys.exit(1)

    print(f"Testing page boundary handling for: {pdf_path}")
    print(f"PyMuPDF4LLM available: {is_pymupdf4llm_available()}")
    print()

    # Extract with both approaches
    print("Extracting with traditional approach...")
    traditional_blocks = extract_with_traditional_approach(pdf_path)

    print("Extracting with PyMuPDF4LLM approach...")
    pymupdf4llm_blocks = extract_with_pymupdf4llm_approach(pdf_path)

    # Compare results
    print("Analyzing and comparing results...")
    comparison_results = compare_text_quality(traditional_blocks, pymupdf4llm_blocks)

    # Generate report
    report = generate_comparison_report(comparison_results, pdf_path)
    print(report)

    # Save detailed results to JSON
    output_file = f"page_boundary_comparison_{Path(pdf_path).stem}.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "pdf_path": pdf_path,
                "pymupdf4llm_available": is_pymupdf4llm_available(),
                "comparison_results": comparison_results,
                "traditional_blocks": traditional_blocks,
                "pymupdf4llm_blocks": pymupdf4llm_blocks,
            },
            f,
            indent=2,
            default=str,
        )

    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
