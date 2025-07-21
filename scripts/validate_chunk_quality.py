#!/usr/bin/env python3
"""
Chunk Quality Validation Script

This script analyzes JSONL output from the PDF chunking pipeline to assess chunk quality,
with specific focus on:
- Chunk size distribution and statistics
- Detection of overly short chunks (3-7 words)
- Conversational pattern analysis
- Dialogue and commentary text handling
- Before/after quality metrics comparison

Usage:
    python scripts/validate_chunk_quality.py <jsonl_file> [--baseline <baseline_jsonl>]
    python scripts/validate_chunk_quality.py --compare <file1.jsonl> <file2.jsonl>
    python scripts/validate_chunk_quality.py --generate <pdf_file> [--traditional] [--enhanced]
"""

import sys
import os
import json
import re
import argparse
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from collections import defaultdict, Counter
import statistics

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pdf_chunker.core import process_document


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load chunks from a JSONL file."""
    chunks = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    chunks.append(chunk)
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {line_num}: {e}", file=sys.stderr)
                    continue
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found", file=sys.stderr)
        return []
    except Exception as e:
        print(f"Error reading file '{filepath}': {e}", file=sys.stderr)
        return []
    
    return chunks


def analyze_chunk_sizes(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze chunk size distribution and statistics."""
    if not chunks:
        return {
            'total_chunks': 0,
            'word_counts': [],
            'char_counts': [],
            'statistics': {}
        }
    
    word_counts = []
    char_counts = []
    
    for chunk in chunks:
        text = chunk.get('text', '')
        if text:
            words = len(text.split())
            chars = len(text)
            word_counts.append(words)
            char_counts.append(chars)
    
    if not word_counts:
        return {
            'total_chunks': len(chunks),
            'word_counts': [],
            'char_counts': [],
            'statistics': {}
        }
    
    # Calculate statistics
    word_stats = {
        'mean': statistics.mean(word_counts),
        'median': statistics.median(word_counts),
        'mode': statistics.mode(word_counts) if word_counts else 0,
        'min': min(word_counts),
        'max': max(word_counts),
        'std_dev': statistics.stdev(word_counts) if len(word_counts) > 1 else 0
    }
    
    char_stats = {
        'mean': statistics.mean(char_counts),
        'median': statistics.median(char_counts),
        'min': min(char_counts),
        'max': max(char_counts),
        'std_dev': statistics.stdev(char_counts) if len(char_counts) > 1 else 0
    }
    
    return {
        'total_chunks': len(chunks),
        'word_counts': word_counts,
        'char_counts': char_counts,
        'statistics': {
            'words': word_stats,
            'characters': char_stats
        }
    }


def detect_short_chunks(chunks: List[Dict[str, Any]], thresholds: Dict[str, int] = None) -> Dict[str, Any]:
    """Detect and categorize short chunks."""
    if thresholds is None:
        thresholds = {
            'very_short': 3,
            'short': 7,
            'minimal': 15
        }
    
    categorized_chunks = {
        'very_short': [],  # ≤ 3 words
        'short': [],       # 4-7 words
        'minimal': [],     # 8-15 words
        'normal': []       # > 15 words
    }
    
    for i, chunk in enumerate(chunks):
        text = chunk.get('text', '')
        word_count = len(text.split()) if text else 0
        
        chunk_info = {
            'index': i,
            'word_count': word_count,
            'char_count': len(text),
            'text_preview': text[:100].replace('\n', ' ') + ('...' if len(text) > 100 else ''),
            'full_text': text
        }
        
        if word_count <= thresholds['very_short']:
            categorized_chunks['very_short'].append(chunk_info)
        elif word_count <= thresholds['short']:
            categorized_chunks['short'].append(chunk_info)
        elif word_count <= thresholds['minimal']:
            categorized_chunks['minimal'].append(chunk_info)
        else:
            categorized_chunks['normal'].append(chunk_info)
    
    return categorized_chunks


def detect_dialogue_patterns(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Detect dialogue and conversational patterns in chunks."""
    dialogue_patterns = {
        'quoted_speech': [],
        'dialogue_attribution': [],
        'conversational_responses': [],
        'potential_fragments': []
    }
    
    # Patterns for detecting dialogue
    quote_pattern = r'"[^"]*"'
    attribution_words = [
        'said', 'replied', 'asked', 'answered', 'continued', 'added', 'noted',
        'observed', 'remarked', 'stated', 'declared', 'exclaimed', 'whispered',
        'shouted', 'muttered', 'explained', 'insisted', 'argued', 'suggested',
        'wondered', 'thought', 'concluded', 'responded', 'commented'
    ]
    
    for i, chunk in enumerate(chunks):
        text = chunk.get('text', '')
        if not text:
            continue
        
        word_count = len(text.split())
        
        # Check for quoted speech
        quotes = re.findall(quote_pattern, text)
        if quotes:
            dialogue_patterns['quoted_speech'].append({
                'index': i,
                'word_count': word_count,
                'quote_count': len(quotes),
                'quotes': quotes[:3],  # First 3 quotes
                'text_preview': text[:100].replace('\n', ' ')
            })
        
        # Check for dialogue attribution
        text_lower = text.lower()
        attribution_found = any(word in text_lower for word in attribution_words)
        if attribution_found and word_count <= 10:
            dialogue_patterns['dialogue_attribution'].append({
                'index': i,
                'word_count': word_count,
                'text_preview': text[:100].replace('\n', ' '),
                'attribution_words': [word for word in attribution_words if word in text_lower]
            })
        
        # Check for conversational responses (short chunks that might be responses)
        if 3 <= word_count <= 8:
            # Look for patterns that suggest this is a response or commentary
            response_indicators = [
                text.strip().endswith('?'),  # Questions
                text.strip().startswith(('Yes', 'No', 'Well', 'Oh', 'Ah', 'Indeed', 'Certainly')),
                any(word in text_lower for word in ['indeed', 'certainly', 'perhaps', 'maybe', 'probably'])
            ]
            
            if any(response_indicators):
                dialogue_patterns['conversational_responses'].append({
                    'index': i,
                    'word_count': word_count,
                    'text_preview': text[:100].replace('\n', ' '),
                    'indicators': [ind for ind, present in zip(['question', 'starter', 'qualifier'], response_indicators) if present]
                })
        
        # Check for potential fragments (very short chunks that might be incomplete)
        if word_count <= 5:
            # Analyze if this looks like a fragment
            is_complete_sentence = (
                text.strip().endswith(('.', '!', '?')) and
                text.strip()[0].isupper() and
                word_count >= 3
            )
            
            if not is_complete_sentence:
                dialogue_patterns['potential_fragments'].append({
                    'index': i,
                    'word_count': word_count,
                    'text_preview': text[:100].replace('\n', ' '),
                    'issues': {
                        'no_end_punctuation': not text.strip().endswith(('.', '!', '?')),
                        'no_capitalization': not (text.strip() and text.strip()[0].isupper()),
                        'too_short': word_count < 3
                    }
                })
    
    return dialogue_patterns


def analyze_chunk_flow(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze text flow and continuity between chunks."""
    flow_analysis = {
        'abrupt_transitions': [],
        'potential_splits': [],
        'continuation_issues': [],
        'flow_score': 0.0
    }
    
    if len(chunks) < 2:
        return flow_analysis
    
    issues_count = 0
    total_transitions = len(chunks) - 1
    
    for i in range(len(chunks) - 1):
        current_chunk = chunks[i]
        next_chunk = chunks[i + 1]
        
        current_text = current_chunk.get('text', '').strip()
        next_text = next_chunk.get('text', '').strip()
        
        if not current_text or not next_text:
            continue
        
        # Check for abrupt transitions
        current_ends_incomplete = not current_text.endswith(('.', '!', '?', ':', ';'))
        next_starts_lowercase = next_text and next_text[0].islower()
        
        if current_ends_incomplete and next_starts_lowercase:
            flow_analysis['abrupt_transitions'].append({
                'current_index': i,
                'next_index': i + 1,
                'current_end': current_text[-30:],
                'next_start': next_text[:30],
                'issue': 'incomplete_sentence_split'
            })
            issues_count += 1
        
        # Check for potential word splits
        current_words = current_text.split()
        next_words = next_text.split()
        
        if (current_words and next_words and
            len(current_words[-1]) < 4 and
            len(next_words[0]) < 8 and
            next_words[0][0].islower()):
            
            flow_analysis['potential_splits'].append({
                'current_index': i,
                'next_index': i + 1,
                'potential_word': f"{current_words[-1]}|{next_words[0]}",
                'current_end': current_text[-20:],
                'next_start': next_text[:20]
            })
            issues_count += 1
        
        # Check for continuation issues (very short chunks followed by related content)
        current_word_count = len(current_words)
        next_word_count = len(next_words)
        
        if (current_word_count <= 5 and next_word_count <= 10 and
            not current_text.endswith(('.', '!', '?'))):
            
            flow_analysis['continuation_issues'].append({
                'current_index': i,
                'next_index': i + 1,
                'current_words': current_word_count,
                'next_words': next_word_count,
                'current_text': current_text,
                'next_text': next_text[:50]
            })
            issues_count += 1
    
    # Calculate flow score (1.0 = perfect, 0.0 = many issues)
    if total_transitions > 0:
        flow_analysis['flow_score'] = max(0.0, 1.0 - (issues_count / total_transitions))
    else:
        flow_analysis['flow_score'] = 1.0
    
    return flow_analysis


def generate_quality_report(chunks: List[Dict[str, Any]], filename: str = "unknown") -> Dict[str, Any]:
    """Generate a comprehensive quality report for chunks."""
    # Always include summary_stats, even if empty
    summary_stats = {
        'total_chunks': len(chunks) if chunks else 0,
        'total_words': 0,
        'total_characters': 0,
        'avg_words_per_chunk': 0,
        'very_short_chunks': 0,
        'short_chunks': 0,
        'dialogue_chunks': 0,
        'flow_issues': 0
    }
    if not chunks:
        return {
            'filename': filename,
            'total_chunks': 0,
            'quality_score': 0.0,
            'quality_factors': [],
            'summary_stats': summary_stats,
            'size_analysis': {},
            'short_chunks': {},
            'dialogue_analysis': {},
            'flow_analysis': {},
            'issues': ['No chunks found'],
            'recommendations': ['Check input file and processing pipeline']
        }
    
    # Analyze different aspects
    size_analysis = analyze_chunk_sizes(chunks)
    short_chunks = detect_short_chunks(chunks)
    dialogue_analysis = detect_dialogue_patterns(chunks)
    flow_analysis = analyze_chunk_flow(chunks)
    
    # Calculate quality score
    quality_factors = []
    issues = []
    recommendations = []
    
    # Factor 1: Chunk size distribution (0.3 weight)
    word_counts = size_analysis['word_counts']
    if word_counts:
        very_short_ratio = len(short_chunks['very_short']) / len(chunks)
        short_ratio = len(short_chunks['short']) / len(chunks)
        
        size_score = 1.0 - (very_short_ratio * 0.8 + short_ratio * 0.4)
        quality_factors.append(('size_distribution', size_score, 0.3))
        
        if very_short_ratio > 0.1:
            issues.append(f"High ratio of very short chunks: {very_short_ratio:.1%}")
            recommendations.append("Consider adjusting minimum chunk size or improving merging logic")
        
        if short_ratio > 0.2:
            issues.append(f"High ratio of short chunks: {short_ratio:.1%}")
            recommendations.append("Review conversational text handling and chunk merging")
    else:
        quality_factors.append(('size_distribution', 0.0, 0.3))
        issues.append("No valid chunks with text content")
    
    # Factor 2: Text flow quality (0.4 weight)
    flow_score = flow_analysis['flow_score']
    quality_factors.append(('text_flow', flow_score, 0.4))
    
    if flow_score < 0.8:
        issues.append(f"Poor text flow continuity: {flow_score:.2f}")
        recommendations.append("Improve page boundary handling and sentence reconstruction")
    
    if len(flow_analysis['abrupt_transitions']) > 0:
        issues.append(f"{len(flow_analysis['abrupt_transitions'])} abrupt transitions detected")
        recommendations.append("Review semantic chunking boundaries")
    
    # Factor 3: Dialogue handling (0.3 weight)
    dialogue_score = 1.0
    total_dialogue_chunks = (len(dialogue_analysis['quoted_speech']) + 
                             len(dialogue_analysis['dialogue_attribution']) +
                             len(dialogue_analysis['conversational_responses']))
    
    if total_dialogue_chunks > 0:
        fragment_ratio = len(dialogue_analysis['potential_fragments']) / total_dialogue_chunks
        dialogue_score = max(0.0, 1.0 - fragment_ratio)
        
        if fragment_ratio > 0.3:
            issues.append(f"High dialogue fragmentation: {fragment_ratio:.1%}")
            recommendations.append("Improve dialogue pattern detection and merging")
    quality_factors.append(('dialogue_handling', dialogue_score, 0.3))
    
    # Calculate overall quality score
    weighted_score = sum(score * weight for _, score, weight in quality_factors)
    overall_quality = max(0.0, min(1.0, weighted_score))
    
    # Generate summary statistics
    summary_stats = {
        'total_chunks': len(chunks),
        'total_words': sum(size_analysis['word_counts']) if size_analysis['word_counts'] else 0,
        'total_characters': sum(size_analysis['char_counts']) if size_analysis['char_counts'] else 0,
        'avg_words_per_chunk': size_analysis['statistics']['words']['mean'] if size_analysis['statistics'] and 'words' in size_analysis['statistics'] and 'mean' in size_analysis['statistics']['words'] else 0,
        'very_short_chunks': len(short_chunks['very_short']) if 'very_short' in short_chunks else 0,
        'short_chunks': len(short_chunks['short']) if 'short' in short_chunks else 0,
        'dialogue_chunks': total_dialogue_chunks,
        'flow_issues': len(flow_analysis['abrupt_transitions']) + len(flow_analysis['potential_splits']) if flow_analysis else 0
    }
    
    return {
        'filename': filename,
        'quality_score': overall_quality,
        'quality_factors': quality_factors,
        'summary_stats': summary_stats,
        'size_analysis': size_analysis,
        'short_chunks': short_chunks,
        'dialogue_analysis': dialogue_analysis,
        'flow_analysis': flow_analysis,
        'issues': issues,
        'recommendations': recommendations
    }


def compare_quality_reports(report1: Dict[str, Any], report2: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two quality reports and generate improvement analysis."""
    comparison = {
        'file1': report1['filename'],
        'file2': report2['filename'],
        'quality_improvement': report2['quality_score'] - report1['quality_score'],
        'improvements': [],
        'regressions': [],
        'summary': {}
    }
    
    # Compare summary statistics
    stats1 = report1['summary_stats']
    stats2 = report2['summary_stats']
    
    comparison['summary'] = {
        'chunk_count_change': stats2['total_chunks'] - stats1['total_chunks'],
        'very_short_change': stats2['very_short_chunks'] - stats1['very_short_chunks'],
        'short_change': stats2['short_chunks'] - stats1['short_chunks'],
        'flow_issues_change': stats2['flow_issues'] - stats1['flow_issues'],
        'dialogue_chunks_change': stats2['dialogue_chunks'] - stats1['dialogue_chunks']
    }
    
    # Analyze improvements and regressions
    if comparison['summary']['very_short_change'] < 0:
        comparison['improvements'].append(f"Reduced very short chunks by {-comparison['summary']['very_short_change']}")
    elif comparison['summary']['very_short_change'] > 0:
        comparison['regressions'].append(f"Increased very short chunks by {comparison['summary']['very_short_change']}")
    
    if comparison['summary']['short_change'] < 0:
        comparison['improvements'].append(f"Reduced short chunks by {-comparison['summary']['short_change']}")
    elif comparison['summary']['short_change'] > 0:
        comparison['regressions'].append(f"Increased short chunks by {comparison['summary']['short_change']}")
    
    if comparison['summary']['flow_issues_change'] < 0:
        comparison['improvements'].append(f"Reduced flow issues by {-comparison['summary']['flow_issues_change']}")
    elif comparison['summary']['flow_issues_change'] > 0:
        comparison['regressions'].append(f"Increased flow issues by {comparison['summary']['flow_issues_change']}")
    
    # Compare quality factors
    factors1 = {name: score for name, score, _ in report1['quality_factors']}
    factors2 = {name: score for name, score, _ in report2['quality_factors']}
    
    for factor_name in factors1:
        if factor_name in factors2:
            improvement = factors2[factor_name] - factors1[factor_name]
            if improvement > 0.05:
                comparison['improvements'].append(f"Improved {factor_name}: {improvement:+.3f}")
            elif improvement < -0.05:
                comparison['regressions'].append(f"Degraded {factor_name}: {improvement:+.3f}")
    
    return comparison


def print_quality_report(report: Dict[str, Any], detailed: bool = True):
    """Print a formatted quality report."""
    print("=" * 80)
    print("CHUNK QUALITY ASSESSMENT REPORT")
    print("=" * 80)
    print(f"File: {report.get('filename', 'unknown')}")
    print(f"Overall Quality Score: {report.get('quality_score', 0.0):.3f}")
    print()
    
    # Summary statistics
    stats = report.get('summary_stats', {})
    print("SUMMARY STATISTICS")
    print("-" * 40)
    print(f"Total Chunks:           {stats.get('total_chunks', 0)}")
    print(f"Total Words:            {stats.get('total_words', 0)}")
    print(f"Total Characters:       {stats.get('total_characters', 0)}")
    print(f"Avg Words per Chunk:    {stats.get('avg_words_per_chunk', 0):.1f}")
    print()
    
    # Chunk size distribution
    print("CHUNK SIZE DISTRIBUTION")
    print("-" * 40)
    total_chunks = stats.get('total_chunks', 0)
    very_short = stats.get('very_short_chunks', 0)
    short = stats.get('short_chunks', 0)
    normal = total_chunks - very_short - short if total_chunks else 0
    print(f"Very Short (≤3 words):  {very_short} ({(very_short/total_chunks*100) if total_chunks else 0:.1f}%)")
    print(f"Short (4-7 words):      {short} ({(short/total_chunks*100) if total_chunks else 0:.1f}%)")
    print(f"Normal (>7 words):      {normal}")
    print()
    
    # Quality factors
    print("QUALITY FACTORS")
    print("-" * 40)
    for factor in report.get('quality_factors', []):
        factor_name, score, weight = factor
        print(f"{factor_name.replace('_', ' ').title():<20} {score:.3f} (weight: {weight:.1f})")
    print()
    
    # Issues and recommendations
    if report.get('issues'):
        print("ISSUES IDENTIFIED")
        print("-" * 40)
        for i, issue in enumerate(report['issues'], 1):
            print(f"{i}. {issue}")
        print()
    
    if report.get('recommendations'):
        print("RECOMMENDATIONS")
        print("-" * 40)
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"{i}. {rec}")
        print()
    
    if detailed and total_chunks:
        # Detailed analysis
        short_chunks = report.get('short_chunks', {})
    
        if short_chunks.get('very_short'):
            print("VERY SHORT CHUNKS (≤3 words)")
            print("-" * 40)
            for chunk in short_chunks['very_short'][:5]:
                print(f"Chunk {chunk['index']}: {chunk['word_count']} words - '{chunk['text_preview']}'")
            if len(short_chunks['very_short']) > 5:
                print(f"... and {len(short_chunks['very_short']) - 5} more")
            print()
    
        if short_chunks.get('short'):
            print("SHORT CHUNKS (4-7 words)")
            print("-" * 40)
            for chunk in short_chunks['short'][:5]:
                print(f"Chunk {chunk['index']}: {chunk['word_count']} words - '{chunk['text_preview']}'")
            if len(short_chunks['short']) > 5:
                print(f"... and {len(short_chunks['short']) - 5} more")
            print()
    
        # Dialogue analysis
        dialogue = report.get('dialogue_analysis', {})
        if any(dialogue.values()):
            print("DIALOGUE ANALYSIS")
            print("-" * 40)
            print(f"Quoted Speech Chunks:      {len(dialogue.get('quoted_speech', []))}")
            print(f"Dialogue Attribution:      {len(dialogue.get('dialogue_attribution', []))}")
            print(f"Conversational Responses:  {len(dialogue.get('conversational_responses', []))}")
            print(f"Potential Fragments:       {len(dialogue.get('potential_fragments', []))}")
            print()
    
        # Flow analysis
        flow = report.get('flow_analysis', {})
        if flow.get('abrupt_transitions') or flow.get('potential_splits'):
            print("TEXT FLOW ISSUES")
            print("-" * 40)
            print(f"Flow Score:           {flow.get('flow_score', 0.0):.3f}")
            print(f"Abrupt Transitions:   {len(flow.get('abrupt_transitions', []))}")
            print(f"Potential Word Splits: {len(flow.get('potential_splits', []))}")
            print(f"Continuation Issues:  {len(flow.get('continuation_issues', []))}")
            print()


def print_comparison_report(comparison: Dict[str, Any]):
    """Print a formatted comparison report."""
    print("=" * 80)
    print("CHUNK QUALITY COMPARISON REPORT")
    print("=" * 80)
    print(f"File 1: {comparison.get('file1', 'unknown')}")
    print(f"File 2: {comparison.get('file2', 'unknown')}")
    print(f"Quality Improvement: {comparison.get('quality_improvement', 0.0):+.3f}")
    print()
    
    # Summary changes
    summary = comparison.get('summary', {})
    print("SUMMARY CHANGES")
    print("-" * 40)
    print(f"Chunk Count Change:      {summary.get('chunk_count_change', 0):+d}")
    print(f"Very Short Chunks:       {summary.get('very_short_change', 0):+d}")
    print(f"Short Chunks:            {summary.get('short_change', 0):+d}")
    print(f"Flow Issues:             {summary.get('flow_issues_change', 0):+d}")
    print(f"Dialogue Chunks:         {summary.get('dialogue_chunks_change', 0):+d}")
    print()
    
    # Improvements
    if comparison.get('improvements'):
        print("IMPROVEMENTS")
        print("-" * 40)
        for improvement in comparison['improvements']:
            print(f"✓ {improvement}")
        print()
    
    # Regressions
    if comparison.get('regressions'):
        print("REGRESSIONS")
        print("-" * 40)
        for regression in comparison['regressions']:
            print(f"✗ {regression}")
        print()
    
    # Overall assessment
    quality_improvement = comparison.get('quality_improvement', 0.0)
    print("OVERALL ASSESSMENT")
    print("-" * 40)
    if quality_improvement > 0.05:
        print("✓ SIGNIFICANT IMPROVEMENT: File 2 shows better chunk quality")
    elif quality_improvement > 0.0:
        print("✓ MODERATE IMPROVEMENT: File 2 shows some improvement")
    elif quality_improvement > -0.05:
        print("≈ SIMILAR QUALITY: Both files have comparable chunk quality")
    else:
        print("✗ DEGRADATION: File 2 shows worse chunk quality")
    print()


def generate_chunks_from_pdf(pdf_path: str, traditional: bool = False, enhanced: bool = True) -> List[Dict[str, Any]]:
    """Generate chunks from a PDF using specified approach."""
    import os

    # Only one approach should be active at a time; if both True, enhanced takes precedence
    if traditional and not enhanced:
        # Traditional: disable conversational text handling
        os.environ['PDF_CHUNKER_USE_PYMUPDF4LLM'] = 'false'
        min_chunk_size = None
        enable_dialogue_detection = False
    else:
        # Enhanced: enable conversational text handling
        os.environ['PDF_CHUNKER_USE_PYMUPDF4LLM'] = 'true'
        min_chunk_size = 8
        enable_dialogue_detection = True

    try:
        # Process the document
        chunks = process_document(
            pdf_path,
            chunk_size=8000,
            overlap=200,
            generate_metadata=True,
            ai_enrichment=False,  # Disable AI for faster processing
            min_chunk_size=min_chunk_size,
            enable_dialogue_detection=enable_dialogue_detection
        )
        return chunks
    except Exception as e:
        print(f"Error processing PDF '{pdf_path}': {e}", file=sys.stderr)
        return []
    finally:
        # Clean up environment
        if 'PDF_CHUNKER_USE_PYMUPDF4LLM' in os.environ:
            del os.environ['PDF_CHUNKER_USE_PYMUPDF4LLM']


def save_chunks_to_jsonl(chunks: List[Dict[str, Any]], output_path: str):
    """Save chunks to a JSONL file."""
    # Remove any double extensions like .traditional.jsonl or .enhanced.jsonl
    base, ext = os.path.splitext(output_path)
    if ext.lower() != ".jsonl":
        output_path = base + ".jsonl"
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                json.dump(chunk, f, ensure_ascii=False)
                f.write('\n')
        print(f"Saved {len(chunks)} chunks to {output_path}")
    except Exception as e:
        print(f"Error saving chunks to '{output_path}': {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description='Validate chunk quality from JSONL output')

    # Main operation modes
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('jsonl_file', nargs='?', help='JSONL file to analyze')
    group.add_argument('--compare', nargs=2, metavar=('FILE1', 'FILE2'),
                      help='Compare two JSONL files')
    group.add_argument('--generate', metavar='PDF_FILE',
                      help='Generate chunks from PDF and analyze')

    # Options for single file analysis
    parser.add_argument('--baseline', metavar='BASELINE_JSONL',
                       help='Baseline JSONL file for comparison')
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed analysis')

    # Options for PDF generation
    parser.add_argument('--traditional', action='store_true',
                       help='Use traditional approach (when generating from PDF)')
    parser.add_argument('--enhanced', action='store_true', default=True,
                       help='Use enhanced approach (default when generating from PDF)')
    parser.add_argument('--save-jsonl', metavar='OUTPUT_PATH',
                       help='Save generated chunks to JSONL file')

    args = parser.parse_args()

    if args.generate:
        # Generate chunks from PDF
        pdf_path = args.generate
        if not os.path.exists(pdf_path):
            print(f"Error: PDF file '{pdf_path}' not found", file=sys.stderr)
            sys.exit(1)

        print(f"Generating chunks from PDF: {pdf_path}")

        # If both --traditional and --enhanced, generate both and save to fixed file names
        if args.traditional and args.enhanced:
            print("Generating traditional approach chunks...")
            traditional_chunks = generate_chunks_from_pdf(pdf_path, traditional=True, enhanced=False)
            trad_path = "chunk_quality_traditional.jsonl"
            save_chunks_to_jsonl(traditional_chunks, trad_path)

            print("Generating enhanced approach chunks...")
            enhanced_chunks = generate_chunks_from_pdf(pdf_path, traditional=False, enhanced=True)
            enh_path = "chunk_quality_enhanced.jsonl"
            save_chunks_to_jsonl(enhanced_chunks, enh_path)

            # Analyze both
            print("\nTraditional Approach Analysis:")
            trad_report = generate_quality_report(traditional_chunks, f"{pdf_path} (traditional)")
            print_quality_report(trad_report, detailed=args.detailed)

            print("\nEnhanced Approach Analysis:")
            enh_report = generate_quality_report(enhanced_chunks, f"{pdf_path} (enhanced)")
            print_quality_report(enh_report, detailed=args.detailed)

            # Compare
            print("\nComparison (Traditional vs Enhanced):")
            comparison = compare_quality_reports(trad_report, enh_report)
            print_comparison_report(comparison)

        else:
            # Generate single approach
            approach = "traditional" if args.traditional else "enhanced"
            chunks = generate_chunks_from_pdf(pdf_path, traditional=args.traditional, enhanced=args.enhanced)
            # Use default file names if not specified
            if args.save_jsonl:
                save_chunks_to_jsonl(chunks, args.save_jsonl)
            else:
                default_path = f"chunk_quality_{approach}.jsonl"
                save_chunks_to_jsonl(chunks, default_path)

            # Analyze
            report = generate_quality_report(chunks, f"{pdf_path} ({approach})")
            print_quality_report(report, detailed=args.detailed)

    elif args.compare:
        # Compare two JSONL files
        file1, file2 = args.compare

        print(f"Loading chunks from {file1}...")
        chunks1 = load_jsonl(file1)

        print(f"Loading chunks from {file2}...")
        chunks2 = load_jsonl(file2)

        if not chunks1 or not chunks2:
            print("Error: Could not load chunks from one or both files", file=sys.stderr)
            sys.exit(1)

        # Generate reports
        report1 = generate_quality_report(chunks1, file1)
        report2 = generate_quality_report(chunks2, file2)

        # Print individual reports
        print(f"\nAnalysis of {file1}:")
        print_quality_report(report1, detailed=args.detailed)

        print(f"\nAnalysis of {file2}:")
        print_quality_report(report2, detailed=args.detailed)

        # Print comparison
        print(f"\nComparison ({file1} vs {file2}):")
        comparison = compare_quality_reports(report1, report2)
        print_comparison_report(comparison)

    else:
        # Analyze single JSONL file
        jsonl_file = args.jsonl_file

        if not os.path.exists(jsonl_file):
            print(f"Error: JSONL file '{jsonl_file}' not found", file=sys.stderr)
            sys.exit(1)

        print(f"Loading chunks from {jsonl_file}...")
        chunks = load_jsonl(jsonl_file)

        if not chunks:
            print("Error: No valid chunks found in file", file=sys.stderr)
            sys.exit(1)

        # Generate and print report
        report = generate_quality_report(chunks, jsonl_file)
        print_quality_report(report, detailed=args.detailed)

        # Compare with baseline if provided
        if args.baseline:
            if not os.path.exists(args.baseline):
                print(f"Warning: Baseline file '{args.baseline}' not found", file=sys.stderr)
            else:
                print(f"\nLoading baseline from {args.baseline}...")
                baseline_chunks = load_jsonl(args.baseline)

                if baseline_chunks:
                    baseline_report = generate_quality_report(baseline_chunks, args.baseline)

                    print(f"\nComparison (Baseline vs Current):")
                    comparison = compare_quality_reports(baseline_report, report)
                    print_comparison_report(comparison)


if __name__ == '__main__':
    main()
