#!/usr/bin/env python3
"""
Duplication Detection Script for PDF Chunking

This script analyzes JSONL chunk files to detect content duplication between chunks.
It uses a sliding window approach to find overlapping text segments and reports
any duplicated content with chunk IDs and positions.

Usage:
    python scripts/detect_duplicates.py <jsonl_file>
    python scripts/detect_duplicates.py output_chunks_pdf.jsonl
"""

import json
import sys
from typing import List, Dict, Set, Tuple
from collections import defaultdict


def load_chunks(jsonl_file: str) -> List[Dict]:
    """Load chunks from JSONL file."""
    chunks = []
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    chunk['_line_number'] = line_num
                    chunks.append(chunk)
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {line_num}: {e}")
                    continue
    except FileNotFoundError:
        print(f"Error: File '{jsonl_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{jsonl_file}': {e}")
        sys.exit(1)
    
    return chunks


def create_sliding_windows(text: str, window_size: int = 50) -> List[str]:
    """Create sliding windows of text for overlap detection."""
    if len(text) < window_size:
        return [text.strip()]
    
    windows = []
    words = text.split()
    
    for i in range(len(words) - window_size + 1):
        window = ' '.join(words[i:i + window_size])
        windows.append(window.strip())
    
    return windows


def detect_duplications(chunks: List[Dict], window_size: int = 50, min_overlap: int = 20) -> List[Dict]:
    """
    Detect content duplications between chunks using sliding window approach.
    
    Args:
        chunks: List of chunk dictionaries
        window_size: Size of sliding window in words
        min_overlap: Minimum overlap size in words to report as duplication
    
    Returns:
        List of duplication reports
    """
    duplications = []
    window_to_chunks = defaultdict(list)
    
    # Create sliding windows for each chunk
    for i, chunk in enumerate(chunks):
        text = chunk.get('text', '')
        if not text.strip():
            continue
            
        windows = create_sliding_windows(text, window_size)
        
        for window_idx, window in enumerate(windows):
            if len(window.split()) >= min_overlap:
                window_to_chunks[window].append({
                    'chunk_index': i,
                    'chunk_line': chunk.get('_line_number', i + 1),
                    'window_index': window_idx,
                    'text_preview': window[:100] + '...' if len(window) > 100 else window
                })
    
    # Find duplicated windows
    for window, chunk_list in window_to_chunks.items():
        if len(chunk_list) > 1:
            # Multiple chunks contain this window - potential duplication
            duplication = {
                'duplicated_text': window[:200] + '...' if len(window) > 200 else window,
                'word_count': len(window.split()),
                'occurrences': []
            }
            
            for occurrence in chunk_list:
                chunk = chunks[occurrence['chunk_index']]
                duplication['occurrences'].append({
                    'chunk_line': occurrence['chunk_line'],
                    'chunk_index': occurrence['chunk_index'],
                    'window_position': occurrence['window_index'],
                    'chunk_id': chunk.get('metadata', {}).get('chunk_id', f"chunk_{occurrence['chunk_index']}"),
                    'chunk_preview': chunk.get('text', '')[:100] + '...' if len(chunk.get('text', '')) > 100 else chunk.get('text', '')
                })
            
            duplications.append(duplication)
    
    # Sort by word count (largest duplications first)
    duplications.sort(key=lambda x: x['word_count'], reverse=True)
    
    return duplications


def analyze_chunk_boundaries(chunks: List[Dict]) -> Dict:
    """Analyze potential boundary issues between consecutive chunks."""
    boundary_issues = []
    
    for i in range(len(chunks) - 1):
        current_chunk = chunks[i]
        next_chunk = chunks[i + 1]
        
        current_text = current_chunk.get('text', '').strip()
        next_text = next_chunk.get('text', '').strip()
        
        if not current_text or not next_text:
            continue
        
        # Check if next chunk starts with end of current chunk
        current_words = current_text.split()
        next_words = next_text.split()
        
        # Look for overlap at boundaries
        max_check = min(20, len(current_words), len(next_words))
        
        for overlap_size in range(max_check, 4, -1):  # Check from large to small overlaps
            current_end = ' '.join(current_words[-overlap_size:])
            next_start = ' '.join(next_words[:overlap_size])
            
            if current_end.lower() == next_start.lower():
                boundary_issues.append({
                    'chunk1_line': current_chunk.get('_line_number', i + 1),
                    'chunk2_line': next_chunk.get('_line_number', i + 2),
                    'chunk1_id': current_chunk.get('metadata', {}).get('chunk_id', f"chunk_{i}"),
                    'chunk2_id': next_chunk.get('metadata', {}).get('chunk_id', f"chunk_{i+1}"),
                    'overlap_words': overlap_size,
                    'overlapping_text': current_end,
                    'issue_type': 'boundary_overlap'
                })
                break
    
    return {
        'boundary_overlaps': boundary_issues,
        'total_boundary_issues': len(boundary_issues)
    }


def generate_report(chunks: List[Dict], duplications: List[Dict], boundary_analysis: Dict, jsonl_file: str):
    """Generate and print the duplication detection report."""
    print(f"\n{'='*80}")
    print(f"DUPLICATION DETECTION REPORT")
    print(f"{'='*80}")
    print(f"File: {jsonl_file}")
    print(f"Total chunks analyzed: {len(chunks)}")
    print(f"Total duplications found: {len(duplications)}")
    print(f"Boundary overlap issues: {boundary_analysis['total_boundary_issues']}")
    
    if duplications:
        print(f"\n{'='*60}")
        print("CONTENT DUPLICATIONS")
        print(f"{'='*60}")
        
        for i, dup in enumerate(duplications[:10], 1):  # Show top 10
            print(f"\n--- Duplication #{i} ---")
            print(f"Duplicated text ({dup['word_count']} words):")
            print(f"  \"{dup['duplicated_text']}\"")
            print(f"Found in {len(dup['occurrences'])} chunks:")
            
            for occ in dup['occurrences']:
                print(f"  • Line {occ['chunk_line']}: {occ['chunk_id']}")
                print(f"    Position: window {occ['window_position']}")
                print(f"    Preview: \"{occ['chunk_preview']}\"")
        
        if len(duplications) > 10:
            print(f"\n... and {len(duplications) - 10} more duplications")
    
    if boundary_analysis['boundary_overlaps']:
        print(f"\n{'='*60}")
        print("BOUNDARY OVERLAP ISSUES")
        print(f"{'='*60}")
        
        for issue in boundary_analysis['boundary_overlaps'][:5]:  # Show top 5
            print(f"\nOverlap between chunks:")
            print(f"  Chunk 1 (Line {issue['chunk1_line']}): {issue['chunk1_id']}")
            print(f"  Chunk 2 (Line {issue['chunk2_line']}): {issue['chunk2_id']}")
            print(f"  Overlapping text ({issue['overlap_words']} words): \"{issue['overlapping_text']}\"")
        
        if len(boundary_analysis['boundary_overlaps']) > 5:
            print(f"\n... and {len(boundary_analysis['boundary_overlaps']) - 5} more boundary issues")
    
    if not duplications and not boundary_analysis['boundary_overlaps']:
        print(f"\n✅ No content duplications or boundary issues detected!")
    else:
        print(f"\n⚠️  Issues detected. Review chunking algorithm for improvements.")
    
    print(f"\n{'='*80}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/detect_duplicates.py <jsonl_file>")
        print("Example: python scripts/detect_duplicates.py output_chunks_pdf.jsonl")
        sys.exit(1)
    
    jsonl_file = sys.argv[1]
    
    print(f"Loading chunks from {jsonl_file}...")
    chunks = load_chunks(jsonl_file)
    
    if not chunks:
        print("No valid chunks found in file.")
        sys.exit(1)
    
    print(f"Analyzing {len(chunks)} chunks for duplications...")
    
    # Detect content duplications
    duplications = detect_duplications(chunks, window_size=50, min_overlap=10)
    
    # Analyze boundary issues
    boundary_analysis = analyze_chunk_boundaries(chunks)
    
    # Generate report
    generate_report(chunks, duplications, boundary_analysis, jsonl_file)


if __name__ == "__main__":
    main()
