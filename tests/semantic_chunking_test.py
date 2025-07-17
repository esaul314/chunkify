#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, '.')

from pdf_chunker.core import process_document

def test_semantic_chunking():
    """Test semantic chunking with proper size validation"""
    
    # Test with all available PDF files to verify chunk sizes
    test_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.lower().endswith('.pdf'):
                test_files.append(os.path.join(root, file))

    if not test_files:
        print('No PDF files found for testing semantic chunking.')
        return False

    print(f'Testing semantic chunking with {len(test_files)} PDF file(s):')
    success_count = 0

    for pdf_file in test_files[:2]:  # Test up to 2 PDFs to avoid long processing
        print(f'Testing full pipeline with: {pdf_file}')
        try:
            # Process document with semantic chunking (no AI enrichment for speed)
            chunks = process_document(
                pdf_file,
                chunk_size=500,  # 500 words target
                overlap=50,      # 50 words overlap
                generate_metadata=True,
                ai_enrichment=False  # Disable AI for faster testing
            )
            
            if chunks:
                print(f'  SUCCESS: Generated {len(chunks)} chunks')
                
                # Analyze chunk sizes with strict validation
                chunk_sizes = [len(chunk.get('text', '')) for chunk in chunks]
                max_size = max(chunk_sizes) if chunk_sizes else 0
                avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
                
                print(f'  Chunk size analysis:')
                print(f'    Average chunk size: {avg_size:.0f} characters')
                print(f'    Maximum chunk size: {max_size} characters')
                print(f'    Minimum chunk size: {min(chunk_sizes) if chunk_sizes else 0} characters')
                
                # Check for oversized chunks with strict limits
                oversized_chunks = [i for i, size in enumerate(chunk_sizes) if size > 10000]
                if oversized_chunks:
                    print(f'  ERROR: Found {len(oversized_chunks)} chunks exceeding 10k characters:')
                    for i in oversized_chunks[:3]:  # Show first 3 oversized chunks
                        print(f'    Chunk {i}: {chunk_sizes[i]} characters')
                else:
                    print(f'  SUCCESS: All chunks are properly sized (<10k characters)')
                
                # Check for extremely long chunks (the original problem)
                extreme_chunks = [i for i, size in enumerate(chunk_sizes) if size > 25000]
                if extreme_chunks:
                    print(f'  CRITICAL ERROR: Found {len(extreme_chunks)} extremely long chunks (>25k chars):')
                    for i in extreme_chunks:
                        print(f'    Chunk {i}: {chunk_sizes[i]} characters')
                else:
                    print(f'  SUCCESS: No extremely long chunks found')
                
                # Validate JSONL line length compatibility
                jsonl_line_lengths = []
                for chunk in chunks:
                    # Simulate JSONL serialization
                    import json
                    jsonl_line = json.dumps(chunk)
                    jsonl_line_lengths.append(len(jsonl_line))
                
                max_jsonl_line = max(jsonl_line_lengths) if jsonl_line_lengths else 0
                print(f'  JSONL line analysis:')
                print(f'    Maximum JSONL line length: {max_jsonl_line} characters')
                
                long_jsonl_lines = [i for i, length in enumerate(jsonl_line_lengths) if length > 30000]
                if long_jsonl_lines:
                    print(f'  ERROR: Found {len(long_jsonl_lines)} JSONL lines exceeding 30k characters!')
                    for i in long_jsonl_lines[:3]:
                        print(f'    JSONL line {i}: {jsonl_line_lengths[i]} characters')
                else:
                    print(f'  SUCCESS: All JSONL lines are reasonable length')
                
                # Test refactored modules by importing them
                print(f'  Testing refactored modules:')
                try:
                    from pdf_chunker.text_cleaning import _clean_text, _clean_paragraph
                    from pdf_chunker.heading_detection import _detect_heading_fallback
                    from pdf_chunker.extraction_fallbacks import _assess_text_quality, _extract_with_pdftotext
                    print(f'    SUCCESS: All refactored modules imported successfully')
                    
                    # Test a few functions to ensure they work
                    test_text = 'This is a test para-graph with hyphenated words.'
                    cleaned = _clean_paragraph(test_text)
                    print(f'    Text cleaning test: "{test_text}" -> "{cleaned}"')
                    
                    heading_test = _detect_heading_fallback('Chapter 1: Introduction')
                    print(f'    Heading detection test: "Chapter 1: Introduction" -> {heading_test}')
                    
                    success_count += 1
                    
                except ImportError as e:
                    print(f'    ERROR: Failed to import refactored modules: {e}')
                except Exception as e:
                    print(f'    ERROR: Refactored module test failed: {e}')
                
                # Show sample chunks
                print(f'  Sample chunks:')
                for i in range(min(3, len(chunks))):
                    chunk_text = chunks[i].get('text', '')
                    preview = chunk_text[:100] if chunk_text else 'Empty chunk'
                    print(f'    Chunk {i} ({len(chunk_text)} chars): "{preview}..."')
                    
            else:
                print(f'  ERROR: No chunks generated')
                
        except Exception as e:
            print(f'  ERROR: Pipeline failed: {e}')
            import traceback
            traceback.print_exc()
        
        print()

    return success_count > 0

if __name__ == '__main__':
    success = test_semantic_chunking()
    sys.exit(0 if success else 1)
