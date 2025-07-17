#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, '.')

from pdf_chunker.parsing import extract_structured_text

def test_pdf_extraction():
    """Test enhanced PDF extraction with fallback strategy"""
    
    # Find PDF files for testing
    test_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.lower().endswith('.pdf'):
                test_files.append(os.path.join(root, file))

    if not test_files:
        print('No PDF files found for testing. Please add a PDF file to test the extraction.')
        return False

    print(f'Found {len(test_files)} PDF file(s) for testing:')
    success_count = 0
    
    for pdf_file in test_files[:3]:  # Test up to 3 PDFs
        print(f'Testing: {pdf_file}')
        try:
            blocks = extract_structured_text(pdf_file)
            
            if blocks:
                print(f'  Extracted {len(blocks)} text blocks')
                
                # Analyze text quality
                total_text = ' '.join(block.get('text', '') for block in blocks)
                lines = total_text.split('\n')
                non_empty_lines = [line for line in lines if line.strip()]
                
                if non_empty_lines:
                    avg_line_length = sum(len(line) for line in non_empty_lines) / len(non_empty_lines)
                    total_chars = sum(len(line) for line in non_empty_lines)
                    total_spaces = sum(line.count(' ') for line in non_empty_lines)
                    space_density = total_spaces / total_chars if total_chars > 0 else 0
                    
                    print(f'  Quality metrics:')
                    print(f'    Average line length: {avg_line_length:.1f} characters')
                    print(f'    Space density: {space_density:.3f}')
                    print(f'    Total blocks: {len(blocks)}')
                    
                    # Check for problematic patterns
                    long_lines = [line for line in non_empty_lines if len(line) > 1000]
                    if long_lines:
                        print(f'  WARNING: Found {len(long_lines)} lines longer than 1000 characters')
                        print(f'    Longest line: {len(max(long_lines, key=len))} characters')
                    
                    if space_density < 0.05:
                        print(f'  WARNING: Very low space density detected ({space_density:.3f})')
                    
                    # Show sample text from first few blocks
                    print(f'  Sample text from first block:')
                    if blocks:
                        sample_text = blocks[0].get('text', '')[:200]
                        print(f'    "{sample_text}..."')
                        
                    # Check extraction method used
                    if blocks and 'source' in blocks[0]:
                        method = blocks[0]['source'].get('method', 'PyMuPDF')
                        print(f'  Extraction method: {method}')
                    
                    # Check heading detection
                    headings = [block for block in blocks if block.get('type') == 'heading']
                    paragraphs = [block for block in blocks if block.get('type') == 'paragraph']
                    print(f'  Block types: {len(headings)} headings, {len(paragraphs)} paragraphs')
                    
                    if headings:
                        print(f'  Sample heading: "{headings[0].get("text", "")[:100]}..."')
                        
                    success_count += 1
                else:
                    print(f'  WARNING: No text content extracted')
            else:
                print(f'  ERROR: Failed to extract structured text')
                
        except Exception as e:
            print(f'  ERROR: {str(e)}')
            import traceback
            traceback.print_exc()
        
        print()

    return success_count > 0

if __name__ == '__main__':
    success = test_pdf_extraction()
    sys.exit(0 if success else 1)
