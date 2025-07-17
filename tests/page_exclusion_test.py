#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, '.')

from pdf_chunker.core import process_document
from pdf_chunker.parsing import extract_structured_text

def test_page_exclusion():
    """Test PDF page exclusion functionality"""
    
    # Test with all available PDF files to verify page exclusion functionality
    test_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.lower().endswith('.pdf'):
                test_files.append(os.path.join(root, file))

    if not test_files:
        print('No PDF files found for testing page exclusion functionality.')
        return False

    print(f'Testing page exclusion functionality with {len(test_files)} PDF file(s):')
    success_count = 0

    for pdf_file in test_files[:2]:  # Test up to 2 PDFs to avoid long processing
        print(f'Testing page exclusion with: {pdf_file}')
        
        try:
            # First, extract without exclusions to get baseline
            print(f'  Baseline extraction (no exclusions):')
            baseline_blocks = extract_structured_text(pdf_file)
            
            if baseline_blocks:
                # Get page numbers from baseline
                baseline_pages = set()
                for block in baseline_blocks:
                    page = block.get('source', {}).get('page')
                    if page:
                        baseline_pages.add(page)
                
                total_pages = max(baseline_pages) if baseline_pages else 0
                print(f'    Total pages in document: {total_pages}')
                print(f'    Baseline blocks extracted: {len(baseline_blocks)}')
                print(f'    Pages with content: {sorted(baseline_pages)}')
                
                if total_pages >= 3:
                    # Test excluding first page
                    print(f'  Test 1: Excluding page 1')
                    try:
                        excluded_blocks = extract_structured_text(pdf_file, exclude_pages='1')
                        excluded_pages = set()
                        for block in excluded_blocks:
                            page = block.get('source', {}).get('page')
                            if page:
                                excluded_pages.add(page)
                        
                        if 1 in excluded_pages:
                            print(f'    ERROR: Page 1 still appears in output!')
                        else:
                            print(f'    SUCCESS: Page 1 successfully excluded')
                            success_count += 1
                        
                        print(f'    Blocks after exclusion: {len(excluded_blocks)}')
                        print(f'    Pages with content: {sorted(excluded_pages)}')
                        
                    except Exception as e:
                        print(f'    ERROR: Page exclusion failed: {e}')
                    
                    # Test full pipeline with page exclusions
                    print(f'  Test 2: Full pipeline with page exclusions')
                    try:
                        chunks = process_document(
                            pdf_file,
                            chunk_size=300,
                            overlap=30,
                            generate_metadata=True,
                            ai_enrichment=False,
                            exclude_pages='1'
                        )
                        
                        if chunks:
                            # Check that excluded pages don't appear in final chunks
                            chunk_pages = set()
                            for chunk in chunks:
                                page = chunk.get('metadata', {}).get('page')
                                if page:
                                    chunk_pages.add(page)
                            
                            if 1 in chunk_pages:
                                print(f'    ERROR: Excluded page 1 appears in final chunks!')
                            else:
                                print(f'    SUCCESS: Page exclusion works in full pipeline')
                                success_count += 1
                            
                            print(f'    Final chunks generated: {len(chunks)}')
                            print(f'    Pages in final chunks: {sorted(chunk_pages)}')
                        else:
                            print(f'    ERROR: No chunks generated in full pipeline')
                            
                    except Exception as e:
                        print(f'    ERROR: Full pipeline with exclusions failed: {e}')
                
                else:
                    print(f'    SKIP: Document too short ({total_pages} pages) for comprehensive testing')
            
            else:
                print(f'    ERROR: No baseline blocks extracted')
                
        except Exception as e:
            print(f'  ERROR: Baseline extraction failed: {e}')
            import traceback
            traceback.print_exc()
        
        print()

    return success_count > 0

if __name__ == '__main__':
    success = test_page_exclusion()
    sys.exit(0 if success else 1)
