#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, '.')

from pdf_chunker.core import process_document
from pdf_chunker.parsing import extract_structured_text

def test_epub_spine_exclusion():
    """Test EPUB spine exclusion functionality"""

    # Use a specific, known-good test EPUB file
    test_epub_path = os.path.join('test_data', 'sample_test.epub')
    if not os.path.exists(test_epub_path):
        print(f'Known-good test EPUB not found at {test_epub_path}.')
        print('Please generate it with scripts/generate_test_epub.py.')
        return False

    print(f'Testing spine-based exclusion functionality with known-good EPUB: {test_epub_path}')
    success_count = 0

    epub_file = test_epub_path
    print(f'Testing spine exclusion with: {epub_file}')

    try:
        # First, extract without exclusions to get baseline
        print(f'  Baseline extraction (no exclusions):')
        baseline_blocks = extract_structured_text(epub_file)

        if baseline_blocks:
            # Get spine information from baseline
            spine_items = set()
            for block in baseline_blocks:
                location = block.get('source', {}).get('location')
                if location:
                    spine_items.add(location)

            total_spines = len(spine_items)
            print(f'    Total spine items in document: {total_spines}')
            print(f'    Baseline blocks extracted: {len(baseline_blocks)}')
            print(f'    Spine items with content: {sorted(list(spine_items)[:5])}...')  # Show first 5

            if total_spines >= 3:
                # Test excluding first spine item
                print(f'  Test 1: Excluding spine item 1')
                try:
                    excluded_blocks = extract_structured_text(epub_file, exclude_pages='1')
                    excluded_spines = set()
                    for block in excluded_blocks:
                        location = block.get('source', {}).get('location')
                        if location:
                            excluded_spines.add(location)

                    # Check if first spine item content is missing
                    first_spine_content = [block for block in baseline_blocks if block.get('source', {}).get('location') == sorted(spine_items)[0]]
                    first_spine_in_excluded = [block for block in excluded_blocks if block.get('source', {}).get('location') == sorted(spine_items)[0]]

                    if first_spine_in_excluded:
                        print(f'    ERROR: First spine item content still appears in output!')
                    else:
                        print(f'    SUCCESS: First spine item successfully excluded')
                        success_count += 1

                    print(f'    Blocks after exclusion: {len(excluded_blocks)}')
                    print(f'    Spine items with content: {len(excluded_spines)}')

                except Exception as e:
                    print(f'    ERROR: Spine exclusion failed: {e}')

                # Test full pipeline with spine exclusions
                print(f'  Test 2: Full pipeline with spine exclusions')
                try:
                    chunks = process_document(
                        epub_file,
                        chunk_size=300,
                        overlap=30,
                        generate_metadata=True,
                        ai_enrichment=False,
                        exclude_pages='1'
                    )

                    if chunks:
                        # Check that excluded spine items don't appear in final chunks
                        chunk_locations = set()
                        for chunk in chunks:
                            location = chunk.get('metadata', {}).get('location')
                            if location:
                                chunk_locations.add(location)

                        sorted_spines = sorted(spine_items)
                        first_spine_in_chunks = sorted_spines[0] in chunk_locations if sorted_spines else False

                        if first_spine_in_chunks:
                            print(f'    ERROR: Excluded first spine item appears in final chunks!')
                        else:
                            print(f'    SUCCESS: Spine exclusion works in full pipeline')
                            success_count += 1

                        print(f'    Final chunks generated: {len(chunks)}')
                        print(f'    Spine items in final chunks: {len(chunk_locations)}')
                    else:
                        print(f'    ERROR: No chunks generated in full pipeline')

                except Exception as e:
                    print(f'    ERROR: Full pipeline with exclusions failed: {e}')

            else:
                print(f'    SKIP: Document too short ({total_spines} spine items) for comprehensive testing')

        else:
            print(f'    ERROR: No baseline blocks extracted')

    except Exception as e:
        print(f'  ERROR: Baseline extraction failed: {e}')
        import traceback
        traceback.print_exc()

    print()
    return success_count > 0

if __name__ == '__main__':
    success = test_epub_spine_exclusion()
    sys.exit(0 if success else 1)
