#!/bin/bash

# Main test orchestrator for PDF chunker test suite

# Source common utilities
source tests/utils/common.sh

# Track overall test results
overall_success=true

# Run PDF extraction tests
print_test_header "enhanced PDF extraction with fallback strategy"
if python3 tests/pdf_extraction_test.py; then
    print_test_result "PDF extraction" "success"
else
    print_test_result "PDF extraction" "failure"
    overall_success=false
fi

echo

# Run AI enrichment tests
print_test_header "enhanced AI enrichment with tag configuration"
if python3 tests/ai_enrichment_test.py; then
    print_test_result "AI enrichment" "success"
else
    print_test_result "AI enrichment" "failure"
    overall_success=false
fi

echo

# Run semantic chunking tests
print_test_header "semantic chunking with proper size validation"
if python3 tests/semantic_chunking_test.py; then
    print_test_result "Semantic chunking" "success"
else
    print_test_result "Semantic chunking" "failure"
    overall_success=false
fi

echo

# Run page exclusion tests
print_test_header "PDF page exclusion functionality"
if python3 tests/page_exclusion_test.py; then
    print_test_result "Page exclusion" "success"
else
    print_test_result "Page exclusion" "failure"
    overall_success=false
fi

echo

# Run EPUB spine exclusion tests
print_test_header "EPUB spine exclusion functionality"
if python3 tests/epub_spine_test.py; then
    print_test_result "EPUB spine exclusion" "success"
else
    print_test_result "EPUB spine exclusion" "failure"
    overall_success=false
fi

echo

# Final verification test
print_test_header "existing PDF page exclusion functionality verification"
python3 -c "
import sys
import os
sys.path.insert(0, '.')

from pdf_chunker.parsing import extract_structured_text

# Gather available PDFs
test_files = [
    os.path.join(root, file)
    for root, _, files in os.walk('.')
    for file in files
    if file.lower().endswith('.pdf')
]

# Choose first multi-page PDF using fitz for accuracy
pdf_file = next(
    (f for f in test_files if __import__('fitz').open(f).__len__() > 1),
    None,
)

if pdf_file:
    print(f'Quick verification with: {pdf_file}')

    try:
        baseline_blocks = extract_structured_text(pdf_file)
        excluded_blocks = extract_structured_text(pdf_file, exclude_pages='1')

        if baseline_blocks and excluded_blocks:
            baseline_pages = {
                b.get('source', {}).get('page')
                for b in baseline_blocks
                if b.get('source', {}).get('page')
            }
            excluded_pages = {
                b.get('source', {}).get('page')
                for b in excluded_blocks
                if b.get('source', {}).get('page')
            }

            if 1 in excluded_pages:
                print('  ERROR: PDF page exclusion broken - page 1 still appears')
                sys.exit(1)
            else:
                print('  SUCCESS: PDF page exclusion functionality preserved')
        else:
            print('  WARNING: Could not verify PDF page exclusion')

    except Exception as e:
        print(f'  ERROR: PDF page exclusion verification failed: {e}')
        sys.exit(1)
else:
    print('  SKIP: No suitable PDF files available for verification')
"

if [ $? -eq 0 ]; then
    print_test_result "PDF page exclusion verification" "success"
else
    print_test_result "PDF page exclusion verification" "failure"
    overall_success=false
fi

echo

# Print final results
if [ "$overall_success" = true ]; then
    echo "All testing completed successfully!"
    exit 0
else
    echo "Some tests failed. Please check the output above."
    exit 1
fi
