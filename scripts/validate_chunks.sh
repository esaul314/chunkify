#!/bin/bash

# Validate chunks script
# Performs structural validation on JSONL chunk files and detects duplicates

# Default file path
JSONL_FILE="${1:-output_chunks_pdf.jsonl}"

# Exit codes
EXIT_SUCCESS=0
EXIT_VALIDATION_FAILED=1
EXIT_FILE_NOT_FOUND=2

# Check if file exists
if [[ ! -f "$JSONL_FILE" ]]; then
    echo "Error: File '$JSONL_FILE' not found" >&2
    exit $EXIT_FILE_NOT_FOUND
fi

echo "Validating chunks in: $JSONL_FILE"

# Initialize counters
total_chunks=0
empty_text_count=0
mid_sentence_count=0
overlong_count=0
validation_failed=0

# Read JSONL file line by line for structural validation
while IFS= read -r line; do
    # Skip empty lines
    [[ -z "$line" ]] && continue
    
    total_chunks=$((total_chunks + 1))
    
    # Extract text field using python for reliable JSON parsing
    text=$(echo "$line" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    print(data.get('text', ''))
except:
    pass
" 2>/dev/null)
    
    # Check for empty text
    if [[ -z "$text" || "$text" =~ ^[[:space:]]*$ ]]; then
        empty_text_count=$((empty_text_count + 1))
        echo "Warning: Empty text in chunk on line $total_chunks" >&2
        validation_failed=1
    fi
    
    # Check for lines starting mid-sentence (no capital letter after whitespace)
    if [[ -n "$text" ]]; then
        # Remove leading whitespace and check first character
        first_char=$(echo "$text" | sed 's/^[[:space:]]*//' | cut -c1)
        if [[ "$first_char" =~ [a-z] ]]; then
            mid_sentence_count=$((mid_sentence_count + 1))
            echo "Warning: Chunk starts mid-sentence on line $total_chunks: '${text:0:50}...'" >&2
            validation_failed=1
        fi
    fi
    
    # Check for over-long chunks (>8000 characters suggests concatenation issues)
    if [[ ${#text} -gt 8000 ]]; then
        overlong_count=$((overlong_count + 1))
        echo "Warning: Overlong chunk (${#text} chars) on line $total_chunks" >&2
        validation_failed=1
    fi
    
done < "$JSONL_FILE"

# Fail fast if file contained no chunks
if [[ $total_chunks -eq 0 ]]; then
    echo "Error: No chunks found in $JSONL_FILE" >&2
    exit $EXIT_VALIDATION_FAILED
fi

echo "Structural validation complete:"
echo "  Total chunks: $total_chunks"
echo "  Empty text: $empty_text_count"
echo "  Mid-sentence starts: $mid_sentence_count" 
echo "  Overlong chunks: $overlong_count"

# Run duplicate detection
echo ""
echo "Running duplicate detection..."
if ! python3 scripts/detect_duplicates.py "$JSONL_FILE"; then
    echo "Error: Duplicate detection failed" >&2
    validation_failed=1
fi

# Final validation result
if [[ $validation_failed -eq 1 ]]; then
    echo ""
    echo "❌ Validation FAILED - anomalies detected in $JSONL_FILE" >&2
    exit $EXIT_VALIDATION_FAILED
else
    echo ""
    echo "✅ Validation PASSED - no structural anomalies detected in $JSONL_FILE"
    exit $EXIT_SUCCESS
fi
