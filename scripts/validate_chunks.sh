#!/bin/bash

# Validate chunks script
# Performs structural validation on JSONL chunk files and detects duplicates

set -euo pipefail

usage() {
    echo "Usage: $0 [-i <jsonl_file>] [-d <document_file>]" >&2
}

DEFAULT_JSONL_FILE="output_chunks_pdf.jsonl"
DEFAULT_DOCUMENT_FILE="sample-local-pdf.pdf"
JSONL_FILE=""
DOCUMENT_FILE=""

while getopts ":i:d:h" opt; do
    case "$opt" in
        i) JSONL_FILE="$OPTARG" ;;
        d) DOCUMENT_FILE="$OPTARG" ;;
        h) usage; exit 0 ;;
        *) usage; exit 1 ;;
    esac
done
shift $((OPTIND-1))

if [[ -z "$JSONL_FILE" && $# -gt 0 ]]; then
    JSONL_FILE="$1"
fi
if [[ -z "$DOCUMENT_FILE" && $# -gt 1 ]]; then
    DOCUMENT_FILE="$2"
fi

JSONL_FILE="${JSONL_FILE:-$DEFAULT_JSONL_FILE}"
DOCUMENT_FILE="${DOCUMENT_FILE:-$DEFAULT_DOCUMENT_FILE}"

ensure_haystack() {
    python3 - <<'PY' >/dev/null 2>&1
import importlib, sys
try:
    importlib.import_module("haystack")
except ModuleNotFoundError:
    sys.exit(1)
PY
    if [[ $? -ne 0 ]]; then
        echo "Installing haystack dependencies..." >&2
        python3 -m pip install --quiet haystack-ai==2.15.1 haystack-experimental==0.10.0
    fi
}

# Ensure directory exists for the provided path
mkdir -p "$(dirname "$JSONL_FILE")"

generate_jsonl() {
    local src="$1"
    local dest="$2"
    ensure_haystack
    PYTHONPATH=. python3 scripts/chunk_pdf.py "$src" > "$dest"
}

# Exit codes
EXIT_SUCCESS=0
EXIT_VALIDATION_FAILED=1
EXIT_FILE_NOT_FOUND=2

# Generate JSONL file if missing or empty
if [[ ! -s "$JSONL_FILE" ]]; then
    if [[ -f "$DOCUMENT_FILE" ]]; then
        echo "Chunk file '$JSONL_FILE' not found or empty. Generating from '$DOCUMENT_FILE'..."
        generate_jsonl "$DOCUMENT_FILE" "$JSONL_FILE"
    else
        echo "Error: File '$JSONL_FILE' not found and document '$DOCUMENT_FILE' unavailable" >&2
        exit $EXIT_FILE_NOT_FOUND
    fi
fi

# Fail fast if the file contains no chunks
if ! grep -q '[^[:space:]]' "$JSONL_FILE"; then
    echo "Error: No chunks found in '$JSONL_FILE'" >&2
    exit $EXIT_VALIDATION_FAILED
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
