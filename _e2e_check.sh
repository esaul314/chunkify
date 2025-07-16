#!/bin/bash

# _apply.sh
# This script runs the document chunking process on a sample file
# and reports its success or failure.
#
# Usage:
#   ./_apply.sh         # Tests the PDF file by default
#   ./_apply.sh pdf     # Explicitly tests the PDF file
#   ./_apply.sh epub    # Explicitly tests the EPUB file

# --- Configuration ---
PDF_FILE="sample_book.pdf"
EPUB_FILE="accessible_epub_3.epub"
SCRIPT_MODULE="scripts.chunk_pdf"
PYTHON_INTERPRETER="./pdf-env/bin/python"

# --- Argument Parsing ---
TEST_TARGET="${1:-pdf}" # Default to 'pdf' if no argument is provided
INPUT_FILE=""
OUTPUT_FILE=""

if [ "$TEST_TARGET" == "pdf" ]; then
    INPUT_FILE="$PDF_FILE"
    OUTPUT_FILE="output_chunks_pdf.jsonl"
elif [ "$TEST_TARGET" == "epub" ]; then
    INPUT_FILE="$EPUB_FILE"
    OUTPUT_FILE="output_chunks_epub.jsonl"
else
    echo "Error: Unknown test target '$TEST_TARGET'. Use 'pdf' or 'epub'."
    exit 1
fi

echo "--- Starting chunking process for $TEST_TARGET file: '$INPUT_FILE' ---"

# --- Pre-flight Checks ---
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Test file '$INPUT_FILE' not found in $(pwd)."
    exit 1
fi

if [ ! -x "$PYTHON_INTERPRETER" ]; then
    echo "Error: Python interpreter '$PYTHON_INTERPRETER' not found or not executable."
    exit 1
fi

# --- Execution ---
# Ensure we're overwriting any existing file by truncating it before we start
echo "Running: $PYTHON_INTERPRETER -m $SCRIPT_MODULE \"$INPUT_FILE\""
# Clear output file first to avoid appending to old results
> "$OUTPUT_FILE"
"$PYTHON_INTERPRETER" -m "$SCRIPT_MODULE" "$INPUT_FILE" > "$OUTPUT_FILE"
EXIT_CODE=$?

# --- Reporting ---
if [ $EXIT_CODE -eq 0 ]; then
    echo "Success: Chunking completed. Output saved to '$OUTPUT_FILE'."
    if [ -s "$OUTPUT_FILE" ]; then
        echo "--- First 10 lines of '$OUTPUT_FILE' ---"
        head -n 10 "$OUTPUT_FILE"
        echo "-------------------------------------------"
    else
        echo "Warning: Output file '$OUTPUT_FILE' is empty."
    fi
    exit 0
else
    echo "Error: Chunking script failed with exit code $EXIT_CODE."
    if [ -s "$OUTPUT_FILE" ]; then
        echo "--- Last 10 lines of '$OUTPUT_FILE' ---"
        tail -n 10 "$OUTPUT_FILE"
        echo "------------------------------------------"
    else
        echo "Output file '$OUTPUT_FILE' is empty or was not created."
    fi
    exit 1
fi
