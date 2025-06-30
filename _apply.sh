#!/bin/bash

# _apply.sh
# This script runs the PDF chunking process on a sample PDF
# and reports its success or failure.

# Configuration
PDF_FILE="Lorem ipsum.pdf"
SCRIPT_PATH="scripts/chunk_pdf.py"
OUTPUT_FILE="output_chunks.json"
# Assuming the script is run from the project root /home/alex/work/AI/pdf_chunker/
PYTHON_INTERPRETER="./pdf-env/bin/python"

echo "Starting PDF chunking process..."

# Check if the PDF file exists
if [ ! -f "$PDF_FILE" ]; then
    echo "Error: Test PDF file '$PDF_FILE' not found in $(pwd)."
    exit 1
fi

# Check if the main script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Chunking script '$SCRIPT_PATH' not found in $(pwd)."
    exit 1
fi

# Check if the Python interpreter exists
if [ ! -x "$PYTHON_INTERPRETER" ]; then
    echo "Error: Python interpreter '$PYTHON_INTERPRETER' not found or not executable in $(pwd)."
    echo "Please ensure the virtual environment 'pdf-env' is set up correctly."
    # Try to fall back to system python3 if local venv python is not found
    if command -v python3 &> /dev/null; then
        echo "Falling back to system 'python3'."
        PYTHON_INTERPRETER="python3"
    else
        exit 1
    fi
fi

# Run the chunking script
echo "Running $PYTHON_INTERPRETER -m scripts.chunk_pdf on $PDF_FILE..."
"$PYTHON_INTERPRETER" -m scripts.chunk_pdf "$PDF_FILE" > "$OUTPUT_FILE"

# Check the exit code of the script
if [ $? -eq 0 ]; then
    echo "Success: PDF chunking completed. Output saved to $OUTPUT_FILE"
    if [ -s "$OUTPUT_FILE" ]; then # Check if file is not empty
        echo "--- First 10 lines of $OUTPUT_FILE ---"
        head -n 10 "$OUTPUT_FILE"
        echo "-------------------------------------"
    else
        echo "Warning: Output file $OUTPUT_FILE is empty."
    fi
    exit 0
else
    echo "Error: PDF chunking script failed with exit code $?."
    if [ -s "$OUTPUT_FILE" ]; then
        echo "--- Last 10 lines of $OUTPUT_FILE ---"
        tail -n 10 "$OUTPUT_FILE"
        echo "------------------------------------"
    else
        echo "Output file $OUTPUT_FILE is empty or was not created."
    fi
    exit 1
fi