#!/bin/bash

# Common test utilities for PDF chunker test suite

# Find files by extension in current directory
find_files_by_extension() {
    local extension="$1"
    local max_files="${2:-10}"
    
    find . -name "*.${extension}" -type f | head -n "$max_files"
}

# Print test section header
print_test_header() {
    local test_name="$1"
    echo "Testing ${test_name}..."
}

# Print test result
print_test_result() {
    local test_name="$1"
    local status="$2"
    
    if [ "$status" = "success" ]; then
        echo "${test_name} testing completed successfully!"
    else
        echo "${test_name} testing completed with issues."
    fi
}

# Check if Python module can be imported
check_python_import() {
    local module="$1"
    python3 -c "import ${module}" 2>/dev/null
    return $?
}

# Run Python test with error handling
run_python_test() {
    local test_script="$1"
    python3 -c "
import sys
import os
sys.path.insert(0, '.')

${test_script}
"
}

# Count files found for testing
count_test_files() {
    local files="$1"
    echo "$files" | wc -l
}
