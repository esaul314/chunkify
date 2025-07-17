import argparse
import json
from pdf_chunker.core import process_document

def main():
    parser = argparse.ArgumentParser("Chunk a document into structured JSONL.")
    parser.add_argument("document_file", help="Path to the document file (PDF or EPUB)")
    parser.add_argument("--chunk_size", type=int, default=400)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument(
        "--exclude-pages",
        type=str,
        help="Page ranges to exclude from processing (e.g., '1,3,5-10,15-20'). For PDF files, excludes pages. For EPUB files, excludes spine indices."
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Set this flag to exclude metadata from the output."
    )
    args = parser.parse_args()

    # The flag is --no-metadata, so we pass the inverse to generate_metadata
    generate_metadata = not args.no_metadata

    chunks = process_document(
        args.document_file,
        args.chunk_size,
        args.overlap,
        generate_metadata=generate_metadata,
        exclude_pages=args.exclude_pages
    )
    
    # Filter out any None or empty chunks
    valid_chunks = [chunk for chunk in chunks if chunk]
    
    # Use a more robust approach for JSONL output
    for chunk in valid_chunks:
        try:
            # Ensure we have a clean, complete JSON object per line
            json_str = json.dumps(chunk, ensure_ascii=False)
            print(json_str)
        except Exception as e:
            # Skip problematic chunks to maintain JSONL integrity
            import sys
            print(f"Error serializing chunk: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
