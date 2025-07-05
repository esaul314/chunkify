import argparse
import json
from pdf_chunker.core import process_document

def main():
    parser = argparse.ArgumentParser("Chunk a document into structured JSONL.")
    parser.add_argument("document_file", help="Path to the document file (PDF or EPUB)")
    parser.add_argument("--chunk_size", type=int, default=400)
    parser.add_argument("--overlap", type=int, default=50)
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
        generate_metadata=generate_metadata
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

