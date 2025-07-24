import argparse
import json
import logging
from pdf_chunker.core import process_document

# Set up logging to see debug messages
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.debug("Starting chunk_pdf script execution")
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
    parser.add_argument(
        "--list-spines",
        action="store_true",
        help="List EPUB spine items with their indices and filenames (EPUB files only)."
    )
    args = parser.parse_args()
    
    logger.debug(f"Processing document: {args.document_file}")
    logger.debug(f"Arguments: chunk_size={args.chunk_size}, overlap={args.overlap}, no_metadata={args.no_metadata}")
    # Handle spine listing for EPUB files
    if args.list_spines:
        if not args.document_file.lower().endswith('.epub'):
            print("Error: --list-spines can only be used with EPUB files.")
            return 1
        
        try:
            from pdf_chunker.epub_parsing import list_epub_spines
            spine_items = list_epub_spines(args.document_file)
            
            print(f"EPUB Spine Structure ({len(spine_items)} items):")
            for item in spine_items:
                print(f"{item['index']:3d}. {item['filename']} - {item['content_preview']}")
            
            return 0
        except Exception as e:
            print(f"Error listing spine items: {e}")
            return 1

    # The flag is --no-metadata, so we pass the inverse to generate_metadata
    generate_metadata = not args.no_metadata
    
    logger.debug(f"Calling process_document with generate_metadata={generate_metadata}")

    chunks = process_document(
        args.document_file,
        args.chunk_size,
        args.overlap,
        generate_metadata=generate_metadata,
        exclude_pages=args.exclude_pages
    )
    
    logger.debug(f"process_document returned {len(chunks)} chunks")
    
    # Filter out any None or empty chunks
    valid_chunks = [chunk for chunk in chunks if chunk]
    
    logger.debug(f"After filtering, have {len(valid_chunks)} valid chunks")
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
