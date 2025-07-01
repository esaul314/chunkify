import argparse
import json
from pdf_chunker.core import process_document

def main():
    parser = argparse.ArgumentParser("Chunk a document into structured JSONL.")
    parser.add_argument("document_file", help="Path to the document file (PDF or EPUB)")
    parser.add_argument("--chunk_size", type=int, default=400)
    parser.add_argument("--overlap", type=int, default=50)
    args = parser.parse_args()

    chunks = process_document(args.document_file, args.chunk_size, args.overlap)
    for chunk in chunks:
        print(json.dumps(chunk, ensure_ascii=False))

if __name__ == "__main__":
    main()

