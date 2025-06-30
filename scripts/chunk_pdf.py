import argparse
import json
from pdf_chunker.core import chunk_pdf

def main():
    parser = argparse.ArgumentParser("Chunk PDF into structured JSON.")
    parser.add_argument("pdf_file", help="Path to PDF file")
    parser.add_argument("--chunk_size", type=int, default=700)
    parser.add_argument("--overlap", type=int, default=100)
    args = parser.parse_args()

    chunks = chunk_pdf(args.pdf_file, args.chunk_size, args.overlap)
    for chunk in chunks:
        print(json.dumps(chunk, ensure_ascii=False))

if __name__ == "__main__":
    main()

