import os
import sys
import json
import litellm
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuration ---
# This is now configured via an explicit init function
UTTERANCE_TYPES = [
    "definition", "explanation", "instruction",
    "example", "opinion", "statement_of_fact",
    "question", "summary", "critique", "unclassified"
]

CLASSIFICATION_PROMPT = f"""
Given the following text, classify its primary utterance type.
Choose the best fit from this list: {UTTERANCE_TYPES}.
Respond with only the chosen type in lowercase.

Text: "{{text_chunk}}"

Utterance Type:
"""

def init_llm():
    """
    Initializes the LLM provider by loading the API key from a .env file.
    """
    load_dotenv() # Load environment variables from .env file
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file or environment.")
    litellm.api_key = api_key

def classify_chunk_utterance(text_chunk: str) -> str:
    """
    Classifies a single text chunk's utterance type using an LLM.
    Returns the classification as a string.
    """
    if not text_chunk or not text_chunk.strip():
        return "unclassified"

    prompt = CLASSIFICATION_PROMPT.format(text_chunk=text_chunk)

    try:
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        classification = response.choices[0].message.content.strip().lower()
        
        return classification if classification in UTTERANCE_TYPES else "unclassified"
        
    except Exception as e:
        print(f"LLM API call failed for a chunk: {e}", file=sys.stderr)
        return "error"

# --- Main execution logic for standalone script ---

def _process_chunk_for_file(chunk: dict) -> dict:
    """Helper function to wrap utterance classification for file processing."""
    classification = classify_chunk_utterance(chunk.get("text", ""))
    if "metadata" not in chunk:
        chunk["metadata"] = {}
    chunk["metadata"]["utterance_type"] = classification
    return chunk

def _process_jsonl_file(input_path: str, output_path: str, max_workers: int = 10):
    """
    Reads a JSONL file, classifies each chunk in parallel, and writes to a new file.
    """
    init_llm() # Ensure LLM is configured when running as script
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        lines = infile.readlines()
        chunks = [json.loads(line) for line in lines]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {executor.submit(_process_chunk_for_file, chunk): chunk for chunk in chunks}
            
            results = [future.result() for future in as_completed(future_to_chunk)]

        for result_chunk in results:
            outfile.write(json.dumps(result_chunk) + '\n')

def main():
    """
    Main function to run the AI enrichment script from the command line.
    """
    if len(sys.argv) != 3:
        print("Usage: python -m pdf_chunker.ai_enrichment <input_file.jsonl> <output_file.jsonl>", file=sys.stderr)
        sys.exit(1)

    input_file, output_file = sys.argv[1], sys.argv[2]
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at '{input_file}'", file=sys.stderr)
        sys.exit(1)
        
    print(f"Starting AI enrichment for '{input_file}'...")
    _process_jsonl_file(input_file, output_file)
    print(f"Enrichment complete. Output saved to '{output_file}'.")

if __name__ == "__main__":
    main()
