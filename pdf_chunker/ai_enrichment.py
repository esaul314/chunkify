import os
import sys
import json
import yaml
from pathlib import Path
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

def _load_tag_configs(config_dir: str = "config/tags") -> dict:
    """
    Load and merge tag configurations from YAML files.
    Returns a dictionary with all available tags organized by category.
    """
    tag_configs = {}
    config_path = Path(config_dir)
    
    if not config_path.exists():
        print(f"Warning: Tag config directory '{config_dir}' not found", file=sys.stderr)
        return tag_configs
    
    # Load all YAML files in the config directory
    for yaml_file in config_path.glob("*.yaml"):
        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    # Merge the configuration, preserving existing categories
                    for category, tags in file_config.items():
                        if category in tag_configs:
                            # Extend existing category with new tags
                            tag_configs[category].extend(tags)
                        else:
                            # Create new category
                            tag_configs[category] = tags.copy()
        except Exception as e:
            print(f"Warning: Failed to load tag config from '{yaml_file}': {e}", file=sys.stderr)
    
    # Remove duplicates from each category while preserving order
    for category in tag_configs:
        seen = set()
        tag_configs[category] = [tag for tag in tag_configs[category] 
                                if not (tag in seen or seen.add(tag))]
    
    return tag_configs


    
def init_llm():
    """
    Initializes the LLM provider by loading the API key from a .env file.
    """
    load_dotenv() # Load environment variables from .env file
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file or environment.")
    litellm.api_key = api_key


def classify_chunk_utterance(text_chunk: str, tag_configs: dict = None) -> dict:
    """
    Classifies a single text chunk's utterance type and assigns relevant tags using an LLM.
    Returns a dictionary with classification and tags.
    """
    if not text_chunk or not text_chunk.strip():
        return {"classification": "unclassified", "tags": []}

    # Load tag configs if not provided
    if tag_configs is None:
        tag_configs = _load_tag_configs()

    # Build the available tags section for the prompt
    available_tags_text = ""
    if tag_configs:
        available_tags_text = "\n\nAvailable tags by category:\n"
        for category, tags in tag_configs.items():
            available_tags_text += f"- {category}: {', '.join(tags)}\n"
        available_tags_text += "\nSelect 2-4 most relevant tags from the available categories."

    prompt = f"""Given the following text, classify its primary utterance type and assign relevant tags.

Classification: Choose the best fit from this list: {UTTERANCE_TYPES}.

{available_tags_text}

Respond in this exact format:
Classification: [chosen_type]
Tags: [tag1, tag2, tag3]

Text: "{text_chunk}"

Response:"""

    try:
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100
        )
        response_text = response.choices[0].message.content.strip()
        
        # Parse the structured response
        classification = "unclassified"
        tags = []
        
        for line in response_text.split('\n'):
            line = line.strip()
            if line.startswith('Classification:'):
                classification = line.split(':', 1)[1].strip().lower()
                if classification not in UTTERANCE_TYPES:
                    classification = "unclassified"
            elif line.startswith('Tags:'):
                tags_text = line.split(':', 1)[1].strip()
                if tags_text and tags_text != '[]':
                    # Parse tags and validate against available tags
                    raw_tags = [tag.strip() for tag in tags_text.replace('[', '').replace(']', '').split(',')]
                    # Validate tags against available vocabulary
                    all_valid_tags = set()
                    for category_tags in tag_configs.values():
                        all_valid_tags.update(category_tags)
                    tags = [tag for tag in raw_tags if tag in all_valid_tags]
        
        return {"classification": classification, "tags": tags}
        
    except Exception as e:
        print(f"LLM API call failed for a chunk: {e}", file=sys.stderr)
        return {"classification": "error", "tags": []}
    
# --- Main execution logic for standalone script ---


def _process_chunk_for_file(chunk: dict, tag_configs: dict = None) -> dict:
    """Helper function to wrap utterance classification and tagging for file processing."""
    result = classify_chunk_utterance(chunk.get("text", ""), tag_configs)
    if "metadata" not in chunk:
        chunk["metadata"] = {}
    chunk["metadata"]["utterance_type"] = result["classification"]
    chunk["metadata"]["tags"] = result["tags"]
    return chunk
    

def _process_jsonl_file(input_path: str, output_path: str, max_workers: int = 10):
    """
    Reads a JSONL file, classifies each chunk in parallel, and writes to a new file.
    """

    init_llm() # Ensure LLM is configured when running as script
    # Load tag configurations once for all chunks
    tag_configs = _load_tag_configs()
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        lines = infile.readlines()
        chunks = [json.loads(line) for line in lines]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {executor.submit(_process_chunk_for_file, chunk, tag_configs): chunk for chunk in chunks}
            
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
