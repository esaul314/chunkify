import json
import os
import sys
from pathlib import Path
from functools import reduce
from typing import Callable

import litellm
import yaml
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuration ---
# This is now configured via an explicit init function
UTTERANCE_TYPES = [
    "definition",
    "explanation",
    "instruction",
    "example",
    "opinion",
    "statement_of_fact",
    "question",
    "summary",
    "critique",
    "unclassified",
]


def _load_tag_configs(config_dir: str = "config/tags") -> dict:
    """Merge YAML tag configurations into a single dictionary."""
    config_path = Path(config_dir)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent.parent / config_path
    if not config_path.exists():
        return {}

    def load_yaml(path: Path) -> dict:
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
                return {k: v for k, v in data.items() if isinstance(v, list)}
        except FileNotFoundError:
            return {}

    def merge_dicts(acc: dict, nxt: dict) -> dict:
        return {key: acc.get(key, []) + nxt.get(key, []) for key in set(acc) | set(nxt)}

    merged = reduce(
        merge_dicts,
        map(load_yaml, config_path.glob("*.yaml")),
        {},
    )

    return {
        k: sorted({tag.strip().lower() for tag in v if isinstance(tag, str)})
        for k, v in merged.items()
    }


def init_llm(api_key: str | None = None) -> Callable[[str], str]:
    """Return a completion function configured with an API key."""
    load_dotenv()
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY not found in .env file or environment.")

    def completion(prompt: str) -> str:
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100,
            api_key=key,
        )
        return response.choices[0].message.content

    return completion


def classify_chunk_utterance(
    text_chunk: str,
    *,
    tag_configs: dict,
    completion_fn: Callable[[str], str],
) -> dict:
    """Classify ``text_chunk`` and assign valid tags using ``completion_fn``."""
    if not text_chunk or not text_chunk.strip():
        return {"classification": "unclassified", "tags": []}

    available_tags_text = ""
    if tag_configs:
        available_tags_text = "\n\nAvailable tags by category:\n"
        for category, tags in tag_configs.items():
            available_tags_text += f"- {category}: {', '.join(tags)}\n"
        available_tags_text += (
            "\nSelect 2-4 most relevant tags from the available categories."
        )

    prompt = f"""Given the following text, classify its primary utterance type and assign relevant tags.

Classification: Choose the best fit from this list: {UTTERANCE_TYPES}.

{available_tags_text}

Respond in this exact format:
Classification: [chosen_type]
Tags: [tag1, tag2, tag3]

Text: "{text_chunk}"

Response:"""

    try:
        response_text = completion_fn(prompt).strip()
        classification = "unclassified"
        tags = []

        for line in response_text.split("\n"):
            line = line.strip()
            if line.startswith("Classification:"):
                classification = line.split(":", 1)[1].strip().lower()
                if classification not in UTTERANCE_TYPES:
                    classification = "unclassified"
            elif line.startswith("Tags:"):
                tags_text = line.split(":", 1)[1].strip()
                raw_tags = [
                    tag.strip().lower()
                    for tag in tags_text.replace("[", "").replace("]", "").split(",")
                    if tag.strip()
                ]
                valid = {t.lower() for tags in tag_configs.values() for t in tags}
                tags = [tag for tag in raw_tags if tag in valid]
        return {"classification": classification, "tags": tags}
    except Exception:
        return {"classification": "error", "tags": []}


# --- Main execution logic for standalone script ---


def _process_chunk_for_file(
    chunk: dict,
    *,
    tag_configs: dict,
    completion_fn: Callable[[str], str],
) -> dict:
    """Helper to wrap utterance classification and tagging for file processing."""
    result = classify_chunk_utterance(
        chunk.get("text", ""), tag_configs=tag_configs, completion_fn=completion_fn
    )
    chunk["utterance_type"] = result["classification"]
    chunk["tags"] = result["tags"]
    if "metadata" not in chunk:
        chunk["metadata"] = {}
    chunk["metadata"]["utterance_type"] = result["classification"]
    chunk["metadata"]["tags"] = result["tags"]
    return chunk


def _process_jsonl_file(
    input_path: str,
    output_path: str,
    completion_fn: Callable[[str], str],
    tag_configs: dict | None = None,
    max_workers: int = 10,
):
    """Read ``input_path`` JSONL, classify chunks, and write to ``output_path``."""
    tag_configs = tag_configs or _load_tag_configs()
    with open(input_path, "r", encoding="utf-8") as infile, open(
        output_path, "w", encoding="utf-8"
    ) as outfile:
        lines = infile.readlines()
        chunks = [json.loads(line) for line in lines]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _process_chunk_for_file,
                    chunk,
                    tag_configs=tag_configs,
                    completion_fn=completion_fn,
                ): idx
                for idx, chunk in enumerate(chunks)
            }
            results = [None] * len(chunks)
            for future in as_completed(futures):
                results[futures[future]] = future.result()

        for result_chunk in results:
            outfile.write(json.dumps(result_chunk) + "\n")


def main() -> None:
    """Run the AI enrichment script from the command line."""
    if len(sys.argv) != 3:
        print(
            "Usage: python -m pdf_chunker.ai_enrichment <input_file.jsonl> <output_file.jsonl>",
            file=sys.stderr,
        )
        sys.exit(1)

    input_file, output_file = sys.argv[1], sys.argv[2]
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at '{input_file}'", file=sys.stderr)
        sys.exit(1)

    print(f"Starting AI enrichment for '{input_file}'...")
    completion_fn = init_llm()
    _process_jsonl_file(input_file, output_file, completion_fn)
    print(f"Enrichment complete. Output saved to '{output_file}'.")


if __name__ == "__main__":
    main()
