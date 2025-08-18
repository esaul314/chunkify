import json
import os
from pathlib import Path
from typing import Callable

from concurrent.futures import ThreadPoolExecutor
from functools import reduce

import yaml

from pdf_chunker.passes.ai_enrich import classify_chunk_utterance


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
    from dotenv import load_dotenv  # lazy import

    load_dotenv()
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY not found in .env file or environment.")
    try:
        import litellm  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("litellm is required for init_llm") from exc

    def completion(prompt: str) -> str:
        response = litellm.completion(  # type: ignore[union-attr]
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100,
            api_key=key,
        )
        return response.choices[0].message.content

    return completion


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
    chunk = {
        **chunk,
        "utterance_type": result["classification"],
        "tags": result["tags"],
    }
    meta = {**chunk.get("metadata", {}), "utterance_type": result["classification"], "tags": result["tags"]}
    return {**chunk, "metadata": meta}


def _process_jsonl_file(
    input_path: str,
    output_path: str,
    completion_fn: Callable[[str], str],
    tag_configs: dict | None = None,
    max_workers: int = 10,
) -> None:
    """Read ``input_path`` JSONL, classify chunks, and write to ``output_path``."""
    tag_configs = tag_configs or _load_tag_configs()
    with (
        open(input_path, "r", encoding="utf-8") as infile,
        open(output_path, "w", encoding="utf-8") as outfile,
    ):
        chunks = [json.loads(line) for line in infile]
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(
                ex.map(
                    lambda c: _process_chunk_for_file(
                        c, tag_configs=tag_configs, completion_fn=completion_fn
                    ),
                    chunks,
                )
            )
        for chunk in results:
            outfile.write(json.dumps(chunk) + "\n")
