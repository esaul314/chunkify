# Source block matching predicate functions.
# Each predicate returns True if ``block`` is considered a match for
# ``chunk_start``. They are composed in ``MATCHERS`` which can be extended
# for additional strategies.
from __future__ import annotations

from typing import Callable, List, Tuple

Matcher = Callable[[str, dict, list[dict]], bool]


def substring_match(chunk_start: str, block: dict, _blocks: list[dict]) -> bool:
    return bool(chunk_start) and chunk_start in block.get("text", "")


def start_match(chunk_start: str, block: dict, _blocks: list[dict]) -> bool:
    block_text = block.get("text", "").strip()
    if not block_text:
        return False
    block_start = block_text[: max(20, len(chunk_start))].replace("\n", " ").strip()
    lower_chunk = chunk_start.lower()
    lower_block = block_start.lower()
    return lower_chunk.startswith(lower_block) or lower_block.startswith(lower_chunk)


def fuzzy_match(chunk_start: str, block: dict, _blocks: list[dict]) -> bool:
    import re

    def normalize(s: str) -> str:
        return re.sub(r"[\W_]+", "", s).lower()

    block_start = block.get("text", "")[: max(20, len(chunk_start))]
    return normalize(block_start).startswith(normalize(chunk_start)[:15])


def overlap_match(chunk_start: str, block: dict, _blocks: list[dict]) -> bool:
    block_text = block.get("text", "")
    return any(chunk_start[:n] in block_text for n in range(30, 10, -5))


def header_match(chunk_start: str, block: dict, blocks: list[dict]) -> bool:
    return (
        block is blocks[0]
        and bool(chunk_start)
        and (
            chunk_start.isupper()
            or chunk_start.startswith("CHAPTER")
            or chunk_start.startswith("SECTION")
        )
    )


MATCHERS: List[Tuple[str, Matcher]] = [
    ("substring match", substring_match),
    ("start match", start_match),
    ("fuzzy match", fuzzy_match),
    ("overlap match", overlap_match),
    ("header/special formatting", header_match),
]

__all__ = [
    "Matcher",
    "MATCHERS",
    "substring_match",
    "start_match",
    "fuzzy_match",
    "overlap_match",
    "header_match",
]
