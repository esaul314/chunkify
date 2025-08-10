# find_glued_words.py
import sys
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Dict, List, Optional, Set, Tuple, Iterator, Any
from funcy import compose, mapcat
from llm_correction import correct_word
import wordninja

# ─── Helpers ───────────────────────────────────────────────


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def extract_candidates(text: str) -> List[str]:
    return [
        m.group()
        for m in re.finditer(r"\b[a-z][^\W\d_]*\b", text)
        if text[max(0, m.start() - 2) : m.start()] != "\n\n"
    ]


def aspell_bad(words: Iterable[str], lang: str = "en_US") -> Set[str]:
    p = subprocess.Popen(
        ["aspell", "-l", lang, "list"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    out, _ = p.communicate("\n".join(words))
    return set(out.splitlines())


# ─── Improved Snippet Extraction ────────────────────────────


def snippet_for(word: str, text: str, window: int = 20) -> Optional[str]:
    pattern = re.compile(r"\b{}\b".format(re.escape(word)))
    match = pattern.search(text)
    if match:
        start, end = match.span()
        before = text[:start].split()[-window:]
        after = text[end:].split()[:window]
        return " ".join(before + [word] + after)
    return None


# ─── Semantic LLM Validation ────────────────────────────────


def semantic_llm_validate(original: str, split: str, snippet: str) -> str:
    prompt = (
        f"In the context '{snippet}', is '{split}' the correct form of '{original}'? "
        f"Reply only with the correct form: either '{split}' or '{original}'."
    )
    return correct_word(original, prompt).strip("'\" ")


# ─── Enhanced Correction Step ───────────────────────────────


@dataclass
class Correction:
    index: int
    word: str
    corrected: str


def enhanced_correction_step(
    words: Iterable[str], texts: List[str], workers: int = 5
) -> Dict[Tuple[int, str], str]:
    pairs = [
        (i, word, text)
        for i, text in enumerate(texts)
        for word in words
        if word in text
    ]

    def correct(args: Tuple[int, str, str]) -> Optional[Correction]:
        i, word, text = args
        snippet = snippet_for(word, text)
        if snippet:
            split_words = " ".join(wordninja.split(word))
            if split_words != word:
                validated = semantic_llm_validate(word, split_words, snippet)
                if validated != word:
                    print(f"[CORRECTED] Record {i}: '{word}' → '{validated}'")
                    return Correction(i, word, validated)
        return None

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results: Iterator[Correction] = filter(
            None, executor.map(correct, pairs)
        )  # type: ignore[arg-type]

    corrections = {(c.index, c.word): c.corrected for c in results}
    return corrections


# ─── Replacement Step ────────────────────────────────────────


def apply_corrections_to_text(text: str, word_map: Dict[str, str]) -> str:
    if not word_map:
        return text
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, word_map.keys())) + r")\b")
    return pattern.sub(lambda m: word_map[m.group()], text)


# ─── Main pipeline ──────────────────────────────────────────


def main(path: str, output_path: str, summary: bool = True) -> None:
    records = load_jsonl(path)
    texts = [record["text"] for record in records]

    candidates_pipeline = compose(
        aspell_bad, lambda texts: set(mapcat(extract_candidates, texts))
    )

    print("[INFO] Extracting suspicious candidates...")
    suspicious = candidates_pipeline(texts)
    print(f"[INFO] Suspicious candidates detected: {len(suspicious)}")

    print("[INFO] Starting enhanced correction step...")
    corrections = enhanced_correction_step(suspicious, texts)

    rec_word_corrections = defaultdict(dict)
    for (i, word), corrected in corrections.items():
        rec_word_corrections[i][word] = corrected

    cleaned_records = [
        {
            **record,
            "text": apply_corrections_to_text(
                record["text"], rec_word_corrections.get(i, {})
            ),
        }
        for i, record in enumerate(records)
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        for record in cleaned_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if summary:
        print("\n[SUMMARY] Detected and corrected glued words:")
        for (i, original), corrected in corrections.items():
            print(f"Record {i}: {original} → {corrected}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.stderr.write(f"Usage: {sys.argv[0]} <input.jsonl> <output.jsonl>\n")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
