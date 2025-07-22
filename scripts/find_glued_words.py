# find_glued_words.py
import sys
import json
import re
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from funcy import compose, cat, mapcat, group_by
import enchant
from nltk.corpus import wordnet
from wordfreq import zipf_frequency
from llm_correction import correct_word

# ─── Helpers ───────────────────────────────────────────────

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def extract_candidates(text):
    return [
        m.group()
        for m in re.finditer(r"\b[^\W\d_]+\b", text)
        if text[max(0, m.start()-2):m.start()] != "\n\n"
    ]

def aspell_bad(words, lang="en_US"):
    p = subprocess.Popen(
        ["aspell", "-l", lang, "list"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    out, _ = p.communicate("\n".join(words))
    return set(out.splitlines())

def enchant_bad(words, lang="en_US"):
    d = enchant.Dict(lang)
    return {w for w in words if not d.check(w)}

def wordnet_bad(words):
    return {w for w in words if not wordnet.synsets(w)}

def freq_bad(words, cutoff=3.0):
    return {w for w in words if zipf_frequency(w, "en") < cutoff}

# ─── LLM Verification Step ──────────────────────────────────

def snippet_for(word, text, window=20):
    tokens = text.split()
    try:
        idx = tokens.index(word)
    except ValueError:
        return None
    start, end = max(0, idx-window), min(len(tokens), idx+window)
    return " ".join(tokens[start:end])

def ai_correction_step(words, texts, workers=5, log=print):
    """
    For each (word, text) pair, call LLM to check for glue errors.
    Returns {(record_idx, word): correction}.
    """
    pairs = [ (i, word, text)
              for i, text in enumerate(texts)
              for word in words if word in text.split() ]

    def correct(args):
        i, word, text = args
        snippet = snippet_for(word, text)
        if snippet:
            log(f"[LLM] Checking word '{word}' in record {i} (context: '{snippet[:60]}...')")  # Truncated
            corrected = correct_word(word, snippet)
            if corrected != word:
                log(f"[LLM] Correction: '{word}' → '{corrected}' (record {i})")
                return (i, word, corrected, snippet)
        return None

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = filter(None, executor.map(correct, pairs))

    # {(record_idx, word): corrected}
    corrections = {(i, word): (corrected, snippet) for i, word, corrected, snippet in results}
    return corrections

# ─── Replacement Step ────────────────────────────────────────

def apply_corrections_to_text(text, word_map):
    # word_map: {word: corrected}
    if not word_map:
        return text
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, word_map.keys())) + r')\b')
    return pattern.sub(lambda m: word_map[m.group()], text)

# ─── Logging ─────────────────────────────────────────────

def log_detected_suspicious(suspicious):
    print("[INFO] Suspicious glued words detected by pipeline:")
    for w in suspicious:
        print("  -", w)
    print()

def log_record_changes(record_idx, changed_words, corrections):
    for w in changed_words:
        corrected, snippet = corrections[(record_idx, w)]
        print(f"[CHANGE] Record {record_idx}: '{w}' → '{corrected}' | context: ...{snippet}...")

# ─── Main pipeline ──────────────────────────────────────────

def main(path, output_path, summary=True):
    records = load_jsonl(path)
    texts = [record["text"] for record in records]

    pipeline = compose(
        freq_bad,
        wordnet_bad,
        enchant_bad,
        aspell_bad,
        lambda texts: set(mapcat(extract_candidates, texts))
    )

    suspicious = pipeline(texts)
    log_detected_suspicious(suspicious)

    corrections = ai_correction_step(suspicious, texts, log=print)

    # Build per-record word correction map
    rec_word_corrections = defaultdict(dict)
    for (i, word), (corrected, _) in corrections.items():
        rec_word_corrections[i][word] = corrected

    cleaned_records = []
    total_changes = 0
    changed_records = []

    for i, record in enumerate(records):
        word_map = rec_word_corrections.get(i, {})
        if word_map:
            log_record_changes(i, word_map, corrections)
            changed_records.append(i)
        cleaned_text = apply_corrections_to_text(record["text"], word_map)
        total_changes += len(word_map)
        cleaned_records.append({**record, "text": cleaned_text})

    with open(output_path, "w", encoding="utf-8") as f:
        for record in cleaned_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if summary:
        print(f"\n[SUMMARY] {total_changes} glued words corrected in {len(changed_records)} records.")
        if changed_records:
            print(f"[SUMMARY] Changed records: {', '.join(map(str, changed_records))}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.stderr.write(f"Usage: {sys.argv[0]} <input.jsonl> <output.jsonl>\n")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])


