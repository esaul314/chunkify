# find_glued_words.py
import sys
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from funcy import compose, cat, mapcat
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
    if word in tokens:
        idx = tokens.index(word)
        start, end = max(0, idx-window), min(len(tokens), idx+window)
        return " ".join(tokens[start:end])
    return None

def ai_correction_step(words, texts, workers=5):
    def correct(pair):
        word, text = pair
        snippet = snippet_for(word, text)
        if snippet:
            corrected = correct_word(word, snippet)
            return (word, corrected) if corrected != word else None
        return None

    word_text_pairs = [(word, text) for text in texts for word in words if word in text]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        corrections = filter(None, executor.map(correct, word_text_pairs))

    return dict(corrections)

# ─── Replacement Step ────────────────────────────────────────

def apply_corrections_to_text(text, corrections):
    pattern = re.compile(r'\\b(' + '|'.join(map(re.escape, corrections.keys())) + r')\\b')
    return pattern.sub(lambda m: corrections[m.group()], text)

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
    corrections = ai_correction_step(suspicious, texts)

    cleaned_records = [
        {**record, "text": apply_corrections_to_text(record["text"], corrections)}
        for record in records
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        for record in cleaned_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if summary:
        print("Detected and corrected glued words:")
        for original, corrected in corrections.items():
            print(f"{original} → {corrected}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.stderr.write(f"Usage: {sys.argv[0]} <input.jsonl> <output.jsonl>\n")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])

