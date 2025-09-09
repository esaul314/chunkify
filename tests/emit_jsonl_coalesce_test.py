from pdf_chunker.passes.emit_jsonl import _rows


def test_leading_fragment_not_dropped():
    doc = {
        "type": "chunks",
        "items": [
            {"text": "fragment without punctuation"},
            {"text": "continues the thought without end"},
            {"text": "Final sentence."},
        ],
    }
    rows = _rows(doc)
    assert rows and rows[0]["text"].startswith("fragment without punctuation")


def test_duplicate_sentence_trimmed():
    doc = {
        "type": "chunks",
        "items": [
            {
                "text": (
                    "the developers didn't want to use the interface. "
                    "Developers are picky about the way they work."
                )
            },
            {
                "text": (
                    "Developers are picky about the way they work. "
                    "Many want nothing to do with the interface."
                )
            },
        ],
    }
    rows = _rows(doc)
    assert sum("Developers are picky" in r["text"] for r in rows) == 1


def test_non_adjacent_duplicate_sentence_trimmed():
    doc = {
        "type": "chunks",
        "items": [
            {"text": "Intro."},
            {
                "text": (
                    "Most engineers don't want to learn a whole new toolset for infrequent tasks."
                )
            },
            {"text": "Filler paragraph in between."},
            {
                "text": (
                    "Most engineers don't want to learn a whole new toolset for infrequent tasks."
                    " Another sentence follows."
                )
            },
        ],
    }
    rows = _rows(doc)
    assert sum("Most engineers don't want to learn" in r["text"] for r in rows) == 1
