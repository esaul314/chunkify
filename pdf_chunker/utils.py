from pathlib import Path

def clean_text(text):
    # Add any cleaning logic needed
    return text.replace("\x0c", "").strip()

def enrich_metadata(filepath):
    def wrapper(chunks):
        base_meta = {"source": Path(filepath).name}
        return [
            {"text": chunk.content}
            #{"text": chunk.content, "metadata": {**base_meta, **chunk.meta}}
            for chunk in chunks
        ]
    return wrapper

