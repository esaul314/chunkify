from toolz import pipe
from .parsing import extract_text
from .splitter import semantic_split
from .utils import clean_text, enrich_metadata

def chunk_pdf(filepath, chunk_size=1000, overlap=100): # Note: chunk_size default is now 700 due to previous change in scripts/chunk_pdf.py, but this function's signature can remain as is.
    return pipe(
        filepath,
        extract_text,
        clean_text,
        lambda text: semantic_split(text, chunk_size, overlap),
        enrich_metadata(filepath)
    )

