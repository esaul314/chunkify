import re
from haystack.schema import Document

def format_chunks_with_metadata(
    haystack_chunks: list[Document], 
    original_blocks: list[dict],
    generate_metadata: bool = True
) -> list[dict]:
    """
    Formats the final chunks and maps them back to their original source metadata.
    """
    char_map = []
    current_pos = 0
    for i, block in enumerate(original_blocks):
        # Text is already cleaned in the parsing stage, so we just calculate length
        text = block["text"]
        start_pos = current_pos
        end_pos = start_pos + len(text)
        char_map.append({"start": start_pos, "end": end_pos, "original_index": i})
        current_pos = end_pos + 2  # Account for '\n\n' separator

    final_chunks = []
    for chunk in haystack_chunks:
        # Text is already cleaned, so we just use the content directly
        final_text = chunk.content.strip()

        if not generate_metadata:
            if final_text:
                final_chunks.append({"text": final_text})
            continue

        chunk_start_char = chunk.meta.get("_split_start_char", 0)
        source_block = None
        for mapping in char_map:
            if mapping["start"] <= chunk_start_char < mapping["end"]:
                source_block = original_blocks[mapping["original_index"]]
                break
        
        metadata = {
            "source_file": source_block.get("source", {}).get("filename", "Unknown") if source_block else "Unknown",
            "source_page": source_block.get("source", {}).get("page") if source_block else None,
            "source_location": source_block.get("source", {}).get("location") if source_block else None,
            "block_type": source_block.get("type", "paragraph") if source_block else "paragraph"
        }

        if final_text:
            final_chunks.append({
                "text": final_text,
                "metadata": {k: v for k, v in metadata.items() if v is not None}
            })
        
    return final_chunks
