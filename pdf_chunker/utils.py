import re

def clean_structured_text(structured_blocks: list[dict]) -> list[dict]:
    """
    Cleans the 'text' field of each dictionary in a list of structured blocks.
    """
    cleaned_blocks = []
    for block in structured_blocks:
        text = block.get("text", "")
        # Normalize whitespace and remove control characters
        cleaned_text = re.sub(r'\s+', ' ', text).strip()
        
        if cleaned_text:
            new_block = block.copy()
            new_block["text"] = cleaned_text
            cleaned_blocks.append(new_block)
            
    return cleaned_blocks
