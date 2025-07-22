# llm_correction.py
import os
from dotenv import load_dotenv
import litellm

load_dotenv()
MODEL = "gpt-3.5-turbo"
litellm.api_key = os.getenv("OPENAI_API_KEY")

def correct_word(word, snippet):
    prompt = (
        f"The following snippet contains the possibly erroneous word '{word}':\n\n"
        f"\"{snippet}\"\n\n"
        "If the word results from words glued together accidentally, correct it. "
        "Otherwise, return the original word unchanged.\n\n"
        "Reply ONLY with the corrected or original word."
    )

    response = litellm.completion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=100
    )
    return response.choices[0].message.content.strip()

