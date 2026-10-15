"""Summarize customer reviews with the Gemini API.

Run:
    python summarize_reviews.py sample_reviews.txt
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from google import genai
from review_logic import build_prompt

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")



def summarize(reviews: str, api_key: str | None = None) -> str:
    """Call Gemini and return the model's text response."""
    clean_text = reviews.strip()
    if not clean_text:
        raise ValueError("The review text is empty.")
    if len(clean_text) > 12000:
        raise ValueError("For this demo, keep the input under 12,000 characters.")

    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Create a key in Google AI Studio and set it "
            "as an environment variable before running this script."
        )

    client = genai.Client(api_key=key)
    response = client.models.generate_content(
        model=MODEL,
        contents=build_prompt(clean_text),
    )
    answer = response.text
    if not answer:
        raise RuntimeError("Gemini returned an empty response. Try a shorter input.")
    return answer.strip()


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python summarize_reviews.py path_to_reviews.txt")
        return 2

    file_path = Path(sys.argv[1])
    if not file_path.is_file():
        print(f"Input file not found: {file_path}")
        return 2

    reviews = file_path.read_text(encoding="utf-8")
    try:
        print(summarize(reviews))
    except Exception as exc:  # friendly CLI error; do not print secrets
        print(f"Could not summarize reviews: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
