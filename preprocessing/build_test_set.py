import re

INPUT_FILE = "data/raw/source_text.txt"     # paste your raw Hindi paragraphs here first
OUTPUT_FILE = "data/raw/test_sentences_devanagari.txt"


def split_into_sentences(text: str):
    # Split on Devanagari danda (।), ?, !
    sentences = re.split(r'(?<=[।?!])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


if __name__ == "__main__":
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw = f.read()

    sentences = split_into_sentences(raw)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for s in sentences:
            f.write(s + "\n")

    print(f"Wrote {len(sentences)} sentences to {OUTPUT_FILE}")