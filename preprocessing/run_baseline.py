from preprocessing.transliterate import HindiTransliterator

INPUT_FILE = "data/raw/test_sentences_devanagari.txt"
OUTPUT_FILE = "data/processed/test_sentences_romanized.txt"


def main():
    t = HindiTransliterator()

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    results = []
    failures = 0
    for line in lines:
        try:
            roman = t.transliterate_sentence(line)
            results.append(roman)
        except Exception as e:
            print(f"FAILED on: {line[:50]}... -> {e}")
            results.append("")
            failures += 1

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(r + "\n")

    print(f"Processed {len(lines)} sentences.")
    print(f"Failures: {failures}")
    print(f"Output written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()