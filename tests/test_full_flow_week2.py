import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from preprocessing.transliterate import HindiTransliterator
from preprocessing.chunker import ProseChunker
from agents.agent1_narrative import NarrativeAnalyst
from memory.scratchpad import Scratchpad


def run_full_flow(story_path: str, session_id: str):
    # Step 1: Read raw Hindi
    with open(story_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    sp = Scratchpad(session_id)
    sp.write("raw_hindi_text", raw_text)

    # Step 2: Transliterate
    transliterator = HindiTransliterator()
    # Transliterate paragraph-by-paragraph to preserve blank-line boundaries
    paragraphs = raw_text.split('\n\n')
    romanized_paragraphs = [
        transliterator.transliterate_sentence(p.strip())
        for p in paragraphs if p.strip()
    ]
    romanized_text = '\n\n'.join(romanized_paragraphs)
    sp.write("romanized_text", romanized_text)

    print("=== ROMANIZED TEXT ===")
    print(romanized_text)
    print()

    # Step 3: Chunk into scenes
    chunker = ProseChunker()
    scene_chunks = chunker.chunk(romanized_text)
    sp.write("scene_chunks", scene_chunks)

    print("=== SCENE CHUNKS ===")
    for c in scene_chunks:
        print(c)
    print()

    # Step 4: Agent 1 - extract event chain
    analyst = NarrativeAnalyst(llm_backend="groq")
    event_chain = analyst.extract_event_chain(scene_chunks)
    sp.write("event_chain", event_chain)

    print("=== EVENT CHAIN ===")
    print(json.dumps(event_chain, indent=2, ensure_ascii=False))

    return event_chain


if __name__ == "__main__":
    run_full_flow("data/raw/test_story_2.txt", "week2_test_story2")