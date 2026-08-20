import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from preprocessing.transliterate import HindiTransliterator
from preprocessing.chunker import ProseChunker
from agents.agent1_narrative import NarrativeAnalyst
from agents.agent2_character import CharacterAnalyst
from agents.agent3_synthesizer import DialogueSynthesizer
from memory.scratchpad import Scratchpad


def run_week4_flow(story_path: str, session_id: str):
    print(f"\n{'='*60}")
    print(f"Week 4 full flow: {story_path}")
    print(f"{'='*60}\n")

    with open(story_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    sp = Scratchpad(session_id)

    # Preprocessing
    transliterator = HindiTransliterator()
    paragraphs = raw_text.split('\n\n')
    romanized_paragraphs = [
        transliterator.transliterate_sentence(p.strip())
        for p in paragraphs if p.strip()
    ]
    romanized_text = '\n\n'.join(romanized_paragraphs)

    chunker = ProseChunker()
    scene_chunks = chunker.chunk(romanized_text)
    sp.write("scene_chunks", scene_chunks)

    # Agent 1
    agent1 = NarrativeAnalyst(llm_backend="groq")
    event_chain = agent1.extract_event_chain(scene_chunks)
    sp.write("event_chain", event_chain)

    # Agent 2
    agent2 = CharacterAnalyst(scratchpad=sp)
    profiles = agent2.build_profiles(scene_chunks, event_chain)
    for events in event_chain:
        for char in events.get("characters", []):
            agent2.update_state(char, events.get("key_events", []),
                                 events.get("scene_goal", ""))

    print("=== CHARACTER PROFILES ===")
    print(json.dumps(profiles, indent=2))

    # Agent 3 (Groq-based Pass 1)
    print("\n=== PASS 1 GENERATION (neutral Hinglish dialogue) ===")
    agent3 = DialogueSynthesizer(llm_backend="groq")
    neutral_script = agent3.generate_pass1_for_story(
        scene_chunks, profiles, event_chain
    )
    sp.write("neutral_script", neutral_script)

    print("\n=== NEUTRAL SCRIPT ===")
    for entry in neutral_script:
        print(f"\n[{entry['scene_id']}]")
        print(entry['dialogue'])


if __name__ == "__main__":
    run_week4_flow("data/raw/test_story_1.txt", "week4_test")