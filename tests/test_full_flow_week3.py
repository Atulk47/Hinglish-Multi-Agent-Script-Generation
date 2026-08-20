import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from preprocessing.transliterate import HindiTransliterator
from preprocessing.chunker import ProseChunker
from agents.agent1_narrative import NarrativeAnalyst
from agents.agent2_character import CharacterAnalyst
from agents.intent_classifier import IntentClassifier
from gates.three_gate_controller import ThreeGateController
from memory.scratchpad import Scratchpad


def run_week3_flow(story_path: str, session_id: str):
    print(f"\n{'='*60}")
    print(f"Running Week 3 flow for: {story_path}")
    print(f"{'='*60}\n")

    with open(story_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    sp = Scratchpad(session_id)

    # --- Preprocessing (Week 1) ---
    transliterator = HindiTransliterator()
    paragraphs = raw_text.split('\n\n')
    romanized_paragraphs = [
        transliterator.transliterate_sentence(p.strip())
        for p in paragraphs if p.strip()
    ]
    romanized_text = '\n\n'.join(romanized_paragraphs)
    sp.write("romanized_text", romanized_text)

    chunker = ProseChunker()
    scene_chunks = chunker.chunk(romanized_text)
    sp.write("scene_chunks", scene_chunks)
    print(f"Chunks: {len(scene_chunks)} scenes\n")

    # --- Agent 1 (Week 2) ---
    analyst = NarrativeAnalyst(llm_backend="groq")
    event_chain = analyst.extract_event_chain(scene_chunks)
    sp.write("event_chain", event_chain)
    print("=== EVENT CHAIN (summary) ===")
    for e in event_chain:
        print(f"  {e['scene_id']}: {e.get('characters',[])} | {e.get('scene_goal','')[:60]}")
    print()

    # --- Agent 2 (Week 3) ---
    char_analyst = CharacterAnalyst(scratchpad=sp)
    profiles = char_analyst.build_profiles(scene_chunks, event_chain)
    print("=== CHARACTER PROFILES ===")
    print(json.dumps(profiles, indent=2))
    print()

    # Update state for each character after each scene
    for events in event_chain:
        for char_name in events.get("characters", []):
            char_analyst.update_state(
                char_name,
                events.get("key_events", []),
                events.get("scene_goal", "")
            )

    # --- Gates test ---
    print("=== GATE TESTS ===")
    intent_clf = IntentClassifier()
    gate_ctrl = ThreeGateController(sp, slang_density_threshold=0.25)

    # Test utterances
    test_cases = [
        ("Arav", "yaar yeh diary bahut interesting hai", "scene_01"),
        ("Arav", "library mein kaafi purani kitabein hain", "scene_01"),
    ]

    for speaker, utterance, scene_id in test_cases:
        result = gate_ctrl.run_all_gates(
            speaker, utterance, scene_id, intent_clf
        )
        print(f"\nSpeaker: {speaker}")
        print(f"Utterance: '{utterance}'")
        print(f"Gate result: {result}")

    print("\n=== DENSITY REPORT ===")
    print(json.dumps(gate_ctrl.get_density_report(), indent=2))


if __name__ == "__main__":
    run_week3_flow("data/raw/test_story_1.txt", "week3_test")