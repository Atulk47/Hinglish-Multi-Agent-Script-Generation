import json
import os
import re
import time
from typing import List
from groq import Groq
from dotenv import load_dotenv
import config

load_dotenv()

PASS1_PROMPT_TEMPLATE = """You are a Hinglish screenplay dialogue writer.
Generate NEUTRAL Hinglish dialogue (NO slang, NO casual fillers like 'yaar'/'bhai') for this scene.

Scene context: {scene_goal}
Location: {location}
Previous event: {narrative_link}
Speaker: {speaker_name} (social role: {speaker_role}, current emotion: {emotion})
Listener: {listener_name}
Key events to cover in dialogue: {key_events}

Rules:
- Write in romanized Hinglish (mix of Hindi words in Roman script + English)
- Neutral register — no slang, no fillers, appropriate for {speaker_role}
- Cover the key events naturally through dialogue
- Output ONLY the dialogue line(s), one per speaker turn
- Format: SPEAKER_NAME: dialogue text
- Maximum 3 dialogue lines total
- No scene directions, no narration, no explanation"""


class DialogueSynthesizer:
    def __init__(self, llm_backend: str = "groq"):
        self.backend = llm_backend
        load_dotenv()
        self.groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
        print("DialogueSynthesizer initialized with Groq backend.")

    def generate_pass1(self, scene: dict, character_profiles: dict,
                        event_chain_entry: dict) -> str:
        characters = event_chain_entry.get("characters", [])
        speaker = characters[0] if characters else "SPEAKER"
        listener = characters[1] if len(characters) > 1 else "narrator"

        profile = {}
        for key in character_profiles:
            if key.lower() == speaker.lower():
                profile = character_profiles[key]
                break

        prompt = PASS1_PROMPT_TEMPLATE.format(
            scene_goal=event_chain_entry.get("scene_goal", ""),
            location=event_chain_entry.get("location", "unknown"),
            narrative_link=event_chain_entry.get("narrative_link", ""),
            speaker_name=speaker,
            speaker_role=profile.get("social_role", "peer"),
            emotion=profile.get("current_emotion", "neutral"),
            listener_name=listener,
            key_events="; ".join(event_chain_entry.get("key_events", []))
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.groq_client.chat.completions.create(
                    model=config.GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=500,
                    reasoning_effort=getattr(config, "GROQ_REASONING_EFFORT", "low"),
                )
                result = response.choices[0].message.content.strip()

                # Ensure output has SPEAKER: format
                if ":" not in result:
                    result = f"{speaker}: {result}"

                return result

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"  Groq call failed ({e}), retrying in 2s...")
                    time.sleep(2)
                else:
                    return f"{speaker}: [generation failed after {max_retries} attempts]"

    def generate_pass1_for_story(self, scene_chunks: list,
                                   character_profiles: dict,
                                   event_chain: list) -> list:
        neutral_script = []
        for chunk, events in zip(scene_chunks, event_chain):
            print(f"  Generating Pass 1 for {chunk['scene_id']}...")
            dialogue = self.generate_pass1(chunk, character_profiles, events)
            neutral_script.append({
                "scene_id": chunk["scene_id"],
                "dialogue": dialogue
            })
            print(f"    -> {dialogue[:80]}...")
            time.sleep(0.5)  # small delay to avoid rate limiting
        return neutral_script


if __name__ == "__main__":
    synth = DialogueSynthesizer()
    print("DialogueSynthesizer loaded successfully.")

    # Quick test
    test_events = {
        "scene_id": "scene_02",
        "key_events": ["Arav finds a dusty cupboard", "Arav discovers a leather diary"],
        "characters": ["Arav"],
        "scene_goal": "Arav discovers a mysterious diary",
        "location": "library",
        "narrative_link": "Following the library setting being established, Arav now explores.",
        "pronoun_map": {}
    }
    test_profiles = {
        "Arav": {
            "name": "Arav", "age": 20, "social_role": "peer",
            "slang_level": 5, "relationship": {}, "current_emotion": "curious",
            "turn_count": 0, "stress_index": 0.0
        }
    }
    result = synth.generate_pass1({}, test_profiles, test_events)
    print("\n=== TEST DIALOGUE ===")
    print(result)