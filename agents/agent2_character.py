from typing import List, Optional
import json

from transformers import pipeline, AutoTokenizer, AutoModel
import torch

from memory.scratchpad import CharacterProfile, Scratchpad


# ── NER model ─────────────────────────────────────────────────────────
NER_MODEL = "Davlan/bert-base-multilingual-cased-ner-hrl"

# Keywords that signal a character's social role
# These appear in the ROMANIZED text (output of transliterator)
SOCIAL_ROLE_KEYWORDS = {
    "elder": [
        "pitaji", "mataji", "dada", "nana", "bade", "uncle", "aunty",
        "dadaji", "naani", "chacha", "maama", "bauji", "ammaji", "taaya",
        "taai", "nani", "dadi", "pardada"
    ],
    "authority": [
        "sir", "madam", "doctor", "inspector", "principal", "sahib",
        "ji", "adhikari", "mantri", "judge", "collector", "officer"
    ],
    "peer": []   # default — no keywords needed
}

NON_CHARACTER_WORDS = {
    "dayari", "diary", "kitab", "kitaben", "alamari", "nakshe", "nakshay",
    "dastavej", "dastavez", "sanket", "log", "chatron", "shodhakarta",
    "shodhakartaon", "nagar", "parishad", "nagar parishad", "vyakti",
    "bhavan", "pustakalay"
}

class CharacterAnalyst:
    def __init__(self, scratchpad: Scratchpad):
        self.scratchpad = scratchpad

        print("Loading NER model (first run downloads ~700MB)...")
        self.ner = pipeline(
            "ner",
            model=NER_MODEL,
            aggregation_strategy="simple",
            device=-1   # CPU
        )
        print("NER model loaded.")

        print("Loading MuRIL model (first run downloads ~357MB)...")
        self.muril_tokenizer = AutoTokenizer.from_pretrained(
            "google/muril-base-cased")
        self.muril_model = AutoModel.from_pretrained(
            "google/muril-base-cased")
        self.muril_model.eval()
        print("MuRIL model loaded.")

    def build_profiles(self, scene_chunks: List[dict],
                        event_chain: List[dict]) -> dict:
        """
        Build CharacterProfile for every character found in the story.
        Uses event_chain (Agent 1 output) as primary source of character names,
        supplemented by NER on the raw romanized text.
        """
        profiles = {}

        for chunk, events in zip(scene_chunks, event_chain):
            # Primary: character names from Agent 1's event chain
            char_names = events.get("characters", [])

            # Secondary: NER on romanized text
            try:
                ner_results = self.ner(chunk["text"])
                ner_names = [
                    e["word"] for e in ner_results
                    if e["entity_group"] == "PER" and len(e["word"]) > 1
                ]
            except Exception as e:
                print(f"  NER failed on scene {chunk['scene_id']}: {e}")
                ner_names = []

            # Merge, deduplicate (case-insensitive)
            seen = {n.lower() for n in profiles.keys()}
            all_names = [n for n in char_names if n.lower() not in NON_CHARACTER_WORDS]
            for name in ner_names:
                if (name.lower() not in seen
                        and name.lower() not in {n.lower() for n in all_names}
                        and name.lower() not in NON_CHARACTER_WORDS):
                    all_names.append(name)

            for name in all_names:
                # Only create profile if not already seen
                if not any(n.lower() == name.lower() for n in profiles):
                    profiles[name] = self._build_initial_profile(
                        name, chunk["text"], events
                    )

        # Save pronoun map from the LAST scene's event chain
        # (it accumulates — last one has the fullest map)
        combined_pronoun_map = {}
        for events in event_chain:
            combined_pronoun_map.update(events.get("pronoun_map", {}))
        self.scratchpad.write("pronoun_map", combined_pronoun_map)

        self.scratchpad.write("character_profiles", profiles)
        return profiles

    def _build_initial_profile(self, name: str, context_text: str,
                                events: dict) -> dict:
        """
        Infer social role from keywords near the character name in the text.
        Returns a plain dict matching CharacterProfile schema.
        """
        text_lower = context_text.lower()
        role = "peer"   # default

        for role_type, keywords in SOCIAL_ROLE_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                role = role_type
                break

        # Slang level: elders/authority → 0, peers → 5 (tunable later)
        slang_level = 0 if role in ["elder", "authority"] else 5

        return {
            "name": name,
            "age": None,
            "social_role": role,
            "slang_level": slang_level,
            "relationship": {},
            "current_emotion": "neutral",
            "turn_count": 0,
            "stress_index": 0.0
        }

    def update_state(self, char_name: str, scene_events: List[str],
                      scene_goal: str):
        """
        Rule-based state updater.
        Call this after each scene to update emotion and stress index.
        """
        profiles = self.scratchpad.read("character_profiles") or {}

        # Case-insensitive lookup
        matched_key = None
        for key in profiles:
            if key.lower() == char_name.lower():
                matched_key = key
                break

        if matched_key is None:
            print(f"  update_state: '{char_name}' not found in profiles, skipping.")
            return

        profile = profiles[matched_key]

        EMOTION_TRIGGERS = {
            "angry":   ["gussa", "naraaz", "chillaya", "maara", "ladai",
                         "krodh", "thappad", "dant"],
            "anxious": ["dar", "ghabra", "pareshan", "chinta", "tension",
                         "fikar", "sehma", "kampa"],
            "happy":   ["khush", "muskuraya", "hansa", "mila", "success",
                         "khushi", "prasan", "aanand"],
            "sad":     ["roya", "udaas", "dukh", "aansu", "chala gaya",
                         "bichad", "dard", "takleef"],
            "urgent":  ["jaldi", "abhi", "turant", "emergency", "rukna",
                         "bhago", "chalo", "hatao"]
        }

        events_text = " ".join(scene_events).lower()

        new_emotion = profile["current_emotion"]   # persist unless triggered
        for emotion, triggers in EMOTION_TRIGGERS.items():
            if any(t in events_text for t in triggers):
                new_emotion = emotion
                break

        profile["current_emotion"] = new_emotion
        profile["turn_count"] = profile.get("turn_count", 0) + 1

        # Stress index: 0.0-1.0, accumulates or decays
        stress = profile.get("stress_index", 0.0)
        if new_emotion in ["anxious", "angry", "urgent"]:
            stress = min(1.0, stress + 0.2)
        elif new_emotion == "happy":
            stress = max(0.0, stress - 0.3)
        # neutral/sad: no change to stress

        profile["stress_index"] = round(stress, 3)
        profiles[matched_key] = profile
        self.scratchpad.write("character_profiles", profiles)
        print(f"  Updated {char_name}: emotion={new_emotion}, stress={stress:.2f}")


if __name__ == "__main__":
    # Quick test — uses scratchpad from Week 2's test session
    # Run: python -m agents.agent2_character
    from memory.scratchpad import Scratchpad
    import json

    sp = Scratchpad("week3_test")

    # Simulate what Agent 1 would have produced for our library story
    fake_event_chain = [
        {
            "scene_id": "scene_01",
            "key_events": ["Arav enters the library", "Arav finds a dusty cupboard"],
            "characters": ["Arav"],
            "scene_goal": "Arav discovers a mysterious diary",
            "location": "library",
            "narrative_link": "This is the first scene.",
            "pronoun_map": {"usne": "Arav", "use": "Arav"}
        },
        {
            "scene_id": "scene_02",
            "key_events": ["Arav reads the diary", "Arav learns about a merchant"],
            "characters": ["Arav"],
            "scene_goal": "Arav becomes curious about the merchant",
            "location": "library",
            "narrative_link": "Following the diary discovery, Arav reads it.",
            "pronoun_map": {"usne": "Arav", "usamen": "diary"}
        }
    ]

    fake_chunks = [
        {"scene_id": "scene_01", "text": "ek din arav nam ka chatr library gaya. use ek dhul se dhaki alamari mili."},
        {"scene_id": "scene_02", "text": "arav utsuk ho gaya. usne dayari padhna shuru kiya."}
    ]

    analyst = CharacterAnalyst(scratchpad=sp)
    profiles = analyst.build_profiles(fake_chunks, fake_event_chain)

    print("\n=== CHARACTER PROFILES ===")
    print(json.dumps(profiles, indent=2))

    print("\n=== UPDATING STATE (scene 1 events) ===")
    analyst.update_state("Arav", fake_event_chain[0]["key_events"],
                          fake_event_chain[0]["scene_goal"])

    print("\n=== PROFILES AFTER UPDATE ===")
    print(json.dumps(sp.read("character_profiles"), indent=2))