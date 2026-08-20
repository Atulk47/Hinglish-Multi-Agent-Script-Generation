from typing import TypedDict, List, Optional
import json
import os


# ── Character profile ──────────────────────────────────────────────────
class CharacterProfile(TypedDict):
    name: str
    age: Optional[int]
    social_role: str        # "peer" | "elder" | "authority"
    slang_level: int         # 0 (none) - 10 (heavy)
    relationship: dict       # {other_char_name: relationship_type}
    current_emotion: str     # "neutral" | "anxious" | "angry" | "happy" | ...
    turn_count: int
    stress_index: float      # 0.0 - 1.0


# ── Scene event ────────────────────────────────────────────────────────
class SceneEvent(TypedDict):
    scene_id: str
    key_events: List[str]
    characters: List[str]
    scene_goal: str
    location: str
    narrative_link: str      # "Following X, Y reacts..."
    pronoun_map: dict


# ── Master LangGraph state ─────────────────────────────────────────────
class PipelineState(TypedDict):
    # Input
    raw_hindi_text: str

    # After preprocessing
    romanized_text: str
    scene_chunks: List[dict]

    # Agent 1 output
    event_chain: List[SceneEvent]

    # Agent 2 output
    character_profiles: dict   # {char_name: CharacterProfile}

    # Agent 3 output (Pass 1)
    neutral_script: List[dict]  # [{scene_id, speaker, dialogue}]

    # Agent 4 output (Pass 2)
    slang_script: List[dict]

    # Evaluation
    validation_scores: List[float]
    failed_scenes: List[str]    # scene_ids that need rewriting
    rewrite_count: int           # guard against infinite loops

    # Final
    final_script: str


class Scratchpad:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.path = f"data/sessions/{session_id}_scratchpad.json"
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._data = {}
        # Load existing data if this session was started before
        if os.path.exists(self.path):
            self.load()

    def write(self, key: str, value):
        self._data[key] = value
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def read(self, key: str):
        return self._data.get(key)

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, 'r', encoding='utf-8') as f:
                self._data = json.load(f)
        return self

    def all(self):
        return self._data


if __name__ == "__main__":
    sp = Scratchpad("test_session")
    sp.write("raw_hindi_text", "तुम क्या कर रहे हो?")
    sp.write("event_chain", [{"scene_id": "scene_01", "key_events": ["test event"]}])
    print("Saved scratchpad:", sp.all())

    sp2 = Scratchpad("test_session")
    print("Reloaded scratchpad:", sp2.all())