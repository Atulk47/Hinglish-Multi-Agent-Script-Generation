from typing import List, Optional
import json
import re

from transformers import pipeline, AutoTokenizer, AutoModel
import torch

from memory.scratchpad import CharacterProfile, Scratchpad
from agents.name_utils import canonical_name, strip_honorific, same_person, display_name


# ── NER model ─────────────────────────────────────────────────────────
NER_MODEL = "Davlan/bert-base-multilingual-cased-ner-hrl"

# Keywords that signal a character's social role
# These appear in the ROMANIZED text (output of transliterator)
SOCIAL_ROLE_KEYWORDS = {
    "elder": [
        "pitaji", "mataji", "dada", "nana", "bade", "uncle", "aunty",
        "dadaji", "naani", "chacha", "maama", "bauji", "ammaji", "taaya",
        "taai", "nani", "dadi", "pardada",
        # English kinship terms the LLM sometimes uses instead of the Hindi name
        "grandfather", "grandpa", "grandmother", "grandma", "granny",
        "father", "dad", "daddy", "mother", "mom", "mummy", "aunt",
        # common address forms
        "amma", "amman", "dadima", "naanima", "papa", "maa", "baba",
    ],
    "authority": [
        "sir", "madam", "doctor", "inspector", "principal", "sahib",
        "ji", "adhikari", "mantri", "judge", "collector", "officer",
        "professor", "prof", "teacher", "faculty", "acharya", "guru",
        "adhyapak", "adhyapika", "prof.",
        "prophaesar", "prophesar", "profesar", "prophae"   # transliterated प्रोफ़ेसर variants
    ],
    "peer": []   # default — no keywords needed
}

# Titles that make a NAME an authority figure ("Coach Varma"). Matched on the NAME only - never on the scene text, or one
# mention of "koch" in a scene would turn every character in that scene into an authority.
TITLE_WORDS = {"coach", "koch", "trainer"}

NON_CHARACTER_WORDS = {
    # objects / places / groups
    "dayari", "diary", "kitab", "kitaben", "alamari", "nakshe", "nakshay",
    "dastavej", "dastavez", "sanket", "log", "chatron", "shodhakarta",
    "shodhakartaon", "nagar", "parishad", "nagar parishad", "vyakti",
    "bhavan", "pustakalay",
    # conversational fillers the LLM sometimes mislabels as a character name
    "yaar", "yara", "bhai", "bhaiya", "dost", "dosto", "doston", "yr",
    # abstract nouns / role words (not proper names)
    "idea", "aidiya", "plan", "team", "group", "project", "startup", "app",
    "data", "event", "stage", "manch", "company", "competition", "hackathon",
    "vijeta", "vijet", "sapna", "sapne", "sapno", "mehnat", "prototype",
    # merchant / shop role words (unnamed roles, not proper names). "kiran"
    # is really किराना (grocery) after schwa-deletion, not the name "Kiran".
    "kirana", "kiran", "kirane", "vyapari", "vyaapari", "byapari", "byopari",
    "dukandar", "dukaandar", "dukaan", "dukan", "shopkeeper", "merchant",
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
        canon_to_key = {}   # canonical name -> display key currently in `profiles`

        # Full romanized story text, used to drop names the LLM hallucinated
        # (a real character's name appears somewhere in the actual story).
        story_text = " ".join(c.get("text", "") for c in scene_chunks).lower()

        def _is_plausible(raw: str) -> bool:
            if not raw:
                return False
            core = canonical_name(raw)
            token = core.split()[0] if core.split() else core
            # blocklist check on both the raw form and the honorific-stripped core
            if (raw.lower() in NON_CHARACTER_WORDS or core in NON_CHARACTER_WORDS
                    or token in NON_CHARACTER_WORDS):
                return False
            if len(core) < 2 or len(token) < 3:
                return False
            # Kinship/role terms (e.g. an LLM writing "Grandfather" for दादाजी) are
            # valid characters even if the English word isn't in the Hindi text.
            kinship = set(SOCIAL_ROLE_KEYWORDS["elder"]) | set(SOCIAL_ROLE_KEYWORDS["authority"]) | TITLE_WORDS
            if token in kinship or core in kinship:
                return True
            # Hallucination guard: otherwise the honorific-stripped name must appear
            # in the story as a WHOLE WORD, not just a fragment inside another word
            # (drops "Vijet" from "vijeta", "Aidiy" from an ideas mention, etc.).
            return re.search(rf"\b{re.escape(token)}\b", story_text) is not None

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

            # Agent-1 names first (more reliable), then NER; merge by canonical
            # identity so "Sharm" / "Prof. Sharma" collapse into one profile.
            for raw in list(char_names) + ner_names:
                if not _is_plausible(raw):
                    continue
                canon = canonical_name(raw)
                _, had_auth = strip_honorific(raw)

                # find an existing group for this person (exact/prefix/fuzzy)
                gkey = None
                for existing in list(profiles.keys()):
                    if same_person(existing, raw):
                        gkey = existing
                        break

                if gkey is None:
                    disp = display_name(raw)
                    prof = self._build_initial_profile(raw, chunk["text"], events)
                    prof["name"] = disp
                    profiles[disp] = prof
                    canon_to_key[canon] = disp
                    continue

                # Merge into the existing profile:
                prof = profiles[gkey]
                # (a) upgrade role if this surface form carries an authority title
                if had_auth and prof.get("social_role") == "peer":
                    prof["social_role"] = "authority"
                    prof["slang_level"] = 0
                # (b) prefer the fullest surface form, then tidy it for display
                display = display_name(self._prefer_display(gkey, raw))
                if display != gkey:
                    profiles[display] = profiles.pop(gkey)
                    profiles[display]["name"] = display
                    for c, k in list(canon_to_key.items()):
                        if k == gkey:
                            canon_to_key[c] = display

        # Save pronoun map from the LAST scene's event chain
        # (it accumulates — last one has the fullest map)
        combined_pronoun_map = {}
        for events in event_chain:
            combined_pronoun_map.update(events.get("pronoun_map", {}))
        self.scratchpad.write("pronoun_map", combined_pronoun_map)

        self.scratchpad.write("character_profiles", profiles)
        return profiles

    @staticmethod
    def _prefer_display(a: str, b: str) -> str:
        """Pick the tidier display name for the same person: keep a title if one
        form has it, otherwise the fuller (longer) surface form."""
        a_core, a_auth = strip_honorific(a)
        b_core, b_auth = strip_honorific(b)
        if a_auth != b_auth:                 # keep the titled form ("Prof. Sharma")
            return a if a_auth else b
        # both titled or both plain -> the longer (more complete) spelling
        return a if len(a.strip()) >= len(b.strip()) else b

    def _build_initial_profile(self, name: str, context_text: str,
                                events: dict) -> dict:
        """
        Infer social role from keywords near the character name in the text.
        Returns a plain dict matching CharacterProfile schema.
        """
        # Match role keywords against the character's NAME as well as the scene
        # text. The name (e.g. "Professor Sharma") often carries the honorific
        # cleanly, whereas transliteration can mangle it in the body text
        # (प्रोफ़ेसर -> "prophaesar"), so the name is the more reliable signal.
        #
        # IMPORTANT: match keywords as WHOLE WORDS, not substrings. Short kinship
        # terms like "nana"/"bade"/"dad" otherwise match inside ordinary Hindi
        # verbs ("banana" = to make, "sunana" = to tell), which would wrongly flag
        # every character in such a scene as an elder.
        # BUG FIXED: the keywords used to be searched in the WHOLE scene text, so one mention of "Nani" in a scene turned every character of that
        # scene into an elder (all slang blocked). A role now comes from the character's OWN name, or - for titles only (teacher, sir, coach, ji) -
        # from the single word right before / after the name in the text ("adhyapak sharma", "sharma sir"). Kinship words (nani, dadi) count only
        # when they ARE the name.
        role = "peer"   # default
        name_toks = re.findall(r"[a-z]+", name.lower())
        ctx = re.findall(r"[a-z]+", (context_text or "").lower())
        adjacent = set()
        for i, t in enumerate(ctx):
            if len(t) >= 3 and t in name_toks:
                adjacent.update(ctx[max(0, i - 1):i] + ctx[i + 1:i + 2])

        def _kw(role_type: str):
            return {k.strip(".").lower() for k in SOCIAL_ROLE_KEYWORDS[role_type]}

        if _kw("elder") & set(name_toks):
            role = "elder"
        elif _kw("authority") & (set(name_toks) | adjacent):
            role = "authority"
        if role == "peer" and any(w in TITLE_WORDS for w in re.findall(r"[a-z]+", name.lower())):
            role = "authority"

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

        # Tolerant lookup (honorifics + name variants merge to one profile)
        matched_key = None
        for key in profiles:
            if key.lower() == char_name.lower() or same_person(key, char_name):
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