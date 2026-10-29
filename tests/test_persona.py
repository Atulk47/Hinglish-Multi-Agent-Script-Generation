"""Unit tests for the Persona Agent (no network): JSON tolerance, name matching, label normalisation, gender second opinion,
lexicon fallback, profile merge + stress dynamics, address sanitising, and the rules-only degradation path."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from types import SimpleNamespace

from agents.agent2b_persona import (PersonaAgent, apply_persona, emotion_at, extract_json, EMOTIONS, _norm_emotion)

CHUNKS = [
    {"scene_id": "scene_01", "text": "mira boli, rohan tum thik ho? rohan dard se karah utha aur bola, chinta mat karo mira. mira ghabra gayi aur ro padi."},
    {"scene_id": "scene_02", "text": "kabir khush hokar bola, hum jeet gaye! sneha khushi se chillayi aur mira muskurayi."},
]
EVENTS = [{"scene_id": "scene_01", "characters": ["Mira", "Rohan"], "key_events": ["Rohan injures his leg", "Mira is scared"]},
          {"scene_id": "scene_02", "characters": ["Kabir", "Sneha", "Mira"], "key_events": ["The team wins"]}]
NAMES = ["Mira", "Rohan", "Kabir", "Sneha"]


class FakeClient:
    def __init__(self, reply=None, boom=False):
        self.reply, self.boom, self.calls = reply, boom, 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls += 1
        if self.boom:
            raise RuntimeError("429 rate limit")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply))])


GOOD = {
    "characters": {"Mira": {"gender": "female", "age_group": "teen", "speech_style": "casual", "traits": ["anxious"]},
                   "Rohan": {"gender": "male", "age_group": "teen", "speech_style": "casual", "traits": ["brave"]},
                   "Kabir": {"gender": "male", "age_group": "teen", "speech_style": "casual", "traits": []},
                   "Sneha": {"gender": "female", "age_group": "child", "speech_style": "casual", "traits": []},
                   "Ghost": {"gender": "male"}},
    "relationships": [{"a": "Sneha", "b": "Kabir", "relation": "younger sister of",
                       "a_to_b": {"address": "Bhaiya", "register": "respectful"}, "b_to_a": {"address": "Yaar", "register": "casual"}},
                      {"a": "Mira", "b": "Rohan", "relation": "teammate",
                       "a_to_b": {"address": "Bhaiya", "register": "respectful"}, "b_to_a": {"address": "Mira", "register": "casual"}},
                      {"a": "Ghost", "b": "Mira", "relation": "x", "a_to_b": {}, "b_to_a": {}}],
    "emotions": {"scene_01": {"Mira": {"emotion": "worried", "intensity": 2}, "Rohan": {"emotion": "sad", "intensity": 9}},
                 "scene_02": {"Kabir": {"emotion": "Joyful", "intensity": 3}, "Nobody": {"emotion": "happy", "intensity": 1}}},
}


def test_extract_json_is_tolerant():
    assert extract_json('```json\n{"a": 1,}\n```')["a"] == 1
    assert extract_json('Sure! here you go: {"a": {"b": 2}} done.')["a"]["b"] == 2
    assert extract_json("no json here") is None and extract_json("") is None


def test_label_normalisation():
    assert _norm_emotion("Worried") == "anxious" and _norm_emotion("joyful") == "happy"
    assert _norm_emotion("???") == "neutral" and _norm_emotion(None) == "neutral"
    assert all(-1 <= v <= 1 and 0 <= a <= 1 for v, a in EMOTIONS.values())


def test_full_merge_with_checks():
    pa = PersonaAgent(llm_client=FakeClient(json.dumps(GOOD)))
    p = pa.analyze(CHUNKS, EVENTS, NAMES)
    assert p["source"] == "llm+rules"
    ch = p["characters"]
    assert ch["Mira"]["gender"] == "female" and ch["Kabir"]["gender"] == "male"
    assert "Ghost" not in ch                                             # unknown names never enter
    em = p["emotions"]
    assert em["scene_01"]["Mira"] == {"emotion": "anxious", "intensity": 2}        # 'worried' -> 'anxious'
    assert em["scene_01"]["Rohan"]["intensity"] == 3                                # 9 clamped
    assert em["scene_02"]["Kabir"]["emotion"] == "happy"
    assert "Nobody" not in em["scene_02"]
    assert em["scene_02"]["Sneha"]["source"] == "lexicon"                           # LLM left her empty -> rules filled
    rel = {(r["a"], r["b"]): r for r in p["relationships"]}
    assert ("Ghost", "Mira") not in rel
    assert rel[("Sneha", "Kabir")]["a_to_b"]["address"] == "Bhaiya"                 # real sibling keeps 'Bhaiya'
    assert rel[("Sneha", "Kabir")]["b_to_a"]["address"] == "Sneha"                  # slang address 'Yaar' replaced by the name
    assert rel[("Mira", "Rohan")]["a_to_b"]["address"] == "Rohan"                   # 'Bhaiya' for a teammate -> name
    assert p["checks"]["unknown_names_dropped"] >= 2


def test_gender_morphology_overrides_a_wrong_llm_answer():
    bad = json.loads(json.dumps(GOOD))
    bad["characters"]["Mira"]["gender"] = "male"                                    # LLM slip; text says boli/gayi/ro padi
    p = PersonaAgent(llm_client=FakeClient(json.dumps(bad))).analyze(CHUNKS, EVENTS, NAMES)
    assert p["characters"]["Mira"]["gender"] == "female"
    assert any(o["name"] == "Mira" for o in p["checks"]["gender_overrides"])


def test_rules_only_when_llm_is_down():
    c = FakeClient(boom=True)
    p = PersonaAgent(llm_client=c).analyze(CHUNKS, EVENTS, NAMES)
    assert p["source"] == "rules-only" and c.calls == 2                              # tried twice, then degraded
    assert p["characters"]["Mira"]["gender"] == "female"                             # from verb forms
    assert p["emotions"]["scene_01"]["Mira"]["emotion"] != "neutral"                 # lexicon: dard / ghabra / ro pad
    assert p["emotions"]["scene_02"]["Kabir"]["emotion"] in ("happy", "excited", "proud")
    assert set(p["emotions"]) == {"scene_01", "scene_02"}


def test_garbage_llm_reply_degrades_not_crashes():
    p = PersonaAgent(llm_client=FakeClient("I cannot do that.")).analyze(CHUNKS, EVENTS, NAMES)
    assert p["source"] == "rules-only" and "Mira" in p["characters"]


def test_apply_persona_merges_profiles_and_stress_dynamics():
    p = PersonaAgent(llm_client=FakeClient(json.dumps(GOOD))).analyze(CHUNKS, EVENTS, NAMES)
    profiles = {n: {"name": n, "social_role": "peer", "slang_level": 5, "current_emotion": "neutral", "stress_index": 0.0,
                    "turn_count": 0} for n in NAMES}
    apply_persona(profiles, p, ["scene_01", "scene_02"])
    mira = profiles["Mira"]
    assert mira["gender"] == "female" and mira["relationship"]["Rohan"]["i_address_them"] == "Rohan"
    assert [e["scene_id"] for e in mira["emotion_arc"]] == ["scene_01", "scene_02"]
    assert mira["emotion_arc"][0]["stress"] > 0                                       # anxious scene raises stress
    assert mira["emotion_arc"][1]["stress"] < mira["emotion_arc"][0]["stress"]        # happy scene relieves it
    assert mira["current_emotion"] == mira["emotion_arc"][-1]["emotion"]
    assert emotion_at(mira, "scene_01")["emotion"] == "anxious"
    assert emotion_at(mira, "scene_99")["emotion"] == mira["current_emotion"]         # unknown scene -> falls back
    assert profiles["Sneha"]["relationship"]["Kabir"]["i_address_them"] == "Bhaiya"


def test_elder_age_group_upgrades_role_to_zero_slang():
    p = {"characters": {"Dada": {"gender": "male", "age_group": "elder", "speech_style": "respectful", "traits": []}},
         "relationships": [], "emotions": {}}
    profiles = {"Dada": {"name": "Dada", "social_role": "peer", "slang_level": 5, "current_emotion": "neutral", "stress_index": 0.0}}
    apply_persona(profiles, p, [])
    assert profiles["Dada"]["social_role"] == "elder" and profiles["Dada"]["slang_level"] == 0


def test_a_kinship_word_in_the_scene_does_not_make_everyone_an_elder():
    """Regression: one mention of 'Nani' in scene 1 used to turn Ananya and Vikram into elders (all slang blocked)."""
    from agents.agent2_character import CharacterAnalyst as C
    c = C.__new__(C)
    txt = "ananya aur uska bada bhai vikram apni nani se bahut pyaar karte the. nani ka janmdin agle hafte tha. adhyapak sharma ne kaha."
    role = lambda n: c._build_initial_profile(n, txt, {})["social_role"]
    assert role("Ananya") == "peer" and role("Vikram") == "peer" and role("Priya") == "peer"
    assert role("Nani") == "elder" and role("Dadaji") == "elder"
    assert role("Sharma") == "authority" and role("Coach Varma") == "authority"


def test_gender_votes_stay_inside_the_clause_and_ignore_auxiliaries():
    pa = PersonaAgent.__new__(PersonaAgent)
    chunks = [{"scene_id": "s1", "text": "ananya ka dil jor se dhadak raha tha. vikram muskuraya aur bola, ananya ghabra gayi aur boli."}]
    names = ["Ananya", "Vikram"]
    assert pa._gender_votes("Ananya", chunks, names) == (2, 0) or pa._gender_votes("Ananya", chunks, names)[1] == 0
    assert pa._gender_votes("Vikram", chunks, names) == (0, 2)
    # "Vikram bola" must never be counted for Ananya, even when the names are adjacent
    assert pa._gender_votes("Ananya", [{"scene_id": "s", "text": "ananya vikram bola"}], names) == (0, 0)
