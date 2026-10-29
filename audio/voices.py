"""
audio/voices.py  -  the VOICE CASTING AGENT.

Given the Persona Agent's output (gender, age group, speech style, traits, social role) it assigns every character
  * a TTS voice (matching gender and language),
  * a personal baseline pitch / speaking rate / volume (child and teen voices higher and quicker, elders lower and slower,
    authority figures lower and a little louder, energetic characters quicker, calm / wise ones slower),
  * and makes sure that no two characters of the same gender end up with the same sound: when the voice bank has fewer voices than
    the cast, characters are spread over distinct (voice, pitch) slots, so e.g. six male characters sound like six people.

Deterministic (no LLM): the same story always gets the same cast, which keeps demos repeatable.
"""
from __future__ import annotations
import hashlib
from typing import Dict

from agents.name_utils import match_profile

# Edge neural voices (verified available): two native Hindi voices plus Indian-English voices.
VOICE_BANK = {
    "hi": {"male": ["hi-IN-MadhurNeural"], "female": ["hi-IN-SwaraNeural"]},
    "en-IN": {"male": ["en-IN-PrabhatNeural"], "female": ["en-IN-NeerjaNeural"]},
}
NARRATOR = {"voice": "en-IN-NeerjaExpressiveNeural", "rate_pct": -4, "pitch_hz": -2, "volume_pct": 0}

AGE_ADJ = {"child": (6, 14, -2), "teen": (4, 9, 0), "young_adult": (1, 3, 0), "adult": (-2, -3, 2), "elder": (-10, -12, 0),
           "unknown": (0, 0, 0)}            # (rate %, pitch Hz, volume %)
TRAIT_RATE = {"energetic": 4, "enthusiastic": 4, "excitable": 4, "youthful": 3, "competitive": 2, "anxious": 2, "talkative": 3,
              "calm": -3, "wise": -4, "supportive": -1, "strict": -2, "authoritative": -2, "gentle": -3, "elderly": -4,
              "fatherly": -2, "protective": -1}
# distinct pitch slots used to separate characters that share one voice (Hz offsets around the age-adjusted pitch)
PITCH_SLOTS = [0, 6, -6, 12, -11, 3, -3, 9]


def _stable(name: str) -> int:
    return int(hashlib.md5(name.lower().encode("utf-8")).hexdigest()[:8], 16)


def cast_voices(profiles: Dict[str, dict], language: str = "hi") -> Dict[str, dict]:
    """-> {character name: {'voice','rate_pct','pitch_hz','volume_pct','gender','why'}}"""
    cast: Dict[str, dict] = {}
    used_slots = {"male": 0, "female": 0, "unknown": 0}
    # oldest-first ordering keeps the main characters on the first (most natural) pitch slots
    order = sorted(profiles.items(), key=lambda kv: -int(kv[1].get("turn_count") or 0))
    for name, p in order:
        gender = p.get("gender") if p.get("gender") in ("male", "female") else None
        if gender is None:                               # unknown gender -> alternate deterministically, flag it
            gender = "female" if _stable(name) % 2 else "male"
            why = "gender unknown: guessed"
        else:
            why = "gender from persona"
        age = p.get("age_group") if p.get("age_group") in AGE_ADJ else "unknown"
        role = p.get("social_role", "peer")
        if age == "unknown":
            age = {"elder": "elder", "authority": "adult"}.get(role, "young_adult")
        ra, pa, va = AGE_ADJ[age]
        if role == "authority":
            ra, pa, va = ra - 1, pa - 2, va + 3
        rate_t = sum(TRAIT_RATE.get(t.lower(), 0) for t in (p.get("traits") or []))
        slot = used_slots[gender]
        used_slots[gender] += 1
        pitch_slot = PITCH_SLOTS[slot % len(PITCH_SLOTS)]
        voices = VOICE_BANK[language][gender]
        voice = voices[slot % len(voices)]
        cast[name] = {"voice": voice, "gender": gender, "age_group": age,
                      "rate_pct": int(ra + max(-6, min(6, rate_t))), "pitch_hz": int(pa + pitch_slot), "volume_pct": int(va),
                      "slot": slot, "why": f"{why}; {age}; {role}"}
    return cast


def voice_for(cast: Dict[str, dict], speaker: str, profiles: Dict[str, dict] | None = None) -> dict:
    """Voice settings for a speaker tag as it appears in the script (case / variant tolerant); a neutral fallback if unknown."""
    for n in cast:
        if n.lower() == speaker.strip().lower():
            return cast[n]
    if profiles:
        prof = match_profile(profiles, speaker)
        for n, p in profiles.items():
            if p is prof and n in cast:
                return cast[n]
    first = speaker.strip().split()[0].lower() if speaker.strip() else ""
    for n in cast:
        if n.lower().split()[0] == first:
            return cast[n]
    return {"voice": VOICE_BANK["hi"]["male"][0], "gender": "male", "rate_pct": 0, "pitch_hz": 0, "volume_pct": 0,
            "age_group": "unknown", "slot": 99, "why": "fallback"}


# ---------------------------------------------------------------------------------------------------------------- expressive (LLM audio) voices
GPT_VOICES = {"female": ["coral", "nova", "shimmer", "sage"], "male": ["ash", "echo", "onyx", "verse", "fable", "alloy"]}
GPT_AGE_PREF = {("female", "child"): ["nova", "coral"], ("female", "teen"): ["coral", "nova"], ("female", "young_adult"): ["coral", "nova", "shimmer"],
                ("female", "adult"): ["shimmer", "sage", "coral"], ("female", "elder"): ["sage", "shimmer"],
                ("male", "child"): ["echo", "ash"], ("male", "teen"): ["ash", "echo", "verse"], ("male", "young_adult"): ["ash", "verse", "echo"],
                ("male", "adult"): ["onyx", "verse", "fable"], ("male", "elder"): ["onyx", "fable", "alloy"]}
GPT_NARRATOR = {"voice": "fable"}
AGE_WORDS = {"child": "a child of about 10", "teen": "a teenager of about 15", "young_adult": "a young person in their early twenties",
             "adult": "a middle-aged adult", "elder": "an elderly person of about 70 with a warm, slightly aged voice", "unknown": "a person"}
DIRECTION = {"neutral": "relaxed and natural", "happy": "bright, warm, smiling voice", "excited": "energetic, quick, bursting with excitement",
             "proud": "warm and steady, a swell of pride", "relieved": "a soft exhale, light and relieved", "affectionate": "gentle and tender",
             "amused": "light, a little laugh in the voice", "sad": "soft, slow, low energy, a catch in the voice", "anxious": "tense, a little hurried, worried",
             "scared": "breathy and slightly shaky, quick, trying not to cry", "angry": "sharp, firm and louder", "frustrated": "tight, exasperated, clipped",
             "surprised": "a sudden intake of breath, wide-eyed", "ashamed": "quiet, hesitant, looking down", "determined": "firm, resolute, steady",
             "jealous": "tight and a little bitter"}
STRENGTH = {1: "slightly", 2: "clearly", 3: "very strongly"}


def assign_gpt_voices(cast: Dict[str, dict]) -> Dict[str, dict]:
    """Give every character a distinct expressive voice of the right gender / age (distinct until the pool of that gender is used up)."""
    used = {"male": set(), "female": set()}
    for name, c in sorted(cast.items(), key=lambda kv: kv[1].get("slot", 0)):
        g = c["gender"] if c["gender"] in ("male", "female") else "male"
        prefs = GPT_AGE_PREF.get((g, c.get("age_group", "young_adult")), []) + GPT_VOICES[g]
        pick = next((v for v in prefs if v not in used[g]), prefs[0])
        used[g].add(pick)
        c["gpt_voice"] = pick
    return cast


# ---------------------------------------------------------------------------------------------------------------- Maya 2 Native voices
# 20 built-in voices (maya-research/maya2-native). Preference lists are ordered: the first voice not yet taken by another character of the same
# gender is used, so a cast sounds like different people. Heard and liked in the listening test: Charvi (teen girl), Satvik (young man),
# Siddhant (adult man); mature / warm voices for elders and adult women.
MAYA_PREF = {("female", "child"): ["Ahana", "Charvi", "Vrinda"], ("female", "teen"): ["Charvi", "Vrinda", "Ahana"],
             ("female", "young_adult"): ["Charvi", "Vrinda", "Ahana", "Jahnavi"], ("female", "adult"): ["Jahnavi", "Vaidehi", "Nivedita", "Tanushree", "Ishwari"],
             ("female", "elder"): ["Malini", "Vaidehi", "Jahnavi"],
             ("male", "child"): ["Nakul", "Satvik", "Shaunak"], ("male", "teen"): ["Nakul", "Satvik", "Shaunak"],
             ("male", "young_adult"): ["Satvik", "Nakul", "Shaunak", "Ritvik"], ("male", "adult"): ["Siddhant", "Abhay", "Akshat", "Ritvik", "Anirban"],
             ("male", "elder"): ["Vasant", "Anirban", "Siddhant"]}
MAYA_POOL = {"female": ["Charvi", "Vrinda", "Ahana", "Jahnavi", "Vaidehi", "Nivedita", "Tanushree", "Ishwari", "Malini", "Kayal"],
             "male": ["Satvik", "Nakul", "Shaunak", "Siddhant", "Abhay", "Akshat", "Ritvik", "Anirban", "Vasant", "Sreekar"]}
MAYA_NARRATOR = {"voice": "Anirban", "lang": "en"}          # scene headings are read in English
# inline emotion tags the model understands "best effort" (off by default: config.AUDIO_MAYA_TAGS)
MAYA_TAGS = {"excited": "[excited]", "happy": "[excited]", "sad": "[sad]", "scared": "[whispers]", "amused": "[laughs]"}


def assign_maya_voices(cast: Dict[str, dict]) -> Dict[str, dict]:
    """Give every character a distinct Maya voice of the right gender / age (distinct until the pool of that gender is used up)."""
    used = {"male": set(), "female": set()}
    for name, c in sorted(cast.items(), key=lambda kv: kv[1].get("slot", 0)):
        g = c["gender"] if c["gender"] in ("male", "female") else "male"
        prefs = MAYA_PREF.get((g, c.get("age_group", "young_adult")), []) + MAYA_POOL[g]
        pick = next((v for v in prefs if v not in used[g]), prefs[0])
        used[g].add(pick)
        c["maya_voice"] = pick
    return cast


def spare_maya_voice(cast: Dict[str, dict], taken, gender: str = "male") -> str:
    """A Maya voice for a speaker with no profile (shopkeeper, doctor ...): adult, and not the voice of anybody already in the cast."""
    busy = {c.get("maya_voice") for c in cast.values()} | set(taken)
    prefs = MAYA_PREF[(gender, "adult")] + MAYA_POOL[gender]
    return next((v for v in prefs if v not in busy), prefs[0])


def acting_instructions(name: str, profile: dict, base: dict, emotion: str, intensity: int, listener: str = "", relation: dict | None = None,
                        scene: str = "", text: str = "") -> str:
    age = AGE_WORDS.get(base.get("age_group", "unknown"), "a person")
    gender = "woman or girl" if base.get("gender") == "female" else "man or boy"
    traits = ", ".join((profile or {}).get("traits") or [])
    who = f"{name}, {age} ({gender})" + (f", {traits}" if traits else "")
    to = ""
    if listener:
        reg = (relation or {}).get("register", "casual")
        rel = (relation or {}).get("relation", "")
        to = f" Speaking to {listener}" + (f" ({rel})" if rel else "") + (", respectfully" if reg == "respectful" else ", casually") + "."
    where = f" Scene: {scene}." if scene else ""
    direction = DIRECTION.get(emotion, "natural")
    pace = "a little faster than normal" if emotion in ("excited", "scared", "anxious", "angry", "surprised") else (
        "slower than normal" if emotion in ("sad", "ashamed", "relieved", "affectionate") else "at a natural conversational pace")
    punct = " The line ends with an exclamation: show real energy." if text.rstrip().endswith("!") else (
        " The line is a question: let the pitch rise at the end." if "?" in text else "")
    return (f"You are a voice actor performing ONE line of a Hindi family film. Speak natural conversational Hindi, saying English words the Indian way. "
            f"Sound like a real person talking, not a narrator or a text reader: natural rhythm, breaths, tiny hesitations where they fit. "
            f"Character: {who}.{to}{where} They feel {STRENGTH.get(int(intensity), 'clearly')} {emotion}: {direction}. Speak {pace}.{punct} "
            f"Never add, drop or change a word.")
