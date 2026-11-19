"""
tools/tag_slang_register.py  -  give every phrase of data/slang/slang_pool.json a `register`:

    urban_youth   modern urban / internet / campus slang and English-origin phrases ("dope", "cringe", "full tension", "red flag", "next level").
                  They are believable in a contemporary, casual story and jarring in a village, historical or otherwise traditional one.
    general       Hindi / Urdu idioms and everyday evaluative words ("jaan nikal gayi", "dil baith raha hai", "paara chadh gaya", "bekaar",
                  "badhiya", "sone pe suhaga"). They fit any setting.

The tag is DATA (a field of the pool), decided by two rules so a new phrase is tagged the same way: (1) any word of English origin makes it urban_youth;
(2) a short list of modern Hindi youth coinages (Mumbai / Delhi / campus: "pakau", "bhaukaal", "jhakaas", ...) is urban_youth too. Everything else is general.
Run again after adding phrases:  python -m tools.tag_slang_register
"""
from __future__ import annotations
import json
import os
import re

POOL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "slang", "slang_pool.json")

ENGLISH_WORDS = {
    "solid", "top", "class", "point", "on", "flop", "sorted", "cringe", "mid", "trash", "dead", "red", "flag", "full", "tension", "fry", "chill",
    "sweat", "mind", "blowing", "mind-blowing", "killer", "next", "level", "lit", "dope", "epic", "awesome", "sick", "legendary", "fire", "cool",
    "op", "banger", "classy", "insane", "god", "god-level", "steal", "wholesome", "scene", "headache", "stress", "anxiety", "panic", "nervous",
    "scary", "bore", "pathetic", "b-grade", "grade", "cakewalk", "smooth", "number", "one", "bro", "boss", "chilling", "vibe", "mood", "savage",
    "legend", "king", "queen", "rock", "rocks", "hit", "super", "superb", "perfect", "best", "worst", "waste", "nonsense", "rubbish", "junk",
}
# modern Hindi youth coinages (Mumbai / Delhi / campus / social-media Hindi), not English but not believable in a period or village story
MODERN_HINDI = {"pakau", "bhaukaal", "jhakaas", "jhakkas", "faadu", "tagda", "dhansu", "dhaansu", "dhasu", "rapchik", "chhaa gaya", "kadak",
                "ek dum kadak", "vella", "dhakkan", "sadela", "phisaddi", "dabba", "kabaad", "jhand", "hawa tight ho gayi", "scene kharab",
                "pataka", "jabra", "dhamakedaar", "bhari"}


# Praise ADJECTIVES describe a THING ("yeh idea lajawaab hai", "mujhe yeh zabardast laga"). In the experiencer frame "mujhe (bahut) ___ lag raha hai" the
# adjective has to name a FEELING, so only the feel-good words fit there ("mujhe mast lag raha hai"); "mujhe bahut lajawaab lag raha hai" is not Hindi.
FEEL_GOOD = {"mast", "ekdum mast", "bindaas", "badiya", "cool", "awesome", "sahi hai"}


# Predicate-noun idioms ("yeh bacchon ka khel HAI", "sone pe suhaga HAI") need a linking verb after them; they cannot replace an adverb ("aasaani se milega" ->
# "*bacchon ka khel milega").
NEEDS_COPULA = {"bacchon ka khel", "left hand ka khel", "sone pe suhaga", "paisa vasool"}


def experiencer_ok(entry: dict) -> bool:
    if entry.get("kind") == "slang_word" and "praise" in (entry.get("slots") or []):
        return entry["phrase"].lower() in FEEL_GOOD
    return True


def register_of(phrase: str) -> str:
    p = phrase.lower().strip()
    if p in MODERN_HINDI:
        return "urban_youth"
    words = set(re.findall(r"[a-z][a-z-]*", p))
    if words & ENGLISH_WORDS:
        return "urban_youth"
    return "general"


def main():
    d = json.load(open(POOL, encoding="utf-8"))
    counts = {"urban_youth": 0, "general": 0}
    for e in d["entries"]:
        e["register"] = register_of(e["phrase"])
        counts[e["register"]] += 1
        if experiencer_ok(e):
            e.pop("experiencer_ok", None)
        else:
            e["experiencer_ok"] = False
        if e["phrase"].lower() in NEEDS_COPULA:
            e["needs_copula"] = True
        else:
            e.pop("needs_copula", None)
    json.dump(d, open(POOL, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(counts)
    print("urban_youth:", sorted(e["phrase"] for e in d["entries"] if e["register"] == "urban_youth"))
    print("general:", sorted(e["phrase"] for e in d["entries"] if e["register"] == "general"))


if __name__ == "__main__":
    main()
