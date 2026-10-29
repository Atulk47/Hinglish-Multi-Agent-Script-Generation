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
Main speaker: {speaker_name} (social role: {speaker_role}, current emotion: {emotion})
Listener: {listener_name}
Key events to cover in dialogue: {key_events}
{cast_block}{source_block}
Rules:
- Write in romanized Hinglish (mix of Hindi words in Roman script + English)
- Use ONLY Roman letters. NEVER write Devanagari (e.g. write "dhanyavaad", not "धन्यवाद")
- Neutral register — no slang, no fillers, appropriate for {speaker_role}
- NEVER use any of these words: {banned_words}
- Every dialogue line must be spoken by a character of the story; never use "Narrator" as a speaker
- STAY ON THE STORY: use only the people, places, objects and events of the original scene. Do NOT invent new plot events, decisions, secrets
  being revealed, new characters or outcomes. You may add small natural touches (greetings, reactions, short connecting sentences) but nothing
  that changes what happens or who knows what.
- Cover the key events naturally through dialogue
- Output ONLY the dialogue line(s), one per speaker turn
- Format: SPEAKER_NAME: dialogue text
- {line_limit}
- No scene directions, no narration, no explanation{style_rules}"""

# Optional style for scenes whose main speaker is a young PEER (config.AGENT3_STYLE = "slotrich"). It keeps the dialogue
# NEUTRAL (no slang, no fillers) but makes it everyday spoken Hinglish with plain feeling/opinion words. Those plain
# words are exactly what the slang editor can naturally replace ("bahut accha" -> a fitting slang), so it creates
# natural openings for varied slang instead of stiff textbook sentences that offer none.
# Slang / fillers the editor is supposed to add LATER. Pass 1 must not contain them (a stronger model like Haiku
# tends to slip "kamaal", "arre" ... into the "neutral" dialogue, which ruins the neutral-vs-slang comparison).
BANNED_WORDS = ["yaar", "bhai", "bro", "boss", "arre", "abe", "abey", "chill", "kamaal", "zabardast", "mast", "solid",
                "kadak", "lajawaab", "jhakaas", "gazab", "faadu", "dhamaal", "fire", "lit", "banger", "makkhan",
                "dhasu", "jalwa", "bawaal", "panga", "locha", "lafda", "ghapla", "jhol", "jhand", "kachra", "flop",
                "sorted"]


def _leaks(text: str):
    import re as _re
    toks = set(_re.findall(r"[a-z]+", text.lower()))
    return sorted(toks & set(BANNED_WORDS))


PEER_STYLE_RULES = """
- Peer characters (friends, classmates) talk like real young people: SHORT everyday sentences (6-14 words), simple common words, not textbook or Sanskritised Hindi (say "matlab" not "maksad", "tareef" not "prashansa", "zaroori" not "avashyak"). Elders and teachers still speak respectfully.
- State feelings and opinions plainly with ordinary words (bahut accha, kharab, bura lag raha hai, chinta, problem, aasaani se, pasand, khush) - still NO slang and NO fillers. Whenever the event involves liking or disliking something, worry, or trouble, include at least one such plain word in the peer's line. In every scene at least two lines from the young characters must say plainly how they feel or what they think of something (e.g. "bahut accha laga", "yeh bahut kharab hai", "mujhe bahut chinta ho rahi hai", "yeh badi problem hai") instead of only stating facts.
- React directly to what just happened instead of describing it formally."""


class DialogueSynthesizer:
    def __init__(self, llm_backend: str = "groq"):
        self.backend = llm_backend
        load_dotenv()
        from llm_client import make_client             # OpenRouter first, Groq fallback
        self.groq_client = make_client("dialogue")
        print("DialogueSynthesizer initialized (OpenRouter-first LLM client).")

    def generate_pass1(self, scene: dict, character_profiles: dict,
                        event_chain_entry: dict) -> str:
        characters = list(event_chain_entry.get("characters", []))
        # Story bible (Persona Agent): only characters who are really IN the scene speak; mentioned-only / absent ones never do
        cast_info = event_chain_entry.get("scene_cast") or {}
        absent = list(cast_info.get("mentioned_only") or [])
        extras = list(cast_info.get("extra_speakers") or [])
        if cast_info.get("present"):
            characters = list(cast_info["present"])
        speaker = characters[0] if characters else "SPEAKER"
        listener = characters[1] if len(characters) > 1 else "narrator"

        from agents.name_utils import match_profile
        profile = match_profile(character_profiles, speaker)

        rich = bool(getattr(config, "AGENT3_RICH", True))
        cast_block, source_block = "", ""
        if rich:
            from agents.name_utils import match_profile as _mp
            from agents.agent2b_persona import emotion_at
            sid = event_chain_entry.get("scene_id")
            cast, rel_lines = [], []
            for c in characters:
                pr = _mp(character_profiles, c) if character_profiles else {}
                em = emotion_at(pr, sid) if pr else {"emotion": "neutral", "intensity": 1}
                g = pr.get("gender", "unknown")
                bits = [pr.get("social_role", "peer")] + ([g] if g in ("male", "female") else []) + [f"feels {em['emotion']}"]
                cast.append(f"{c} ({', '.join(bits)})")
                for other, rel in (pr.get("relationship") or {}).items():
                    if any(other.lower() == x.lower() for x in characters):
                        rel_lines.append(f"{c} to {other}: {rel.get('relation', '')}; calls them '{rel.get('i_address_them', '')}', "
                                         f"speaks {rel.get('register', 'casual')}ly")
            if cast:
                cast_block = ("Characters present in this scene: " + "; ".join(cast) +
                              ". Every character who SPEAKS in the original text below must get a line, in their own voice; "
                              "elders and teachers/coaches speak respectfully, friends speak casually. Use the correct Hindi verb "
                              "forms for each speaker's gender (a woman says 'main aati hoon', 'maine dekha', a man says 'main aata hoon'). "
                              "Let each character's feeling show in their line.\n")
                if rel_lines:
                    cast_block += "Relationships (use these address terms and this level of respect): " + " | ".join(rel_lines[:8]) + "\n"
            if extras:
                cast_block += ("Unnamed people who speak in the original text (give them a line, label them with this role): "
                               + ", ".join(extras) + ".\n")
            if absent:
                cast_block += ("NOT PRESENT in this scene (they are only mentioned or are elsewhere): " + ", ".join(absent) +
                               ". They must NOT speak and nobody may speak TO them or act as if they are there.\n")
            for sc in (event_chain_entry.get("secrets") or []):
                cast_block += (f"SECRET that must stay hidden: {sc['what']} - it is hidden from {', '.join(sc['hidden_from'])}. "
                               f"Those people must not learn it, hint at it or be told; it may be discussed only among others who know.\n")
            if event_chain_entry.get("feedback"):
                cast_block += ("A REVIEWER REJECTED YOUR PREVIOUS ATTEMPT for these problems - fix every one of them: "
                               + event_chain_entry["feedback"] + "\n")
            deva = (scene or {}).get("deva_text") or ""
            src = (scene or {}).get("text") or ""
            if deva.strip():
                source_block = ("ORIGINAL SCENE (Hindi, exactly as written by the author). Keep its facts, its order of events and its "
                                "spoken lines; turn narration into what people would actually SAY to each other, do not have a "
                                "character narrate facts to the listener:\n" + deva.strip()[:2000] + "\n")
            elif src.strip():
                source_block = ("Original scene text (auto-romanized, so spellings can be off: koch=coach, tim=team, kap=cup, "
                                "maich=match, skul=school). Keep the facts and the spoken lines of this text; turn narration into "
                                "what people would actually SAY to each other, do not have a character narrate facts to the listener:\n"
                                + src.strip()[:1800] + "\n")
        n_lines = int(getattr(config, "AGENT3_MAX_LINES", 6)) if rich else 3
        prompt = PASS1_PROMPT_TEMPLATE.format(
            cast_block=cast_block, source_block=source_block,
            line_limit=f"Maximum {n_lines} dialogue lines total, a natural back-and-forth between the characters",
            scene_goal=event_chain_entry.get("scene_goal", ""),
            location=event_chain_entry.get("location", "unknown"),
            narrative_link=event_chain_entry.get("narrative_link", ""),
            speaker_name=speaker,
            speaker_role=profile.get("social_role", "peer"),
            emotion=(__import__("agents.agent2b_persona", fromlist=["emotion_at"]).emotion_at(profile, event_chain_entry.get("scene_id"))["emotion"]
                     if profile else "neutral"),
            listener_name=listener,
            key_events="; ".join(event_chain_entry.get("key_events", [])),
            banned_words=", ".join(BANNED_WORDS),
            style_rules=(PEER_STYLE_RULES
                         if (getattr(config, "AGENT3_STYLE", "default") == "slotrich"
                             and str(profile.get("social_role", "peer")) == "peer") else "")
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.groq_client.chat.completions.create(
                    model=config.GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=900 if rich else 500,
                    reasoning_effort=getattr(config, "GROQ_REASONING_EFFORT", "low"),
                )
                result = response.choices[0].message.content.strip()

                # keep Pass 1 neutral: if banned slang/filler words slipped in, ask again (max 2 times)
                for _ in range(2):
                    leaks = _leaks(result)
                    if not leaks:
                        break
                    r2 = self.groq_client.chat.completions.create(
                        model=config.GROQ_MODEL, temperature=0.3, max_tokens=500,
                        messages=[{"role": "user", "content": prompt + "\n\nIMPORTANT: your previous answer used banned words ("
                                   + ", ".join(leaks) + "). Write the dialogue again WITHOUT any banned word."}])
                    again = (r2.choices[0].message.content or "").strip()
                    if again:
                        result = again

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