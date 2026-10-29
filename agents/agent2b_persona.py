"""
agents/agent2b_persona.py   -  "Persona Agent": who each character IS and how they FEEL, scene by scene.

WHY: Agent 2 only knew a character's name and a keyword-based role. Its emotion/stress fields stayed "neutral / 0" because the
rules matched romanized-Hindi trigger words against Agent 1's English summaries. Gender (needed for correct Hindi verb forms,
for 'bhai' vs 'yaar', and later for voice casting), relationships (who is whose sister / teacher / rival -> 'aap' vs 'tum',
'Bhaiya') and a real emotional arc were all missing.

HOW (an agent that reasons + rules that verify it):
  1. ONE structured LLM call reads the whole story (Devanagari paragraphs when they align with the scenes, plus the romanized
     scene text) and returns   characters{gender, age_group, speech_style, traits}
                                relationships[{a, b, relation, a_to_b{address, register}, b_to_a{...}}]
                                emotions{scene_id: {character: {emotion, intensity 1-3}}}.
  2. Deterministic CHECKS on the answer (names matched to the real profiles, labels normalised to a fixed emotion set,
     intensities clamped, impossible values dropped).
  3. A rule-based SECOND OPINION: Hindi verb morphology votes on gender (boli/gayi/rahi vs bola/gaya/raha) and a bilingual
     emotion lexicon scores each scene. Strong morphology evidence overrides an LLM gender slip; the lexicon fills any scene the
     LLM left empty.
  4. If the LLM is unreachable the whole thing degrades to the rule-based persona (never to "all neutral").

The result is merged into the Agent-2 profiles (gender, age_group, speech_style, traits, relationship, emotion_arc,
current_emotion, stress_index) and written to the scratchpad as "persona". Valence/arousal numbers per emotion drive the
stress dynamics and (later) the TTS style.
"""
from __future__ import annotations
import json
import re
from typing import Dict, List, Optional

import config
from agents.name_utils import same_person, canonical_name

# ── emotion model ────────────────────────────────────────────────────────────────────────────────────────────────
# label -> (valence -1..1, arousal 0..1). Fixed vocabulary = comparable across scenes, usable by TTS and slang intensity.
EMOTIONS: Dict[str, tuple] = {
    "neutral": (0.0, 0.2), "happy": (0.7, 0.5), "excited": (0.8, 0.9), "proud": (0.8, 0.5), "relieved": (0.5, 0.3),
    "affectionate": (0.7, 0.3), "amused": (0.6, 0.6), "sad": (-0.7, 0.2), "anxious": (-0.5, 0.7), "scared": (-0.7, 0.9),
    "angry": (-0.7, 0.9), "frustrated": (-0.6, 0.7), "surprised": (0.1, 0.8), "ashamed": (-0.6, 0.4),
    "determined": (0.3, 0.7), "jealous": (-0.5, 0.6),
}
SYNONYMS = {"worried": "anxious", "nervous": "anxious", "tense": "anxious", "afraid": "scared", "fear": "scared",
            "fearful": "scared", "terrified": "scared", "joyful": "happy", "glad": "happy", "cheerful": "happy",
            "delighted": "happy", "thrilled": "excited", "enthusiastic": "excited", "eager": "excited", "upset": "sad",
            "heartbroken": "sad", "worry": "anxious", "mad": "angry", "furious": "angry", "irritated": "frustrated",
            "annoyed": "frustrated", "disappointed": "sad", "guilty": "ashamed", "embarrassed": "ashamed",
            "shocked": "surprised", "astonished": "surprised", "hopeful": "determined", "resolute": "determined",
            "loving": "affectionate", "caring": "affectionate", "emotional": "sad", "tearful": "sad", "grateful": "affectionate",
            "calm": "neutral", "serious": "neutral", "curious": "surprised", "confident": "determined", "anxiety": "anxious"}
SLANG_ADDRESS = {"yaar", "bhai", "bro", "boss", "dost", "bhaiya", "bhaiyya"} - {"bhaiya", "bhaiyya"}
SIBLING_ADDRESS = {"bhaiya", "bhaiyya", "didi", "bhai", "behen", "dida"}
AGE_GROUPS = ("child", "teen", "young_adult", "adult", "elder")
GENDERS = ("male", "female", "unknown")
STYLES = ("casual", "respectful", "formal")

# ── rule-based second opinion ───────────────────────────────────────────────────────────────────────────────────────
FEM_VERB = re.compile(r"\b(boli|bolti|kehti|kahi|gayi|aayi|ayi|thi|rahi|karti|dekhti|hui|hoon\s+aur\s+main\s+chali|chali|dauri|"
                      r"daudi|ro\s+padi|muskurayi|hansi|ghabrayi|sahmi|khadi|baithi|boli|puchhi|pooch\s+rahi|kar\s+rahi)\b")
MASC_VERB = re.compile(r"\b(bola|bolta|kehta|kaha|gaya|aaya|ayaa|tha|raha|karta|dekhta|hua|chala|dauda|muskuraya|hansa|ghabraya|"
                       r"khada|baitha|puchha|kar\s+raha|ro\s+pada)\b")

LEXICON = {
    "sad": ["roya", "roi", "rone", "udaas", "udas", "dukh", "dukhi", "aansu", "aasu", "dard", "bichad", "nam", "ro pad", "ro rah",
            "cries", "cry", "tears", "sad", "pain", "grief", "mourn", "weeping"],
    "anxious": ["dar ", "ghabra", "pareshan", "chinta", "tension", "fikar", "sehma", "kamp", "bechain", "worried", "anxious",
                "nervous", "afraid of", "fear that"],
    "scared": ["darr", "dara", "dare", "dar lag", "scared", "terrified", "frightened"],
    "angry": ["gussa", "naraaz", "chillaya", "krodh", "thappad", "angry", "furious", "shouts", "yells", "rage"],
    "happy": ["khush", "muskura", "hansa", "hansi", "khushi", "prasann", "aanand", "happy", "smiles", "laughs", "celebrate", "joy"],
    "excited": ["utsahit", "josh", "ujagar", "excited", "thrilled", "enthusiastic", "cheer", "dhamaal"],
    "proud": ["garv", "naaz", "gaurav", "proud", "pride"],
    "relieved": ["raahat", "rahat", "relieved", "relief", "saans li"],
    "surprised": ["hairan", "achanak", "chaunk", "surprised", "shocked", "suddenly"],
    "frustrated": ["jhunjhal", "jhunjh", "frustrat", "nirash", "naakaam", "bekaar", "annoyed", "irritat"],
    "ashamed": ["sharminda", "sharm", "pachtaa", "maafi", "maaf", "ashamed", "apologi", "regret", "guilt"],
    "determined": ["thaan", "sankalp", "hausla", "himmat", "determined", "resolve", "promise"],
    "jealous": ["jalan", "jealous", "envy", "irshya"],
}

PROMPT = """You are a character-psychology analyst for a screenplay adaptation. Read the story scenes below (each scene has the original
Hindi text when available, then a romanized version) and describe the CHARACTERS and how they FEEL.

Known characters (use EXACTLY these names): {names}

{scenes}

Return ONLY one JSON object (no markdown fences, no commentary) with exactly this shape:
{{
  "characters": {{ "<Name>": {{"gender": "male|female|unknown", "age_group": "child|teen|young_adult|adult|elder",
                              "speech_style": "casual|respectful|formal", "traits": ["2-3 short personality words"]}} }},
  "relationships": [ {{"a": "<Name>", "b": "<Name>", "relation": "e.g. younger sister of / grandfather of / teammate / rival / coach of",
                      "a_to_b": {{"address": "how a addresses b, e.g. Bhaiya / Dadaji / first name / Sir", "register": "casual|respectful"}},
                      "b_to_a": {{"address": "how b addresses a", "register": "casual|respectful"}}}} ],
  "emotions": {{ "<scene_id>": {{ "<Name>": {{"emotion": "one of: {emotions}", "intensity": 1}} }} }},
  "scene_cast": {{ "<scene_id>": {{"present": ["names physically IN the scene: they act or could speak there"],
                                  "mentioned_only": ["names that are only talked ABOUT, or are elsewhere in this scene"],
                                  "extra_speakers": ["Roman-letter role of an UNNAMED person who speaks in the scene, e.g. Dukaandaar"]}} }},
  "secrets": [ {{"what": "a secret / surprise / hidden plan stated in the story", "hidden_from": ["Names who must NOT know it"],
                "scenes": ["scene ids in which it is still a secret"]}} ]
}}
Rules:
- gender: infer from Hindi verb forms / pronouns / kinship (e.g. 'boli', 'gayi', 'rahi' = female; 'bola', 'gaya' = male).
- age_group: child, teen, young_adult, adult, elder. Grandparents are elder; teachers/coaches adult unless stated.
- speech_style: how THIS person normally speaks to others: casual (friends/siblings), respectful (to elders), formal (authority figures).
- relationships: list every pair of characters who interact. 'register' is how the speaker talks TO the other person.
- emotions: for EVERY scene list EVERY character present in that scene with the emotion they mainly feel there and an intensity
  1 (mild), 2 (clear), 3 (very strong). Use "neutral" only when the character is truly emotionless in that scene.
- Never invent characters that are not in the known list.
- scene_cast: a character is "present" ONLY if the text shows them in that scene's place. Someone who is merely talked about (the person a
  surprise is being planned for), or who is away (gone to the temple, at work), is "mentioned_only" and must not appear in "present".
  Unnamed people who speak (a shopkeeper, a doctor) go in "extra_speakers".
- secrets: only genuine secrets or surprises that the story itself sets up (e.g. a birthday surprise kept from the birthday person). Use []
  when there are none. Never invent one.
"""


def _norm_emotion(label) -> str:
    s = str(label or "").strip().lower()
    s = SYNONYMS.get(s, s)
    return s if s in EMOTIONS else "neutral"


def _clamp_intensity(x) -> int:
    try:
        return max(1, min(3, int(round(float(x)))))
    except Exception:
        return 1


def extract_json(raw: str) -> Optional[dict]:
    """Tolerant JSON extraction: fences, leading/trailing prose, trailing commas."""
    if not raw:
        return None
    txt = re.sub(r"```(?:json)?", "", raw)
    a, b = txt.find("{"), txt.rfind("}")
    if a < 0 or b <= a:
        return None
    blob = txt[a:b + 1]
    for attempt in (blob, re.sub(r",\s*([}\]])", r"\1", blob)):
        try:
            d = json.loads(attempt)
            return d if isinstance(d, dict) else None
        except Exception:
            continue
    return None


class PersonaAgent:
    def __init__(self, llm_client=None):
        self.client = llm_client
        if self.client is None and getattr(config, "PERSONA_USE_LLM", True):
            try:
                from llm_client import make_client
                self.client = make_client("extract")
            except Exception:
                self.client = None
        self.last_source = "none"

    # ─────────────────────────────────────────────────────────────────────────────────────── public
    def analyze(self, scene_chunks: List[dict], event_chain: List[dict], profile_names: List[str],
                devanagari_paragraphs: Optional[List[str]] = None) -> dict:
        """-> persona dict {characters, relationships, emotions, source, checks}. Never raises."""
        names = self._known_names(profile_names, event_chain)
        scene_ids = [c["scene_id"] for c in scene_chunks]
        present = {e["scene_id"]: [n for n in (e.get("characters") or [])] for e in event_chain}
        llm = None
        if self.client is not None and names:
            llm = self._ask(names, scene_chunks, devanagari_paragraphs)
        persona = self._merge(llm, names, scene_chunks, event_chain, scene_ids, present)
        persona["source"] = self.last_source
        return persona

    # ─────────────────────────────────────────────────────────────────────────────────────── LLM call
    def _known_names(self, profile_names, event_chain):
        out, seen = [], set()
        for n in list(profile_names) + [c for e in event_chain for c in (e.get("characters") or [])]:
            if isinstance(n, str) and n.strip() and n.strip().lower() not in seen:
                seen.add(n.strip().lower())
                out.append(n.strip())
        return out

    def _ask(self, names, scene_chunks, deva):
        use_deva = bool(deva) and len(deva) == len(scene_chunks)
        blocks = []
        for i, ch in enumerate(scene_chunks):
            part = f"[{ch['scene_id']}]\n"
            if use_deva:
                part += "Hindi: " + deva[i].strip()[:900] + "\n"
            part += "Romanized: " + ch["text"].strip()[:900]
            blocks.append(part)
        prompt = PROMPT.format(names=", ".join(names), scenes="\n\n".join(blocks), emotions=", ".join(EMOTIONS))
        for attempt in range(2):
            try:
                resp = self.client.chat.completions.create(
                    model=config.GROQ_MODEL, messages=[{"role": "user", "content": prompt}],
                    temperature=0.0 if attempt == 0 else 0.3, max_tokens=4500)
                d = extract_json(resp.choices[0].message.content or "")
                if d and isinstance(d.get("characters"), dict):
                    self.last_source = "llm+rules"
                    return d
            except Exception as e:                           # network / quota / bad request
                print(f"  [persona] LLM call failed ({str(e)[:90]})")
        self.last_source = "rules-only"
        return None

    # ─────────────────────────────────────────────────────────────────────────────────────── checks + second opinion
    def _match(self, name: str, names: List[str]) -> Optional[str]:
        if not isinstance(name, str):
            return None
        for n in names:
            if n.lower() == name.strip().lower():
                return n
        for n in names:
            if same_person(n, name):
                return n
        return None

    # MAIN verbs only: auxiliaries (tha/thi/raha/rahi/hua/hui) agree with whatever the clause is about ("Ananya ka dil dhadak raha tha" is about the
    # heart, not about Ananya), and a verb after ne/ko/ka/ki/ke/se/mein/par does not agree with the name either.
    FEM_TOK = {"boli", "bolti", "kehti", "kahi", "gayi", "aayi", "ayi", "karti", "dekhti", "chali", "dauri", "daudi", "muskurayi", "hansi",
               "ghabrayi", "sahmi", "khadi", "baithi", "puchhi", "lauti", "chillayi", "ruki", "rukin"}
    MASC_TOK = {"bola", "bolta", "kehta", "gaya", "aaya", "karta", "dekhta", "chala", "dauda", "muskuraya", "hansa", "ghabraya", "khada", "baitha",
                "puchha", "lauta", "chillaya", "ruka"}
    STOP_TOK = {"ka", "ki", "ke", "ko", "ne", "se", "mein", "men", "par", "tak", "liye"}

    def _gender_votes(self, name: str, scene_chunks, all_names: Optional[List[str]] = None) -> tuple:
        """Feminine vs masculine MAIN verbs that follow the character's name within the same clause (<= 5 words; the window ends at punctuation,
        at the next character's name, or at a postposition)."""
        f = m = 0
        key = (canonical_name(name).split() or [name.lower()])[0]
        others = {(canonical_name(n).split() or [n.lower()])[0] for n in (all_names or []) if n != name} - {key}
        for ch in scene_chunks:
            toks = re.findall(r"[a-z]+|[.!?,;]", ch["text"].lower())
            for i, t in enumerate(toks):
                if t != key:
                    continue
                for w in toks[i + 1:i + 6]:
                    if w in ".!?,;" or w in others or w in self.STOP_TOK:
                        break
                    f += w in self.FEM_TOK
                    m += w in self.MASC_TOK
        return f, m

    def _lexicon_emotion(self, text: str, key_events: List[str], names: List[str]) -> Dict[str, tuple]:
        """Scene-level fallback: the strongest-scoring emotion in the scene text + events, applied to every present character."""
        blob = (text + " " + " ".join(key_events or [])).lower()
        score = {}
        for emo, words in LEXICON.items():
            s = sum(blob.count(w) for w in words)
            if s:
                score[emo] = s
        if not score:
            return {}
        emo = max(score, key=score.get)
        inten = 3 if score[emo] >= 4 else (2 if score[emo] >= 2 else 1)
        return {n: (emo, inten) for n in names}

    def _merge(self, llm, names, scene_chunks, event_chain, scene_ids, present) -> dict:
        checks = {"llm_ok": bool(llm), "unknown_names_dropped": 0, "labels_normalised": 0, "gender_overrides": [],
                  "emotion_filled_by_rules": 0, "fallback_gender": 0}
        characters, relationships, emotions = {}, [], {sid: {} for sid in scene_ids}
        raw_chars = (llm or {}).get("characters") or {}
        for n in names:
            rec = None
            for k, v in raw_chars.items():
                if self._match(k, [n]) and isinstance(v, dict):
                    rec = v
                    break
            rec = rec or {}
            g = str(rec.get("gender", "unknown")).lower()
            g = g if g in ("male", "female") else "unknown"
            f, m = self._gender_votes(n, scene_chunks, names)
            strong = (f >= 3 and m == 0) or (m >= 3 and f == 0)               # the LLM reads the Hindi itself: only unanimous evidence overrides it
            vote = "female" if f > m else ("male" if m > f else "unknown")
            if g == "unknown" and vote != "unknown":
                g = vote
                checks["fallback_gender"] += 1
            elif strong and vote != g:
                checks["gender_overrides"].append({"name": n, "llm": g, "rules": vote, "votes": [f, m]})
                g = vote
            age = str(rec.get("age_group", "")).lower()
            age = age if age in AGE_GROUPS else "unknown"
            style = str(rec.get("speech_style", "")).lower()
            style = style if style in STYLES else "casual"
            traits = [str(t)[:20] for t in (rec.get("traits") or []) if isinstance(t, str)][:3]
            characters[n] = {"gender": g, "age_group": age, "speech_style": style, "traits": traits,
                             "gender_votes": [f, m]}
        # relationships
        seen_pairs = set()
        for r in (llm or {}).get("relationships") or []:
            if not isinstance(r, dict):
                continue
            a, b = self._match(r.get("a"), names), self._match(r.get("b"), names)
            if not a or not b or a == b or (a, b) in seen_pairs:
                checks["unknown_names_dropped"] += (not a or not b)
                continue
            seen_pairs.add((a, b))

            def side(x):
                x = x if isinstance(x, dict) else {}
                reg = str(x.get("register", "casual")).lower()
                return {"address": str(x.get("address", ""))[:30], "register": reg if reg in ("casual", "respectful") else "casual"}
            rel = {"a": a, "b": b, "relation": str(r.get("relation", ""))[:60],
                   "a_to_b": side(r.get("a_to_b")), "b_to_a": side(r.get("b_to_a"))}
            sibling = bool(re.search(r"brother|sister|sibling|cousin|bhai|behen|didi", rel["relation"].lower()))
            for key, other in (("a_to_b", b), ("b_to_a", a)):
                addr = rel[key]["address"].strip().lower().rstrip("!,.")
                # slang-like address words would leak slang into the NEUTRAL pass; sibling words only fit real siblings
                if addr in SLANG_ADDRESS or (addr in SIBLING_ADDRESS and not sibling):
                    rel[key]["address"] = other.split()[0]
            relationships.append(rel)
        # emotions
        raw_em = (llm or {}).get("emotions") or {}
        for sid in scene_ids:
            per = raw_em.get(sid) if isinstance(raw_em.get(sid), dict) else {}
            for k, v in per.items():
                n = self._match(k, names)
                if not n:
                    checks["unknown_names_dropped"] += 1
                    continue
                v = v if isinstance(v, dict) else {"emotion": v}
                lab = _norm_emotion(v.get("emotion"))
                if lab != str(v.get("emotion", "")).strip().lower():
                    checks["labels_normalised"] += 1
                emotions[sid][n] = {"emotion": lab, "intensity": _clamp_intensity(v.get("intensity", 1))}
        for ev, ch in zip(event_chain, scene_chunks):
            sid = ev["scene_id"]
            chars = [self._match(c, names) or c for c in (ev.get("characters") or [])]
            missing = [c for c in chars if c in names and c not in emotions[sid]]
            if missing:
                lex = self._lexicon_emotion(ch["text"], ev.get("key_events"), missing)
                for n, (emo, inten) in lex.items():
                    emotions[sid][n] = {"emotion": emo, "intensity": inten, "source": "lexicon"}
                    checks["emotion_filled_by_rules"] += 1
                for n in missing:
                    emotions[sid].setdefault(n, {"emotion": "neutral", "intensity": 1, "source": "default"})
        # who is actually IN each scene (vs only mentioned) and what must stay secret from whom: the story bible Agent 3 and the critic obey
        raw_sc = (llm or {}).get("scene_cast") if isinstance((llm or {}).get("scene_cast"), dict) else {}
        scene_cast = {}
        for ev in event_chain:
            sid = ev["scene_id"]
            rec = raw_sc.get(sid) if isinstance(raw_sc.get(sid), dict) else {}
            pres = [m for m in (self._match(x, names) for x in (rec.get("present") or [])) if m]
            ment = [m for m in (self._match(x, names) for x in (rec.get("mentioned_only") or [])) if m and m not in pres]
            extras = [re.sub(r"[^A-Za-z .'-]", "", str(x)).strip()[:24] for x in (rec.get("extra_speakers") or []) if isinstance(x, str)]
            extras = [x for x in extras if len(x) >= 3][:3]
            fallback = False
            if not pres:                                                   # LLM silent / failed: Agent 1's list, as before
                pres = [self._match(c, names) or c for c in (ev.get("characters") or [])]
                fallback = True
            scene_cast[sid] = {"present": list(dict.fromkeys(pres)), "mentioned_only": ment, "extra_speakers": extras, "fallback": fallback}
        secrets = []
        for sc in (llm or {}).get("secrets") or []:
            if not isinstance(sc, dict):
                continue
            hidden = [m for m in (self._match(x, names) for x in (sc.get("hidden_from") or [])) if m]
            what = str(sc.get("what", "")).strip()[:180]
            if what and hidden:
                scs = [s for s in (sc.get("scenes") or []) if s in scene_ids] or list(scene_ids)
                secrets.append({"what": what, "hidden_from": hidden, "scenes": scs})
        checks["secrets"] = len(secrets)
        checks["scene_cast_fallback"] = sum(1 for v in scene_cast.values() if v["fallback"])
        return {"characters": characters, "relationships": relationships, "emotions": emotions, "scene_cast": scene_cast,
                "secrets": secrets, "checks": checks}


# ── applying the persona to Agent-2 profiles ───────────────────────────────────────────────────────────────────────
def apply_persona(profiles: dict, persona: dict, scene_ids: List[str]) -> dict:
    """Merge persona fields into the profiles IN PLACE and run the valence/arousal stress dynamics. Returns profiles."""
    chars = persona.get("characters") or {}
    rels = persona.get("relationships") or []
    for pname, prof in profiles.items():
        key = next((n for n in chars if n.lower() == pname.lower() or same_person(n, pname)), None)
        if key:
            c = chars[key]
            prof["gender"] = c["gender"]
            prof["age_group"] = c["age_group"]
            prof["speech_style"] = c["speech_style"]
            prof["traits"] = c["traits"]
            if c["age_group"] == "elder" and prof.get("social_role") == "peer":
                prof["social_role"] = "elder"
                prof["slang_level"] = 0
        prof.setdefault("gender", "unknown")
        prof["relationship"] = {}
        prof["emotion_arc"] = []
        stress = 0.0
        for sid in scene_ids:
            per = (persona.get("emotions") or {}).get(sid, {})
            k2 = next((n for n in per if n.lower() == pname.lower() or same_person(n, pname)), None)
            if not k2:
                continue
            e = per[k2]
            val, aro = EMOTIONS.get(e["emotion"], (0.0, 0.2))
            inten = e["intensity"]
            # stress rises with negative-valence, high-arousal emotion; positive emotion relieves it
            delta = (-val) * aro * (0.35 + 0.2 * inten)
            stress = max(0.0, min(1.0, stress * 0.8 + max(delta, 0) - max(-delta, 0) * 0.6))
            prof["emotion_arc"].append({"scene_id": sid, "emotion": e["emotion"], "intensity": inten,
                                        "valence": val, "arousal": aro, "stress": round(stress, 3)})
        if prof["emotion_arc"]:
            last = prof["emotion_arc"][-1]
            prof["current_emotion"] = last["emotion"]
            prof["stress_index"] = last["stress"]
            prof["turn_count"] = len(prof["emotion_arc"])
    for r in rels:
        for me, other, side, rev in ((r["a"], r["b"], r["a_to_b"], r["b_to_a"]), (r["b"], r["a"], r["b_to_a"], r["a_to_b"])):
            pk = next((k for k in profiles if k.lower() == me.lower() or same_person(k, me)), None)
            if pk:
                profiles[pk]["relationship"][other] = {"relation": r["relation"], "i_address_them": side["address"],
                                                       "register": side["register"], "they_address_me": rev["address"]}
    return profiles


def emotion_at(profile: dict, scene_id: str) -> dict:
    """{'emotion','intensity'} the profile has in this scene (neutral/1 if unknown)."""
    for e in profile.get("emotion_arc") or []:
        if e["scene_id"] == scene_id:
            return {"emotion": e["emotion"], "intensity": e["intensity"]}
    return {"emotion": profile.get("current_emotion", "neutral"), "intensity": 1}
