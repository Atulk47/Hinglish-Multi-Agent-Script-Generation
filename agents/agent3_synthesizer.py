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
- Write in romanized Hinglish (mix of Hindi words in Roman script + English) the way people really TALK: short spoken sentences, everyday words, English words where Indians say them (feedback, deadline, remote, plan), not written or textbook Hindi. A character speaks TO the others, never describes the scene to an audience
- Use ONLY Roman letters. NEVER write Devanagari (e.g. write "dhanyavaad", not "धन्यवाद")
- Neutral register — no slang, no fillers, appropriate for {speaker_role}
- NEVER use any of these words: {banned_words}
- Every dialogue line must be spoken by a character of the story. NEVER write a line labelled Narrator / Narration / Voice / Author: narration is
  either turned into what the characters themselves say, or left out. Never write notes, corrections or explanations to yourself.
- A gesture, expression or action in the original (nodding, smiling, hugging, crying, looking away) is something the character DOES, not something
  they say. Never write the character announcing their own gesture or manner ("I say yes happily", "I nod", "I smile", "main khushi se haan karta hoon"):
  write what a person would actually say ("Haan, ...", "Theek hai Amma") or nothing at all.
- STAY ON THE STORY: use only the people, places, objects and events of the original scene. Do NOT invent new plot events, decisions, secrets
  being revealed, new characters or outcomes. Beyond what the original says, add at most a brief greeting or a one-word acknowledgement ("Haan",
  "Theek hai") where two people answer each other: do NOT add thanks, promises, opinions, jokes, questions, agreements or explanations the original
  does not contain. In particular do NOT state any place, destination, journey, sum of money, term of a deal or family relationship that the
  original scene does not state.
- DESCRIPTION IS NOT SPEECH: background about a place, a time, a person's habits or a situation ("the village had a market", "he always lent money",
  "a long queue stood outside") is said by nobody. Turn only what HAPPENS between the characters (a request, an answer, a decision, a reaction the
  original describes) into dialogue. EVERY line the original quotes must still be said, and every key event needs its line; what you must not do is add
  lines beyond those. A scene that only describes may get one or two lines or none.
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


def _leaks(text: str, source: str = ""):
    """Banned slang / filler words in the neutral pass. A word the STORY itself contains ("super bro" is in the original) is not a leak: it is the author's word."""
    import re as _re
    toks = set(_re.findall(r"[a-z]+", text.lower()))
    return sorted((toks & set(BANNED_WORDS)) - set(_re.findall(r"[a-z]+", (source or "").lower())))


PEER_STYLE_RULES = """
- Peer characters (friends, classmates) talk like real young people: SHORT everyday sentences (6-14 words), simple common words, not textbook or Sanskritised Hindi (say "matlab" not "maksad", "tareef" not "prashansa", "zaroori" not "avashyak"). Elders and teachers still speak respectfully.
- Where the ORIGINAL shows a feeling or an opinion (a feeling word, an exclamation, a reaction, "khush hua", "chinta thi"), say it plainly with an ordinary word (bahut accha, kharab, bura lag raha hai, chinta, pasand, khush) - still NO slang and NO fillers.
- Do NOT add a feeling or an opinion the original does not show. A line that only reports a fact, a plan or an action stays a fact, a plan or an action. Never tack "mujhe bahut accha laga" / "yeh bahut accha hai" onto a line to make it sound warm; use such a phrase only when the original says the person liked it.
- React directly to what just happened instead of describing it formally."""


# The ORIGINAL (pre-2026-11-18) rules, kept behind AGENT3_STYLE="slotrich_pad": they REQUIRE at least two plainly stated feelings per scene, which gives the slang editor many
# openings (slang on about twice as many lines) but also pads lines with "mujhe bahut accha laga" the story never says (judged as invented). Fidelity vs slang coverage: see docs/BASELINE.md.
PEER_STYLE_RULES_PAD = """
- Peer characters (friends, classmates) talk like real young people: SHORT everyday sentences (6-14 words), simple common words, not textbook or Sanskritised Hindi (say "matlab" not "maksad", "tareef" not "prashansa", "zaroori" not "avashyak"). Elders and teachers still speak respectfully.
- State feelings and opinions plainly with ordinary words (bahut accha, kharab, bura lag raha hai, chinta, problem, aasaani se, pasand, khush) - still NO slang and NO fillers. Whenever the event involves liking or disliking something, worry, or trouble, include at least one such plain word in the peer's line. In every scene at least two lines from the young characters must say plainly how they feel or what they think of something (e.g. "bahut accha laga", "yeh bahut kharab hai", "mujhe bahut chinta ho rahi hai", "yeh badi problem hai") instead of only stating facts.
- Put the feeling lines in the talk between peers (friend to friend, sibling to sibling), not in a line to an elder, a teacher or a shopkeeper: lines to them stay plain and respectful.
- React directly to what just happened instead of describing it formally."""


# Words that appear in English prose but (almost) never in romanized Hinglish dialogue. The model sometimes appends a note to itself
# ("Narrator's listener is not a character in the story, so the scene's sole speaker is Arav ...") that would otherwise be spoken and subtitled.
_EN_PROSE = {"the", "is", "are", "was", "were", "not", "of", "an", "so", "who", "whom", "which", "that", "this", "as", "his", "her", "their",
             "they", "sole", "character", "characters", "story", "narrator", "narrator's", "listener", "scene", "scene's", "speaker", "speakers",
             "dialogue", "note", "because", "therefore", "addresses", "himself", "herself"}
_NOTE_START = re.compile(r"^\s*[\(\[]?\s*(note|nb|comment|explanation|assistant)\b\s*[:\)\]-]", re.I)


def _is_english_meta(text: str) -> bool:
    """True for a stretch of English prose / a self-note (>= 6 words, at least 3 prose-only words making up >= 20% of it)."""
    if _NOTE_START.match(text):
        return True
    w = re.findall(r"[A-Za-z']+", text.lower())
    if len(w) < 6:
        return False
    hits = sum(x in _EN_PROSE for x in w)
    return hits >= 3 and hits / len(w) >= 0.2


def strip_meta(text: str) -> str:
    """Remove model self-notes from a Pass-1 scene. A line without a 'Speaker:' prefix that is English prose is dropped; inside a spoken
    line, an English-prose SENTENCE is cut. Dialogue itself (Hinglish, with ordinary English loanwords) is never touched, and a scene is
    never emptied by this."""
    keep = []
    for ln in text.split("\n"):
        if _NOTE_START.match(ln):                                # "(Note: ...)" is not a speaker called Note
            continue
        m = re.match(r"^(\s*[^:\n]{1,40}:\s*)(.*)$", ln)
        if m is None:
            if ln.strip() and _is_english_meta(ln):
                continue
            keep.append(ln)
            continue
        head, body = m.group(1), m.group(2)
        sents = re.split(r"(?<=[.!?])\s+", body)
        good = [s for s in sents if not _is_english_meta(s)]
        if good and len(good) != len(sents):
            ln = head + " ".join(good)
        elif not good and sents:
            continue                                            # the whole "line" was a self-note
        keep.append(ln)
    out = "\n".join(keep).strip()
    return out if out else text


_NON_SPEAKERS = {"narrator", "narration", "narrators", "voice", "voiceover", "author", "speaker", "note", "notes", "wait", "correct", "correct output",
                 "output", "scene", "location", "stage direction", "direction", "dialogue", "translation", "explanation", "comment", "assistant",
                 "system", "revised", "revision", "fixed", "answer", "corrected", "rewrite", "rewritten", "final",
                 # fields of the writer's own prompt that a model sometimes echoes as if they were speakers
                 "listener", "main speaker", "scene context", "key events", "previous event", "characters", "character", "person", "someone", "everyone", "all"}
_LINE = re.compile(r"^\s*[*#_>\-\s]*([^:\n]{1,40}?)[*_\s]*:\s*(.*)$")


def _name_matches(label: str, names) -> str:
    """-> the canonical name from `names` that `label` denotes (case, spelling variants, 'Horiram' ~ 'Hori'), or ''."""
    from agents.name_utils import same_person
    lab = label.strip().strip("*#_ ").strip()
    low = lab.lower()
    for n in names:
        if n.lower() == low:
            return n
    for n in names:
        if same_person(n, lab) or n.lower().split()[0] == low.split()[0]:
            return n
    return ""


def _name_in_source(label: str, source: str) -> bool:
    """A speaker label that is not in the cast is still a real character if his NAME is in the scene's own text (Agent 1 can miss a person; the story still
    names him). 'Asif', a name invented by the writer, is not in the text, so it is rejected."""
    import difflib
    toks = set(re.findall(r"[a-z]+", (source or "").lower()))
    words = [w for w in re.findall(r"[a-z]+", label.lower())]
    if not toks or not words or any(len(w) < 3 for w in words):
        return False
    skel = lambda w: re.sub(r"(.)\1+", r"\1", re.sub(r"[aeiouywh]", "", w.replace("ph", "f").replace("c", "k").replace("q", "k").replace("j", "g")))      # j / g are one sound in "engineer" ~ "injiniyar"; w / h are silent or inconsistent in transliteration ("owner" ~ "onar")
    skels = {skel(t) for t in toks if len(skel(t)) >= 2}
    # fuzzy spelling OR the same consonant skeleton ("Team Lead" ~ the transliteration "tim lid": English loanwords written in Devanagari come back differently spelled)
    per_word = all(w in toks or difflib.get_close_matches(w, toks, n=1, cutoff=0.86) or (len(skel(w)) >= 3 and skel(w) in skels) for w in words)
    if per_word:
        return True
    # a MULTI-word role ("Team Lead", "Product Owner") against English loanwords written in Devanagari ("tim lid", "prodakt onar"): the whole label's consonant
    # skeleton must occur in the source's (selective enough at 4+ consonants)
    src_all = skel("".join(re.findall(r"[a-z]+", (source or "").lower())))
    lab_all = skel("".join(words))
    return len(words) > 1 and len(lab_all) >= 4 and lab_all in src_all


def sanitize_pass1(text: str, allowed, extras=(), canonical=(), source: str = "") -> str:
    """Make a Pass-1 scene clean screenplay data, for ANY story. A line survives only if its speaker is a real character of the story (matched to the
    canonical name, so HORI / Horiram / Hori become one name). Dropped: 'Narrator' lines (narration is never a speaker), the model's notes to
    itself ('Wait, that violates the rule: ...'), unlabeled stray lines, and English prose inside a spoken line. Unlabeled Hinglish that follows
    a spoken line is a wrapped continuation and is joined to it. If nothing is left the result is EMPTY (the caller retries or keeps the scene without dialogue); raw model text is never passed on."""
    names = [n for n in list(allowed) + list(extras) if isinstance(n, str) and n.strip()]
    out = []
    for ln in (text or "").split("\n"):
        if not ln.strip():
            continue
        if _NOTE_START.match(ln):
            continue
        m = _LINE.match(ln)
        label = m.group(1).strip() if m else ""
        generic = (not m) or label.lower() in _NON_SPEAKERS or label.lower().split()[0] in {"wait", "note", "correct", "revised", "actually", "hmm"}
        looks_like_name = bool(m) and len(label.split()) <= 3 and re.match(r"^[A-Za-z][A-Za-z .'-]*$", label) is not None
        if generic or not looks_like_name:
            if not m and out and not _is_english_meta(ln):            # wrapped continuation of the previous spoken line
                from agents.grounding import narrates_own_gesture as _ng
                cont = " ".join(x for x in re.split(r"(?<=[.!?])\s+", ln.strip()) if not _is_english_meta(x) and not _ng(x))
                if cont:
                    out[-1] = out[-1] + " " + cont
            continue
        canon = _name_matches(label, names) if names else ""
        if not canon:
            if source and _name_in_source(label, source):
                canon = label.strip().title()                         # named in the scene's own text, only missing from the cast list
            elif names or source:
                continue                                               # a speaker who is neither in the cast nor in the text (no cast at all used to accept ANYTHING)
            else:
                canon = label.title()                                  # no cast and no source to check against (a bare call)
        if canonical:                                                  # one name per person across the whole script ("Hori" / "HoriRam" -> the profile's name)
            canon = _name_matches(canon, list(canonical)) or canon
        body = m.group(2).strip()
        if not body:
            continue                                                   # "Dhaniya:" with nothing after it (a silent character) is not a line
        sents = re.split(r"(?<=[.!?])\s+", body)
        from agents.grounding import narrates_own_gesture
        good = [x for x in sents if not _is_english_meta(x) and not narrates_own_gesture(x)]
        if not good:
            continue
        line_out = f"{canon}: " + " ".join(good)
        if line_out in out:
            continue                                                   # the writer sometimes repeats a line verbatim
        out.append(line_out)
    res = "\n".join(out).strip()
    return res


def line_budget(source: str, max_lines: int = 6) -> int:
    """How many dialogue lines a scene can honestly carry: one per spoken line of the source, plus a little for reactions. A one-sentence scene
    must not be padded to six lines (that padding is where invented facts come from)."""
    src = source or ""
    speech = len(re.findall(r"[—―]", src)) + len(re.findall(r"[\"“][^\"”]{2,}[\"”]", src))
    sentences = max(1, len([x for x in re.split(r"[।.!?]+", src) if x.strip()]))
    return max(2, min(int(max_lines), speech + 1 + sentences // 2))


# Slang that leaked into the NEUTRAL pass and survived the retries. Unambiguous Hindi slang only (words like "fire", "lit", "solid", "boss", "bhai" can be
# ordinary English / kinship words, so they are never rewritten). The writer sometimes "corrects" itself in the line ("bahut mast nahi, bahut accha hai").
_LEAK_POS = {"kamaal", "zabardast", "mast", "kadak", "lajawaab", "lajawab", "jhakaas", "gazab", "faadu", "dhamaal", "jalwa", "makkhan", "bawaal"}
_LEAK_NEG = {"kachra", "jhand", "flop"}
_LEAK_TROUBLE = {"panga", "locha", "lafda", "ghapla", "jhol"}
_LEAK_FILLER = {"yaar", "bro", "arre", "abe", "abey"}


_GLUED = re.compile(r"\b(school|college|hostel|office|class|project|team|ground|match|canteen|library|hospital|station|market|room|lab|city|phone|video|exam|test|plan|game|group|club|mall|park)"
                    r"(mein|men|ko|ka|ki|ke|se|par|pe|tak|ne)\b", re.I)


def fix_glued_words(text: str) -> str:
    """The writer sometimes glues an English noun to the Hindi postposition ("Schoolmein", "teamne"): put the space back. Only a fixed list of common nouns, so no real word is split."""
    return _GLUED.sub(lambda m: m.group(1) + " " + m.group(2), text or "")


# Plain everyday words for each analysed emotion: exactly the kind of word the slang editor can swap (agents/edit_gate slots). Used only by the padded style (AGENT3_STYLE=slotrich_pad).
EMOTION_WORDS = {
    "happy": "bahut khushi ho rahi hai / bahut accha laga / bahut maza aa raha hai", "excited": "bahut excited hoon / bahut maza aa raha hai",
    "proud": "bahut garv hai / bahut accha laga", "amused": "bahut maza aa raha hai", "relieved": "bahut accha laga / bahut tension thi",
    "affectionate": "bahut pasand aaya", "scared": "bahut darr lag raha hai / bahut chinta ho rahi hai", "anxious": "bahut chinta ho rahi hai / bahut tension ho rahi hai",
    "angry": "bahut gussa aa raha hai", "frustrated": "bahut bura lag raha hai / bahut gussa aa raha hai", "sad": "bahut bura lag raha hai",
    "ashamed": "bahut bura lag raha hai",
}


def feelings_block(characters, profiles, scene_id) -> str:
    """The Persona Agent's emotion arc, made usable: for every PEER in the scene whose analysed emotion has a plain everyday word, ask for ONE line that says it. Peers only (slang is
    never added to an elder or an authority), only feelings the analysis lists (nothing invented). The lines must be between peers, because a line to an elder stays plain anyway."""
    from agents.name_utils import match_profile
    from agents.agent2b_persona import emotion_at
    lines = []
    for c in characters or []:
        pr = match_profile(profiles, c) if profiles else {}
        if not pr or str(pr.get("social_role", "peer")) != "peer":
            continue
        em = emotion_at(pr, scene_id)
        words = EMOTION_WORDS.get(em["emotion"])
        if words:
            lines.append(f"- {c} feels {em['emotion']}: give {c} ONE line that says so in plain words, e.g. \"{words}\"")
    if not lines:
        return ""
    return ("FEELINGS TO VOICE (from the story analysis; peers only; never invent a feeling that is not listed here):" + chr(10) + chr(10).join(lines) + chr(10)
            + "Put these feeling lines in the talk BETWEEN peers (to a friend, sibling or classmate), not in a line to an elder, a teacher or a shopkeeper." + chr(10))


def clean_markup(text: str) -> str:
    """Markdown the writer sometimes leaves in the dialogue ("Sharma: * * Tumhari mehnat ...", "**Hamid**: ..."): asterisks are never part of spoken Hinglish, and they showed up in the
    screenplay and the subtitles. Removes every asterisk and a leading bullet dash, and tidies the spaces."""
    out = []
    for ln in (text or "").split(chr(10)):
        ln = re.sub(r"\*+", "", ln)
        ln = re.sub(r"^\s*[-•]\s+", "", ln)
        ln = re.sub(r"(:\s*)[-•]\s+", r"", ln, count=1)
        out.append(re.sub(r"[ 	]{2,}", " ", ln).rstrip())
    return chr(10).join(out)


def strip_diacritics(text: str) -> str:
    """Pass-1 is plain Roman letters: a model that copies a scholarly romanisation ("Ramu" -> "Ramū", "dhanyāvaad") is mapped back to ASCII. Only marks on Latin
    letters are removed; Devanagari (never expected here) and symbols such as the rupee sign are left alone."""
    import unicodedata
    out, prev = [], ""
    for ch in unicodedata.normalize("NFD", text or ""):
        if unicodedata.category(ch) == "Mn" and prev and ord(prev) < 0x250:
            continue
        out.append(ch)
        prev = ch
    return unicodedata.normalize("NFC", "".join(out))


def fix_self_reference(text: str, aliases=()) -> str:
    """A speaker who names HIMSELF (or his nickname) as the subject - "Rahul: Rocky ne Gita ki ball rok di" (Rocky = Rahul) - is narrating, not talking: the
    subject becomes "maine". Only the "<own name> ne" form is touched (it is always a subject), and only for the speaker's own names."""
    amap = {}
    for al in aliases or []:
        if isinstance(al, dict) and al.get("alias") and al.get("canonical"):
            amap.setdefault(al["canonical"].lower(), set()).add(al["alias"].lower())
            amap.setdefault(al["alias"].lower(), set()).add(al["canonical"].lower())
    out = []
    for line in (text or "").split(chr(10)):
        sp, sep, utt = line.partition(":")
        if sep:
            mine = {sp.strip().lower()} | amap.get(sp.strip().lower(), set())
            mine = {m for m in mine if len(m) >= 3}
            if mine:
                rx = re.compile(r"(^|[.!?]\s+)?(?<![A-Za-z])(?:" + "|".join(re.escape(m) for m in sorted(mine, key=len, reverse=True)) + r")\s+ne(?![A-Za-z])", re.I)
                utt = rx.sub(lambda m: (m.group(1) or "") + ("Maine" if (m.group(1) or utt[:m.start()].strip() == "") else "maine"), utt)
            line = sp + sep + utt
        out.append(line)
    return chr(10).join(out)


def neutralize_leaks(text: str) -> str:
    """Last line of defence after the retries: remove a self-correction ("X nahi,"), delete fillers, and swap leaked slang for its plain word."""
    all_slang = "|".join(sorted(_LEAK_POS | _LEAK_NEG | _LEAK_TROUBLE, key=len, reverse=True))
    t = re.sub(r"(?:(?:bahut|ekdum|kaafi|bilkul)\s+)?\b(?:" + all_slang + r")\s+(?:nahi|nahin|nhi)\s*,\s*", "", text or "", flags=re.I)
    t = re.sub(r"\b(?:" + "|".join(_LEAK_FILLER) + r")\b\s*,?\s*", "", t, flags=re.I)

    def swap(m):
        w = m.group(0).lower()
        rep = "accha" if w in _LEAK_POS else "kharab" if w in _LEAK_NEG else "problem"
        return rep.capitalize() if m.group(0)[:1].isupper() else rep
    t = re.sub(r"\b(?:" + all_slang + r")\b", swap, t, flags=re.I)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return re.sub(r"(?m)^(\s*[^:\n]{1,40}:)\s*([a-z])", lambda m: m.group(1) + " " + m.group(2).upper(), t)


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
        everyone = list(character_profiles or {})
        speaker = characters[0] if characters else "(the person who speaks in the original text - name them as the text does)"
        others = [c for c in characters if c != characters[0]] if characters else []
        listener = others[0] if others else ("another character of the story who is in this scene (use the name the original text gives; if it names "
                                             "nobody else, the speaker thinks aloud or talks to someone off-screen). It is never a narrator")

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
                if any(str((_mp(character_profiles, c) or {}).get("social_role", "")) == "elder" for c in characters):
                    cast_block += "An elder (grandparent, parent) speaks to a child or a younger person with 'tum' or 'tu', never 'aap'; the young speak to the elder with 'aap'.\n"
                if getattr(config, "AGENT3_STYLE", "default") == "slotrich_pad":
                    cast_block += feelings_block(characters, character_profiles, sid)
            if extras:
                cast_block += ("Unnamed people who speak in the original text (give them a line, label them with this role): "
                               + ", ".join(extras) + ".\n")
            if absent:
                cast_block += ("NOT PRESENT in this scene (they are only mentioned or are elsewhere): " + ", ".join(absent) +
                               ". They must NOT speak and nobody may speak TO them or act as if they are there.\n")
            for al in (event_chain_entry.get("aliases") or []):
                cast_block += (f"SAME PERSON: \"{al['alias']}\" is only another name for {al['canonical']}. They are ONE character: never speak of them as two people "
                               f"(do not write {al['canonical']} talking about {al['alias']} as if someone else); use {al['canonical']}.\n")
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
                                "character narrate facts to the listener:\n" + deva.strip()[:3400] + "\n")
            elif src.strip():
                source_block = ("Original scene text (auto-romanized, so spellings can be off: koch=coach, tim=team, kap=cup, "
                                "maich=match, skul=school). Keep the facts and the spoken lines of this text; turn narration into "
                                "what people would actually SAY to each other, do not have a character narrate facts to the listener:\n"
                                + src.strip()[:3000] + "\n")
        n_lines = (line_budget(deva or src or "", int(getattr(config, "AGENT3_MAX_LINES", 6)))) if rich else 3
        if rich:                                                       # every grounded key event of the scene needs a line to be said in (events lost to a too-small budget)
            n_lines = max(n_lines, min(int(getattr(config, "AGENT3_MAX_LINES", 6)), len(event_chain_entry.get("key_events") or []) + 1))
        if rich:
            from agents.grounding import spoken_lines
            said = spoken_lines(deva) or spoken_lines(src)
            if said:
                source_block += ("WORDS ACTUALLY SPOKEN in the original (keep their meaning and their order; translate them faithfully into neutral Hinglish and give "
                                 "each to the character who says it - do not replace them with something else; a line marked (reported) is indirect speech: "
                                 "write it as that character's own direct words):" + chr(10)
                                 + chr(10).join(f"{i}. {q}" for i, q in enumerate(said, 1)) + chr(10))
                n_lines = max(n_lines, min(int(getattr(config, "AGENT3_MAX_LINES", 6)), len(said) + 1))
            from agents.grounding import gloss_lines
            gl = gloss_lines(src)
            if gl:
                source_block += ("WORD MEANINGS (romanised Hindi words that are easy to misread - use exactly these meanings): " + "; ".join(gl) + chr(10))
            from agents.grounding import gesture_cues
            acts = gesture_cues(deva) or gesture_cues(src)
            if acts:
                source_block += ("GESTURES AND ACTIONS in the original (the character DOES these; never turn them into words the character says about themselves - no 'I nod', "
                                 "'I say yes happily', 'I smile'):" + chr(10) + chr(10).join("- " + a for a in acts) + chr(10))
            from agents.grounding import unspoken_cues
            quiet = unspoken_cues(deva) or unspoken_cues(src)
            if quiet:
                source_block += ("NOT SPOKEN ALOUD in the original (a feeling hidden, a silence, a thought, an unanswered question): never turn these into spoken "
                                 "lines - a character who hid something does not say it, a character who stayed silent says nothing:" + chr(10)
                                 + chr(10).join("- " + q for q in quiet) + chr(10))
        prompt = PASS1_PROMPT_TEMPLATE.format(
            cast_block=cast_block, source_block=source_block,
            line_limit=f"At most {n_lines} dialogue lines in total (fewer is fine - never pad a short scene), a natural back-and-forth between the characters",
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
            style_rules=((PEER_STYLE_RULES_PAD if getattr(config, "AGENT3_STYLE", "default") == "slotrich_pad" else PEER_STYLE_RULES)
                         if (getattr(config, "AGENT3_STYLE", "default") in ("slotrich", "slotrich_pad")
                             and str(profile.get("social_role", "peer")) == "peer") else "")
        )

        if getattr(config, "AGENT3_STYLE", "default") == "slotrich_pad":
            # The padded style asks the peers to state how they feel; two other sentences of the prompt used to forbid exactly that ("do NOT add ... opinions", "never pad a
            # short scene"), the writer obeyed the stricter one and the lines carried no feeling word the slang editor could swap (found by measuring: 72% of lines had none).
            prompt = re.sub(r"do NOT add thanks, promises, opinions, jokes, questions, agreements or explanations the original\s+does not contain\.",
                            "do NOT add new events, thanks, promises, jokes, questions or explanations the original does not contain. DO let a character say how they FEEL "
                            "about what just happened, in one plain everyday word, whenever the scene clearly has that feeling (excited, worried, scared, relieved, proud, "
                            "upset, pleased).", prompt)
            prompt = prompt.replace(" (fewer is fine - never pad a short scene)", " (fewer is fine)")

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
                    leaks = _leaks(result, (scene or {}).get("text") or "")
                    if not leaks:
                        break
                    r2 = self.groq_client.chat.completions.create(
                        model=config.GROQ_MODEL, temperature=0.3, max_tokens=1400,
                        messages=[{"role": "user", "content": prompt + "\n\nIMPORTANT: your previous answer used banned words ("
                                   + ", ".join(leaks) + "). Write the dialogue again WITHOUT any banned word."}])
                    again = (r2.choices[0].message.content or "").strip()
                    if again:
                        result = again

                # the profiles (NER-verified by Agent 2) define who the people of the story are; Agent 1's raw name list is trusted only when there are no profiles
                allowed = [c for c in everyone if not any(c.lower() == a.lower() for a in absent)] + list(extras) + ([] if everyone else list(characters))
                result = sanitize_pass1(strip_meta(fix_glued_words(strip_diacritics(clean_markup(result)))), allowed, canonical=everyone, source=(scene or {}).get("text") or "")   # only real characters speak; no Narrator, no notes to itself

                result = fix_self_reference(result, (event_chain_entry or {}).get("aliases"))     # "Rahul: Rocky ne ..." (Rocky = Rahul) -> "Rahul: Maine ..."

                if _leaks(result, (scene or {}).get("text") or ""):              # still there after the retries: neutralise instead of shipping slang into the neutral pass
                    result = neutralize_leaks(result)

                # Ensure output has SPEAKER: format
                if result and ":" not in result:
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