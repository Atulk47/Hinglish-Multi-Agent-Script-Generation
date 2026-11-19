"""
agents/grounding.py  -  PROVENANCE for everything an agent writes into the shared scratchpad.

A shared scratchpad lets agents build on each other, but it also lets one agent's mistake become every later agent's "fact". Seen on a real run
(a classical Hindi novel excerpt read through auto-romanisation): "forehead filled with hair" (the source says wrinkles), "something is locked at the
door" (the source says tied), a location "where the object is locked", and a secret ("the cow was bought with a shady loan") that the story never
mentions - each then steered the dialogue writer and the critic. The rule that stops this, for ANY story:

    a claim is accepted into the scratchpad only if it carries a quotation from the story and that quotation really is in the story.

This module is the deterministic part of that rule (no model call): normalising text, checking that a quoted span is supported by the source
(Devanagari original and/or its romanisation), and cleaning a location down to "Unknown" unless it is both grounded and unhedged.
"""
from __future__ import annotations
import re
import unicodedata
from typing import Iterable, List

_PUNCT = re.compile(r"[‒-―\-_\"'`‘’“”.,;:!?()\[\]{}|/\\*#~^<>=+।॥…]+")
_TOK = re.compile(r"[ऀ-ॿ]+|[a-z0-9]+")
_ZW = re.compile(r"[​‌‍﻿]")


def norm(s: str) -> str:
    """Lower-case, NFC, no zero-width characters, nukta-less Devanagari (the LLM and the source often disagree on the dot), punctuation -> space."""
    s = unicodedata.normalize("NFC", str(s or ""))
    s = _ZW.sub("", s)
    s = unicodedata.normalize("NFD", s).replace("़", "")                     # drop the nukta so  ज़ == ज
    s = unicodedata.normalize("NFC", s).lower()
    s = _PUNCT.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def tokens(s: str) -> List[str]:
    return _TOK.findall(norm(s))


def span_supported(span: str, *sources: str, min_recall: float = 0.75) -> bool:
    """True when `span` (a quotation the model says it copied from the story) is really in at least one of the sources.
    One-word spans must occur as a word; longer spans need >= `min_recall` of their words in the source AND at least one adjacent word pair in
    the same order (a bag of common words is not a quotation)."""
    st = tokens(span)
    if not st:
        return False
    for src in sources:
        tt = tokens(src)
        if not tt:
            continue
        tset = set(tt)
        if len(st) == 1:
            if len(st[0]) >= 3 and st[0] in tset:
                return True
            continue
        recall = sum(w in tset for w in st) / len(st)
        pairs = set(zip(tt, tt[1:]))
        if recall >= min_recall and any(p in pairs for p in zip(st, st[1:])):
            return True
    return False


_HEDGE = re.compile(r"\b(likely|probably|possibly|maybe|perhaps|presumably|unspecified|unknown|unclear|not\s+(stated|specified|mentioned|given)|"
                    r"n/?a|none|somewhere|where\s+.{0,40}\b(is|was|are|were|lies|lying|present)\b|object|someone|something)\b", re.I)


def clean_location(location: str, evidence: str, *sources: str) -> str:
    """A scene location is kept only if it is unhedged AND the model quoted the words of the story that name / show the place."""
    loc = str(location or "").strip().strip("*").strip()
    if not loc or _HEDGE.search(loc) or len(loc.split()) > 8:
        return "Unknown"
    if not span_supported(evidence, *sources):
        return "Unknown"
    return loc


def is_unknown_location(location: str) -> bool:
    s = str(location or "").strip().strip("*").strip().lower()
    return (not s) or s in ("unknown", "unspecified", "n/a", "none", "unspecified location") or bool(_HEDGE.search(s))


def ground_events(events: Iterable[str], evidence: Iterable[str], *sources: str):
    """-> (kept_events, dropped_events). With no evidence list at all (a model that ignored the instruction) nothing is dropped, so a non-compliant
    model cannot blank a scene; with evidence, an event whose quotation is not in the story is removed."""
    events = [str(e) for e in (events or [])]
    ev = list(evidence or [])
    if not ev:
        return events, []
    kept, dropped = [], []
    for i, e in enumerate(events):
        q = ev[i] if i < len(ev) else ""
        (kept if span_supported(q, *sources) else dropped).append(e)
    return kept, dropped


# ── the author's own spoken words ─────────────────────────────────────────────────────────────────────────────────────────
_SPEECH_VERBS = (r"कहा|कही|कहती|कहता|कहते|कहने|बोला|बोली|बोले|बोलते|बोलती|पूछा|पूछी|पूछते|पुकारा|चिल्लाया|चिल्लायी|चीखा|चीखी|"
                 r"उत्तर दिया|जवाब दिया|डाँटा|डांटा|समझाया|बताया|बोल उठा|बोल उठी|कह उठा|कह उठी|"
                 r"kaha|kahi|kehti|kehta|bola|boli|bole|poochha|puchha|puchhi|pukara|chillaya|chillayi|dante|samjhaya|bataya|jawab diya|uttar diya")
_DASH_SPEECH = re.compile(r"(?:(?:" + _SPEECH_VERBS + r")[^\n—―–\"“]{0,25}|^\s*)[—―–-]\s*([^\n]+)", re.I | re.M)
_QUOTED = re.compile(r"[\"“]([^\"”\n]{3,}?)[\"”]")


_REPORTED = re.compile(r"(?:" + _SPEECH_VERBS + r"|याद दिलाया|कहने लगा|कहने लगी|पूछने लगा|पूछने लगी)\s+कि\s+([^\u0964!?\n]{4,})"
                       r"|(?:kaha|bola|boli|poochha|puchha|puchhi|bataya|samjhaya|yaad dilaya|jawab diya)\s+ki\s+([^.!?\n]{4,})", re.I)


def reported_speech(text: str, limit: int = 8) -> List[str]:
    """Indirect speech, the usual way Hindi prose reports a conversation without quotation marks: "कबीर ने मज़ाक उड़ाते हुए कहा कि आरव हर चीज़ को इतना गंभीर क्यों
    लेता है", "आरव ने उसे याद दिलाया कि पिछली परीक्षा में भी ...". Returns the reported clauses."""
    out: List[str] = []
    for m in _REPORTED.finditer(text or ""):
        q = (m.group(1) or m.group(2) or "").strip()
        if q and q not in out:
            out.append(q)
    return out[:limit]


def spoken_lines(text: str, limit: int = 8) -> List[str]:
    """The words the AUTHOR puts in characters' mouths, found by Hindi prose convention (a dash after a speech verb - "धनिया बोली—अरे, चलो।" - or a
    dialogue paragraph that opens with a dash, or quotation marks). They are handed to the dialogue writer as text that must survive translation, so a
    scene cannot drop or replace what a character actually says. A dash inside narration ("एक ही लालसा थी—किसी तरह ...") is not speech."""
    out: List[str] = []
    for m in _QUOTED.finditer(text or ""):
        q = m.group(1).strip()
        if q and q not in out:
            out.append(q)
    for m in _DASH_SPEECH.finditer(text or ""):
        q = m.group(1).strip()
        if q and q not in out:
            out.append(q)
    for q in reported_speech(text, limit):
        mark = "(reported) " + q
        if mark not in out and q not in out:
            out.append(mark)
    return out[:limit]


# ── secrets must be CONCEALMENT, not just any quotable sentence ──────────────────────────────────────────────────────────
_CONCEAL = re.compile(r"छिप|छुप|राज़|राज\b|रहस्य|गुप्त|सरप्राइज|सरप्राइज़|चुपके|मत\s*बता|न\s*बता|नहीं\s*बता|किसी\s*को\s*(?:न|नहीं|मत)|पता\s*न\s*चल|"
                      r"भनक|खबर\s*न|surprise|secret|chhup|chup\s*ke|chupke|raaz|raz\b|rahasya|gupt|mat\s*bata|na\s*bata|nahi\s*bata|kisi\s*ko\s*(?:na|nahi|mat)|"
                      r"pata\s*na\s*chal|bhanak|hidden|conceal|keep\s*it\s*quiet", re.I)


def shows_concealment(evidence: str) -> bool:
    """A 'secret' is accepted only if the quotation behind it speaks of hiding / surprise / not telling, in Hindi or Roman script."""
    return bool(_CONCEAL.search(evidence or ""))


_SPEAKS = re.compile(r"(?:" + _SPEECH_VERBS + r")", re.I)


def speaks(quote: str) -> bool:
    """The quotation contains a speech verb (कहा / बोला / पूछा ... or kaha / bola ...): the person it is about is really speaking."""
    return bool(_SPEAKS.search(quote or ""))


_W = r"(?:[\u0900-\u0963\u0966-\u097F]{1,40}|[A-Za-z]{1,40})"        # a word: bounded length, so a 40 000-character token cannot make the match quadratic
_SUBJECT_BEFORE_SPEECH = re.compile(r"(?<![\u0900-\u0963\u0966-\u097FA-Za-z])(" + _W + r"(?:\s+" + _W + r"){0,3})\s+(?:" + _SPEECH_VERBS + r")", re.I)


def speech_subjects(text: str) -> List[str]:
    """Words that stand in the (up to) 3 positions before a speech verb: 'दुकानदार ने कहा', 'dukaandaar ne kaha', 'धनिया बोली'. Deterministic evidence that a
    person / role really speaks in the text, independent of what a model claims."""
    out: List[str] = []
    for m in _SUBJECT_BEFORE_SPEECH.finditer(text or ""):
        out += re.findall(r"[ऀ-ॣ०-ॿ]+|[A-Za-z]+", m.group(1))
    return out


def role_speaks_in(role: str, *sources: str, min_ratio: float = 0.72) -> bool:
    """True if `role` (Roman letters, e.g. 'Dukaandaar') matches, after transliteration, a word that precedes a speech verb in the sources."""
    import difflib
    key = re.sub(r"[^a-z]", "", (role or "").lower())
    if len(key) < 3:
        return False
    tr = None
    for src in sources:
        for w in speech_subjects(src):
            if re.search(r"[ऀ-ॣ०-ॿ]", w):
                if tr is None:
                    from preprocessing.transliterate import HindiTransliterator
                    tr = HindiTransliterator()
                w = tr.transliterate_sentence(w)
            wk = re.sub(r"[^a-z]", "", w.lower())
            if len(wk) >= 3 and (difflib.SequenceMatcher(None, key, wk).ratio() >= min_ratio):
                return True
    return False



# -- what the author says was NOT said aloud ---------------------------------------------------------------------------------
_UNSPOKEN = re.compile(
    r"छिपा|छुपा|छिप|चुपचाप|चुप\s+(?:खड़|बैठ|हो|रह[ा-ी]|था|थी)|मौन|खामोश|ख़ामोश|बिना\s+कुछ\s+(?:कहे|बोले|कहा)|कुछ\s+(?:न|नहीं)\s+(?:बोल|कह)|"
    r"(?:उत्तर|जवाब)\s+नहीं\s+दिया|कोई\s+(?:उत्तर|जवाब)|मन\s+ही\s+मन|मन\s+में|सोचा|सोचती|सोचता|सोचने|शब्द\s+नहीं\s+निकल|दबा\s+ली|रोक\s+ली|"
    r"नहीं\s+कहा|नहीं\s+बताया|कहना\s+चाह|"
    r"chhup|chhip|chupchap|chup\s+(?:khad|baith|ho|rah|tha|thi)|maun|khamosh|bina\s+kuch\s+(?:kahe|bole|kaha)|kuch\s+(?:na|nahi)\s+(?:bol|kah)|"
    r"(?:uttar|jawab)\s+nahi\s+diya|man\s+hi\s+man|mann\s+mein|socha|sochti|sochta|shabd\s+nahi\s+nikal|nahi\s+kaha|nahi\s+bataya|kehna\s+chah", re.I)
_SENT_SPLIT = re.compile(r"(?<=[।!?.])\s+")


def unspoken_cues(text: str, limit: int = 4) -> List[str]:
    """Sentences of the source that say something was hidden, kept silent, only thought or never answered ("उसने अपनी चिंता छिपा ली", "वह चुप रहा",
    "कुछ कहना चाहती थी, लेकिन शब्द नहीं निकल रहे थे"). A speech sentence is never one of them. They are given to the dialogue writer and to the critic as
    text that must NOT turn into spoken lines: a character who hid a feeling does not say it aloud."""
    out: List[str] = []
    for sent in _SENT_SPLIT.split(re.sub(r"\s+", " ", text or "")):
        sent = sent.strip()
        if len(sent) < 8 or spoken_lines(sent):
            continue
        if _UNSPOKEN.search(sent):
            out.append(sent)
    return out[:limit]



# -- gestures and actions: things a character DOES, not things they SAY --------------------------------------------------------
_GESTURE = re.compile(
    r"सिर\s+(?:हिला|झुका|उठा)|गर्दन\s+हिला|मुस्कुरा|मुस्करा|मुस्कराहट|हँस|हंस|रो\s+(?:पड़|दि)|आँसू|आंसू|गले\s+(?:से\s+)?लगा|सीने\s+से\s+लगा|चूम|"
    r"सिर\s+पर\s+हाथ|हाथ\s+फेर|हाथ\s+जोड़|पैर\s+छू|कंधे\s+उचका|मुँह\s+फेर|ठंडी\s+साँस|गहरी\s+साँस|आँखें\s+(?:झुका|बंद|चुरा)|नज़रें\s+(?:झुका|चुरा)|"
    r"भौंहें|मुट्ठी|कसकर\s+(?:पकड़|खींच)|"
    r"sir\s+(?:hila|jhuka)|gardan\s+hila|muskura|muskara|hans\s+(?:pad|diy)|ro\s+pad|aansu|gale\s+(?:se\s+)?laga|chum|sir\s+par\s+haath|haath\s+jod|"
    r"pair\s+chhu|kandhe\s+uchka|munh\s+fer|thandi\s+saans|gehri\s+saans|aankhein\s+(?:jhuka|band)", re.I)


def gesture_cues(text: str, limit: int = 4) -> List[str]:
    """Narration of what a character DOES with face and body ("हामिद ने खुशी से सिर हिलाया", "अमीना ने उसके सिर पर हाथ फेर दिया"). The part before the spoken
    words is kept ("... सिर हिलाया और बोला"). Handed to the writer as actions that must not become the character's own speech."""
    out: List[str] = []
    for sent in _SENT_SPLIT.split(re.sub(r"\s+", " ", text or "")):
        narration = re.split(r"[—―–\"“]", sent, maxsplit=1)[0].strip()
        if len(narration) >= 8 and _GESTURE.search(narration):
            out.append(narration)
    return out[:limit]


# a speaker announcing his OWN gesture or manner as if it were speech ("main khushi se haan karta hoon", "main sir hilata hoon", "main muskura raha hoon")
_SELF_NARRATION = re.compile(
    r"\b(?:main|hum)\b[^.!?\n]{0,25}\b(?:sir|gardan|haath|aankhein?|palkein?)\s+(?:hila|jhuka|jod|band|utha)\w*|"
    r"\b(?:main|hum)\b[^.!?\n]{0,20}\b(?:muskura|muskara|hans|ro|gale\s+lag|chum)\w*\s+(?:raha|rahi|rahe|ta|ti|te)\s*(?:hoon|hun|hain)\b|"
    r"\b(?:haan|haami|sehmati|na|inkaar)\s+kar(?:ta|ti|te)\s+(?:hoon|hun|hain)\b|"
    r"\b(?:khushi|prem|pyaar|dukh|gusse|sharm|garv)\s+se\s+(?:haan|ji|sir|sehmat)\b", re.I)


def narrates_own_gesture(sentence: str) -> bool:
    return bool(_SELF_NARRATION.search(sentence or ""))


# ---- aliases: "Rahul, known to his friends as Rocky" -----------------------------------------------------------------------------------------
_ALIAS_CUE = re.compile(r"\bnaa?m (?:se|ka|par)\b|\bke naa?m\b|\burf\b|kahal|kahat|kahe jat|kehl|kehat|pukar|bulat|bulaya|jana jata|jane jate|janati|"
                        r"pahachan|ke rup men|nickname|alias|called|known as", re.I)


def _skel(w: str) -> str:
    w = w.lower().replace("ph", "f").replace("c", "k").replace("q", "k")
    return re.sub(r"(.)\1+", r"\1", re.sub(r"[aeiouywh]", "", w))


def _name_among(name: str, toks: List[str]) -> bool:
    import difflib
    n = re.sub(r"[^a-z]", "", name.lower())
    if len(n) < 2:
        return False
    for t in toks:
        if t == n or (t[:1] == n[:1] and len(_skel(n)) >= 2 and _skel(t) == _skel(n)) or difflib.SequenceMatcher(None, t, n).ratio() >= 0.84:
            return True
    return False


def names_one_person(quote: str, a: str, b: str) -> bool:
    """True when ONE quotation from the story contains both names (spelling-tolerant: Devanagari loanwords come back differently romanised) AND a cue that
    one is another name for the other (nam se / kehlate / pukarte / urf / known as ...). Two people who merely appear together are not aliases."""
    q = str(quote or "")
    if re.search(r"[\u0900-\u097F]", q):
        try:
            from preprocessing.transliterate import HindiTransliterator
            q = HindiTransliterator().transliterate_sentence(q)
        except Exception:
            return False
    toks = re.findall(r"[a-z]+", q.lower())
    if not (a and b and a.lower() != b.lower() and _name_among(a, toks) and _name_among(b, toks)):
        return False
    text = " ".join(toks)
    # the cue must sit in the same stretch as the names (within 8 words of both), not somewhere else in a longer quotation
    for m in _ALIAS_CUE.finditer(text):
        at = len(text[:m.start()].split())
        near = lambda nm, w: any(_name_among(nm, [t]) and abs(k - at) <= w for k, t in enumerate(toks))
        if near(a, 8) and near(b, 8) and (near(a, 2) or near(b, 2)):        # and one of them is the name right next to the cue ("Rocky ke nam se")
            return True
    return False


def name_grounded(label: str, source: str, min_tokens: int = 5) -> bool:
    """A character name must come from the story: found in the (romanised) text, spelling-tolerant (Geeta/Gita, Rocky/roki, "Engineer"/"injiniyar", titles ignored).
    A source too short to judge (fewer than min_tokens words) vouches for everything."""
    toks = re.findall(r"[a-z]+", (source or "").lower())
    if len(toks) < min_tokens:
        return True
    if span_supported(label, source):
        return True
    try:
        from agents.agent3_synthesizer import _name_in_source
        if _name_in_source(label, source):
            return True
    except Exception:
        pass
    words = [w for w in re.findall(r"[a-z]+", str(label).lower()) if w not in {"mr", "mrs", "ms", "dr", "miss", "shri", "smt"}]
    if bool(words) and all(_name_among(w, toks) for w in words):
        return True
    compact = "".join(words)                                  # "DadaJi" / "Damad Sahab" against "dada ji" / "damad sahab" in the text: spacing differs, letters do not
    return len(compact) >= 6 and compact in "".join(toks)


# ---- glossary of easily misread romanised Hindi / Urdu words --------------------------------------------------------------------------------------
_GLOSS = None


def gloss_lines(roman_text: str, limit: int = 6):
    """['riyayat = leniency / concession ...', ...] for the glossary words that occur in the (romanised) scene text. Whole words only; at most `limit`."""
    global _GLOSS
    if _GLOSS is None:
        import json as _json
        import os as _os
        try:
            _GLOSS = _json.load(open(_os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "data", "lexicon", "hindi_gloss.json"), encoding="utf-8"))
        except Exception:
            _GLOSS = {}
    toks = set(re.findall(r"[a-z]+", (roman_text or "").lower()))
    return [f"{w} = {m}" for w, m in _GLOSS.items() if w in toks][:limit]
