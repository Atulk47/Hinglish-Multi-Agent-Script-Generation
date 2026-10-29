"""
agents/edit_gate.py
Deterministic, local "is this edit natural?" gate for the slang editor - replaces the Groq LLM judge.

WHY: the Groq judge (gpt-oss-20b) rejected clearly good local edits ("Yeh kamaal idea hai, Arav." -> 2/5)
and is one more cloud dependency. The rules below were derived from blind ratings of 213 real model
outputs (see evaluation/score_adapters.py): filler-only edits were good 83% of the time, plain insertions
in the middle / at the tail 0-10%, and substitutions only when the replaced word is an evaluative or
emotion word (accha, uttam, kharab, chinta ...) and NOT a content word (a name, "intezaar", "avashyak").

An edit passes only if it is one of:
  1. FILLER      a mood-neutral filler (yaar/bhai/arre/chill/...) placed at the start, after a comma, or
                 right after a name/vocative - and the line is not formal ("aap", "chaliye" ...)
  2. SUBSTITUTE  the slang REPLACES an evaluative word (bahut accha -> zabardast, uttam -> badiya), or an
                 emotion word with a matching emotion idiom (chinta -> full tension), or a problem noun with
                 a matching problem slang (samasya -> panga); adding only a natural intensifier (ekdum/toh)
Everything else - appended idioms, mid-sentence insertions, replaced content words, extra rewritten words -
is rejected, so the line stays neutral.
"""
import difflib
import re

FILLERS = {"yaar", "bhai", "bro", "boss", "abey", "arre", "chill", "kasam se"}
# CORE = a genuine evaluative word that a slang can naturally replace. MINOR = intensifiers / particles that may be
# removed ONLY together with a core word ("bahut accha" -> "zabardast"); alone, "bhi"/"bahut" being swapped for a slang
# is nonsense ("main bhi interested" -> "main mast interested").
CORE_EVAL = {"accha", "achha", "acchi", "achhi", "acha", "achi", "uttam", "sundar", "kharab", "bekaar", "bekar",
             "badhiya", "shandar", "shaandaar", "prashansaniya", "sarahniya", "behtareen", "bura", "buri", "ghatiya",
             "boring"}
MINOR_EVAL = {"bahut", "kaafi", "bilkul", "ekdum", "bhi", "hi", "toh", "se"}
# words that are ALREADY slang: replacing one with another slang adds nothing ("zabardast" -> "dhamaal")
EXISTING_SLANG = {"zabardast", "kamaal", "mast", "solid", "kadak", "lajawaab", "jhakaas", "gazab", "faadu",
                  "dhamaal", "fire", "lit", "banger", "makkhan", "dhasu", "jalwa", "bawaal", "panga", "locha",
                  "lafda", "ghapla", "jhol", "gadbad", "flop", "jhand", "kachra", "bakwaas", "sorted",
                  "bindaas", "faltu", "wahiyat", "chakkar", "jhamela"}
EVAL_WORDS = CORE_EVAL | MINOR_EVAL | EXISTING_SLANG            # (kept for older imports)
# female names: "bhai" must never follow one (the first demo produced "Riy bhai")
FEMALE_NAMES = {"meera", "sneha", "ananya", "pooja", "isha", "riya", "tanvi", "divya", "neha", "kavya", "simran",
                "mehak", "aditi", "sakshi", "naina", "payal", "diya", "riy", "anika", "sana", "priya", "nisha",
                "shreya", "kriti", "aarti", "sita", "radha", "anjali", "sunita", "komal", "ritu", "pallavi"}
CTX_WORDS = {"sahi", "theek"}                      # replaceable only as a predicate ("... sahi hai")
FEEL_WORDS = {"chinta", "chintit", "nirasha", "niraasha", "ghabrahat", "pareshani", "pareshan"}
PROB_WORDS = {"samasya", "dikkat", "problem"}
EASE_WORDS = {"aasaani", "asaani", "aasani"}                     # "aasaani se" -> "chutki mein"
FEEL_SLANG = {"full tension", "dil baith raha hai", "jaan nikal gayi", "fatti padi hai", "dimaag fry",
              "pasine chhoot gaye", "haath-pair phool gaye"}
PROB_SLANG = {"panga", "locha", "lafda", "ghapla", "jhol", "gadbad", "chakkar", "jhamela"}
EASE_SLANG = {"chutki mein", "bacchon ka khel", "left hand ka khel"}
INTENSIFIERS = {"ekdum", "toh", "bilkul", "kaafi", "bahut"}
OPENER_WORDS = {"chaliye", "chalo", "theek", "hai", "haan", "ok", "accha", "achha"}
FORMAL = re.compile(r"\b(aap|aapka|aapki|aapke|aapko|chaliye|kijiye|kariye|sir|madam|ji)\b")


def _toks(s):
    return re.findall(r"[A-Za-z']+|[.,!?;]", s)


def describe(line, out, slang):
    a, b = _toks(line.lower()), _toks(out.lower())
    removed, added = [], []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if tag in ("delete", "replace"):
            removed += a[i1:i2]
        if tag in ("insert", "replace"):
            added += b[j1:j2]
    sl = slang.lower().split()
    return {"removed": [w for w in removed if w.isalpha()],
            "extra": [w for w in added if w.isalpha() and w not in sl]}


GLUE = {"ekdum", "bada", "badi", "bilkul", "ek", "toh", "ho", "raha", "rahi", "rahe", "hai", "hain", "tha", "thi", "hua",
        "hui", "gaya", "gayi", "ka", "ki", "ke", "mein", "se", "kaafi", "bahut", "full", "lag", "lagta", "lagti"}
REMOVABLE_NOUNS = {"situation", "halat", "condition", "baat", "cheez"}


def phrase_swap_ok(line, out, slang):
    """Safe 'idiom for phrase' substitution: ONE contiguous span of <= 5 words that contains a slot word is replaced by a
    mood slang of the matching kind (bahut mushkil situation -> bada panga; bahut dar lag raha hai -> fatti padi hai).
    Everything outside the span must be identical; only simple glue words (ekdum / ho / raha / hai ...) may come with the
    slang; the span may not touch a negation. Slot detection already excludes formal, elder-addressed and health lines."""
    sl = slang.lower().strip()
    if not sl or sl in FILLERS:
        return False
    a, b = _toks(line.lower()), _toks(out.lower())
    p = 0                                           # common prefix / suffix, so everything outside ONE span is identical
    while p < min(len(a), len(b)) and a[p] == b[p]:
        p += 1
    s = 0
    while s < min(len(a), len(b)) - p and a[len(a) - 1 - s] == b[len(b) - 1 - s]:
        s += 1
    st = sl.split()
    span_a = span_b = end_a = None
    for sp in range(s, -1, -1):                     # the slang may itself end in a word the sentence also has ("... padi hai")
        sb = b[p:len(b) - sp]
        if any(sb[x:x + len(st)] == st for x in range(len(sb) - len(st) + 1)):
            span_a, span_b, end_a = a[p:len(a) - sp], sb, len(a) - sp
            break
    if span_a is None or not span_a:
        return False
    if len(span_a) > 5 or any(not t.isalpha() for t in span_a + span_b):
        return False
    slots = [s for s in slot_candidates(line) if s["word"] in span_a and sl in [x.lower() for x in s["slangs"]]]
    if not slots:
        return False
    if any(t in NEGATORS for t in a):               # any negation anywhere in the sentence: too risky for a phrase swap
        return False
    k = next((x for x in range(len(span_b) - len(st) + 1) if span_b[x:x + len(st)] == st), -1)
    extras = span_b[:k] + span_b[k + len(st):]
    if len(extras) > 3 or any(t not in GLUE for t in extras):
        return False
    slot_words = {s["word"] for s in slots}
    return all(t in slot_words or t in MINOR_EVAL or t in GLUE or t in REMOVABLE_NOUNS for t in span_a)


# grammatical gender of the noun-like mood slangs; a verb or adjective that disagrees ("thodi dimaag fry ho rahi hai") is rejected
SLANG_GENDER = {"dimaag fry": "m", "full tension": "f", "kachra": "m", "flop": "m", "jhand": "m", "panga": "m", "locha": "m",
                "lafda": "m", "ghapla": "m", "jhol": "m", "gadbad": "f", "chakkar": "m", "jhamela": "m"}
_FEM_FORMS = {"rahi", "gayi", "hui", "thi", "badi", "thodi", "kaafi badi"}
_MASC_FORMS = {"raha", "gaya", "hua", "tha", "bada", "thoda"}


def agreement_ok(out, slang):
    """Only where the slang noun IS the grammatical subject/adjective target: the adjective right before it (badi/bada,
    thodi/thoda) and a 'ho <form>' verb right after it. A predicate noun ("film ekdum flop thi") agrees with the film, not the slang."""
    g = SLANG_GENDER.get(slang.lower())
    if not g:
        return True
    toks = re.findall(r"[a-z']+", out.lower())
    st = slang.lower().split()
    k = next((i for i in range(len(toks) - len(st) + 1) if toks[i:i + len(st)] == st), -1)
    if k < 0:
        return True
    bad = _FEM_FORMS if g == "m" else _MASC_FORMS
    before = toks[k - 1] if k > 0 else None
    after = toks[k + len(st):k + len(st) + 2]
    if before in bad and before in ("badi", "bada", "thodi", "thoda"):
        return False
    if after[:1] == ["ho"] and len(after) == 2 and after[1] in bad:
        return False
    return True


def rule_gate(line, out, slang):
    """-> (ok, reason). Deterministic; no network. Strict single-word rules first, then the safe phrase swap."""
    ok, why = _strict_gate(line, out, slang)
    if not ok and why in ("rewrote-other-words", "only-minor-words-replaced", "replaced-content-word", "replaced-idiom-word",
                          "emotion-word-needs-emotion-idiom") and phrase_swap_ok(line, out, slang):
        ok, why = True, "phrase-swap"
    if ok and slang and not agreement_ok(out, slang):
        return False, "gender-agreement"
    return ok, why


def _strict_gate(line, out, slang):
    sl = slang.lower()
    d = describe(line, out, slang)
    if [w for w in d["extra"] if w not in INTENSIFIERS]:
        return False, "rewrote-other-words"
    lt = _toks(line.lower())
    if sl in FILLERS and any(w in RESPECT_ADDR for w in lt):
        return False, "filler-to-elder"
    if sl in FILLERS:
        # an opener may be swapped for a filler ("Chaliye, ..." / "Theek hai, ..." -> "Arre, ...")
        if [w for w in d["removed"] if w not in OPENER_WORDS] or d["extra"]:
            return False, "filler-with-rewrite"
        if FORMAL.search(line.lower()) and not d["removed"]:
            return False, "filler-on-formal-line"
        ot = _toks(out.lower())
        first = sl.split()[0]
        k = ot.index(first) if first in ot else -1
        prev = ot[k - 1] if k > 0 else None
        nxt = ot[k + len(sl.split())] if 0 <= k and k + len(sl.split()) < len(ot) else None
        names = [w.lower() for w in re.findall(r"\b[A-Z][a-z]+\b", line)]
        # natural spots: sentence start, after a comma, right after a name (vocative), or closing a
        # clause just before a comma ("Chinta mat karo yaar, ...")
        if first == "bhai" and prev in FEMALE_NAMES:
            return False, "bhai-after-female-name"
        if k <= 1 or prev == "," or prev in names or nxt == ",":
            return True, "filler"
        return False, "filler-mid-sentence"
    if not d["removed"]:
        return False, "appended-or-inserted-slang"          # tail / mid insertions were 0-10% good
    if any(w in EXISTING_SLANG for w in d["removed"]):
        return False, "replaced-existing-slang"
    if not any(w in CORE_EVAL or w in CTX_WORDS or w in FEEL_WORDS or w in PROB_WORDS or w in EASE_WORDS
               for w in d["removed"]):
        return False, "only-minor-words-replaced"
    for w in d["removed"]:
        if w in CORE_EVAL or w in MINOR_EVAL:
            continue
        if w in CTX_WORDS:
            i = lt.index(w) if w in lt else -1
            if 0 <= i < len(lt) - 1 and lt[i + 1] in ("hai", "tha", "hoga", "hain", "lag"):
                continue
            return False, "replaced-idiom-word"
        if w in FEEL_WORDS and sl in FEEL_SLANG:
            continue
        if w in PROB_WORDS and sl in PROB_SLANG:
            continue
        if w in EASE_WORDS and sl in EASE_SLANG:
            continue
        return False, "replaced-content-word"
    if any(w in FEEL_WORDS for w in d["removed"]) and sl not in FEEL_SLANG:
        return False, "emotion-word-needs-emotion-idiom"
    return True, "substitution"


# ── SLOT TYPES: what a mood slang can naturally REPLACE, and which slangs fit each kind of slot ─────────────────────
POS_SLANG = ["bindaas", "paisa vasool", "zabardast", "kamaal", "mast", "kadak", "solid", "lajawaab", "jhakaas", "gazab", "faadu", "top class",
             "on point", "ek number", "dhamaal"]
NEG_SLANG = ["ghatiya", "bakwaas", "kachra", "flop", "jhand", "bekaar", "faltu", "wahiyat", "third class"]
POS_ADJ = {"accha", "achha", "acchi", "achhi", "acha", "achi", "achcha", "achchi", "achche", "acche", "achhe",
           "badhiya", "shandar", "shaandaar", "behtareen"}          # formal praise (uttam, sarahniya ...) is NOT a slot
RESPECT = {"aap", "aapki", "aapka", "aapke", "aapko", "aapne", "aapse", "aapaka", "aapaki", "aapake", "aapane", "ji"}
NEG_ADJ = {"kharab", "bekaar", "bekar", "bura", "buri", "ghatiya", "boring"}
WORRY = {"chinta", "nirasha", "niraasha", "ghabrahat", "pareshani", "dar"}      # "dar" only before lag/lagta (see slot_candidates)
TROUBLE_PHRASE = {"mushkil", "mushkilat", "musibat"}               # only replaceable as a PHRASE ("bahut mushkil situation" -> "bada panga")
SLOT_TABLE = [                      # (name, sign, trigger words, slangs that fit this slot)
    ("praise", +1, POS_ADJ, POS_SLANG),
    ("criticism", -1, NEG_ADJ, NEG_SLANG),
    ("worry", -1, WORRY, sorted(FEEL_SLANG)),
    ("trouble", -1, PROB_WORDS, sorted(PROB_SLANG)),
    ("trouble", -1, TROUBLE_PHRASE, sorted(PROB_SLANG)),
    ("ease", +1, EASE_WORDS, sorted(EASE_SLANG)),
]
# words that mean the line is addressed to an elder / teacher / coach: never add slang or fillers ("Papa yaar, ...")
RESPECT_ADDR = {"sar", "sir", "madam", "maam", "papa", "mummy", "mumma", "dadaji", "dadi", "dada", "nani", "nana", "naniji",
                "pitaji", "maa", "coach", "koch", "guruji", "uncle", "aunty"}
SERIOUS_SCENE = {"aspatal", "hospital", "doctor", "dakter", "tabiyat", "bimar", "bimari", "icu", "dawai", "operation"}
NEGATORS = {"nahi", "nahin", "nhi", "mat", "na", "naa"}
HEALTH = {"tabiyat", "tabeeyat", "sehat", "haalat", "halat", "bimar", "bimari", "dard", "chot", "ghayal", "hospital",
          "aspatal", "doctor", "marij"}              # "tabiyat kharab" is a fact about health, not an opinion: no slang


def slot_candidates(sentence: str):
    """The places in a sentence where a MOOD slang can replace a word, each with the slangs that fit THAT slot:
    [{'word','type','sign','slangs'}]. A word under negation ("accha nahi", "chinta mat karo", "koi problem nahi") is
    never a slot - replacing it would flip the meaning or sound absurd."""
    toks = re.findall(r"[a-z']+|,", sentence.lower())
    out = []
    if any(t in RESPECT or t in RESPECT_ADDR for t in toks):   # 'aap' / addressing an elder or coach: no slang
        return out
    if any(t in HEALTH for t in toks):           # illness / injury / hospital lines are serious: no mood slang
        return out
    for i, t in enumerate(toks):
        around = [x for x in toks[max(0, i - 2):i] + toks[i + 1:i + 4]]
        if any(x in NEGATORS for x in around):
            continue
        for name, sign, words, slangs in SLOT_TABLE:
            if t in words:
                if name == "praise" and toks[i + 1:i + 2] == [","]:
                    continue                                        # "Accha, chalo ..." is an interjection, not an opinion
                if t == "dar" and toks[i + 1:i + 2] not in (["lag"], ["lagta"], ["lagti"], ["lagne"]):
                    continue                                        # "dar" must be the fear noun, not a part of something else
                if t in ("bura", "buri") and toks[i + 1:i + 2] in (["maan"], ["maane"], ["mat"]):
                    continue
                out.append({"word": t, "type": name, "sign": sign, "slangs": list(slangs)})
        if t in CTX_WORDS:              # "idea sahi hai" -> "idea solid hai"; not the agreement "Theek hai, ..."
            if i >= 2 and toks[i - 1] != "," and toks[i + 1:i + 2] and toks[i + 1] in ("hai", "tha", "hoga", "hain", "lag"):
                out.append({"word": t, "type": "predicate", "sign": +1, "slangs": POS_SLANG[:8] + ["sorted"]})
    return out


def has_slot(sentence: str) -> bool:
    """Can a MOOD-slang edit possibly pass rule_gate for this sentence? (= it has a slot, see slot_candidates)"""
    return bool(slot_candidates(sentence))


# How strong each mood slang is (1 mild .. 3 strong) - used to match slang strength to the emotion's intensity.
INTENSITY = {"badiya": 1, "sahi hai": 1, "mast": 1, "solid": 2, "on point": 2, "top class": 2, "lajawaab": 2, "kadak": 2,
             "bindaas": 2, "paisa vasool": 2, "zabardast": 2, "kamaal": 2, "ek number": 2, "jhakaas": 3, "gazab": 3,
             "faadu": 3, "dhamaal": 3, "sorted": 1,
             "bekaar": 1, "faltu": 1, "bakwaas": 2, "ghatiya": 2, "flop": 2, "wahiyat": 2, "third class": 2, "kachra": 3,
             "jhand": 3,
             "full tension": 1, "dimaag fry": 2, "dil baith raha hai": 2, "pasine chhoot gaye": 3, "fatti padi hai": 3,
             "jaan nikal gayi": 3, "haath-pair phool gaye": 3,
             "gadbad": 1, "jhamela": 1, "chakkar": 1, "ghapla": 2, "jhol": 2, "panga": 2, "locha": 2, "lafda": 2,
             "chutki mein": 1, "bacchon ka khel": 2, "left hand ka khel": 2}
NOUN_GENDER = {"panga": "m", "locha": "m", "lafda": "m", "ghapla": "m", "jhol": "m", "gadbad": "f", "chakkar": "m",
               "jhamela": "m"}
