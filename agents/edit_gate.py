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
EVAL_WORDS = {"accha", "achha", "acchi", "achhi", "acha", "achi", "bahut", "kaafi", "bilkul", "ekdum", "uttam",
              "sundar", "kharab", "bekaar", "bekar", "badhiya", "badiya", "shandar", "shaandaar", "prashansaniya",
              "sarahniya", "behtareen", "bura", "buri", "ghatiya", "bakwaas", "boring", "zabardast", "kamaal",
              "mast", "solid", "bhi", "hi"}
CTX_WORDS = {"sahi", "theek"}                      # replaceable only as a predicate ("... sahi hai")
FEEL_WORDS = {"chinta", "chintit", "nirasha", "niraasha", "ghabrahat", "pareshani", "pareshan"}
PROB_WORDS = {"samasya", "dikkat"}
FEEL_SLANG = {"full tension", "dil baith raha hai", "jaan nikal gayi", "fatti padi hai", "dimaag fry"}
PROB_SLANG = {"panga", "locha", "lafda", "ghapla", "jhol", "gadbad"}
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


def rule_gate(line, out, slang):
    """-> (ok, reason). Deterministic; no network."""
    sl = slang.lower()
    d = describe(line, out, slang)
    if [w for w in d["extra"] if w not in INTENSIFIERS]:
        return False, "rewrote-other-words"
    lt = _toks(line.lower())
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
        if k <= 1 or prev == "," or prev in names or nxt == ",":
            return True, "filler"
        return False, "filler-mid-sentence"
    if not d["removed"]:
        return False, "appended-or-inserted-slang"          # tail / mid insertions were 0-10% good
    for w in d["removed"]:
        if w in EVAL_WORDS:
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
        return False, "replaced-content-word"
    if any(w in FEEL_WORDS for w in d["removed"]) and sl not in FEEL_SLANG:
        return False, "emotion-word-needs-emotion-idiom"
    return True, "substitution"


def has_slot(sentence: str) -> bool:
    """Can a MOOD-slang edit (anything that is not a filler) possibly pass rule_gate for this sentence?
    rule_gate only accepts such an edit if it REPLACES an evaluative / emotion / problem word that is in the
    sentence. If the sentence contains none, every mood-slang output would be rejected, so the (slow, ~20 s)
    local generation can be skipped with no change to the final result."""
    toks = set(re.findall(r"[a-z']+", sentence.lower()))
    return bool(toks & (EVAL_WORDS | CTX_WORDS | FEEL_WORDS | PROB_WORDS))
