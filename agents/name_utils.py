"""
agents/name_utils.py
Character-name canonicalisation + tolerant profile lookup.

The LLM (Agent 1) and NER often refer to the SAME person by slightly different
surface forms across scenes — "Sharm", "Prof. Sharma", "Prof. Sharm". Left as-is
these become separate character profiles and the gates mis-match speakers. These
helpers give a single canonical key per person and a lookup that tolerates
honorifics + minor spelling variants, so profiles merge and speaker→profile
matching stays robust.

Pure stdlib (re + difflib) so any module can import it without loading models.
"""
from __future__ import annotations
import re
from difflib import SequenceMatcher

# Leading titles stripped when deriving a canonical name. Longer forms first so
# "professor" is removed before a bare "prof" match can leave "essor" behind.
HONORIFICS = [
    # English / standard titles
    "professor", "prof.", "prof", "doctor", "dr.", "dr", "shri", "sri", "smt.",
    "smt", "kumari", "km.", "pandit", "pt.", "master", "mast.", "mr.", "mr",
    "mrs.", "mrs", "ms.", "ms", "sir", "madam", "seth", "thakur", "chaudhary",
    "acharya", "guru", "adhyapak",
    # transliterated Devanagari forms the romanizer emits (प्रोफ़ेसर etc.)
    "prophaesar", "prophesar", "profesar", "prophae",
]

# Surface honorifics that specifically imply an AUTHORITY role.
AUTHORITY_HONORIFICS = {
    "professor", "prof", "prof.", "doctor", "dr", "dr.", "sir", "madam",
    "master", "acharya", "guru", "adhyapak", "pandit", "pt.",
    "prophaesar", "prophesar", "profesar", "prophae",
}


def strip_honorific(name: str):
    """Return (name_without_leading_titles, had_authority_honorific)."""
    n = (name or "").strip()
    had_auth = False
    changed = True
    while changed and n:
        changed = False
        low = n.lower()
        for h in HONORIFICS:
            # leading title, optionally followed by '.' / space
            if low == h or low.startswith(h + " ") or low.startswith(h + "."):
                if h.rstrip(".") in {x.rstrip(".") for x in AUTHORITY_HONORIFICS}:
                    had_auth = True
                n = n[len(h):].lstrip(" .").strip()
                changed = True
                break
    return n, had_auth


# Transliteration artifacts that make an ugly display name; drop these for display
# but keep clean standard titles (Prof., Dr.) as-is.
_ARTIFACT_TITLES = ("prophaesar", "prophesar", "profesar", "prophae")


def display_name(name: str) -> str:
    """A tidy human-facing name: strip transliterated-title artifacts and
    title-case, but leave clean standard titles intact.
    'Prophaesar Sharm' -> 'Sharm';  'Prof. Sharma' -> 'Prof. Sharma'."""
    n = (name or "").strip()
    low = n.lower()
    for a in _ARTIFACT_TITLES:
        if low.startswith(a + " ") or low.startswith(a + "."):
            n = n[len(a):].lstrip(" .").strip()
            break
    # title-case only bare (untitled) names, so "Prof. Sharma" isn't mangled
    if not any(n.lower().startswith(h) for h in ("prof", "dr", "shri", "sir", "smt")):
        n = " ".join(w.capitalize() for w in n.split())
    return n or name


def canonical_name(name: str) -> str:
    """Lowercase, honorific-free, punctuation-free key used to compare names."""
    core, _ = strip_honorific(name or "")
    return re.sub(r"[^a-z0-9 ]", "", core.lower()).strip()


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def same_person(a: str, b: str, fuzz: float = 0.86) -> bool:
    """True if two surface names likely denote the same character."""
    ca, cb = canonical_name(a), canonical_name(b)
    if not ca or not cb:
        return False
    if ca == cb:
        return True
    # one canonical name is a prefix of the other: "sharm" vs "sharma"
    if ca.startswith(cb) or cb.startswith(ca):
        return True
    return _ratio(ca, cb) >= fuzz


def match_profile(profiles: dict, speaker: str) -> dict:
    """Find the profile for `speaker`, tolerating honorifics + minor variants.
    Order: exact canonical → prefix → best fuzzy (>=0.86). Returns {} if none."""
    if not profiles or not speaker:
        return {}
    cs = canonical_name(speaker)
    if not cs:
        return {}
    for k, v in profiles.items():          # exact canonical
        if canonical_name(k) == cs:
            return v
    for k, v in profiles.items():          # prefix (Sharm ~ Sharma)
        ck = canonical_name(k)
        if ck and (ck.startswith(cs) or cs.startswith(ck)):
            return v
    best, best_v = 0.0, {}                  # fuzzy
    for k, v in profiles.items():
        r = _ratio(canonical_name(k), cs)
        if r > best:
            best, best_v = r, v
    return best_v if best >= 0.86 else {}
