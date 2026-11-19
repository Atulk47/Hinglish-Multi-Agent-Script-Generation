"""
evaluation/edit_audit.py  -  EDIT AUDITOR: an independent check of every slang edit, from the saved neutral / slang text alone.

Why it exists: embedding similarity (MuRIL cosine) is ~0.999 for ANY two lines that share most of their words, so it cannot tell
"yeh bahut bura hua" -> "yeh ekdum wahiyat hua" (good) from "mujhe ekdum jhand lag raha hai" on a fear line, or from slang on a line
about a father's health. The auditor looks at what the edit actually DID:

  * locality       one small contiguous change (not a rewrite, not an appended clause)
  * preservation   names, numbers, negation, question-vs-statement are untouched
  * polarity       the slang's mood (praise / criticism / worry / trouble) matches the word it replaced
  * context        no slang on health / hospital lines, on lines that address an elder / teacher ("aap", "papa"), or from elder speakers
  * rule gate      the deterministic edit_gate.rule_gate is re-run on (neutral, edited, slang), so a pipeline that skipped it is caught
  * style          crude slang, stacked fillers, a filler on a line that already has one

No model, no network, no training: it runs in milliseconds and is deterministic.  `audit_scene` audits a whole scene;
`repair_scene` reverts every edit that fails a BLOCKING check to the neutral line (always safe: the neutral line is the Pass-1 text).
"""
from __future__ import annotations
import difflib
import re
from typing import Dict, List, Optional

from agents import edit_gate as eg
from evaluation.compare import _parse_turns

POS_SIGN = set(eg.POS_SLANG) | {"sab set hai", "sorted", "badiya"}
NEG_SIGN = set(eg.NEG_SLANG) | eg.FEEL_SLANG | eg.PROB_SLANG
EASE_SIGN = set(eg.EASE_SLANG)
KNOWN_SLANG = POS_SIGN | NEG_SIGN | EASE_SIGN | eg.FILLERS | eg.EXISTING_SLANG          # intensifiers (ekdum, bahut ...) are NOT slang
CRUDE = {"jhand", "ghanta", "kachra"}                    # edgy: fine between close peers, never from / to anybody else
POS_WORDS = set(eg.POS_ADJ) | {"sundar", "uttam", "shandar", "behtareen"}
NEG_WORDS = set(eg.NEG_ADJ) | set(eg.WORRY) | set(eg.FEEL_WORDS) | set(eg.PROB_WORDS) | {"mushkil", "musibat", "kharab"}
SERIOUS_ROLES = {"elder", "authority"}
BLOCKING = {"non-local", "negation-changed", "entity-changed", "question-changed", "serious-context", "respect-context", "elder-speaker",
            "polarity-flip", "rule-gate", "crude-to-nonpeer", "feeling-slot-misfit"}          # slang-context-mismatch is a WARNING (taste: the judge decides)
VALUE_WORDS = {"paisa", "paise", "paisay", "ticket", "tickets", "price", "keemat", "daam", "kharcha", "kharch", "rupaye", "rupees", "mehnga", "mehengi",
               "sasta", "sasti", "value", "worth", "cost", "bill", "fees", "fee", "deal", "offer", "discount"}
CONTEXT_SLANG = {"paisa vasool": VALUE_WORDS}                 # slangs that only make sense in a given context (judge finding: "mujhe ekdum paisa vasool lag raha hai")
FEELING_EXEMPT = set(eg.FEEL_SLANG) | set(eg.PROB_SLANG) | set(eg.FILLERS)
for _x in eg.EXTRA_SLOTS.values():                              # joy / pride / anger / like ... slots from the pool file
    (POS_SIGN if _x["sign"] > 0 else NEG_SIGN).update(_x["slangs"])
    (POS_WORDS if _x["sign"] > 0 else NEG_WORDS).update(_x["triggers"])
    FEELING_EXEMPT |= set(_x["slangs"])
    KNOWN_SLANG |= set(_x["slangs"])
_END = ",!?"


def _words(s: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9']+|[.,!?;]", s or "")


def _sign(word_list, sets) -> int:
    pos = any(w in POS_WORDS for w in word_list)
    neg = any(w in NEG_WORDS for w in word_list)
    return 1 if pos and not neg else -1 if neg and not pos else 0


def diff_edit(neutral: str, edited: str) -> dict:
    """The change between two lines: removed / added words and the number of separate changed regions."""
    a, b = _words(neutral), _words(edited)
    la, lb = [w.lower() for w in a], [w.lower() for w in b]
    removed, added, regions = [], [], 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, la, lb, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        regions += 1
        removed += a[i1:i2]
        added += b[j1:j2]
    return {"removed": removed, "added": added, "regions": regions}


def _infer_slang(added: List[str]) -> Optional[str]:
    text = " ".join(w.lower() for w in added if w not in ".,!?;")
    best = None
    for s in sorted(KNOWN_SLANG, key=lambda x: (-len(x), x)):          # longest first, then alphabetical: set order must not decide (PYTHONHASHSEED)
        if re.search(r"\b" + re.escape(s) + r"\b", text):
            best = s
            break
    return best


def _infer_from_text(neutral: str, edited: str) -> Optional[str]:
    """Fallback when the word diff splits a hyphenated idiom ("dil garden-garden ho gaya"): the longest known slang that the edit has and the line did not."""
    n, e = (neutral or "").lower(), (edited or "").lower()
    for s in sorted(KNOWN_SLANG, key=lambda x: (-len(x), x)):
        pat =r"(?<![a-z])" + re.escape(s) + r"(?![a-z])"
        if re.search(pat, e) and not re.search(pat, n):
            return s
    return None


def _addressed_elder(text: str) -> bool:
    """An elder / teacher term used as a FORM OF ADDRESS (vocative: at the start of the line or followed by , ! ?) or 'aap'. A mere mention
    ("Sir ne bola tha", "Dadaji ko yaad kiya", "Aryan Sir ki team") is not addressing anybody."""
    return eg.addresses_elder(text)                  # one shared rule for the gate and the audit


def _health_is_subject(text: str) -> bool:
    """Health / injury words that the line is ABOUT. "chot ke bawajood ..." (despite the injury) only mentions it in passing."""
    toks = re.findall(r"[a-z']+", (text or "").lower())
    for i, t in enumerate(toks):
        if t in eg.HEALTH or t in eg.SERIOUS_SCENE:
            if "bawajood" in toks[i + 1:i + 4] or "baad" in toks[i + 1:i + 3] and "bhi" in toks[i + 1:i + 4]:
                continue
            return True
    return False


def _feeling_misfit(edited: str, sl: str) -> bool:
    """"Mujhe ekdum <crude slang> lag raha hai": a crude word ('jhand', 'kachra', 'ghanta') is never a FEELING ("mujhe ekdum jhand lag raha hai").
    Deliberately narrow: milder quality slang in that slot ("mujhe ekdum ghatiya laga") occurs in the gold data and is left to the judge - rules encode only
    objective failures, not taste inferred from a few examples."""
    if sl not in CRUDE:
        return False
    toks = re.findall(r"[a-z']+", (edited or "").lower())
    st = sl.split()
    k = next((i for i in range(len(toks) - len(st) + 1) if toks[i:i + len(st)] == st), -1)
    if k < 0:
        return False
    before = toks[max(0, k - 3):k]
    after = toks[k + len(st):k + len(st) + 3]
    return "mujhe" in before and any(t.startswith("lag") for t in after)


# a capitalised word that merely STARTS a later sentence ("Mujhe bahut khushi ho rahi hai", "Bahut accha laga") is not a name: common non-name starters
_SENTENCE_STARTERS = ({"mujhe", "main", "hum", "humein", "hamein", "yeh", "woh", "tum", "aaj", "ab", "bahut", "kaafi", "ekdum", "bilkul", "itna", "kitna", "sach", "mera", "meri", "mere"}
                      | set(eg.MINOR_EVAL) | set(eg.POS_ADJ) | set(eg.NEG_ADJ) | set(eg.FEEL_WORDS) | set(eg.PROB_WORDS) | set(eg.EASE_WORDS))


def audit_edit(neutral: str, edited: str, slang: Optional[str] = None, role: Optional[str] = None, scene_text: str = "",
               peer_ok: bool = True) -> dict:
    """-> {changed, ok, flags: [{code, blocking, detail}], removed, added, slang}.  ok = no BLOCKING flag."""
    neutral, edited = (neutral or "").strip(), (edited or "").strip()
    if neutral.lower() == edited.lower():
        return {"changed": False, "ok": True, "flags": [], "removed": [], "added": [], "slang": None}
    d = diff_edit(neutral, edited)
    removed = [w.lower() for w in d["removed"] if w.isalnum() or "'" in w]
    added = [w.lower() for w in d["added"] if w.isalnum() or "'" in w]
    sl = (slang or "").lower().strip() or _infer_slang(d["added"]) or _infer_from_text(neutral, edited)
    flags: List[dict] = []

    def flag(code, detail=""):
        flags.append({"code": code, "blocking": code in BLOCKING, "detail": detail})

    ntoks = [w.lower() for w in _words(neutral)]
    slang_words = (sl or "").split()
    extra_added = [w for w in added if w not in slang_words and w not in eg.INTENSIFIERS]
    # locality: one small change (an appended clause or a rewrite changes meaning in ways no embedding reliably shows)
    if d["regions"] > 2 or len(extra_added) > 3 or len(removed) > (8 if eg.is_emotion_slang(sl) else 6):       # an emotion idiom may replace a 7-word span (+ a possessive)
        flag("non-local", f"{d['regions']} regions, +{len(added)} -{len(removed)} words")
    # preservation
    if any(w in eg.NEGATORS for w in removed) or any(w in eg.NEGATORS for w in added if w not in ntoks and w not in slang_words):     # "pair zameen par nahi hain" is a happy idiom, not a negation
        flag("negation-changed")
    first = (_words(neutral) or [""])[0]
    if any(w.isdigit() for w in removed) or any(w[:1].isupper() and w != first and w.lower() not in _SENTENCE_STARTERS for w in d["removed"] if w.isalpha()):
        flag("entity-changed", " ".join(d["removed"]))             # a number or a (non-initial) capitalised name was replaced
    if neutral.rstrip().endswith("?") != edited.rstrip().endswith("?"):
        flag("question-changed")
    # context
    ctx = set(ntoks) | set(re.findall(r"[a-z']+", scene_text.lower()))
    if _health_is_subject(neutral):
        flag("serious-context", "health / hospital line")
    if _addressed_elder(neutral):
        flag("respect-context", "line addresses an elder / teacher or uses 'aap'")
    elif any(w in eg.RESPECT_ADDR for w in ntoks):
        flag("elder-mentioned", "an elder / teacher is only mentioned")
    if role in SERIOUS_ROLES:
        flag("elder-speaker", str(role))
    # polarity: what was replaced vs the slang's mood
    if sl and removed:
        ws = _sign(removed, None)
        ss = 1 if sl in POS_SIGN else -1 if sl in NEG_SIGN else 0
        if ws and ss and ws != ss:
            flag("polarity-flip", f"replaced {removed} by '{sl}'")
    if sl and sl not in eg.FILLERS and not removed:
        ss = 1 if sl in POS_SIGN else -1 if sl in NEG_SIGN else 0
        cs = _sign([w for w in ntoks], None)
        if ss and cs and ss != cs:
            flag("tone-mismatch", f"'{sl}' on a line that reads {'positive' if cs > 0 else 'negative'}")
    if sl and _feeling_misfit(edited, sl):
        flag("feeling-slot-misfit", f"'{sl}' as the first-person feeling")
    if sl in CONTEXT_SLANG and not (set(re.findall(r"[a-z']+", neutral.lower())) & CONTEXT_SLANG[sl]):
        flag("slang-context-mismatch", f"'{sl}' needs a price / value context")
    # the deterministic acceptance rules, re-run independently of whoever produced the edit
    if sl:
        ok, why = eg.rule_gate(neutral, edited, sl)
        if not ok:
            flag("rule-gate", why)
    elif extra_added:
        flag("unknown-insertion", " ".join(extra_added))
    # style
    if sl in CRUDE:
        flag("crude-to-nonpeer" if not peer_ok else "crude", sl)
    if sl in eg.FILLERS and any(w in eg.FILLERS for w in ntoks):
        flag("stacked-filler", sl)
    return {"changed": True, "ok": not any(f["blocking"] for f in flags), "flags": flags, "removed": d["removed"], "added": d["added"],
            "slang": sl}


def _lines(dialogue: str) -> List[str]:
    return [ln.strip() for ln in (dialogue or "").split("\n") if ln.strip()]


def audit_scene(neutral_scene: dict, slang_scene: dict, profiles: Optional[dict] = None) -> List[dict]:
    """One audit per speaker turn: [{index, speaker, neutral, edited, **audit_edit}] (unchanged turns are included with changed=False)."""
    from agents.name_utils import match_profile
    profiles = profiles or {}
    n_turns, s_turns = _parse_turns(neutral_scene.get("dialogue", "")), _parse_turns(slang_scene.get("dialogue", ""))
    decisions = slang_scene.get("decisions") or []
    scene_text = " ".join(t["text"] for t in n_turns)
    out = []
    for i, s in enumerate(s_turns):
        n = n_turns[i]["text"] if i < len(n_turns) else ""
        dec = decisions[i] if i < len(decisions) else {}
        prof = match_profile(profiles, s["speaker"]) or {}
        a = audit_edit(n, s["text"], dec.get("slang"), role=prof.get("social_role"), scene_text=scene_text)
        out.append({"index": i, "speaker": s["speaker"], "neutral": n, "edited": s["text"], **a})
    return out


def repair_scene(neutral_scene: dict, slang_scene: dict, audits: List[dict]) -> tuple:
    """Revert every turn whose edit failed a blocking check to the neutral text. -> (new slang_scene, [reverted audits])."""
    bad = {a["index"]: a for a in audits if a["changed"] and not a["ok"]}
    if not bad:
        return slang_scene, []
    raw = [ln for ln in (slang_scene.get("dialogue") or "").split("\n")]
    n_turns = _parse_turns(neutral_scene.get("dialogue", ""))
    s_turns = _parse_turns(slang_scene.get("dialogue", ""))          # same parser as audit_scene: turn i here == audit index i
    replace, drop = {}, set()
    for i in bad:
        if i < len(s_turns) and i < len(n_turns):
            first, rest = s_turns[i]["lines"][0], s_turns[i]["lines"][1:]
            m = re.match(r"^\s*([A-Za-z][A-Za-z_. ]*?)\s*:", raw[first])
            if m:
                replace[first] = f"{m.group(1).strip()}: {n_turns[i]['text']}"
                drop.update(rest)                                      # continuation lines of that turn are part of the reverted text
    lines = [replace.get(no, ln) for no, ln in enumerate(raw) if no not in drop]
    new = dict(slang_scene)
    new["dialogue"] = "\n".join(lines)
    decs = [dict(d) for d in (slang_scene.get("decisions") or [])]
    for i, a in bad.items():
        if i < len(decs):
            decs[i].update({"applied": False, "reverted_by_validator": [f["code"] for f in a["flags"] if f["blocking"]]})
    new["decisions"] = decs
    return new, list(bad.values())
