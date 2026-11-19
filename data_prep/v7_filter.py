"""
data_prep/v7_filter.py  -  the deterministic safety checks a teacher edit must pass before it may become training data (and the same checks score any model in the benchmark).

An edit is accepted when ALL hold:
  * the situation allows slang at all (data_prep.v7_format.context_allows_edit): an edit made where none is allowed is rejected, not "fixed";
  * roman letters only, a sane length, at most MAX EDITS changed regions;
  * the existing edit audit (evaluation/edit_audit.py) raises no BLOCKING flag: locality, negation, names / numbers, question form, polarity, serious or respect context,
    and for pool phrases the runtime rule gate (frames, agreement, meaning checks);
  * gendered verb forms of the original are kept; at most one filler in the line; nothing crude.
Pool slang is checked by the full runtime gate; slang outside the pool is checked by the structural rules above plus (in v7_judge.py) an LLM judge from a different model family.
"""
from __future__ import annotations
import os
import re
import sys
from typing import Dict, List

sys.path.insert(0, os.getcwd())
from evaluation.edit_audit import audit_edit, diff_edit

FILLERS = {"yaar", "arre", "are", "bro", "bhai", "abey", "abe", "boss", "oye", "re", "dost"}
GENDERED = {"raha", "rahi", "rahe", "gaya", "gayi", "gaye", "tha", "thi", "the", "hoon", "karta", "karti", "karte", "chahta", "chahti", "chahte", "aaya", "aayi", "aaye",
            "gaya", "diya", "diye", "liya", "liye", "bola", "boli", "bole", "kiya", "kiye", "gira", "giri", "ruka", "ruki", "hua", "hui", "hue", "lagta", "lagti", "lagte"}
PROFANE = {"chutiya", "madarchod", "behenchod", "bhosdi", "gandu", "randi", "harami", "kamina", "saala", "saali", "lund", "gaand", "bc", "mc"}
BLOCKING_EXTRA = {"unknown-insertion-many"}


def _toks(s: str) -> List[str]:
    return re.findall(r"[a-z']+", (s or "").lower())


def check(ctx: Dict, edited: str) -> Dict:
    """-> {'ok': bool, 'changed': bool, 'reasons': [..], 'flags': [..]}. 'ok' with changed=False is a valid "keep as is"."""
    line = (ctx.get("line") or "").strip()
    e = (edited or "").strip()
    out = {"ok": False, "changed": False, "reasons": [], "flags": []}
    if not e:
        out["reasons"].append("empty")
        return out
    if e.lower() == line.lower():
        out["ok"], out["changed"] = True, False
        return out
    out["changed"] = True
    R = out["reasons"]
    if not ctx.get("allows_edit", True):
        R.append("edit-where-none-allowed")
    if re.search(r"[ऀ-ॿ]", e):
        R.append("devanagari")
    ratio = len(e) / max(1, len(line))
    if ratio > 1.6 or ratio < 0.7:
        R.append("length")
    d = diff_edit(line, e)
    if d["regions"] > int(ctx.get("max_edits") or 2):
        R.append("too-many-edits")
    a = audit_edit(line, e, None, role=(ctx.get("speaker") or {}).get("role"), peer_ok=True)
    out["flags"] = [f["code"] for f in a["flags"]]
    # the pool's runtime rule gate judges a whole diff as ONE swap; for a line with several edits (a swap plus a filler) it would reject a legitimate line, so it applies only to
    # single-region edits. Every other check (locality, negation, names, polarity, register, gender forms ...) applies to all edits.
    out["soft"] = []
    for f in a["flags"]:
        if not f["blocking"]:
            continue
        if f["code"] == "rule-gate" and d["regions"] > 1:
            continue
        if f["code"] == "rule-gate" and "appended-or-inserted" in str(f.get("detail", "")):
            out["soft"].append("appended-tail")               # earlier research: such tails were mostly judged bad; keep ONLY if the judge gives 5/5 (see v7_teacher.accepted)
            continue
        R.append("audit:" + f["code"])
    lt, et = set(_toks(line)), set(_toks(e))
    if [w for w in (lt - et) if w in GENDERED]:
        R.append("gender-form-changed")
    if sum(1 for w in _toks(e) if w in FILLERS) - sum(1 for w in _toks(line) if w in FILLERS) > 1:
        R.append("stacked-filler")
    if et & PROFANE:
        R.append("profanity")
    out["ok"] = not R
    return out


def is_filler_only(ctx: Dict, edited: str) -> bool:
    """True when the only thing added is a filler word (counted separately so fillers cannot dominate the data)."""
    added = [w for w in _toks(edited) if w not in set(_toks(ctx.get("line", "")))]
    return bool(added) and all(w in FILLERS for w in added)
