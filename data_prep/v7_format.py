"""
data_prep/v7_format.py  -  the contract of the v7 slang editor: a CONTEXT-CONDITIONED LINE EDITOR.

v6 picks ONE phrase from a code-chosen shortlist for ONE clause (or abstains). v7 gets the whole situation and writes the edited LINE itself:

    SETTING   modern urban casual | campus | workplace | traditional (village / period / formal)
    SPEAKER   name (peer | elder | authority, gender, emotion + intensity)
    LISTENER  name (role, gender) and the REGISTER the speaker uses with them (casual | respectful)
    SCENE     one-line gist, plus the previous line
    MAX EDITS 1..3
    HINTS     optional slang from the pool that fits (the model may use them or other natural slang)
    LINE      the neutral Hinglish line

  ->  the same line with 0..MAX EDITS small edits (a feeling word swapped for slang, an idiom, a light interjection), or the line UNCHANGED.

The model must answer UNCHANGED whenever slang is wrong: an elder / authority speaker or listener, a respectful register, a traditional setting, grief, an apology, thanks, a health line.
`context_allows_edit` is the deterministic version of that rule: it decides which training examples are "keep as is" (no teacher call needed) and rejects any teacher edit made
where none is allowed.
"""
from __future__ import annotations
import re
from typing import Dict, List, Optional

SYSTEM_V7 = ("You are a Hinglish (romanized Hindi-English) dialogue editor for young characters. You get a line, who says it to whom, the speaker's emotion, the setting and "
             "optionally some slang hints. Rewrite the line with at most MAX EDITS small, natural edits: swap a plain feeling or opinion word for fitting slang, use an idiom, or add "
             "one light interjection. Keep the meaning, names, numbers, gender forms and the question or statement form EXACTLY the same. Prefer varied, natural slang over a stock "
             "filler. If slang would be wrong or forced (an elder, teacher or boss is speaking or listening, respectful register, a traditional setting, grief, an apology, thanks, "
             "illness) reply with the line UNCHANGED. Reply with ONLY the line.")

SERIOUS_EMOTIONS = {"sad", "ashamed"}
ELDER_ROLES = {"elder", "authority"}
_TITLES = re.compile(r"\b(sir|ma'?am|madam|professor|prof|teacher|principal|warden|doctor|dr|inspector|collector|dukaandaar|shopkeeper|coach|uncle|aunty|nani|nana|dadi|dada|"
                     r"dadaji|dadiji|naniji|nanaji|papa|mummy|maa|pitaji|babuji|chacha|chachi|mausi|bua|tau|tai)\b", re.I)
_SERIOUS_LINE = re.compile(r"\b(sorry|maafi|maaf|dhanyavaad|dhanyawad|shukriya|thank|thanks|mubarak|hospital|doctor|bimar|bimaar|ghayal|chot|dard|mar gaya|maut|shok)\b", re.I)


def role_of(name: str, age_group: Optional[str] = None, social_role: Optional[str] = None) -> str:
    """peer | elder | authority from the profile when there is one, else from the age group and the title in the name."""
    if social_role in ("peer", "elder", "authority"):
        return social_role
    if age_group == "elder":
        return "elder"
    return "authority" if _TITLES.search(name or "") else "peer"


def context_allows_edit(ctx: Dict) -> bool:
    """False when slang would be wrong in this situation (the gold answer is then the line unchanged)."""
    sp, li = ctx.get("speaker") or {}, ctx.get("listener") or {}
    if sp.get("role") in ELDER_ROLES or li.get("role") in ELDER_ROLES or li.get("register") == "respectful":
        return False
    if (ctx.get("setting") or {}).get("traditional"):
        return False
    if sp.get("emotion") in SERIOUS_EMOTIONS and int(sp.get("intensity") or 1) >= 3:
        return False
    line = ctx.get("line", "")
    if _SERIOUS_LINE.search(line) or re.search(r"\baap\b|\bji\b", line, re.I):
        return False
    return True


def _who(p: Dict, extra: str = "") -> str:
    bits = [p.get("role", "peer")] + ([p["gender"]] if p.get("gender") in ("male", "female") else []) + ([extra] if extra else [])
    return f"{p.get('name', '?')} ({', '.join(b for b in bits if b)})"


def render_user(ctx: Dict) -> str:
    st = ctx.get("setting") or {}
    setting = "traditional (village / period / formal speech)" if st.get("traditional") else f"modern {st.get('milieu') or 'urban'}, {st.get('formality') or 'casual'}"
    sp, li = ctx.get("speaker") or {}, ctx.get("listener") or {}
    emo = f"{sp.get('emotion', 'neutral')} {sp.get('intensity', 1)}"
    lines = [f"SETTING: {setting}",
             f"SPEAKER: {_who(sp, 'feels ' + emo)}",
             f"LISTENER: {_who(li, 'register ' + (li.get('register') or 'casual'))}" if li.get("name") else "LISTENER: the other characters (casual)"]
    if ctx.get("gist"):
        lines.append(f"SCENE: {ctx['gist']}")
    if ctx.get("previous"):
        lines.append(f"PREVIOUS LINE: {ctx['previous']}")
    lines.append(f"MAX EDITS: {int(ctx.get('max_edits') or 2)}")
    if ctx.get("hints"):
        lines.append("HINTS (optional): " + ", ".join(ctx["hints"]))
    lines.append(f"LINE: {ctx['line']}")
    return "\n".join(lines)


def make_chat(ctx: Dict, target: str) -> Dict:
    return {"messages": [{"role": "system", "content": SYSTEM_V7}, {"role": "user", "content": render_user(ctx)}, {"role": "assistant", "content": target}]}


def parse_output(text: str, line: str) -> str:
    """The model's reply -> the line (first non-empty line, quotes and a leading 'Name:' removed); empty or runaway output -> the original line (a safe abstention)."""
    t = (text or "").strip().strip('"').strip()
    t = t.split("\n")[0].strip().strip('"').strip()
    if not t or len(t) > 3 * max(20, len(line)):
        return line
    return t
