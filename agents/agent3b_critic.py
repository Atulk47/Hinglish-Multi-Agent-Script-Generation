"""
agents/agent3b_critic.py  -  the FIDELITY CRITIC: is the generated dialogue still THE AUTHOR'S STORY?

Agent 3 writes dialogue from a scene. Left alone, a language model happily adds plot: it let the birthday person speak in the scene where the
children plan her surprise, and let her "find out" about it. This critic reads the original scene, the story bible (who is present, who is only
mentioned, which secrets exist) and the dialogue, and reports only CLEAR problems:

    absent_speaker  a character who is not in the scene speaks, or is spoken to as if present
    secret_leak     someone who must not know a secret learns it, is told it or is hinted at it
    invented_event  the dialogue states a plot event / decision / outcome / person that the scene does not contain
    contradiction   the dialogue contradicts the source (who / what / where / when)
    missing_event   a key event of the scene is not covered at all

Two layers: a deterministic check (a speaker label that is not in the scene's cast - free, instant) and one LLM call that judges the rest.
Harmless embellishment (greetings, reactions, rephrasing) is explicitly allowed. The pipeline regenerates a scene with the reviewer's feedback
(up to config.AGENT3_FIX_ATTEMPTS times), keeps the attempt with the fewest problems, and finally deletes any line spoken by an absent character.
"""
from __future__ import annotations
import json
import re
from typing import List, Optional

from agents.name_utils import same_person

PROMPT = """You are a strict CONTINUITY EDITOR checking a screenplay scene against the author's original story.

ORIGINAL SCENE (Hindi):
{source}

Characters PRESENT in the scene: {present}
Characters NOT present (only mentioned, or elsewhere): {absent}
{secrets}Key events that must happen in the scene: {events}

DIALOGUE TO CHECK:
{dialogue}

Report ONLY clear problems, of exactly these types:
- absent_speaker : a NOT-present character speaks, or someone talks to / answers an absent character as if they were there
- secret_leak    : the person a secret is hidden FROM is present and hears it, or a line has them learn / guess / be hinted at it.
                   Characters who already know the secret may talk about it freely when the hidden-from person is NOT present: that is NOT a leak.
- invented_event : the dialogue adds a SIGNIFICANT plot fact that is not in the original scene: a new event, a new decision, a different outcome,
                   a new person or a new important object
- contradiction  : the dialogue contradicts the original scene (who did what, where, when)
- missing_event  : an IMPORTANT key event of the scene is not covered by the dialogue at all
Do NOT report harmless things: greetings, small reactions, feelings, wishes, small plans or promises ("let's start tomorrow", "we will always
be together"), details of the weather or setting, rephrasing, polite filler - the writer may add small natural touches.
Return ONLY JSON: {{"issues": [{{"line": <line number, 0 if the whole scene>, "type": "<one of the types>", "detail": "<short, concrete>"}}]}}
Use an empty list if the dialogue is faithful."""

TYPES = {"absent_speaker", "secret_leak", "invented_event", "contradiction", "missing_event"}


def parse_lines(dialogue: str) -> List[tuple]:
    out = []
    for ln in (dialogue or "").split("\n"):
        if ":" in ln:
            spk, utt = ln.split(":", 1)
            if spk.strip() and utt.strip() and len(spk.split()) <= 3:
                out.append((spk.strip().strip("*#_ "), utt.strip()))
    return out


def _is(speaker: str, names: List[str]) -> bool:
    s = speaker.strip().lower()
    for n in names:
        if n.lower() == s or n.lower().split()[0] == s.split()[0] or same_person(n, speaker):
            return True
    return False


class FidelityCritic:
    def __init__(self, client=None):
        self.client = client
        if self.client is None:
            try:
                from llm_client import make_client
                self.client = make_client("extract")
            except Exception:
                self.client = None

    def check(self, source: str, dialogue: str, present: List[str], absent: List[str], secrets: Optional[List[dict]] = None,
              key_events: Optional[List[str]] = None, extras: Optional[List[str]] = None) -> dict:
        lines = parse_lines(dialogue)
        issues = []
        allowed = list(present) + list(extras or [])
        if present:
            for i, (spk, _) in enumerate(lines, 1):
                if spk.lower() in ("narrator", "bachche", "sab", "sabhi", "all", "everyone", "children", "kids") or _is(spk, allowed):
                    continue
                if _is(spk, absent):
                    issues.append({"line": i, "type": "absent_speaker", "detail": f"{spk} is not in this scene (only mentioned or elsewhere)"})
                elif not re.match(r"^[A-Za-z .'-]+$", spk):
                    continue
                else:
                    issues.append({"line": i, "type": "absent_speaker", "detail": f"{spk} is not one of the characters of this scene"})
        used_llm = False
        if self.client is not None and lines:
            sec = ""
            for s in (secrets or []):
                sec += f"SECRET: {s['what']} - must stay hidden from {', '.join(s['hidden_from'])}.\n"
            prompt = PROMPT.format(source=(source or "")[:2200], present=", ".join(present) or "(unknown)",
                                   absent=", ".join(absent) or "(none)", secrets=sec, events="; ".join(key_events or [])[:700] or "(see the scene)",
                                   dialogue="\n".join(f"{i}. {s}: {u}" for i, (s, u) in enumerate(lines, 1)))
            try:
                r = self.client.chat.completions.create(model="x", messages=[{"role": "user", "content": prompt}],
                                                        temperature=0.0, max_tokens=900)
                txt = r.choices[0].message.content or ""
                arr = json.loads(re.search(r"\{.*\}", txt, re.DOTALL).group()).get("issues") or []
                used_llm = True
                have = {(x["line"], x["type"]) for x in issues}
                for it in arr:
                    t = str(it.get("type", "")).strip()
                    if t in TYPES:
                        try:
                            ln = int(it.get("line", 0))
                        except Exception:
                            ln = 0
                        if (ln, t) not in have:
                            issues.append({"line": ln, "type": t, "detail": str(it.get("detail", ""))[:160]})
            except Exception:
                used_llm = False
        return {"ok": not issues, "issues": issues, "llm_used": used_llm}

    @staticmethod
    def drop_absent_lines(dialogue: str, present: List[str], extras: Optional[List[str]] = None) -> str:
        """Last resort: delete lines spoken by someone who is not in the scene."""
        allowed = list(present) + list(extras or [])
        keep = []
        for ln in (dialogue or "").split("\n"):
            if ":" in ln and present:
                spk = ln.split(":", 1)[0].strip().strip("*#_ ")
                if re.match(r"^[A-Za-z .'-]+$", spk) and spk.lower() not in ("narrator",) and not _is(spk, allowed) and len(spk.split()) <= 3 \
                        and not spk.lower() in ("bachche", "sab", "sabhi", "all", "everyone", "children", "kids"):
                    continue
            keep.append(ln)
        return "\n".join(keep)
