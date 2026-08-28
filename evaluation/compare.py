"""
evaluation/compare.py
Build a line-by-line NEUTRAL (Pass 1)  vs  SLANG (Pass 2) comparison from a run.

The pipeline already stores `neutral_script` and `slang_script` separately; this
pairs them per speaker turn so a demo/panel can see, for every line, the original
sentence, the slangified sentence, and exactly which slang word was inserted.

Pure stdlib (only `re`) so it can be imported anywhere without loading models.
"""
from __future__ import annotations
import re

# A speaker turn looks like  "ROHAN: kuch text"  (speaker may contain spaces/dots,
# e.g. "PROFESSOR SHARMA:").  Anything before the first colon is the speaker.
_TURN = re.compile(r"^\s*([A-Za-z][A-Za-z_. ]*?)\s*:\s*(.*)$")


def _parse_turns(dialogue: str):
    """Split a multi-line 'SPEAKER: line' block into ordered {speaker, text} turns.
    Lines that don't start with a speaker tag are folded into the previous turn."""
    turns = []
    for raw in (dialogue or "").split("\n"):
        line = raw.strip()
        if not line:
            continue
        m = _TURN.match(line)
        if m:
            turns.append({"speaker": m.group(1).strip().upper(), "text": m.group(2).strip()})
        elif turns:                      # continuation of the previous speaker's line
            turns[-1]["text"] = (turns[-1]["text"] + " " + line).strip()
    return turns


def build_comparison(neutral_script, slang_script):
    """Return a list of scenes, each with paired neutral/slang lines.

    [{scene_id, location, lines: [
        {speaker, neutral, slang, slang_added, slang_applied, intent,
         gate_stopped, changed}, ...]}]
    Turns are aligned by position (Agent 4 rewrites line-by-line, so the neutral
    and slang scripts have the same turns in the same order)."""
    neutral_by_id = {s["scene_id"]: s for s in (neutral_script or [])}
    out = []
    for slang in (slang_script or []):
        sid = slang.get("scene_id")
        n_turns = _parse_turns(neutral_by_id.get(sid, {}).get("dialogue", ""))
        s_turns = _parse_turns(slang.get("dialogue", ""))
        decisions = slang.get("decisions", []) or []
        lines = []
        for i, s in enumerate(s_turns):
            n_text = n_turns[i]["text"] if i < len(n_turns) else ""
            dec = decisions[i] if i < len(decisions) else {}
            lines.append({
                "speaker": s["speaker"],
                "neutral": n_text,
                "slang": s["text"],
                "slang_added": dec.get("slang"),
                "slang_applied": dec.get("applied"),
                "intent": dec.get("intent"),
                "gate_stopped": dec.get("gate_stopped"),
                "changed": n_text.strip() != s["text"].strip(),
            })
        out.append({
            "scene_id": sid,
            "location": (slang.get("events") or {}).get("location", ""),
            "lines": lines,
        })
    return out


def print_comparison(comparison):
    """Pretty terminal print of the comparison (used by the runner and demos)."""
    for scene in comparison:
        print(f"\n[{scene['scene_id']}]  {scene.get('location','')}")
        for ln in scene["lines"]:
            tag = ""
            if ln["slang_added"] and ln["slang_applied"]:
                tag = f"   (+slang: {ln['slang_added']})"
            elif ln["gate_stopped"]:
                tag = f"   (no slang - blocked at {ln['gate_stopped']})"
            print(f"  {ln['speaker']}:")
            print(f"     neutral : {ln['neutral']}")
            print(f"     slang   : {ln['slang']}{tag}")
