"""
data_prep/v7_corpus.py  -  neutral lines WITH their real context, taken from the saved pipeline runs (UI traces and stress-run outputs).

    python -m data_prep.v7_corpus build            ->  data/v7/contexts.jsonl  (+ a printed summary)

Every spoken line of every saved run becomes one context example: the line, the speaker (role, gender, the Persona Agent's emotion for that scene), the listener (the previous speaker)
and the register the speaker uses with them, the setting, the scene gist, the previous line. Optional slang HINTS come from the pool through the runtime slot detection. The `story` field is
the split key: whole stories are held out for evaluation, so the benchmark measures generalisation, not memorisation.
"""
from __future__ import annotations
import argparse
import glob
import hashlib
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.getcwd())
from data_prep.v7_format import context_allows_edit, role_of

OUT = "data/v7"


def _story_id(text_first: str, name: str) -> str:
    return name.split("_")[1] if name.startswith("ui_") and False else hashlib.md5((text_first or name).encode("utf-8")).hexdigest()[:8]


def _people(run: dict) -> dict:
    """name -> {role, gender, age} from the profiles (UI traces) or the Persona characters (stress runs)."""
    people = {}
    for n, p in (run.get("character_profiles") or {}).items():
        people[n] = {"role": role_of(n, p.get("age_group"), p.get("social_role")), "gender": p.get("gender")}
    persona = run.get("persona") or (run.get("result") or {}).get("persona") or {}
    for n, p in (persona.get("characters") or {}).items():
        people.setdefault(n, {"role": role_of(n, p.get("age_group")), "gender": p.get("gender")})
    return people


def _register(run: dict, a: str, b: str) -> str:
    prof = (run.get("character_profiles") or {}).get(a) or {}
    for other, rel in (prof.get("relationship") or {}).items():
        if other.lower() == b.lower():
            return rel.get("register") or "casual"
    persona = run.get("persona") or (run.get("result") or {}).get("persona") or {}
    for r in persona.get("relationships") or []:
        if str(r.get("a", "")).lower() == a.lower() and str(r.get("b", "")).lower() == b.lower():
            return (r.get("a_to_b") or {}).get("register") or "casual"
        if str(r.get("b", "")).lower() == a.lower() and str(r.get("a", "")).lower() == b.lower():
            return (r.get("b_to_a") or {}).get("register") or "casual"
    return "casual"


def _emotion(run: dict, name: str, scene: str) -> tuple:
    prof = (run.get("character_profiles") or {}).get(name) or {}
    for e in prof.get("emotion_arc") or []:
        if e.get("scene_id") == scene:
            return e.get("emotion", "neutral"), int(e.get("intensity") or 1)
    persona = run.get("persona") or (run.get("result") or {}).get("persona") or {}
    e = ((persona.get("emotions") or {}).get(scene) or {}).get(name)
    if isinstance(e, dict):
        return e.get("emotion", "neutral"), int(e.get("intensity") or 1)
    return "neutral", 1


def contexts_from_run(run: dict, name: str) -> list:
    from agents import edit_gate as eg
    neutral = run.get("neutral_script") or (run.get("result") or {}).get("neutral") or []
    chain = {e["scene_id"]: e for e in (run.get("event_chain") or (run.get("result") or {}).get("event_chain") or [])}
    persona = run.get("persona") or (run.get("result") or {}).get("persona") or {}
    setting = persona.get("setting") or {}
    eg.set_setting(setting)
    people = _people(run)
    story = hashlib.md5(json.dumps([s.get("dialogue", "")[:60] for s in neutral][:3], ensure_ascii=False).encode()).hexdigest()[:8]
    out = []
    for sc in neutral:
        sid = sc.get("scene_id")
        prev_sp, prev_line = None, ""
        gist = (chain.get(sid) or {}).get("scene_goal") or ""
        for raw in (sc.get("dialogue") or "").split("\n"):
            if ":" not in raw or not raw.strip():
                continue
            sp, utt = [x.strip() for x in raw.split(":", 1)]
            if not utt:
                continue
            spp = people.get(sp) or {"role": role_of(sp), "gender": None}
            lis = people.get(prev_sp) if prev_sp and prev_sp != sp else None
            emo, inten = _emotion(run, sp, sid)
            ctx = {"story": story, "source": name, "scene": sid, "setting": {k: setting.get(k) for k in ("era", "milieu", "formality", "traditional")},
                   "speaker": {"name": sp, "role": spp["role"], "gender": spp.get("gender"), "emotion": emo, "intensity": inten},
                   "listener": ({"name": prev_sp, "role": lis["role"], "gender": lis.get("gender"), "register": _register(run, sp, prev_sp)} if lis else {}),
                   "gist": gist, "previous": prev_line, "line": utt}
            cands = [c for c in eg.slot_candidates(utt)]
            ctx["hints_all"] = list(dict.fromkeys(s for c in cands for s in c["slangs"]))[:80] if cands else []
            ctx["hints"] = ctx["hints_all"][:6]
            ctx["allows_edit"] = context_allows_edit(ctx)
            ctx["has_slot"] = bool(cands)
            out.append(ctx)
            prev_sp, prev_line = sp, utt
    eg.set_setting(None)
    return out


def build() -> list:
    files = [f for f in glob.glob("outputs/ui_*_trace.json")] + [f for f in glob.glob("outputs/stress*/s*.json") + glob.glob("outputs/stress*/h*.json") + glob.glob("outputs/stress*/d*.json") + glob.glob("outputs/stress*/_*.json")]
    seen, rows = set(), []
    for f in sorted(files):
        try:
            run = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if not (run.get("neutral_script") or (run.get("result") or {}).get("neutral")):
            continue
        for c in contexts_from_run(run, os.path.basename(f)):
            key = (c["speaker"]["name"], c["line"].lower())
            if key in seen or len(c["line"].split()) < 3:
                continue
            seen.add(key)
            rows.append(c)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    a = ap.parse_args()
    rows = build()
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "contexts.jsonl"), "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            r["id"] = f"c{i:05d}"
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} unique context examples from {len({r['story'] for r in rows})} stories -> {OUT}/contexts.jsonl")
    print("context allows an edit:", sum(r["allows_edit"] for r in rows), "| must stay unchanged:", sum(not r["allows_edit"] for r in rows), "| has a swappable slot:", sum(r["has_slot"] for r in rows))
    print("speaker roles:", dict(Counter(r["speaker"]["role"] for r in rows)), "| traditional settings:", sum(bool((r["setting"] or {}).get("traditional")) for r in rows))
    print("emotions:", dict(Counter(r["speaker"]["emotion"] for r in rows).most_common(8)))


if __name__ == "__main__":
    main()
