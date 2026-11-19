"""
evaluation/director_ab.py  -  PAIRED test of the Slang Director prompt: the same neutral scripts, the old prompt (v3) against the current one (v4).

    python -m evaluation.director_ab --src outputs/stress4 --out outputs/director_ab.json

Only Director calls are made (no local models, about $0.01 per scene). For every scene both arms plan the SAME dialogue, so the difference is the prompt alone.
Reported per arm: nominated slang lines, filler-only lines, refusals that give a quota-style reason, and SAFETY counts: nominations on speakers the Persona marks
as authority / slang level 0 (should be 0), and filler-only nominations on a thanks / apology / question line (should be 0).
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

QUOTA = re.compile(r"already|only one|one nomin|enough|\bcap\b|limit|spread|too many|consecutive|back-to-back|variety|allowed|density", re.I)
THANKS_Q = re.compile(r"dhanyavaad|shukriya|thank|maaf|sorry|\?\s*$", re.I)


def old_system(new: str) -> str:
    s = new
    s = s.replace('''
     (the one exception is a FILLER-ONLY line, see below).''', "")
    i, j = s.index("Density:"), s.index("Answer with JSON only")
    s = s[:i] + '''Density: a lively scene between friends usually has slang on about a third of its lines (2 to 4 out of 6); a formal, serious or one-sided scene
may have none. Spread nominations over different speakers. Adjacent lines are fine when both really express feeling (genuine banter or a shared
reaction). Slang the story already used is a mild reason to prefer a different mood, never a reason to refuse a line that fits.

''' + s[j:]
    s = s.replace('"slang": false, "filler": false, "register"', '"slang": false, "register"')
    s = s.replace('\n"filler" is true only for a FILLER-ONLY line (then "slang" is false).', "")
    return s


def old_user(u: str) -> str:
    return re.sub(r"cap: (\d+) slang line\(s\) - nominate the \d+ best if more fit, do not hold back below it", r"nominate at most \1", u)


def profiles_for(run):
    from agents.agent2b_persona import apply_persona
    persona = run["result"]["persona"]
    names = list(persona.get("characters", {}))
    sids = [e.get("scene_id") for e in run["result"]["event_chain"]]
    return apply_persona({n: {"name": n, "social_role": "peer"} for n in names}, persona, sids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="outputs/stress4")
    ap.add_argument("--out", default="outputs/director_ab.json")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    import config
    config.LLM_COST_CAP_USD = 3.0
    from agents import slang_director as sd, edit_gate

    class Pad:
        def __init__(self, p):
            self.p = p

        def read(self, k):
            return self.p if k == "character_profiles" else None

    NEW_SYSTEM = sd.SYSTEM
    OLD_SYSTEM = old_system(NEW_SYSTEM)
    assert OLD_SYSTEM != NEW_SYSTEM and "FILLER-ONLY" not in OLD_SYSTEM
    cache = os.path.join(ROOT, "outputs", "_director_cache_ab.json")
    res = {"old": [], "new": []}
    files = sorted(glob.glob(os.path.join(ROOT, a.src, "*.json")))
    files = [f for f in files if os.path.basename(f) != "report.json"]
    if a.only:
        files = [f for f in files if any(o in os.path.basename(f) for o in a.only)]
    for f in files:
        run = json.load(open(f, encoding="utf-8"))
        sid = os.path.splitext(os.path.basename(f))[0]
        if not run["result"]["neutral"]:
            continue
        profs = profiles_for(run)
        edit_gate.set_setting((run["result"]["persona"] or {}).get("setting"))
        events = {e["scene_id"]: e for e in run["result"]["event_chain"]}
        for arm in ("old", "new"):
            sd.SYSTEM = OLD_SYSTEM if arm == "old" else NEW_SYSTEM
            d = sd.SlangDirector(Pad(profs), cache_path=cache)
            orig_bc = sd.build_context
            if arm == "old":
                sd.build_context = lambda *x, **k: old_user(orig_bc(*x, **k))
            try:
                for sc in run["result"]["neutral"]:
                    items = sd.dialogue_items(sc["dialogue"])
                    if not items:
                        continue
                    plan = d.plan_scene(sc["scene_id"], sc["dialogue"], events.get(sc["scene_id"]))
                    if not plan:
                        continue
                    for i, (sp, utt) in enumerate(items):
                        v = plan["lines"][i]
                        prof = next((p for k, p in profs.items() if k.lower() == sp.lower()), {}) or {}
                        bad_speaker = (prof.get("social_role") in ("authority", "elder")) or prof.get("slang_level") == 0
                        res[arm].append({"story": sid, "scene": sc["scene_id"], "speaker": sp, "line": utt, "slang": v["slang"], "filler": bool(v.get("filler")),
                                         "quota_refusal": (not v["slang"]) and bool(QUOTA.search(v["why"])), "why": v["why"],
                                         "unsafe_speaker": bool((v["slang"] or v.get("filler")) and bad_speaker),
                                         "filler_on_thanks_or_question": bool(v.get("filler") and THANKS_Q.search(utt))})
                    d.note_used([])
            finally:
                sd.build_context = orig_bc
        print(sid, "done", flush=True)
    sd.SYSTEM = NEW_SYSTEM
    json.dump(res, open(os.path.join(ROOT, a.out), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for arm in ("old", "new"):
        r = res[arm]
        n = max(1, len(r))
        print(f"{arm}: lines={len(r)} slang={sum(x['slang'] for x in r)} ({sum(x['slang'] for x in r)/n:.1%}) filler_only={sum(x['filler'] for x in r)} "
              f"quota_refusals={sum(x['quota_refusal'] for x in r)} unsafe_speaker={sum(x['unsafe_speaker'] for x in r)} "
              f"filler_on_thanks_or_question={sum(x['filler_on_thanks_or_question'] for x in r)}")
    try:
        import llm_client
        print(llm_client.cost_summary())
    except Exception:
        pass


if __name__ == "__main__":
    main()
