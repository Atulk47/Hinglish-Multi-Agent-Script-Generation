"""Low-RAM before/after: replay demo_001's neutral dialogue through the NEW Agent 4.
Uses only Groq (no NER/MuRIL), so it runs under memory pressure."""
import json
from memory.scratchpad import Scratchpad
from agents.intent_classifier import GroqIntentClassifier
from agents.agent4_slang_rewriter import SlangRewriter

trace = json.load(open("outputs/demo_001_trace.json", encoding="utf-8"))
neutral = trace["neutral_script"]
old_slang = {s["scene_id"]: s["dialogue"] for s in trace["slang_script"]}

sp = Scratchpad("slang_compare")
sp.write("character_profiles", trace["character_profiles"])

rw = SlangRewriter(sp, GroqIntentClassifier(), backend="groq")
new_slang = rw.rewrite_script([{"scene_id": s["scene_id"], "dialogue": s["dialogue"]}
                               for s in neutral])
new_map = {s["scene_id"]: s["dialogue"] for s in new_slang}

print("\n" + "=" * 70)
print("  OLD (monotonous)   vs   NEW (diverse)   — same neutral input")
print("=" * 70)
for s in neutral:
    sid = s["scene_id"]
    print(f"\n[{sid}]")
    print("  OLD:")
    for l in old_slang.get(sid, "").split("\n"):
        print("    " + l)
    print("  NEW:")
    for l in new_map.get(sid, "").split("\n"):
        print("    " + l)
