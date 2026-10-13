"""
evaluation/rerun_editor.py
Re-run ONLY the slang editor (Agent 4) on the saved Pass-1 dialogue of an existing trace.
Use it to test editor changes without regenerating Agents 1-3 (saves Groq tokens and ~10 min).

    python -u -m evaluation.rerun_editor outputs/final_a_v3_trace.json --out outputs/final_a_v5_editor_only.json
Prints every line the editor changed (neutral -> slang) and the per-scene engine split.
"""
import argparse, json, os
from memory.scratchpad import Scratchpad
from agents.intent_classifier import GroqIntentClassifier
from agents.agent4_slang_rewriter import SlangRewriter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    t = json.load(open(args.trace, encoding="utf-8"))
    sp = Scratchpad("rerun_editor")
    sp.write("character_profiles", t.get("character_profiles") or {})
    sr = SlangRewriter(sp, GroqIntentClassifier())
    neutral = t["neutral_script"]
    slang = sr.rewrite_script(neutral)

    print("\n" + "=" * 78 + "\nCHANGED LINES  (neutral -> slang)\n" + "=" * 78)
    n_changed = 0
    for sc_n, sc_s in zip(neutral, slang):
        for dec, ln_n, ln_s in zip(sc_s["decisions"], sc_n["dialogue"].split("\n"), sc_s["dialogue"].split("\n")):
            if dec.get("applied"):
                n_changed += 1
                print(f"\n[{sc_s['scene_id']}] {dec['speaker']}  (+{dec.get('slang')}, {dec.get('engine')})")
                print("  neutral:", ln_n.split(":", 1)[-1].strip())
                print("  slang  :", ln_s.split(":", 1)[-1].strip())
    print(f"\n{n_changed} line(s) changed | stats: {sr.stats}")
    if args.out:
        json.dump({"slang_script": slang}, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("saved", args.out)


if __name__ == "__main__":
    main()
