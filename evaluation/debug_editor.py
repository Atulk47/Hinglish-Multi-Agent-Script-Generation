"""
evaluation/debug_editor.py
LOCAL-MODEL-ONLY trace: for every gate-eligible peer line of a saved trace, show
  clause chosen -> tone -> shortlist offered -> what the Qwen LoRA actually wrote -> which guard accepted/rejected it.
No Groq fallback. Use it to see WHY the trained editor does or doesn't produce an edit.

    python -u -m evaluation.debug_editor outputs/final_a_v3_trace.json
"""
import argparse, json, os, re
os.environ["QWEN_GROQ_FALLBACK"] = "0"
import config
from memory.scratchpad import Scratchpad
from agents.intent_classifier import GroqIntentClassifier
from agents.agent4_slang_rewriter import SlangRewriter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--max", type=int, default=14)
    args = ap.parse_args()
    t = json.load(open(args.trace, encoding="utf-8"))
    sp = Scratchpad("debug_editor")
    sp.write("character_profiles", t.get("character_profiles") or {})
    sr = SlangRewriter(sp, GroqIntentClassifier())
    n = 0
    for sc in t["neutral_script"]:
        for ln in sc["dialogue"].split("\n"):
            if ":" not in ln or n >= args.max:
                continue
            spk, utt = [x.strip() for x in ln.split(":", 1)]
            dec = sr.gate.run_all_gates(spk, utt, sc["scene_id"], sr.intent_clf)
            if not dec["go"] or spk.upper() in ("SPEAKER", "NARRATOR") or "DADI" in spk.upper():
                continue
            span = sr._pick_segment(utt)
            if not span:
                print(f"\n[{sc['scene_id']}] {spk}: no editable sentence"); continue
            core = utt[span[0]:span[1]].strip()
            pol = sr._clause_polarity(core)
            sl = sr._fit_shortlist(core, dec["intent"], "neutral", pol)
            print(f"\n[{sc['scene_id']}] {spk}\n  sentence : {core}\n  tone     : {pol:+d}   shortlist: {sl}")
            if not sl:
                continue
            n += 1
            out = sr._qwen_edit(core, sl)
            ok, used, why = sr._vet_edit(core, out, sl, pol)
            print(f"  qwen out : {out}\n  verdict  : {'ACCEPTED' if ok else 'rejected: ' + why}  (slang={used})")


if __name__ == "__main__":
    main()
