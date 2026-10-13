"""
evaluation/editor_benchmark.py - slang-editor-only consistency test across DIFFERENT narratives.

Feeds the editor (Agent 4) the saved Pass-1 dialogue of several different stories, line by
line, with the real gates/models, and reports - per story and overall - how it behaves:
  coverage      = edited lines / gate-eligible lines
  engine split  = local LoRA editor vs Groq fallback
  neutral because = why lines were left unedited (no-fit / drift / judge / ...)
It also writes a CSV for HUMAN rating of every edited line (the precision number for the paper).
If Groq errored at any point (rate/daily cap) the run is flagged INVALID: judge/rerank/tone
degrade silently otherwise, and the numbers would mean nothing.

PowerShell:
    $env:GROQ_MODEL="openai/gpt-oss-120b"
    python -m evaluation.editor_benchmark                  # 6 eligible lines per story
    python -m evaluation.editor_benchmark --per-story 4
"""
import argparse
import csv
import json
import os
import re
import time
from collections import Counter

from memory.scratchpad import Scratchpad
from agents.intent_classifier import GroqIntentClassifier
from agents.agent4_slang_rewriter import SlangRewriter

STORIES = {                      # narrative -> saved Pass-1 trace (different plots / registers)
    "library (story 1)":      "outputs/orch_full_trace.json",
    "himalaya (story 2)":     "outputs/orch_s2_trace.json",
    "exhibition (story 3)":   "outputs/orch_s3_trace.json",
    "hackathon (long story)": "outputs/long_test_trace.json",
    "science fair (demo)":    "outputs/loratest8_trace.json",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-story", type=int, default=6, help="gate-eligible lines per story")
    ap.add_argument("--out", default="outputs/editor_benchmark")
    args = ap.parse_args()

    sp = Scratchpad("editor_bench")
    sr = SlangRewriter(sp, GroqIntentClassifier())
    print("backend:", sr.backend, "\n")
    rows, per_story = [], {}
    t0 = time.time()

    for story, path in STORIES.items():
        if not os.path.exists(path):
            print("skip (missing):", path); continue
        t = json.load(open(path, encoding="utf-8"))
        sp.write("character_profiles", t.get("character_profiles") or {})
        sr.reset()
        elig = edited = blocked = 0
        engines, why = Counter(), Counter()
        for sc in t["neutral_script"]:
            if elig >= args.per_story:
                break
            if "[generation failed" in sc.get("dialogue", ""):
                continue
            for ln in sc["dialogue"].split("\n"):
                if elig >= args.per_story:
                    break
                if ":" not in ln or not ln.split(":", 1)[1].strip():
                    continue
                spk, utt = [x.strip() for x in ln.split(":", 1)]
                new, meta = sr.rewrite_line(spk, utt, sc["scene_id"])
                if not meta.get("go") or "skipped" in (meta.get("reason") or ""):
                    blocked += 1; continue
                elig += 1
                if meta.get("applied"):
                    edited += 1
                    engines[meta.get("engine", "?")] += 1
                    rows.append({"story": story, "speaker": spk, "neutral": utt, "edited": new,
                                 "clause": meta.get("edited_clause", ""), "engine": meta.get("engine"),
                                 "human_rating_1to5": "", "human_notes": ""})
                else:
                    note = meta.get("editor_note") or ""
                    for tok in re.findall(r"[a-z\-:]+(?::\d)?", note.replace("kept neutral", "")):
                        if tok.strip(":-"):
                            why[tok.split(":")[0] if tok.startswith("judge") else tok] += 1
        per_story[story] = {"eligible": elig, "edited": edited, "gate_blocked": blocked,
                            "coverage": round(edited / elig, 2) if elig else None,
                            "engines": dict(engines), "left_neutral_because": dict(why)}
        print(f"{story:26s} eligible {elig:2d} | edited {edited:2d} ({100*edited/max(elig,1):3.0f}%) | "
              f"gate-blocked {blocked:2d} | {dict(engines)}")

    tot_e = sum(v["eligible"] for v in per_story.values())
    tot_d = sum(v["edited"] for v in per_story.values())
    eng = Counter()
    for v in per_story.values():
        eng.update(v["engines"])
    errs = sr.stats.get("groq_errors", 0)
    summary = {"stories": per_story, "overall": {
        "eligible": tot_e, "edited": tot_d, "coverage": round(tot_d / max(tot_e, 1), 2),
        "engines": dict(eng), "groq_errors": errs, "seconds": round(time.time() - t0)},
        "valid": errs == 0}
    os.makedirs("outputs", exist_ok=True)
    json.dump(summary, open(args.out + ".json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    with open(args.out + "_review.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["story"])
        w.writeheader(); w.writerows(rows)

    print("\n" + "=" * 70)
    print(f"OVERALL  eligible {tot_e} | edited {tot_d} ({100*tot_d/max(tot_e,1):.0f}%) | "
          f"engines {dict(eng)} | {summary['overall']['seconds']}s")
    if errs:
        print(f"!! INVALID RUN: {errs} Groq call(s) failed (rate/daily cap) - judge/rerank/tone "
              f"were degraded. Do not use these numbers; retry later or on the other model pool.")
    print(f"Saved {args.out}.json and {args.out}_review.csv (team: fill human_rating_1to5)")
    print("=" * 70)


if __name__ == "__main__":
    main()
