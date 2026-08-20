"""
run_pipeline.py — end-to-end demo runner for the Hinglish MAS pipeline (Groq-first).

Usage:
    python run_pipeline.py --story data/raw/test_story_1.txt --session demo_001
"""
import argparse
import json
import os
import sys
import time

# Windows consoles default stdout to cp1252, which can't encode the pipeline's
# box-drawing chars (──) or Devanagari text -> UnicodeEncodeError. Force UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from orchestrator.pipeline import HinglishPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--story", default="data/raw/test_story_1.txt")
    ap.add_argument("--session", default="demo_001")
    args = ap.parse_args()

    with open(args.story, encoding="utf-8") as f:
        story = f.read()

    print("=" * 64)
    print("  HINGLISH MULTI-AGENT SCRIPT GENERATOR  (Groq-first)")
    print("=" * 64)
    print(f"Story : {args.story}")
    print(f"Session: {args.session}\n")

    t0 = time.time()
    pipe = HinglishPipeline(session_id=args.session)
    final = pipe.run(story)
    elapsed = time.time() - t0

    # Pull intermediate artifacts from the scratchpad for a full trace
    sp = pipe.scratchpad
    print("\n" + "=" * 64)
    print("  ROMANIZED INPUT")
    print("=" * 64)
    print(sp.read("romanized_text"))

    print("\n" + "=" * 64)
    print("  PASS 1  (neutral Hinglish)   vs   PASS 2  (slang)")
    print("=" * 64)
    neutral = {s["scene_id"]: s["dialogue"] for s in (sp.read("neutral_script") or [])}
    for scene in (sp.read("slang_script") or []):
        sid = scene["scene_id"]
        print(f"\n[{sid}]")
        print("  neutral:")
        for l in neutral.get(sid, "").split("\n"):
            print(f"    {l}")
        print("  slang:")
        for l in scene["dialogue"].split("\n"):
            print(f"    {l}")

    print("\n" + "=" * 64)
    print("  FINAL SCREENPLAY")
    print("=" * 64)
    print(final)

    os.makedirs("outputs", exist_ok=True)
    txt_path = f"outputs/{args.session}_script.txt"
    json_path = f"outputs/{args.session}_trace.json"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(final)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "romanized_text": sp.read("romanized_text"),
            "event_chain": sp.read("event_chain"),
            "character_profiles": sp.read("character_profiles"),
            "neutral_script": sp.read("neutral_script"),
            "slang_script": sp.read("slang_script"),
            "final_script": final,
        }, f, ensure_ascii=False, indent=2)

    print(f"\nDone in {elapsed:.1f}s.")
    print(f"Saved: {txt_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
