"""
view_demo.py — pretty-print a run's per-module output for demos / screenshots.

Usage:
    python view_demo.py                       # defaults to outputs/demo11_trace.json
    python view_demo.py outputs/<session>_trace.json

Prints: character roles, per-scene neutral-vs-slang comparison, and the
validation diagnoses — each clean enough to screenshot for a slide.
"""
import json
import sys

from evaluation.compare import print_comparison


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "outputs/demo11_trace.json"
    with open(path, encoding="utf-8") as f:
        t = json.load(f)

    print("=" * 60)
    print("  CHARACTER PROFILES  (Agent 2)")
    print("=" * 60)
    for name, p in t.get("character_profiles", {}).items():
        print(f"  {name:18s} -> role={p.get('social_role','?'):9s} "
              f"slang_level={p.get('slang_level','?')}  "
              f"emotion={p.get('current_emotion','?')}  "
              f"stress={p.get('stress_index','?')}")

    print("\n" + "=" * 60)
    print("  NEUTRAL (Pass 1)  vs  SLANG (Pass 2)   (Agent 3 -> Gates -> Agent 4)")
    print("=" * 60)
    print_comparison(t.get("comparison", []))

    diag = t.get("scene_diagnoses")
    if diag:
        print("\n" + "=" * 60)
        print("  VALIDATION DIAGNOSES  (MuRIL Validator + Redo)")
        print("=" * 60)
        for sid, d in diag.items():
            print(f"  {sid}: {d.get('failure_mode','?'):14s} "
                  f"sim={d.get('similarity','?')} "
                  f"line_min={d.get('line_min_similarity','?')} "
                  f"recall={d.get('event_recall','?')} "
                  f"cov={d.get('slang_coverage','?')} -> "
                  f"{'PASS' if d.get('passed') else 'RE-DO'}")


if __name__ == "__main__":
    main()
