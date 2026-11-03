"""
evaluation/evaluate.py
Evaluation harness for the Hinglish slang pipeline.

Runs on a saved pipeline trace (outputs/<session>_trace.json) so it needs NO
Groq calls and does not re-run the pipeline — except the optional --judge pass,
which uses Groq to rate whether each scene's slang fits the intended emotion
(the metric that catches context-mismatches like appending "kangal" [broke] to a
line about being honored).

Usage:
    python -m evaluation.evaluate outputs/orch_full_trace.json
    python -m evaluation.evaluate outputs/orch_full_trace.json --judge
    python -m evaluation.evaluate outputs/*.json          # aggregate several
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys

# Windows cp1252 stdout can't encode Devanagari/box chars -> force UTF-8.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from dotenv import load_dotenv
load_dotenv()   # so --judge can read GROQ_API_KEY from .env

import config
from evaluation.metrics import load_slang_lexicon, scene_slang_metrics, aggregate

JUDGE_PROMPT = """You rate inserted Hindi-English (Hinglish) slang in a screenplay scene on TWO axes, each 1-5.
Neutral (Pass 1): {neutral}
Slang   (Pass 2): {slang}
Scene intent: {intent}
  emotion_fit: does the added slang match the line's emotion/intent? (1=contradicts, 3=neutral, 5=perfect fit)
  placement  : does it read naturally / well-placed? (1=jarring tacked-on, 5=seamless & fluent)
Answer ONLY JSON: {{"emotion_fit": <1-5 int>, "placement": <1-5 int>, "reason": "<=8 words"}}"""


def load_pairs(trace_path):
    """Return neutral/slang scene pairs, preserving decisions for line-level checks."""
    t = json.load(open(trace_path, encoding="utf-8"))
    neutral = {s["scene_id"]: s for s in t.get("neutral_script", [])}
    pairs = []
    for s in t.get("slang_script", []):
        sid = s["scene_id"]
        n = neutral.get(sid, {})
        ev = s.get("events") or n.get("events") or {}
        neutral_scene = {**n, "scene_id": sid, "events": ev}
        slang_scene = {**s, "scene_id": sid, "events": ev}
        pairs.append((sid, neutral_scene, slang_scene, ev.get("scene_goal", "")))
    return pairs


def judge_scene(client, neutral, slang, intent):
    import re
    try:
        resp = client.chat.completions.create(
            model=config.GROQ_MODEL, temperature=0.0, max_tokens=120,
            reasoning_effort=getattr(config, "GROQ_REASONING_EFFORT", "low"),
            timeout=30,
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                neutral=neutral, slang=slang, intent=intent or "neutral")}])
    except Exception as e:
        # e.g. Groq daily-token limit — don't crash the whole eval, just skip judging
        print(f"             judge: SKIPPED ({type(e).__name__})")
        return None
    m = re.search(r"\{.*\}", resp.choices[0].message.content or "", re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+", help="trace .json file(s)")
    ap.add_argument("--judge", action="store_true",
                    help="add Groq LLM-judge context-appropriateness (uses API)")
    ap.add_argument("--lexicon", default=None,
                    help="slang CSV to score against (defaults to FELIX_DATASET_CLEAN)")
    ap.add_argument("--out", default=None, help="write report JSON here")
    args = ap.parse_args()

    trace_files = []
    for pat in args.traces:
        trace_files.extend(glob.glob(pat))
    if not trace_files:
        print("no trace files matched:", args.traces)
        return

    phrases, single = load_slang_lexicon(args.lexicon)
    print(f"slang lexicon: {len(phrases)} phrases / {len(single)} words\n")

    from evaluation.muril_validator import MuRILValidator
    validator = MuRILValidator()

    client = None
    if args.judge:
        from groq import Groq
        client = Groq(api_key=os.environ["GROQ_API_KEY"], timeout=30.0, max_retries=1)

    scenes = []
    for tf in sorted(trace_files):
        for tup in load_pairs(tf):
            scenes.append(tup)

    all_scene_metrics, sims, recalls = [], [], []
    changed_line_sims = []
    changed_line_count = 0
    fits, places = [], []
    print(f"{'scene':<12}{'sim':>7}{'editSim':>8}{'changed':>8}{'recall':>8}"
          f"{'cover':>7}{'dens':>6}{'valid':>7}{'cmi':>6}  slang")
    print("-" * 93)

    for sid, neutral_scene, slang_scene, intent in scenes:
        neutral = neutral_scene.get("dialogue", "")
        slang = slang_scene.get("dialogue", "")
        sm = scene_slang_metrics(neutral, slang, phrases, single)
        val = validator.diagnose_scene(neutral_scene, slang_scene)
        sims.append(val["similarity"])
        recalls.append(val["event_recall"])
        scene_changed_sims = [
            line["similarity"] for line in val["per_line"]
            if (line.get("neutral") or "").strip().casefold()
            != (line.get("slang") or "").strip().casefold()
        ]
        changed_line_sims.extend(scene_changed_sims)
        changed_line_count += len(scene_changed_sims)
        all_scene_metrics.append(sm)
        validity = f"{sm['validity']:>7.2f}" if sm["validity"] is not None else f"{'N/A':>7}"
        edit_sim = (sum(scene_changed_sims) / len(scene_changed_sims)
                    if scene_changed_sims else None)
        edit_sim_text = f"{edit_sim:>8.3f}" if edit_sim is not None else f"{'N/A':>8}"
        print(f"{sid:<12}{val['similarity']:>7.3f}{edit_sim_text}{len(scene_changed_sims):>8d}"
              f"{val['event_recall']:>8.2f}"
              f"{sm['coverage']:>7.2f}{sm['density']:>6.2f}{validity}"
              f"{sm['cmi']:>6.1f}  {', '.join(sm['detected']) or '-'}")
        if client is not None:
            j = judge_scene(client, neutral, slang, intent)
            if j is not None:
                fits.append(float(j.get("emotion_fit", 0)))
                places.append(float(j.get("placement", 0)))
                print(f"             judge: emotion_fit={j.get('emotion_fit')}/5 "
                      f"placement={j.get('placement')}/5  {j.get('reason', '')}")

    agg = aggregate(all_scene_metrics)
    n = max(len(sims), 1)
    report = {
        "traces": trace_files,
        "n_scenes": len(sims),
        "semantic_sim_mean": round(sum(sims) / n, 4),
        "semantic_sim_min": round(min(sims), 4) if sims else 0.0,
        "changed_line_count": changed_line_count,
        "changed_line_semantic_sim_mean": (
            round(sum(changed_line_sims) / len(changed_line_sims), 4)
            if changed_line_sims else None
        ),
        "changed_line_semantic_sim_min": (
            round(min(changed_line_sims), 4) if changed_line_sims else None
        ),
        "changed_line_similarity_definition": (
            "MuRIL cosine similarity over dialogue turns whose neutral and edited text differ; "
            "this is a supporting proxy, not a factual entailment guarantee."
        ),
        "event_recall_mean": round(sum(recalls) / n, 4),
        **agg,
    }
    if fits:
        report["judge_emotion_fit"] = round(sum(fits) / len(fits), 2)   # /5
        report["judge_placement"] = round(sum(places) / len(places), 2)  # /5

    print("\n" + "=" * 48 + "\n  AGGREGATE\n" + "=" * 48)
    for k, v in report.items():
        if k in ("traces",):
            continue
        print(f"  {k:<22} {v}")

    out = args.out or "evaluation/reports/eval_report.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(report, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
