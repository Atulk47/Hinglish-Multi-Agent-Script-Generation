"""
tools/export_maya_jobs.py  -  turns finished stories (pipeline traces) into ONE Kaggle notebook that renders all their lines with Maya voices.

    python -m tools.export_maya_jobs --trace outputs/prof2_trace.json --trace outputs/cricket5_trace.json --name demo
    -> notebooks/maya_batch_demo.py      upload to Kaggle (GPU T4 x1, Internet on), run all, download maya_batch.zip
    python -m tools.import_maya_zip maya_batch.zip            -> the stories now play with Maya voices offline

Both versions of every story (with slang / neutral first pass) are exported by default, because the UI renders both. Lines whose clip is
already in the cache are skipped (use --all to include them again). The text sent to Maya is exactly what the app will ask for later
(audio.tts.maya_key), so the clips are found by their key.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from audio import synthesize, tts

TEMPLATE = "notebooks/_maya_batch_template.py"


def collect_jobs(traces, variants=("slang", "neutral"), include_cached=False, text_mode="auto", narrator=True):
    """-> (jobs, stats): unique render jobs [{key, voice, lang, text, speaker, scene}] and counts."""
    jobs, seen, total, cached = [], set(), 0, 0
    for t in traces:
        trace = json.load(open(t, encoding="utf-8")) if isinstance(t, str) else t
        for v in variants:
            plan, _ = synthesize.prepare_plan(trace, v, narrator, text_mode)
            for p, j in zip(plan, synthesize.plan_jobs(plan)):
                key = tts.maya_key(j)
                total += 1
                if key in seen:
                    continue
                seen.add(key)
                if not include_cached and os.path.exists(os.path.join(tts.MAYA_CACHE, key + ".wav")):
                    cached += 1
                    continue
                jobs.append({"key": key, "voice": j["maya_voice"], "lang": j["maya_lang"], "text": tts.maya_text(j),
                             "speaker": p["speaker"], "scene": p["scene_id"]})
    return jobs, {"lines": total, "unique": len(seen), "already_cached": cached, "to_render": len(jobs)}


def build_notebook(jobs) -> str:
    return open(TEMPLATE, encoding="utf-8").read().replace("__JOBS_JSON__", json.dumps(jobs, ensure_ascii=False, indent=0))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", action="append", required=True, help="pipeline trace json (repeat for several stories)")
    ap.add_argument("--name", default="batch")
    ap.add_argument("--variant", choices=["slang", "neutral", "both"], default="both")
    ap.add_argument("--all", action="store_true", help="also include lines that are already cached")
    ap.add_argument("--no-narrator", action="store_true")
    a = ap.parse_args()
    variants = ("slang", "neutral") if a.variant == "both" else (a.variant,)
    jobs, st = collect_jobs(a.trace, variants, a.all, narrator=not a.no_narrator)
    print(f"{st['lines']} lines in {len(a.trace)} stories -> {st['unique']} unique, {st['already_cached']} already cached, {st['to_render']} to render")
    if not jobs:
        print("Nothing to render: every line is already in the cache.")
        return
    out = f"notebooks/maya_batch_{a.name}.py"
    open(out, "w", encoding="utf-8").write(build_notebook(jobs))
    by = {}
    for j in jobs:
        by.setdefault((j["speaker"], j["voice"]), 0)
        by[(j["speaker"], j["voice"])] += 1
    print("casting:", "; ".join(f"{s} -> {v} ({n})" for (s, v), n in sorted(by.items())))
    print(f"notebook -> {out}   (~{sum(len(j['text']) for j in jobs) / 14 / 60:.0f}-{sum(len(j['text']) for j in jobs) / 14 / 60 * 1.5 + 2:.0f} min of GPU time + 5-10 min install)")


if __name__ == "__main__":
    main()
