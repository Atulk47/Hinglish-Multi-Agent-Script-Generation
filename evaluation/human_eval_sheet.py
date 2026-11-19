"""
evaluation/human_eval_sheet.py  -  a BLIND rating sheet for native speakers, built from saved outputs (no model calls, no cost).

    python -m evaluation.human_eval_sheet            # outputs/human_eval/sheet.csv + key.csv + README.md

Each item = one Hindi story + two Hinglish scripts (A / B, order randomised per item) made from it. The rater never sees which system wrote which: that mapping lives only in
key.csv (keep it away from the raters). Items: the pipeline against ONE Opus call with the rules in the prompt (opus_rules) and against the plain-prompt Opus call (opus_simple), on
the same 7 stories. Raters fill: more_faithful, more_natural, slang_fit, overall (each A / B / tie) and optional comments. Analyse with evaluation.human_eval_sheet.score(key, filled).
"""
from __future__ import annotations
import csv
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

COLUMNS = ["id", "story_hindi", "script_A", "script_B", "more_faithful", "more_natural", "slang_fit", "overall", "comments"]
README = """# Native-speaker rating sheet

You will see {n} items. Each has a Hindi story and two Hinglish (Roman-script) scripts made from it, A and B. Please do not try to guess which is machine-made or which system wrote which.

For each item fill the four columns with A, B or tie:
- more_faithful : which script stays closer to the story (fewer invented lines / events / speakers, fewer missing events)?
- more_natural  : which sounds more like real people talking (Hinglish a speaker would actually say)?
- slang_fit     : which uses slang (or none) better for these people, their relationships and the situation (no slang to elders / in serious moments; not forced)?
- overall       : which would you prefer as a screenplay?
Add anything odd in `comments` (a wrong word, a wrong gender form, a line nobody would say).

Time: about 3 minutes per item. Two or three raters are enough; rate independently. Return the filled sheet.csv. Keep key.csv away from the raters.
"""


def load_pipeline_script(path):
    run = json.load(open(path, encoding="utf-8"))
    return "\n".join(l.strip() for sc in run["slang_script"] for l in sc["dialogue"].split("\n") if ":" in l and l.strip())


def build(pipeline_dir, baseline_json, out_dir, seed=2026):
    rnd = random.Random(seed)
    base = json.load(open(baseline_json, encoding="utf-8"))
    os.makedirs(out_dir, exist_ok=True)
    rows, key = [], []
    i = 0
    for sid in sorted(base):
        src = os.path.join(ROOT, "data", "stress", sid + ".txt")
        pj = os.path.join(ROOT, pipeline_dir, sid + ".json")
        if not (os.path.exists(src) and os.path.exists(pj)):
            continue
        story = open(src, encoding="utf-8").read().strip()
        pipe = load_pipeline_script(pj)
        for arm in ("opus_rules", "opus_simple"):
            other = (base[sid].get(arm) or {}).get("raw") or (base[sid].get(arm) or {}).get("text")
            if not other:
                continue
            i += 1
            flip = rnd.random() < 0.5
            a, b = (other, pipe) if flip else (pipe, other)
            iid = f"item{i:02d}"
            rows.append({"id": iid, "story_hindi": story, "script_A": a, "script_B": b, "more_faithful": "", "more_natural": "", "slang_fit": "", "overall": "", "comments": ""})
            key.append({"id": iid, "story": sid, "A": arm if flip else "pipeline", "B": "pipeline" if flip else arm})
    rnd.shuffle(rows)                                                   # item order does not reveal the story order either
    with open(os.path.join(out_dir, "sheet.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out_dir, "key.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "story", "A", "B"])
        w.writeheader()
        w.writerows(key)
    open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8").write(README.format(n=len(rows)))
    return len(rows)


def score(key_csv, filled_csvs):
    """-> {'pipeline vs opus_rules': {'more_faithful': {'pipeline': n, 'other': n, 'tie': n}, ...}, ...} pooled over all raters; unreadable cells are skipped."""
    key = {r["id"]: r for r in csv.DictReader(open(key_csv, encoding="utf-8-sig"))}
    out = {}
    for path in filled_csvs:
        for r in csv.DictReader(open(path, encoding="utf-8-sig")):
            k = key.get(r["id"])
            if not k:
                continue
            other = k["A"] if k["B"] == "pipeline" else k["B"]
            bucket = out.setdefault(f"pipeline vs {other}", {c: {"pipeline": 0, "other": 0, "tie": 0} for c in COLUMNS[4:8]})
            for c in COLUMNS[4:8]:
                v = (r.get(c) or "").strip().lower()
                if v == "tie":
                    bucket[c]["tie"] += 1
                elif v in ("a", "b"):
                    who = k[v.upper()]
                    bucket[c]["pipeline" if who == "pipeline" else "other"] += 1
    return out


def main():
    n = build("outputs/stress_full_v4", "outputs/baseline_vs_pipeline_v2.json", os.path.join(ROOT, "outputs", "human_eval"))
    print(f"{n} items -> outputs/human_eval/sheet.csv (give to raters), key.csv (keep), README.md")


if __name__ == "__main__":
    main()
