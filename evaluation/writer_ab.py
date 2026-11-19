"""
evaluation/writer_ab.py  -  PAIRED before/after test of the Pass-1 writer on the same stories.

    python -m evaluation.writer_ab --before outputs/stress6 --after outputs/stress9 --judge-model openai/gpt-oss-120b

For each story both neutral scripts are compared with the SAME judge against the same reference (the saved `opus_rules` script of baseline_vs_pipeline.json), so the
judge, the reference and the stories are fixed and only the writer differs. Reported: padding rate (lines that tack "mujhe bahut accha laga" on, counted by regex, no judge
involved), lines, and the judge's invented / dropped / naturalness. A cheap judge is a direction check; confirm the headline with evaluation.baseline_vs_pipeline.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PAD = re.compile(r"mujhe (?:bahut |kaafi )?(?:accha|achha)(?: laga| lag raha)|(?:bahut|kaafi) (?:accha|achha) (?:laga|lag raha)|yeh (?:bahut )?(?:accha|achha) (?:hai|idea)", re.I)


def load(d, sid):
    run = json.load(open(os.path.join(ROOT, d, sid + ".json"), encoding="utf-8"))
    return [l.strip() for sc in run["result"]["neutral"] for l in sc["dialogue"].split("\n") if ":" in l and l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default="outputs/stress6")
    ap.add_argument("--after", default="outputs/stress9")
    ap.add_argument("--reference", default="outputs/baseline_vs_pipeline.json")
    ap.add_argument("--judge-model", default="openai/gpt-oss-120b")
    ap.add_argument("--out", default="outputs/writer_ab.json")
    a = ap.parse_args()
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    import config
    import llm_client
    from evaluation.baseline_vs_pipeline import JUDGE_SYSTEM
    config.LLM_COST_CAP_USD = 3.0
    config.OPENROUTER_MODELS["judge_ext"] = a.judge_model
    ref = json.load(open(os.path.join(ROOT, a.reference), encoding="utf-8"))
    sids = [s for s in ref if os.path.exists(os.path.join(ROOT, a.before, s + ".json")) and os.path.exists(os.path.join(ROOT, a.after, s + ".json"))]
    rnd = random.Random(11)

    def judge(job):
        sid, arm = job
        mine = "\n".join(load(a.before if arm == "before" else a.after, sid))
        other = ref[sid]["opus_rules"]["text"]
        flip = rnd.random() < 0.5
        X, Y = (other, mine) if flip else (mine, other)
        text = open(os.path.join(ROOT, "data", "stress", sid + ".txt"), encoding="utf-8").read().strip()
        user = f"STORY:\n{text}\n\nSCRIPT X:\n{X}\n\nSCRIPT Y:\n{Y}"
        try:
            out, _ = llm_client.complete("judge_ext", [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}], temperature=0.0, max_tokens=4000, only="openrouter")
            v = json.loads(out[out.find("{"):out.rfind("}") + 1])
        except Exception as e:
            return sid, arm, {"error": str(e)[:80]}
        mine_key = ("Y" if flip else "X")
        return sid, arm, {k: v.get(mine_key, {}).get(k) for k in ("n_invented", "n_dropped", "naturalness", "slang_fit")}

    with ThreadPoolExecutor(max_workers=4) as ex:
        res = list(ex.map(judge, [(s, arm) for s in sids for arm in ("before", "after")]))
    rows = {s: {} for s in sids}
    for sid, arm, v in res:
        rows[sid][arm] = v
    for sid in sids:
        for arm, d in (("before", a.before), ("after", a.after)):
            lines = load(d, sid)
            rows[sid][arm].update(lines=len(lines), padded=sum(bool(PAD.search(l)) for l in lines))
    json.dump(rows, open(os.path.join(ROOT, a.out), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{'story':26s} {'lines b/a':10s} {'padded b/a':11s} {'invented b/a':13s} {'dropped b/a':12s} {'natural b/a'}")
    tot = {"before": {"lines": 0, "padded": 0, "inv": [], "drop": [], "nat": []}, "after": {"lines": 0, "padded": 0, "inv": [], "drop": [], "nat": []}}
    for sid in sids:
        b, af = rows[sid]["before"], rows[sid]["after"]
        print(f"{sid[:26]:26s} {b['lines']:>3}/{af['lines']:<6} {b['padded']:>3}/{af['padded']:<7} {str(b.get('n_invented')):>5}/{str(af.get('n_invented')):<7} {str(b.get('n_dropped')):>4}/{str(af.get('n_dropped')):<7} {b.get('naturalness')}/{af.get('naturalness')}")
        for arm, r in (("before", b), ("after", af)):
            tot[arm]["lines"] += r["lines"]
            tot[arm]["padded"] += r["padded"]
            for k, key in (("inv", "n_invented"), ("drop", "n_dropped"), ("nat", "naturalness")):
                if isinstance(r.get(key), (int, float)):
                    tot[arm][k].append(r[key])
    mean = lambda l: round(sum(l) / len(l), 2) if l else None
    for arm in ("before", "after"):
        t = tot[arm]
        print(f"{arm:7s} lines={t['lines']} padded={t['padded']} ({t['padded'] / max(1, t['lines']):.1%}) invented={mean(t['inv'])} dropped={mean(t['drop'])} naturalness={mean(t['nat'])}")
    print(llm_client.cost_summary())


if __name__ == "__main__":
    main()
