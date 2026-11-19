"""
evaluation/eval_judge.py  -  VALIDATING THE JUDGE with known answers (no human needed), per criterion, with confidence intervals.

    python -m evaluation.eval_judge [--n-good 100] [--n-events 120]

J1  known-answer perturbations: GOOD edits (gold edits of the held-out kaggle_v5/eval_slots sentences) and the same edits corrupted five ways
    (hallucinated clause, content-word swap, negation flip, unrelated line, wrong-mood slang).  For each judge (and the 2-judge ensemble):
    corruption detection rate and good-edit acceptance, Wilson 95% intervals; agreement between the two judge families (Cohen's kappa).
J2  event coverage: a scene's dialogue against its OWN key events (should be covered) and against ANOTHER scene's events (should not):
    accuracy of the LLM event check vs the keyword and centred-MuRIL checks of the validator (same pairs).

Results -> outputs/judge_eval.json.  The corruptions are produced by evaluation.eval_validator.corruptions (the same generator used for the audit), so every
metric in the project is tested against the same controlled failures.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

load_dotenv()
from evaluation import edit_audit, judge
from evaluation.eval_validator import corruptions
from evaluation.stats import cohen_kappa, fmt_rate, wilson


def build_sets(n_good: int):
    rows = [json.loads(l) for l in open("kaggle_v5/eval_slots.jsonl", encoding="utf-8")]
    good = [r for r in rows if r.get("expect") == "edit" and r.get("gold") and r["gold"] != r["line"]]
    pool = [r["line"] for r in rows]
    rng = random.Random(11)
    items = []
    for r in good:
        slang = edit_audit._infer_slang(edit_audit.diff_edit(r["line"], r["gold"])["added"])
        cs = corruptions(rng, r["line"], r["gold"], slang, pool)
        items.append({"line": r["line"], "good": r["gold"], "corrupt": {k: v for k, v in cs.items() if v and v.strip() != r["gold"].strip()}})
    random.Random(3).shuffle(items)
    return items[:n_good]


def j1(n_good: int, models):
    data = build_sets(n_good)
    jobs = [("good", d["line"], d["good"]) for d in data] + [(k, d["line"], v) for d in data for k, v in d["corrupt"].items()]
    res = judge.judge_pairs([(a, b) for _, a, b in jobs], models=models)
    out = {"n_good": len(data), "models": list(models), "per_judge": {}, "ensemble": {}, "agreement": {}}
    kinds = sorted({k for k, _, _ in jobs if k != "good"})
    views = {m: (lambda r, m=m: judge._accept(r["models"].get(m))) for m in models}
    views["ENSEMBLE(all accept)"] = lambda r: r["accept"]
    views["ANY accepts"] = lambda r: r["accept_any"]
    for name, accept in views.items():
        g = [accept(r) for (k, _, _), r in zip(jobs, res) if k == "good"]
        row = {"good_accepted": fmt_rate(sum(g), len(g)), "detect": {}}
        tot_k = tot_n = 0
        for kd in kinds:
            bad = [not accept(r) for (k, _, _), r in zip(jobs, res) if k == kd]
            row["detect"][kd] = fmt_rate(sum(bad), len(bad))
            tot_k += sum(bad)
            tot_n += len(bad)
        row["overall_detect"] = fmt_rate(tot_k, tot_n)
        out["per_judge" if name in models else "ensemble"][name] = row
    if len(models) == 2:
        a = [judge._accept(r["models"].get(models[0])) for r in res]
        b = [judge._accept(r["models"].get(models[1])) for r in res]
        out["agreement"] = {"cohen_kappa_between_judges": round(cohen_kappa(a, b), 3), "raw_agreement": round(sum(x == y for x, y in zip(a, b)) / len(a), 3)}
    out["unparsed_answers"] = judge.STATS["unparsed"]
    return out


def j2(n_events: int, model):
    from evaluation.muril_validator import MuRILValidator
    scenes = []
    for p in sorted(glob.glob("outputs/*_trace.json")):
        try:
            t = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        for sc in t.get("neutral_script") or []:
            ev = (sc.get("events") or {}).get("key_events") or []
            if ev and sc.get("dialogue") and "[generation failed" not in sc["dialogue"].lower():
                scenes.append((sc["dialogue"], ev))
    rng = random.Random(4)
    pairs = []
    for i, (d, evs) in enumerate(scenes):
        j = rng.choice([k for k in range(len(scenes)) if k != i])
        pairs.append((d, rng.choice(evs), True))
        pairs.append((d, rng.choice(scenes[j][1]), False))
    rng.shuffle(pairs)
    pairs = pairs[:n_events]
    v = MuRILValidator()
    judged = [judge.event_covered(d, e, model) for d, e, _ in pairs]
    judge.save_cache()
    kw = [v._keyword_hit(d, e) for d, e, _ in pairs]
    sem = [v.event_similarity(d, e) >= v.event_sim_min for d, e, _ in pairs]
    truth = [t for _, _, t in pairs]

    def acc(pred):
        ok = [(p == t) for p, t in zip(pred, truth) if p is not None]
        return fmt_rate(sum(ok), len(ok))
    return {"n_pairs": len(pairs), "accuracy_llm_event_check": acc(judged), "accuracy_keyword": acc(kw), "accuracy_centred_muril": acc(sem),
            "unparsed": sum(1 for x in judged if x is None)}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-good", type=int, default=100)
    ap.add_argument("--n-events", type=int, default=120)
    ap.add_argument("--models", default=",".join(judge.DEFAULT_MODELS))
    ap.add_argument("--skip-events", action="store_true")
    a = ap.parse_args()
    models = tuple(m.strip() for m in a.models.split(",") if m.strip())
    out = {"judge_version": judge.JUDGE_VERSION, "j1_known_answer": j1(a.n_good, models)}
    if not a.skip_events:
        out["j2_event_coverage"] = j2(a.n_events, models[0])
    out["judge_cost_usd"] = round(judge.STATS["cost"], 4)
    os.makedirs("outputs", exist_ok=True)
    json.dump(out, open("outputs/judge_eval.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
