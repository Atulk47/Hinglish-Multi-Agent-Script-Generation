"""
data_prep/v7_bench.py  -  one benchmark for every slang editor: it scores a file of answers on the held-out TEST stories (data/v7/test.jsonl).

    python -m data_prep.v7_bench gen-llm --model google/gemini-3.6-flash --label zeroshot_flash --budget 0.3     # any prompted model, same prompt as v7 (no training)
    python -m data_prep.v7_bench score --label v7_1_5b --answers eval_outputs_v7_1_5b.jsonl [--judge --budget 0.4]
    python -m data_prep.v7_bench table                                                                          # side-by-side table of every scored label

An answers file is JSONL: {"id": "<test row id>", "output": "<the edited line>"}  (the Kaggle script writes it; `teacher` is scored from data/v7/teacher_raw.jsonl).
Metrics (all per answer, then summed; "eligible" = the situation allows slang):
  forced_keep_ok     share of the FORBIDDEN-situation rows (elder / respectful / grief / thanks ...) left unchanged       (safety; must be ~100%)
  edit_rate          share of eligible rows the model changed
  valid_of_edits     share of its changes that pass every deterministic check (v7_filter: locality, names, numbers, negation, gender forms, register ...)
  accepted_per_100   valid changes the cross-family judge accepts (natural >= 4, meaning kept, register ok) per 100 eligible rows   (the headline; needs --judge)
  distinct_slang     different added words among the valid changes (variety), and top_share = the share of the single most common one
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.getcwd())
from dotenv import load_dotenv
load_dotenv(".env")
import config
import llm_client as L
from data_prep import v7_filter, v7_teacher
from data_prep.v7_assemble import added_key

D = "data/v7"
OUTD = f"{D}/bench"


def _read(p):
    return [json.loads(x) for x in open(p, encoding="utf-8")] if os.path.exists(p) else []


def _answers(label: str, path: str | None):
    if label == "teacher":
        raw = {r["id"]: r["edited"] for r in _read(f"{D}/teacher_raw.jsonl")}
        return raw
    return {r["id"]: r["output"] for r in _read(path or f"{OUTD}/{label}.jsonl")}


def score(label: str, path: str | None = None, judge: bool = False, budget: float = 0.4, judge_model: str = "google/gemini-3.6-flash") -> dict:
    test = _read(f"{D}/test.jsonl")
    ans = _answers(label, path)
    forced = [r for r in test if r["kind"] == "keep_forced"]
    elig = [r for r in test if r["kind"] != "keep_forced" and r["id"] in ans]
    res = {"label": label, "test_rows": len(test), "answered": sum(r["id"] in ans for r in test)}
    kept = sum(1 for r in forced if r["id"] in ans and ans[r["id"]].strip().lower() == r["line"].strip().lower())
    res["forced_keep_ok"] = round(kept / max(1, sum(r["id"] in ans for r in forced)), 3) if any(r["id"] in ans for r in forced) else None
    changed, valid, soft_bad, added, judge_items = 0, 0, 0, Counter(), []
    for r in elig:
        c = dict(r["ctx"], max_edits=r.get("max_edits") or 2)
        v = v7_filter.check(c, ans[r["id"]])
        if not v["changed"]:
            continue
        changed += 1
        if v["ok"]:
            valid += 1
            added[added_key(r["line"], ans[r["id"]])] += 1
            judge_items.append((c, {"id": r["id"], "edited": ans[r["id"]]}))
    res.update(eligible=len(elig), edit_rate=round(changed / max(1, len(elig)), 3), valid_of_edits=round(valid / max(1, changed), 3), valid_edits=valid,
               distinct_slang=len(added), top_share=round(added.most_common(1)[0][1] / max(1, valid), 3) if added else 0, top=added.most_common(5))
    res["valid_per_100"] = round(100 * valid / max(1, len(elig)), 1)
    if judge:
        res.update(_judge(label, judge_items, len(elig), budget, judge_model))
    os.makedirs(OUTD, exist_ok=True)
    json.dump(res, open(f"{OUTD}/score_{label}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return res


def _judge(label, items, n_elig, budget, model):
    from data_prep.v7_format import render_user
    cache_p = f"{OUTD}/judge_{label}.jsonl"
    done = {r["id"]: r for r in _read(cache_p)}
    todo = [(c, r) for c, r in items if r["id"] not in done]
    config.OPENROUTER_MODELS["judge_ext"] = model
    c0 = float(L.STATS["cost"])
    os.makedirs(OUTD, exist_ok=True)
    with open(cache_p, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), 8):
            if float(L.STATS["cost"]) - c0 >= budget:
                print("judge budget reached")
                break
            batch = todo[i:i + 8]
            body = (v7_teacher.NL * 2).join(f"### id: {c['id']}" + v7_teacher.NL + render_user(c) + v7_teacher.NL + f"EDITED: {r['edited']}" for c, r in batch)
            msgs = [{"role": "system", "content": v7_teacher.JUDGE_SYSTEM}, {"role": "user", "content": body}]
            try:
                text = L.complete("judge_ext", msgs, temperature=0.0, max_tokens=6000, only="openrouter")[0]
            except Exception as e:
                print("judge call failed:", str(e)[:100])
                continue
            for x in v7_teacher._json_array(text):
                if isinstance(x, dict) and str(x.get("id")) in {c["id"] for c, _ in batch}:
                    done[str(x["id"])] = {"id": str(x["id"]), "natural": x.get("natural"), "meaning_kept": x.get("meaning_kept"), "register_ok": x.get("register_ok")}
                    f.write(json.dumps(done[str(x["id"])]) + "\n")
            f.flush()
    ids = {r["id"] for _, r in items}
    j = [done[i] for i in ids if i in done]
    ok = [x for x in j if (x.get("natural") or 0) >= 4 and x.get("meaning_kept") is True and x.get("register_ok") is True]
    return {"judged": len(j), "accepted": len(ok), "accepted_per_100": round(100 * len(ok) / max(1, n_elig), 1), "mean_natural": round(sum((x.get("natural") or 0) for x in j) / max(1, len(j)), 2)}


def gen_llm(model: str, label: str, budget: float):
    """A prompted model answers every test row with the SAME system prompt the v7 model was trained with (the no-training baseline)."""
    from data_prep.v7_format import SYSTEM_V7, render_user
    test = _read(f"{D}/test.jsonl")
    config.OPENROUTER_MODELS["bench"] = model
    c0 = float(L.STATS["cost"])
    os.makedirs(OUTD, exist_ok=True)
    out = []
    for r in test:
        if float(L.STATS["cost"]) - c0 >= budget:
            print("budget reached")
            break
        msgs = [{"role": "system", "content": SYSTEM_V7}, {"role": "user", "content": render_user(r["ctx"])}]
        try:
            t = L.complete("bench", msgs, temperature=0.0, max_tokens=300, only="openrouter")[0]
        except Exception as e:
            print("call failed:", str(e)[:100])
            continue
        out.append({"id": r["id"], "output": (t or "").strip().split("\n")[0].strip().strip('"')})
    with open(f"{OUTD}/{label}.jsonl", "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"{len(out)} answers -> {OUTD}/{label}.jsonl | cost ${float(L.STATS['cost']) - c0:.3f}")


def table():
    rows = [json.load(open(os.path.join(OUTD, f), encoding="utf-8")) for f in sorted(os.listdir(OUTD)) if f.startswith("score_")]
    cols = ["label", "answered", "forced_keep_ok", "edit_rate", "valid_of_edits", "valid_per_100", "accepted_per_100", "mean_natural", "distinct_slang", "top_share"]
    print(" | ".join(cols))
    for r in rows:
        print(" | ".join(str(r.get(c, "-")) for c in cols))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen-llm", "score", "table"])
    ap.add_argument("--label", default="x")
    ap.add_argument("--answers")
    ap.add_argument("--model", default="google/gemini-3.6-flash")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--budget", type=float, default=0.4)
    a = ap.parse_args()
    if a.cmd == "gen-llm":
        gen_llm(a.model, a.label, a.budget)
    elif a.cmd == "score":
        print(json.dumps(score(a.label, a.answers, a.judge, a.budget), ensure_ascii=False, indent=1))
    else:
        table()


if __name__ == "__main__":
    main()
