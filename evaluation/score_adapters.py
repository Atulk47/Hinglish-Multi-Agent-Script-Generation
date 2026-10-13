"""
evaluation/score_adapters.py
Fair, LOCAL-ONLY before/after comparison of slang-editor adapters on HELD-OUT data
(no Groq fallback, no reranker help: the model alone, same inputs for every adapter).

Inputs (data/felix_dataset/v3/): eval_real.jsonl, eval_synth.jsonl, eval_outputs_<tag>.jsonl
(generated on Kaggle by notebooks/train_slang_editor_lora_qwen_v3.py for tags base / old / new).

For every output:
  1. runtime guards  - slang NEWLY inserted from the shortlist, drift guard, polarity-flip guard
  2. blind judge     - a strong LLM rates naturalness 1-5 + whether the meaning is preserved
                       (the judge never sees which adapter wrote the line)
Reported per adapter, separately for REAL pipeline sentences and synthetic held-out sentences:
  good_edit_rate   : good (judge>=4 & meaning kept) / all rows where a slang fits    <- headline
  accept_rate      : passes the runtime guards / same rows
  precision        : good / accepted
  bad_edit_rate    : accepted by the guards but judged bad (what would still reach a script)
  abstain_rate     : unchanged on rows where nothing fits (higher is better)
  forced_edit_rate : edited although nothing fits (lower is better)

  python -m evaluation.score_adapters --tags base,old,new --max-usd 1.5
"""
from __future__ import annotations
import argparse, collections, csv, hashlib, json, os, random, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from data_prep.v3_common import Teacher, parse_json, read_jsonl, keeps_meaning, newly_inserted, norm
from data_prep.slang_bank_v3 import BANK
from data_prep.label_v3 import SYSTEM, JUDGE_PROMPT
from agents.agent4_slang_rewriter import SlangRewriter

D = "data/felix_dataset/v3/"
POS = {s for s, v in BANK.items() if v[0] == "P"}
NEG = {s for s, v in BANK.items() if v[0] == "N"}


def guard(line, shortlist, out):
    """-> (accepted, used_slang, reason)"""
    if not out or norm(out) == norm(line):
        return False, None, "unchanged"
    used = next((s for s in shortlist if newly_inserted(line, out, s)), None)
    if used is None:
        return False, None, "off-list"
    if not keeps_meaning(line, out):
        return False, used, "drift"
    pol = SlangRewriter._line_polarity(line)
    if (pol < 0 and used.lower() in POS) or (pol > 0 and used.lower() in NEG):
        return False, used, "polarity-flip"
    return True, used, ""


def key(line, out):
    return hashlib.md5((line + "||" + out).encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="base,old,new")
    ap.add_argument("--max-usd", type=float, default=1.5)
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--min-score", type=float, default=4.0)
    ap.add_argument("--out", default="outputs/adapter_comparison")
    ap.add_argument("--human-n", type=int, default=60)
    ap.add_argument("--judge", choices=["api", "session"], default="session",
                    help="api = Anthropic judge (needs credit); session = Claude judges in the coding session "
                         "(dump -> verdict file -> apply)")
    ap.add_argument("--stage", choices=["dump", "apply", "report"], default="report")
    ap.add_argument("--keep", default=D + "review/keep_adapters.json",
                    help="session judge verdicts: {'keep': [pair ids that are good]}")
    args = ap.parse_args()
    tags = [t for t in args.tags.split(",") if os.path.exists(f"{D}eval_outputs_{t}.jsonl")]
    ev = {r["id"]: r for n in ("eval_real.jsonl", "eval_synth.jsonl", "eval_filler.jsonl") if os.path.exists(D + n) for r in read_jsonl(D + n)}
    outs = {t: {o["id"]: o["output"] for o in read_jsonl(f"{D}eval_outputs_{t}.jsonl")} for t in tags}
    print("adapters:", tags, "| eval rows:", len(ev))

    # ---- guards, then blind judging of every accepted output ----
    cache_p = D + "judge_cache.json"
    cache = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
    todo, verdicts = {}, {}
    for t in tags:
        for i, r in ev.items():
            o = outs[t].get(i, "")
            ok, used, why = guard(r["line"], r["shortlist"], o)
            verdicts[(t, i)] = {"accepted": ok, "slang": used, "why": why, "out": o}
            if ok and r["src"] != "filler" and key(r["line"], o) not in cache:
                todo[key(r["line"], o)] = (r["line"], o, used)
    items = list(todo.items())
    if args.judge == "session":
        # Blind pairs: the judge never sees which adapter produced which output.
        pair_file = D + "review/adapter_pairs.json"
        os.makedirs(D + "review", exist_ok=True)
        if args.stage == "dump":
            pairs = [{"id": n, "key": k, "line": c[0], "out": c[1], "slang": c[2]} for n, (k, c) in enumerate(items)]
            json.dump(pairs, open(pair_file, "w", encoding="utf-8"), ensure_ascii=False)
            with open(D + "review/adapter_pairs.txt", "w", encoding="utf-8") as f:
                for q in pairs:
                    f.write("#%d\n O: %s\n E: %s   [%s]\n" % (q["id"], q["line"], q["out"], q["slang"]))
            print(f"{len(pairs)} unique accepted (line,output) pairs to judge -> {D}review/adapter_pairs.txt")
            return
        if args.stage == "apply":
            pairs = json.load(open(pair_file, encoding="utf-8"))
            keep = set(json.load(open(args.keep))["keep"])
            for q in pairs:
                cache[q["key"]] = ({"score": 4.0, "meaning": True} if q["id"] in keep
                                   else {"score": 2.0, "meaning": False})
            json.dump(cache, open(cache_p, "w"))
            print(f"applied {len(keep)} good / {len(pairs)} judged -> cache")
            return
        if items:
            print(f"WARNING: {len(items)} accepted outputs are not judged yet - run --stage dump, judge, "
                  "then --stage apply. They count as NOT good in this report.")
        teacher = type("NoApi", (), {"cost": 0.0})()
    else:
        teacher = Teacher(args.model, args.max_usd)
        def run(chunk):
            txt = teacher.ask(SYSTEM, JUDGE_PROMPT.format(items="\n".join(
                '%d. ORIGINAL: "%s"\n   EDITED: "%s"\n   SLANG: %s' % (n + 1, c[0], c[1], c[2])
                for n, (_, c) in enumerate(chunk))), 2500)
            arr = parse_json(txt) or []
            return [(chunk[int(a["i"]) - 1][0], {"score": float(a.get("score", 0)), "meaning": bool(a.get("meaning", False))})
                    for a in arr if isinstance(a, dict) and "i" in a and 1 <= int(a["i"]) <= len(chunk)]
        with ThreadPoolExecutor(6) as ex:
            for fu in as_completed([ex.submit(run, items[s:s + 25]) for s in range(0, len(items), 25)]):
                try:
                    for k, v in fu.result():
                        cache[k] = v
                except Exception as e:
                    print("  judge batch failed:", str(e)[:100])
    json.dump(cache, open(cache_p, "w"))

    # ---- metrics ----
    report = {}
    for t in tags:
        for src in ("real", "synth", "filler", "all"):
            rows = [(i, r) for i, r in ev.items() if src == "all" or r["src"] == src]
            edit_rows = [(i, r) for i, r in rows if r["expect"] == "edit"]
            abst_rows = [(i, r) for i, r in rows if r["expect"] == "abstain"]
            acc = good = 0
            scores = []
            for i, r in edit_rows:
                v = verdicts[(t, i)]
                if v["accepted"]:
                    acc += 1
                    j = cache.get(key(r["line"], v["out"]))
                    if r["src"] == "filler":
                        # filler rows are scored by the deterministic gate (placement rules), no judge needed
                        from agents.edit_gate import rule_gate
                        if rule_gate(r["line"], v["out"], v["slang"] or "")[0]:
                            good += 1
                            scores.append(5.0)
                    elif j:
                        scores.append(j["score"])
                        if j["score"] >= args.min_score and j["meaning"]:
                            good += 1
            ab_ok = sum(norm(verdicts[(t, i)]["out"]) == norm(r["line"]) for i, r in abst_rows)
            forced = sum(verdicts[(t, i)]["accepted"] for i, r in abst_rows)
            n_e, n_a = max(len(edit_rows), 1), max(len(abst_rows), 1)
            report.setdefault(t, {})[src] = {
                "edit_rows": len(edit_rows), "abstain_rows": len(abst_rows),
                "good_edit_rate": round(good / n_e, 3), "accept_rate": round(acc / n_e, 3),
                "precision": round(good / acc, 3) if acc else None,
                "bad_edit_rate": round((acc - good) / n_e, 3),
                "mean_judge_score": round(sum(scores) / len(scores), 2) if scores else None,
                "abstain_rate": round(ab_ok / n_a, 3), "forced_edit_rate": round(forced / n_a, 3)}
    os.makedirs("outputs", exist_ok=True)
    json.dump(report, open(args.out + ".json", "w"), indent=2)

    print("\n" + "=" * 92)
    for src in ("real", "synth", "filler"):
        print(f"\n{src.upper()} held-out sentences")
        print(f"{'adapter':8s} {'edit rows':>9s} {'GOOD edit':>10s} {'accepted':>9s} {'precision':>9s} "
              f"{'bad(reach script)':>18s} {'abstain ok':>11s} {'forced edit':>12s}")
        for t in tags:
            m = report[t][src]
            print(f"{t:8s} {m['edit_rows']:9d} {m['good_edit_rate']:10.2f} {m['accept_rate']:9.2f} "
                  f"{(m['precision'] if m['precision'] is not None else float('nan')):9.2f} "
                  f"{m['bad_edit_rate']:18.2f} {m['abstain_rate']:11.2f} {m['forced_edit_rate']:12.2f}")
    print("=" * 92)

    # ---- blind A/B sheet for the team (old vs new, REAL sentences, both produced an accepted edit) ----
    if "old" in tags and "new" in tags:
        rng = random.Random(1)
        cand = [i for i, r in ev.items() if r["src"] == "real" and r["expect"] == "edit"
                and (verdicts[("old", i)]["accepted"] or verdicts[("new", i)]["accepted"])]
        rng.shuffle(cand)
        with open(args.out + "_human_ab.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "original", "option_1", "option_2", "better (1/2/tie/neither)", "notes"])
            key_rows = []
            for i in cand[:args.human_n]:
                a, b = outs["old"][i], outs["new"][i]
                flip = rng.random() < 0.5
                w.writerow([i, ev[i]["line"], b if flip else a, a if flip else b, "", ""])
                key_rows.append({"id": i, "option_1_is": "new" if flip else "old"})
        json.dump(key_rows, open(args.out + "_human_ab_KEY.json", "w"), indent=1)
        print(f"\nblind A/B sheet: {args.out}_human_ab.csv  (answer key: ..._KEY.json - do NOT show raters)")
    print(f"\njudge cost ~${teacher.cost:.2f} | report: {args.out}.json")


if __name__ == "__main__":
    main()
