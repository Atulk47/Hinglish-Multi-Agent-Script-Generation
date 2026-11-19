"""
tools/check_determinism.py  -  is the slang editor repeatable?  (same input -> exactly the same output, whatever was asked before it)

    python -m tools.check_determinism model    [--n 60] [--gguf PATH]          # A: the GGUF editor alone
    python -m tools.check_determinism pipeline fin_hostel fin_sci [--hashseeds 0 1 2] [--out-dir outputs/determinism]   # B: whole editing stage, several processes

A  Takes N real prompts (kaggle_v6/eval_*.jsonl) and runs them four ways: in order, in REVERSE order, shuffled with an unrelated prompt between every call
   (a different "previous prompt" for each), and in order again.  A deterministic editor gives the same text for a prompt in all four passes.  Run once
   with prefix reuse ON (what llama.cpp does by default) and once OFF (what GGUFEditor now does): the difference is the proof that the cache was the cause.
B  Re-runs `evaluation.ablation --modes agentic` on frozen Pass-1 traces in separate processes with different PYTHONHASHSEED values (Python randomises str hashes
   per process, so any set-order dependence shows up) and compares every slang dialogue and every decision.  Needs the Director / tone caches filled (a first
   run fills them; afterwards the LLM is not asked again).
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import random
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _prompts(n: int, seed: int = 3):
    from agents.agent4_slang_rewriter import QWEN_SYSTEM_V2, QWEN_USER_V2          # the exact prompt the pipeline sends
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "kaggle_v6", "eval_*.jsonl"))):
        for l in open(f, encoding="utf-8"):
            if l.strip():
                d = json.loads(l)
                if d.get("line") and d.get("shortlist"):
                    rows.append((QWEN_SYSTEM_V2, QWEN_USER_V2.replace("{slangs}", ", ".join(d["shortlist"])).replace("{line}", d["line"])))
    random.Random(seed).shuffle(rows)
    return rows[:n]


def check_model(n: int, gguf: str | None) -> dict:
    import config
    from agents.gguf_editor import GGUFEditor
    path = gguf or os.path.join(ROOT, config.QWEN_GGUF_PATH.lstrip("./"))
    ed = GGUFEditor(path)
    P = _prompts(n)
    poison = _prompts(n, seed=99)
    rng = random.Random(1)

    def run(order, between=False):
        out = {}
        for i in order:
            if between:                                                          # an unrelated prompt first: a different "previous call" every time
                ed.generate(*poison[rng.randrange(len(poison))], 60)
            out[i] = ed.generate(P[i][0], P[i][1], 80)
        return out

    res = {"n_prompts": len(P), "gguf": os.path.basename(path), "config": ed.config}
    for label, reuse in (("prefix_reuse_ON (llama.cpp default)", "1"), ("prefix_reuse_OFF (GGUFEditor now)", "0")):
        os.environ["QWEN_GGUF_REUSE_PREFIX"] = reuse
        idx = list(range(len(P)))
        shuffled = idx[:]
        random.Random(2).shuffle(shuffled)
        passes = [run(idx), run(idx[::-1]), run(shuffled, between=True), run(idx)]
        diff = [i for i in idx if len({p[i] for p in passes}) > 1]
        res[label] = {"prompts_with_differing_output": len(diff), "share": round(len(diff) / len(P), 4),
                      "examples": [{"prompt_tail": P[i][1][-160:], "outputs": sorted({p[i] for p in passes})} for i in diff[:5]]}
        print(f"{label}: {len(diff)}/{len(P)} prompts changed output across the 4 passes", flush=True)
    return res


def check_pipeline(stories, hashseeds, out_dir) -> dict:
    py = sys.executable
    runs = {}
    for hs in hashseeds:
        d = os.path.join(out_dir, f"hs{hs}")
        for s in stories:
            env = dict(os.environ, PYTHONHASHSEED=str(hs), PYTHONIOENCODING="utf-8")
            print(f"--- PYTHONHASHSEED={hs} {s}", flush=True)
            r = subprocess.run([py, "-m", "evaluation.ablation", os.path.join("outputs", f"{s}_trace.json"), "--out-dir", d, "--modes", "agentic"],
                               cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
            if r.returncode != 0:
                print(r.stdout[-1500:], r.stderr[-1500:])
                raise SystemExit(f"ablation failed for {s} (PYTHONHASHSEED={hs})")
        runs[hs] = d
    ref_hs = hashseeds[0]
    res = {"stories": stories, "hashseeds": hashseeds, "identical": True, "differences": []}
    for s in stories:
        def load(hs):
            t = json.load(open(os.path.join(runs[hs], f"{s}_trace_agentic.json"), encoding="utf-8"))
            return {sc["scene_id"]: (sc["dialogue"], [(d.get("slang"), d.get("applied")) for d in sc.get("decisions", [])]) for sc in t["slang_script"]}
        ref = load(ref_hs)
        for hs in hashseeds[1:]:
            cur = load(hs)
            for sid in ref:
                if ref[sid] != cur.get(sid):
                    res["identical"] = False
                    res["differences"].append({"story": s, "scene": sid, "hashseed_a": ref_hs, "hashseed_b": hs,
                                               "a": ref[sid][0][:400], "b": (cur.get(sid) or ("",))[0][:400]})
    print("PIPELINE:", "IDENTICAL across processes" if res["identical"] else f"{len(res['differences'])} scene(s) differ", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("model")
    a.add_argument("--n", type=int, default=60)
    a.add_argument("--gguf", default=None)
    b = sub.add_parser("pipeline")
    b.add_argument("stories", nargs="+")
    b.add_argument("--hashseeds", nargs="+", type=int, default=[0, 1])
    b.add_argument("--out-dir", default=os.path.join("outputs", "determinism"))
    args = ap.parse_args()
    if args.cmd == "model":
        res = check_model(args.n, args.gguf)
        dest = os.path.join(ROOT, "outputs", "determinism_model.json")
    else:
        res = check_pipeline(args.stories, args.hashseeds, args.out_dir)
        dest = os.path.join(ROOT, "outputs", "determinism_pipeline.json")
    json.dump(res, open(dest, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved", dest)


if __name__ == "__main__":
    main()
