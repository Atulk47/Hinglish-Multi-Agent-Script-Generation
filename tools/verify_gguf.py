"""
tools/verify_gguf.py  -  is the quantized editor as good as the original, and how much faster is it?

    python tools/verify_gguf.py --gguf models/gguf/slang-editor-v4-Q4_K_M.gguf --eval kaggle_v4 --ref data/felix_dataset/v3/eval_outputs_new.jsonl

Runs the held-out evaluation rows through the GGUF model (greedy, same prompt) and reports
  * exact-match rate against the reference outputs of the full-precision model (from the Kaggle run),
  * strict-valid-edit rate and abstain rate (same definitions as the training notebook), for GGUF and for the reference,
  * seconds per edit.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from agents.agent4_slang_rewriter import QWEN_SYSTEM_V2, QWEN_USER_V2, SlangRewriter
from agents.edit_gate import rule_gate
from agents.gguf_editor import GGUFEditor


def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def used_slang(line, sl, out):
    return next((s for s in sl if s.lower() in out.lower() and s.lower() not in line.lower()), None)


def strict_ok(line, sl, out):
    return norm(out) != norm(line) and used_slang(line, sl, out) is not None and SlangRewriter._keeps_meaning(line, out)


def gate_ok(line, sl, out):
    u = used_slang(line, sl, out)
    return bool(u) and rule_gate(line, out, u)[0] and SlangRewriter._keeps_meaning(line, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--eval", required=True, help="folder with eval_*.jsonl")
    ap.add_argument("--ref", default=None, help="eval_outputs_new.jsonl of the full-precision model (id -> output)")
    ap.add_argument("--max", type=int, default=0, help="limit rows (0 = all)")
    a = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(os.path.join(a.eval, "eval_*.jsonl"))):
        rows += [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    if a.max and len(rows) > a.max:                      # stratified-by-eval-file random sample (the file order is not representative)
        import random
        rng = random.Random(0)
        by = {}
        for r in rows:
            by.setdefault(r.get("src", "?"), []).append(r)
        picked = []
        for k, lst in by.items():
            picked += rng.sample(lst, max(1, round(a.max * len(lst) / len(rows))))
        rows = picked[:a.max]
    ref = {}
    if a.ref and os.path.exists(a.ref):
        for l in open(a.ref, encoding="utf-8"):
            if l.strip():
                d = json.loads(l)
                ref[d["id"]] = d["output"]
    ed = GGUFEditor(a.gguf)
    t0 = time.time()
    outs = []
    for r in rows:
        user = QWEN_USER_V2.replace("{slangs}", ", ".join(r["shortlist"])).replace("{line}", r["line"])
        n_line = ed.n_tokens(r["line"])
        o = ed.generate(QWEN_SYSTEM_V2, user, max(24, min(110, int(n_line * 1.4) + 12)))
        outs.append(o.split("\n")[0].strip().strip('"'))
    dt = time.time() - t0
    print(f"{os.path.basename(a.gguf)}: {len(rows)} rows in {dt:.0f}s = {dt / len(rows):.2f} s/edit")
    both = [(r, o, ref[r["id"]]) for r, o in zip(rows, outs) if r["id"] in ref]
    if both:
        same = sum(norm(o) == norm(rf) for _, o, rf in both)
        print(f"exact match with the full-precision outputs: {same}/{len(both)} = {same / len(both):.1%}")
        same_dec = sum((norm(o) == norm(r["line"])) == (norm(rf) == norm(r["line"])) for r, o, rf in both)
        print(f"same edit-vs-abstain decision: {same_dec}/{len(both)} = {same_dec / len(both):.1%}")
    for name, picks in (("GGUF", [(r, o) for r, o in zip(rows, outs)]), ("reference", [(r, ref[r["id"]]) for r in rows if r["id"] in ref])):
        if not picks:
            continue
        e = [(r, o) for r, o in picks if r["expect"] == "edit"]
        ab = [(r, o) for r, o in picks if r["expect"] == "abstain"]
        print(f"[{name:9s}] valid edits {sum(strict_ok(r['line'], r['shortlist'], o) for r, o in e)}/{len(e)}"
              f" | gate-valid {sum(bool(gate_ok(r['line'], r['shortlist'], o)) for r, o in e)}/{len(e)}"
              f" | abstain kept {sum(norm(o) == norm(r['line']) for r, o in ab)}/{len(ab)}")
    json.dump([{"id": r["id"], "output": o} for r, o in zip(rows, outs)],
              open(a.gguf.replace(".gguf", "_eval_outputs.json"), "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__":
    main()
