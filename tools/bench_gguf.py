"""
tools/bench_gguf.py - how fast can the GGUF slang editor go without changing a single output?

    python tools/bench_gguf.py --gguf models/gguf/slang-editor-v5-Q5_K_M.gguf --eval kaggle_v5 --n 40

Tries: threads (4/6/8/10), prompt-lookup speculative decoding (the answer mostly COPIES the input sentence, so the next tokens can be proposed
from the prompt and verified in one pass), and prefix caching (system prompt + instruction header are identical for every call).
Reports seconds per edit and whether the outputs are identical to the plain baseline.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.agent4_slang_rewriter import QWEN_SYSTEM_V2, QWEN_USER_V2


def rows(folder, n, seed=0):
    allr = []
    for f in sorted(glob.glob(os.path.join(folder, "eval_*.jsonl"))):
        allr += [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    random.Random(seed).shuffle(allr)
    return allr[:n]


def run(llm, rs, label):
    outs, t0 = [], time.time()
    for r in rs:
        user = QWEN_USER_V2.replace("{slangs}", ", ".join(r["shortlist"])).replace("{line}", r["line"])
        n_line = len(llm.tokenize(r["line"].encode(), add_bos=False))
        res = llm.create_chat_completion(messages=[{"role": "system", "content": QWEN_SYSTEM_V2}, {"role": "user", "content": user}],
                                         temperature=0.0, max_tokens=max(24, min(110, int(n_line * 1.4) + 12)), stop=["<|im_end|>"])
        outs.append((res["choices"][0]["message"]["content"] or "").strip().split("\n")[0].strip().strip('"'))
    dt = time.time() - t0
    return outs, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    from llama_cpp import Llama
    from llama_cpp.llama_cache import LlamaRAMCache
    from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
    rs = rows(a.eval, a.n)
    base_out = None
    LK = LlamaPromptLookupDecoding
    configs = [("baseline (10 threads)", dict(n_threads=10), False, False),
               ("lookup 10 / ngram 2", dict(n_threads=10, draft_model=LK(num_pred_tokens=10, max_ngram_size=2)), True, False),
               ("lookup 6 / ngram 2", dict(n_threads=10, draft_model=LK(num_pred_tokens=6, max_ngram_size=2)), True, False),
               ("lookup 14 / ngram 3", dict(n_threads=10, draft_model=LK(num_pred_tokens=14, max_ngram_size=3)), True, False)]
    print(f"{'config':32s} {'s/edit':>7s}  identical-to-baseline")
    for label, kw, _, cache in configs:
        llm = Llama(model_path=a.gguf, n_ctx=1024, n_batch=256, verbose=False, seed=7, **kw)
        if cache:
            llm.set_cache(LlamaRAMCache(capacity_bytes=256 << 20))
        run(llm, rs[:2], label)                                   # warm-up
        outs, dt = run(llm, rs, label)
        if base_out is None:
            base_out = outs
        same = sum(x == y for x, y in zip(outs, base_out))
        print(f"{label:32s} {dt / len(rs):7.2f}  {same}/{len(rs)}", flush=True)
        del llm


if __name__ == "__main__":
    main()
