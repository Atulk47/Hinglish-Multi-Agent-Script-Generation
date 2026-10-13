"""
evaluation/bench_int8.py  -  int8 dynamic quantisation of the merged slang editor: speed + does it change the edits?
Converts layer by layer (low peak RAM), then compares against the GPU outputs of the same adapter
(data/felix_dataset/v3/eval_outputs_new.jsonl, produced on Kaggle) on N eval rows.

    python -u -m evaluation.bench_int8 --n 30
"""
import argparse, json, time, torch
import torch.nn as nn
import config
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from agents.agent4_slang_rewriter import QWEN_SYSTEM_V2, QWEN_USER_V2

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=30); args = ap.parse_args()
tok = AutoTokenizer.from_pretrained(config.QWEN_BASE_MODEL)


def quant_linear(lin: nn.Linear) -> nn.Module:
    f = nn.Linear(lin.in_features, lin.out_features, bias=lin.bias is not None)
    f.weight.data = lin.weight.data.float()
    if lin.bias is not None:
        f.bias.data = lin.bias.data.float()
    q = torch.ao.quantization.quantize_dynamic(nn.Sequential(f), {nn.Linear}, dtype=torch.qint8)
    return q[0]


def quantize_model(model):
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, nn.Linear):
                setattr(mod, cname, quant_linear(child))
    for p in model.parameters():                       # norms / embeddings -> fp32 so dtypes line up
        p.data = p.data.float()
    for b in model.buffers():
        if b.dtype.is_floating_point:
            b.data = b.data.float()
    return model


def prompt(line, sl):
    user = QWEN_USER_V2.replace("{slangs}", ", ".join(sl)).replace("{line}", line)
    return tok.apply_chat_template([{"role": "system", "content": QWEN_SYSTEM_V2}, {"role": "user", "content": user}],
                                   add_generation_prompt=True, return_tensors="pt", return_dict=True)


def gen(model, line, sl, lookup=True):
    enc = prompt(line, sl)
    n_line = len(tok(line, add_special_tokens=False)["input_ids"])
    kw = dict(prompt_lookup_num_tokens=10) if lookup else {}
    with torch.no_grad():
        o = model.generate(**enc, max_new_tokens=max(24, min(110, int(n_line * 1.4) + 12)), do_sample=False,
                           pad_token_id=tok.eos_token_id, **kw)
    return tok.decode(o[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip().split("\n")[0].strip().strip('"')


t = time.time()
base = AutoModelForCausalLM.from_pretrained(config.QWEN_BASE_MODEL, dtype=torch.bfloat16)
m = PeftModel.from_pretrained(base, config.QWEN_LORA_PATH).merge_and_unload().eval()
m = quantize_model(m).eval()
print(f"load + merge + int8 quantise: {time.time()-t:.0f}s", flush=True)

ev = {r["id"]: r for n in ("eval_real.jsonl", "eval_synth.jsonl", "eval_filler.jsonl")
      for r in map(json.loads, open("data/felix_dataset/v3/" + n, encoding="utf-8"))}
gpu = {r["id"]: r["output"] for r in map(json.loads, open("data/felix_dataset/v3/eval_outputs_new.jsonl", encoding="utf-8"))}
ids = [i for i in ev if i in gpu][:: max(1, len(gpu) // args.n)][: args.n]
same = tot = 0; secs = 0.0
for i in ids:
    r = ev[i]; t = time.time()
    out = gen(m, r["line"], r["shortlist"])
    dt = time.time() - t; secs += dt; tot += 1
    ok = out.strip() == gpu[i].strip(); same += ok
    print(f"  {dt:5.1f}s {'SAME ' if ok else 'DIFF '} {out[:90]}", flush=True)
print(f"\nint8 + lookup: {secs/tot:.1f}s per edit | identical to the GPU fp16 output on {same}/{tot} rows")
