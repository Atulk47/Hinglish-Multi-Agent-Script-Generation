"""
evaluation/bench_speed.py   -  which CPU settings make the local slang editor fastest?
Measures seconds per edit for the SAME sentence under several configurations and checks that the OUTPUT TEXT
is identical to the baseline (a speed-up must not change the edit).

    python -u -m evaluation.bench_speed
"""
import copy, time, sys, os, gc
import torch
import config
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from agents.agent4_slang_rewriter import QWEN_SYSTEM_V2, QWEN_USER_V2

LINE = "Chinta mat karo, yeh sirf ek choti si galti hai; hum isse seekhenge aur phir se try karenge, zaroor safal honge."
SL = ["yaar", "arre", "bro", "chill", "boss", "bhai"]
LINE2 = "Yeh bahut accha idea hai, Arav."
SL2 = ["badiya", "kamaal", "zabardast", "jhakaas", "solid"]

tok = AutoTokenizer.from_pretrained(config.QWEN_BASE_MODEL)


def prompt(line, sl):
    user = QWEN_USER_V2.replace("{slangs}", ", ".join(sl)).replace("{line}", line)
    return tok.apply_chat_template([{"role": "system", "content": QWEN_SYSTEM_V2}, {"role": "user", "content": user}],
                                   add_generation_prompt=True, return_tensors="pt", return_dict=True)


def run(model, label, **gen_kw):
    outs, t_all = [], 0.0
    for line, sl in ((LINE, SL), (LINE2, SL2)):
        enc = prompt(line, sl)
        n_line = len(tok(line, add_special_tokens=False)["input_ids"])
        t = time.time()
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=max(24, min(110, int(n_line * 1.4) + 12)), do_sample=False,
                               pad_token_id=tok.eos_token_id, **gen_kw)
        dt = time.time() - t
        new = o[0][enc["input_ids"].shape[1]:]
        outs.append(tok.decode(new, skip_special_tokens=True).strip().split("\n")[0])
        t_all += dt
        print(f"   [{label}] {dt:5.1f}s  {len(new):3d} tokens  {len(new)/dt:4.1f} tok/s", flush=True)
    return outs, t_all


def load(dtype):
    base = AutoModelForCausalLM.from_pretrained(config.QWEN_BASE_MODEL, dtype=dtype)
    return PeftModel.from_pretrained(base, config.QWEN_LORA_PATH).eval()


results = {}
print("== bf16, adapter NOT merged, 10 threads (current setting)")
m = load(torch.bfloat16)
base_out, results["baseline"] = run(m, "baseline")
print("   outputs:", base_out)

print("== bf16, adapter MERGED into the weights")
m = m.merge_and_unload().eval()
for th in (10, 6, 4, 2):
    torch.set_num_threads(th)
    o, t = run(m, f"merged bf16 threads={th}")
    results[f"merged bf16 t{th}"] = t
    print("   same text as baseline:", o == base_out)
torch.set_num_threads(10)

print("== merged bf16 + prompt-lookup decoding (reuses the input sentence as draft tokens)")
for th in (10, 4):
    torch.set_num_threads(th)
    try:
        o, t = run(m, f"merged bf16 + lookup t={th}", prompt_lookup_num_tokens=10)
        results[f"merged bf16 lookup t{th}"] = t
        print("   same text as baseline:", o == base_out)
    except Exception as e:
        print("   lookup failed:", str(e)[:120])
torch.set_num_threads(10)

print("== merged, float32 (needs ~6 GB free RAM)")
try:
    m32 = m.to(torch.float32)
    for th in (10, 4):
        torch.set_num_threads(th)
        o, t = run(m32, f"merged fp32 threads={th}")
        results[f"merged fp32 t{th}"] = t
        print("   same text as baseline:", o == base_out)
    o, t = run(m32, "merged fp32 + lookup", prompt_lookup_num_tokens=10)
    results["merged fp32 lookup"] = t
    print("   same text as baseline:", o == base_out)
except Exception as e:
    print("   fp32 failed:", str(e)[:150])

print("\nSUMMARY (total seconds for the 2 test sentences; lower is better)")
for k, v in sorted(results.items(), key=lambda kv: kv[1]):
    print(f"  {v:6.1f}s  {k}")
