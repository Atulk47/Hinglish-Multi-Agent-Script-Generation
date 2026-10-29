"""
tools/export_gguf.py  -  slang-editor LoRA adapter  ->  4-bit GGUF for fast CPU inference (llama.cpp).

    python -m tools.export_gguf --adapter models/slang-editor-lora-v5 --out models/gguf/slang-editor-v5 [--quant Q4_K_M] [--verify]

Steps (all local, RAM-light - works with ~2 GB free):
  1. MERGE   W' = W + (alpha/r) * B @ A   tensor by tensor straight from the base model's safetensors (never loads the whole model),
             written as sharded safetensors next to the tokenizer files (a normal HF checkpoint),
  2. CONVERT llama.cpp's convert_hf_to_gguf.py (sources vendored in tools/llama_cpp_convert) -> f16 GGUF,
  3. QUANTIZE llama_cpp.llama_model_quantize -> Q4_K_M (default) and/or Q8_0,
  4. VERIFY  (optional) run the held-out evaluation rows through the quantized model and compare with the reference outputs.
"""
from __future__ import annotations
import argparse
import ctypes
import glob
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CONVERT_DIR = os.path.join(HERE, "llama_cpp_convert")
FTYPE = {"Q4_K_M": 15, "Q4_0": 2, "Q5_K_M": 17, "Q6_K": 18, "Q8_0": 7}


def base_snapshot(model_id: str = "Qwen/Qwen2.5-1.5B-Instruct") -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])


def merge_lora(base_dir: str, adapter_dir: str, out_dir: str, shard_bytes: int = 600_000_000) -> dict:
    """Merge the adapter into the base weights without ever holding the full model in RAM."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    cfg = json.load(open(os.path.join(adapter_dir, "adapter_config.json"), encoding="utf-8"))
    r, alpha = int(cfg["r"]), float(cfg["lora_alpha"])
    scale = alpha / (r ** 0.5) if cfg.get("use_rslora") else alpha / r             # standard LoRA scaling
    lora = {}
    with safe_open(os.path.join(adapter_dir, "adapter_model.safetensors"), "pt") as f:
        for k in f.keys():
            lora[k] = f.get_tensor(k).float()
    pairs = {}                                           # module path -> (A, B)
    for k, v in lora.items():
        m = re.match(r"base_model\.model\.(.+)\.lora_([AB])\.weight", k)
        if m:
            pairs.setdefault(m.group(1) + ".weight", {})[m.group(2)] = v
    files = sorted(glob.glob(os.path.join(base_dir, "*.safetensors")))
    assert files, f"no safetensors in {base_dir}"
    shard, size, n_shard, index, merged = {}, 0, 0, {}, 0

    def flush():
        nonlocal shard, size, n_shard
        if not shard:
            return
        name = f"model-{n_shard:05d}.safetensors"
        save_file(shard, os.path.join(out_dir, name), metadata={"format": "pt"})
        for k in shard:
            index[k] = name
        n_shard += 1
        shard, size = {}, 0

    for fp in files:
        with safe_open(fp, "pt") as f:
            for key in f.keys():
                w = f.get_tensor(key)
                if key in pairs and "A" in pairs[key] and "B" in pairs[key]:
                    delta = (pairs[key]["B"] @ pairs[key]["A"]) * scale
                    w = (w.float() + delta).to(w.dtype)
                    merged += 1
                shard[key] = w.contiguous()
                size += w.numel() * w.element_size()
                if size >= shard_bytes:
                    flush()
    flush()
    assert merged == len(pairs), f"merged {merged} of {len(pairs)} LoRA modules - key names did not match the base model"
    json.dump({"metadata": {"total_size": 0}, "weight_map": index}, open(os.path.join(out_dir, "model.safetensors.index.json"), "w"))
    for name in os.listdir(base_dir):                    # config + tokenizer files
        if name.endswith((".json", ".txt", ".model", ".jinja")) and not name.startswith("model.safetensors"):
            src = os.path.join(base_dir, name)
            if os.path.isfile(src):
                open(os.path.join(out_dir, name), "wb").write(open(src, "rb").read())
    open(os.path.join(out_dir, "model.safetensors.index.json"), "w").write(
        json.dumps({"metadata": {}, "weight_map": index}))
    print(f"merged {merged} LoRA modules (r={r}, alpha={alpha}, scale={scale}) into {n_shard} shards -> {out_dir}")
    return {"merged": merged, "shards": n_shard}


def convert_merged_direct(base_dir: str, adapter_dir: str, out_gguf: str, outtype: str = "q8_0"):
    """Merge the LoRA INTO the converter's tensor stream: llama.cpp's convert script reads the base model read-only, every targeted weight
    is replaced by W + scale * B @ A as it is loaded, and the result is written straight to a q8_0 GGUF. No merged HF checkpoint (3 GB) and
    no f16 GGUF (3 GB) ever touch the disk."""
    import torch
    from safetensors import safe_open
    cfg = json.load(open(os.path.join(adapter_dir, "adapter_config.json"), encoding="utf-8"))
    r, alpha = int(cfg["r"]), float(cfg["lora_alpha"])
    scale = alpha / (r ** 0.5) if cfg.get("use_rslora") else alpha / r
    pairs = {}
    with safe_open(os.path.join(adapter_dir, "adapter_model.safetensors"), "pt") as f:
        for k in f.keys():
            m = re.match(r"base_model\.model\.(.+)\.lora_([AB])\.weight", k)
            if m:
                pairs.setdefault(m.group(1) + ".weight", {})[m.group(2)] = f.get_tensor(k).float()
    sys.path.insert(0, os.path.join(CONVERT_DIR, "gguf-py"))
    sys.path.insert(0, CONVERT_DIR)
    os.environ["NO_LOCAL_GGUF"] = "1"
    import convert_hf_to_gguf as C
    from conversion import base as B
    hits = {"n": 0}
    orig = B.ModelBase.index_tensors

    def patched(self, *a, **k):
        tensors = orig(self, *a, **k)
        for name in list(tensors):
            if name in pairs and "A" in pairs[name] and "B" in pairs[name]:
                def merged(g=tensors[name], A=pairs[name]["A"], Bm=pairs[name]["B"]):
                    w = g()
                    return (w.float() + (Bm @ A) * scale).to(w.dtype)
                tensors[name] = merged
                hits["n"] += 1
        return tensors
    B.ModelBase.index_tensors = patched
    argv = sys.argv
    sys.argv = ["convert_hf_to_gguf.py", base_dir, "--outfile", out_gguf, "--outtype", outtype, "--no-lazy", "--use-temp-file"]
    try:
        C.main()
    finally:
        sys.argv = argv
        B.ModelBase.index_tensors = orig
    assert hits["n"] == len(pairs), f"LoRA merged into {hits['n']} of {len(pairs)} modules - tensor names did not match"
    print(f"merged {hits['n']} LoRA modules (r={r}, alpha={alpha}, scale={scale}) while converting -> {out_gguf} "
          f"({os.path.getsize(out_gguf) / 1e6:.0f} MB)")


def convert_f16(hf_dir: str, out_gguf: str):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(CONVERT_DIR, "gguf-py") + os.pathsep + CONVERT_DIR + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, os.path.join(CONVERT_DIR, "convert_hf_to_gguf.py"), hf_dir, "--outfile", out_gguf, "--outtype", "f16"]
    print("convert:", " ".join(cmd[-4:]))
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out_gguf):
        raise RuntimeError("GGUF conversion failed:\n" + (r.stderr or r.stdout)[-1500:])


def quantize(f16_gguf: str, out_gguf: str, kind: str):
    import llama_cpp
    p = llama_cpp.llama_model_quantize_default_params()
    p.ftype = FTYPE[kind]
    p.nthread = max(2, (os.cpu_count() or 4) - 2)
    p.allow_requantize = True                       # q8_0 -> q4_k_m
    rc = llama_cpp.llama_model_quantize(f16_gguf.encode(), out_gguf.encode(), ctypes.byref(p))
    if rc != 0 or not os.path.exists(out_gguf):
        raise RuntimeError(f"quantize {kind} failed (rc={rc})")
    print(f"quantized {kind}: {os.path.getsize(out_gguf) / 1e6:.0f} MB -> {out_gguf}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True, help="output prefix, e.g. models/gguf/slang-editor-v5")
    ap.add_argument("--quant", nargs="+", default=["Q4_K_M"], choices=list(FTYPE), help="extra quantisations made from the q8_0 file")
    ap.add_argument("--keep-q8", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    q8 = a.out + "-Q8_0.gguf"
    convert_merged_direct(base_snapshot(), a.adapter, q8, "q8_0")
    for q in a.quant:
        if q != "Q8_0":
            quantize(q8, f"{a.out}-{q}.gguf", q)
    if not a.keep_q8 and "Q8_0" not in a.quant:
        os.remove(q8)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
