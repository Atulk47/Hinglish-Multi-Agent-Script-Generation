"""
tools/calibrate_validator.py  -  fits the "centring" that makes MuRIL similarities informative (no training, ~2-4 minutes on CPU).

Problem: raw MuRIL sentence embeddings share one big common direction, so the cosine of ANY two Hinglish lines is ~0.99 and a validator built on it
cannot tell a faithful edit from a different sentence ("anisotropy").  Fix (Mu & Viswanath 2018, "all-but-the-top"): subtract the mean embedding of a
corpus of the project's own lines and remove the top few principal directions, then take the cosine.

Corpus: every dialogue line (speaker tag stripped) of every saved pipeline trace (neutral + slang) plus the judged edit lines in data/felix_dataset/v3.
Writes data/validator/muril_centering.npz (mean + removed directions; ~25 KB, safe to commit) and data/validator/centering_meta.json.

    python -m tools.calibrate_validator
"""
from __future__ import annotations
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

import config

OUT = "data/validator/muril_centering.npz"
K_REMOVE = 5
MAX_LINES = 3000


def corpus_lines() -> list:
    seen, out = set(), []

    def add(t):
        t = re.sub(r"^\s*[A-Za-z][A-Za-z_. ]*?\s*:\s*", "", (t or "").strip())
        if 6 <= len(t) <= 400 and t not in seen:
            seen.add(t)
            out.append(t)
    for p in sorted(glob.glob("outputs/*_trace.json")):
        try:
            tr = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        for key in ("neutral_script", "slang_script"):
            for sc in tr.get(key) or []:
                for ln in (sc.get("dialogue") or "").split("\n"):
                    add(ln)
    for f in glob.glob("data/felix_dataset/v3/judged*.jsonl"):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            add(r.get("line"))
            add(r.get("edit"))
    return out


def main():
    from transformers import AutoModel, AutoTokenizer
    lines = corpus_lines()
    import random
    random.Random(0).shuffle(lines)
    lines = lines[:MAX_LINES]
    print(f"{len(lines)} corpus lines")
    tok = AutoTokenizer.from_pretrained(config.MURIL_MODEL_PATH)
    model = AutoModel.from_pretrained(config.MURIL_MODEL_PATH).eval()
    vecs = []
    for i in range(0, len(lines), 24):
        enc = tok(lines[i:i + 24], return_tensors="pt", truncation=True, max_length=256, padding=True)
        with torch.no_grad():
            h = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        vecs.append(((h * m).sum(1) / m.sum(1).clamp(min=1e-9)).numpy())
        if (i // 24) % 20 == 0:
            print(f"  {i}/{len(lines)}", flush=True)
    X = np.concatenate(vecs)
    mu = X.mean(axis=0)
    Xc = X - mu
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    V = vt[:K_REMOVE]
    def cos(a, b):
        return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(X), size=(2000, 2))
    raw = np.mean([cos(X[i], X[j]) for i, j in idx if i != j])
    P = Xc - (Xc @ V.T) @ V
    cen = np.mean([cos(P[i], P[j]) for i, j in idx if i != j])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, mu=mu.astype(np.float32), V=V.astype(np.float32))
    meta = {"n_lines": len(lines), "k_removed": K_REMOVE, "top_singular_values": [round(float(x), 2) for x in s[:8]],
            "variance_share_removed": round(float((s[:K_REMOVE] ** 2).sum() / (s ** 2).sum()), 4),
            "mean_cosine_random_pair_raw": round(float(raw), 4), "mean_cosine_random_pair_centred": round(float(cen), 4)}
    json.dump(meta, open("data/validator/centering_meta.json", "w"), indent=1)
    print(json.dumps(meta, indent=1), "\n->", OUT)


if __name__ == "__main__":
    main()
