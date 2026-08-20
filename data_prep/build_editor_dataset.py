"""
data_prep/build_editor_dataset.py
Turn the slang dataset into seq2seq training data for the mT5 "slang editor".

Input  format (MUST match agents/agent4_slang_rewriter.py :: _mt5_edit):
    "insert slang | control: {CONTROL} | {neutral}"
Target format:
    "{output}"

Merges the cleaned dataset with any augmented CSVs (same schema) you pass in,
de-duplicates, splits train/val, and writes JSONL ready for Kaggle training.

Usage:
    python -m data_prep.build_editor_dataset
    python -m data_prep.build_editor_dataset --extra data/felix_dataset/augmented_v2.csv
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import random

import config


def read_clean(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            neutral = (r.get("neutral") or "").strip()
            output = (r.get("output") or "").strip()
            control = (r.get("control") or "ASSERT_NEUTRAL").strip().upper() or "ASSERT_NEUTRAL"
            if neutral and output and neutral.lower() != output.lower():
                out.append((neutral, output, control))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=config.FELIX_DATASET_CLEAN)
    ap.add_argument("--extra", nargs="*", default=[], help="extra augmented CSVs (clean schema)")
    ap.add_argument("--outdir", default="data/felix_dataset")
    ap.add_argument("--val-frac", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    rows = read_clean(args.src)
    for p in args.extra:
        rows += read_clean(p)

    # de-duplicate on (input, target)
    seen, uniq = set(), []
    for neutral, output, control in rows:
        src = f"insert slang | control: {control} | {neutral}"
        key = (src, output)
        if key in seen:
            continue
        seen.add(key)
        uniq.append({"input": src, "target": output, "control": control})

    random.shuffle(uniq)
    n_val = int(len(uniq) * args.val_frac)
    val, train = uniq[:n_val], uniq[n_val:]

    os.makedirs(args.outdir, exist_ok=True)
    for name, split in [("editor_train.jsonl", train), ("editor_val.jsonl", val)]:
        with open(os.path.join(args.outdir, name), "w", encoding="utf-8") as f:
            for ex in split:
                f.write(json.dumps({"input": ex["input"], "target": ex["target"]},
                                   ensure_ascii=False) + "\n")

    controls = sorted({r["control"] for r in uniq})
    print(f"total pairs: {len(uniq)}  | train: {len(train)}  val: {len(val)}")
    print(f"distinct control tokens: {len(controls)}")
    print("sample input :", train[0]["input"] if train else "(none)")
    print("sample target:", train[0]["target"] if train else "(none)")
    print(f"wrote {args.outdir}/editor_train.jsonl and editor_val.jsonl")


if __name__ == "__main__":
    main()
