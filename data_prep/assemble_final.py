"""
data_prep/assemble_final.py
Merge all clean slang-pair sources into ONE final dataset:
  - salvaged (pre-rated >=4), hand-authored gold, and the vetted generated rows.
Normalises columns, case-normalises slang, de-duplicates on the output line,
drops empties / no-change rows, and writes a single clean CSV.

Usage:
  python -m data_prep.assemble_final --out data/felix_dataset/final_dataset.csv \
      --src data/felix_dataset/salvaged_good_rows.csv data/felix_dataset/claude_generated.csv \
            data/felix_dataset/gen_claude_vetted.csv data/felix_dataset/gen_haiku_vetted.csv \
            data/felix_dataset/gen_quality_vetted.csv
"""
from __future__ import annotations
import argparse, csv, os, re
from collections import Counter


def norm_key(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def read(path):
    rows = []
    if not os.path.exists(path):
        print(f"  (skip, not found: {path})"); return rows
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            neutral = (r.get("neutral") or "").strip()
            output = (r.get("output") or "").strip()
            control = (r.get("control") or "ASSERT_NEUTRAL").strip().upper() or "ASSERT_NEUTRAL"
            slang = (r.get("slangs") or r.get("slang") or "").strip()
            src = (r.get("source") or os.path.basename(path)).strip()
            if neutral and output and neutral.lower() != output.lower():
                rows.append({"neutral": neutral, "output": output, "slangs": slang,
                             "control": control, "source": src})
    print(f"  {os.path.basename(path):34s} -> {len(rows)} rows")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--out", default="data/felix_dataset/final_dataset.csv")
    args = ap.parse_args()

    allrows = []
    print("Reading sources:")
    for p in args.src:
        allrows += read(p)

    # de-duplicate on the output line (case/space-insensitive)
    seen, uniq = set(), []
    for r in allrows:
        k = norm_key(r["output"])
        if k in seen:
            continue
        seen.add(k); uniq.append(r)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cols = ["neutral", "output", "slangs", "control", "source"]
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(uniq)

    bysrc = Counter(r["source"] for r in uniq)
    byctrl = Counter(r["control"] for r in uniq)
    print(f"\nFINAL: {len(uniq)} unique rows (from {len(allrows)} merged)")
    print("  by source :", dict(bysrc))
    print("  distinct slangs:", len({r["slangs"].lower() for r in uniq if r["slangs"]}))
    print("  distinct controls:", len(byctrl))
    print("saved:", args.out)


if __name__ == "__main__":
    main()
