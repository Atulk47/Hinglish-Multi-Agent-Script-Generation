"""
tools/apply_slang_review.py  -  apply the native speaker's answers (from outputs/slang_pool_review/review.html) to the slang pool.

    python -m tools.apply_slang_review slang_review_answers.json [--dry-run]

* Section B (new candidates): Keep -> promoted to status "core" in data/slang/slang_pool.json (with the control labels the older pools use, so the polarity guards and
  the control-based shortlists know the phrase); Drop -> marked "rejected" in data/slang/slang_candidates.json; Fix -> listed for manual follow-up (not promoted).
* Section A (existing phrases): Drop -> status "disabled" (no longer used); Fix -> listed for manual follow-up; Keep -> unchanged.
Nothing is deleted: the pool file keeps every entry with its status.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

POOL = "data/slang/slang_pool.json"
CANDS = "data/slang/slang_candidates.json"
CONTROLS = {"praise": ["ASSERT_POSITIVE"], "criticism": ["ASSERT_NEGATIVE"], "worry": ["ASSERT_ANXIETY", "ASSERT_STRESS"],
            "trouble": ["ASSERT_FRUSTRATION"], "ease": ["ASSERT_POSITIVE"],
            "joy": ["ASSERT_HAPPY"], "pride": ["ASSERT_HAPPY"], "like": ["ASSERT_POSITIVE"], "anger": ["ASSERT_ANGER"]}


def apply(answers: list, dry: bool = False) -> dict:
    pool = json.load(open(POOL, encoding="utf-8"))
    cands = json.load(open(CANDS, encoding="utf-8"))
    pe = {e["phrase"].lower(): e for e in pool["entries"]}
    ce = {c["phrase"].lower(): c for c in cands["entries"]}
    rep = {"promoted": [], "rejected": [], "fix_new": [], "disabled": [], "fix_existing": [], "unanswered": []}
    for a in answers:
        p, dec, note = a["phrase"], (a.get("decision") or "").lower(), a.get("note", "")
        existing = a.get("section", "").startswith("A.")
        if not dec:
            rep["unanswered"].append(p)
            continue
        if existing:
            e = pe.get(p.lower())
            if dec == "drop" and e:
                e["status"] = "disabled"
                rep["disabled"].append(p)
            elif dec == "fix":
                rep["fix_existing"].append((p, note))
            continue
        c = ce.get(p.lower())
        if not c:
            continue
        if dec == "keep":
            slot = c["slots"][0]
            entry = {k: c.get(k) for k in ("phrase", "slots", "sign", "intensity", "kind", "gender", "frame", "context")}
            entry["intensity"] = entry["intensity"] or 2
            entry["controls"] = CONTROLS.get(slot, []) + (["ASSERT_EXCITED"] if slot == "praise" and entry["intensity"] == 3 else [])
            entry["status"] = "core"
            entry["note"] = "; ".join(x for x in (c.get("do_not_use_when"), c.get("risk"), note) if x)
            if p.lower() not in pe:
                pool["entries"].append(entry)
                pe[p.lower()] = entry
            c["status"] = "promoted"
            rep["promoted"].append(p)
        elif dec == "drop":
            c["status"] = "rejected"
            rep["rejected"].append(p)
        elif dec == "fix":
            c["note"] = note
            rep["fix_new"].append((p, note))
    if not dry:
        json.dump(pool, open(POOL, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        json.dump(cands, open(CANDS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("answers")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    rep = apply(json.load(open(a.answers, encoding="utf-8")), a.dry_run)
    for k, v in rep.items():
        print(f"{k:13s} {len(v):3d}  {v if len(v) <= 40 else v[:40]}")
    if not a.dry_run:
        from agents import edit_gate as eg
        print("pool now:", len(eg.POOL_ENTRIES), "core entries |", len(eg.POS_SLANG), "praise,", len(eg.NEG_SLANG), "criticism,", len(eg.FEEL_SLANG), "worry,",
              len(eg.PROB_SLANG), "trouble,", len(eg.EASE_SLANG), "ease")


if __name__ == "__main__":
    main()
