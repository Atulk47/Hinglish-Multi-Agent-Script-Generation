"""
data_prep/review_pack.py
In-session (no API credit) review harness. Claude, working in the coding session, acts as the
strict judge / teacher: it reads compact candidate lists and writes verdict files; this script
turns those into the same `judged` schema the dataset builder reads.

  dump  : python -m data_prep.review_pack dump  labeled2.jsonl  --start 0 --n 150   -> prints candidates
  apply : python -m data_prep.review_pack apply labeled2.jsonl  verdicts.json judged2.jsonl
          verdicts.json = {"keep": [3, 7, ...]}  (candidate numbers that score >= 4 AND keep the meaning)

  python -m data_prep.review_pack newlabels  sentences.jsonl  labels.jsonl  out_judged.jsonl
          labels.jsonl rows: {"id": n, "slang": "NONE"|<bank slang>, "edit": "...", "also_ok": [...],
                              "alt": {"slang","edit"}|null, "alt2": ...}   -> validated with the same
          runtime guards (newly inserted, drift guard, punctuation) and stored as judged rows.
"""
from __future__ import annotations
import json, os, sys
from data_prep.v3_common import read_jsonl, write_jsonl, norm
from data_prep.label_v3 import clean_bank, validate_edit, fix_punct

D = "data/felix_dataset/v3/"


def candidates(rows):
    """Flat, stable numbering of every EDIT candidate (main, alt, alt2)."""
    out = []
    for ri, r in enumerate(rows):
        if r["label"] != "EDIT":
            continue
        out.append((ri, "main", r["clause"], r["edit"], r["slang"]))
        for ak in ("alt", "alt2"):
            if r.get(ak):
                out.append((ri, ak, r["clause"], r[ak]["edit"], r[ak]["slang"]))
    return out


def dump(inp, start, n):
    rows = read_jsonl(D + inp)
    c = candidates(rows)
    for k in range(start, min(start + n, len(c))):
        ri, kind, o, e, s = c[k]
        print(f"#{k}\n O: {o}\n E: {e}   [{s}]")
    print(f"-- {min(start + n, len(c))}/{len(c)}")


def apply(inp, verdict, out):
    rows = read_jsonl(D + inp)
    keep = set(json.load(open(verdict))["keep"])
    c = candidates(rows)
    ok = {"score": 4.0, "meaning": True, "by": "claude-session"}
    bad = {"score": 2.0, "meaning": False, "by": "claude-session"}
    for k, (ri, kind, o, e, s) in enumerate(c):
        tgt = rows[ri] if kind == "main" else rows[ri][kind]
        tgt["judge"] = ok if k in keep else bad
        tgt["keep"] = k in keep
    write_jsonl(D + out, rows)
    print(f"kept {len(keep)}/{len(c)} candidates -> {D}{out}")


def newlabels(sent_file, label_file, out):
    bank = clean_bank()
    names = {k: v[0] for k, v in bank.items()}
    sents = {r["n"]: r for r in read_jsonl(D + sent_file)}
    rows, rej = [], 0
    for lab in read_jsonl(D + label_file):
        s = sents.get(lab["id"])
        if not s:
            continue
        clause = s["clause"]
        rec = dict(s, bank=list(names.values()), label="NONE", slang=None, edit=None, also_ok=[],
                   alt=None, alt2=None, why="")
        slang = str(lab.get("slang", "")).strip()
        if slang.upper() == "NONE" or not slang:
            rows.append(rec); continue
        sl = names.get(slang.lower())
        ok, why = validate_edit(clause, str(lab.get("edit", "")).strip(), sl, set(names.values()))
        if not ok:
            rec["label"] = "REJECT"; rec["why"] = why; rows.append(rec); rej += 1; continue
        rec.update(label="EDIT", slang=sl, edit=fix_punct(clause, lab["edit"].strip()),
                   judge={"score": 4.0, "meaning": True, "by": "claude-session"}, keep=True)
        rec["also_ok"] = [names[x.lower()] for x in lab.get("also_ok", []) if x.lower() in names][:4]
        for ak in ("alt", "alt2"):
            a = lab.get(ak)
            if isinstance(a, dict):
                a_sl = names.get(str(a.get("slang", "")).strip().lower())
                if a_sl and validate_edit(clause, str(a.get("edit", "")).strip(), a_sl, set(names.values()))[0]:
                    rec[ak] = {"slang": a_sl, "edit": fix_punct(clause, a["edit"].strip()),
                               "judge": {"score": 4.0, "meaning": True, "by": "claude-session"}, "keep": True}
        rows.append(rec)
    write_jsonl(D + out, rows)
    from collections import Counter
    print(dict(Counter(r["label"] for r in rows)), "| guard-rejected:", rej, "->", D + out)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "dump":
        import argparse
        ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("inp")
        ap.add_argument("--start", type=int, default=0); ap.add_argument("--n", type=int, default=150)
        a = ap.parse_args(); dump(a.inp, a.start, a.n)
    elif cmd == "apply":
        apply(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "newlabels":
        newlabels(sys.argv[2], sys.argv[3], sys.argv[4])
