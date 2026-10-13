"""
data_prep/hand_to_judged.py
Convert the in-session, hand-written (Claude-as-teacher) pairs in data/felix_dataset/v3/hand/*.jsonl
into the `judged` schema the builder reads. Every pair goes through the SAME runtime guards the
pipeline applies (slang newly inserted from the bank, drift guard, punctuation); the slang actually
inserted is detected from the edit text (so a mislabelled slang cannot slip in).

  python -m data_prep.hand_to_judged            # -> data/felix_dataset/v3/judged_hand.jsonl
"""
from __future__ import annotations
import glob, json, re
from collections import Counter
from data_prep.v3_common import read_jsonl, write_jsonl, norm
from data_prep.label_v3 import clean_bank, validate_edit, fix_punct

D = "data/felix_dataset/v3/"
OK = {"score": 4.0, "meaning": True, "by": "claude-session"}


def detect(sentence, edit, names):
    """The bank slang newly inserted by the edit (longest first)."""
    low_s, low_e = sentence.lower(), edit.lower()
    for k in sorted(names, key=len, reverse=True):
        if re.search(r"(?<![a-z])" + re.escape(k) + r"(?![a-z])", low_e) and k not in low_s:
            return names[k]
    return None


def main():
    bank = clean_bank()
    names = {k: v[0] for k, v in bank.items()}
    allnames = set(names.values())
    rows, rej, seen = [], Counter(), set()
    for fn in sorted(glob.glob(D + "hand/*.jsonl")):
        for ln in open(fn, encoding="utf-8"):
            if not ln.strip():
                continue
            o = json.loads(ln)
            s = o["s"].strip()
            if norm(s) in seen:
                rej["duplicate"] += 1
                continue
            seen.add(norm(s))
            rec = {"scene_id": "hand_" + fn.split("hand_")[-1][:2], "speaker": "", "line": s, "clause": s,
                   "bank": list(names.values()), "label": "NONE", "slang": None, "edit": None,
                   "also_ok": [], "alt": None, "alt2": None, "why": ""}
            if str(o.get("slang", "")).upper() == "NONE":
                rows.append(rec); continue
            ed = o["edit"].strip()
            sl = detect(s, ed, names)
            ok, why = validate_edit(s, ed, sl, allnames) if sl else (False, "no-bank-slang-inserted")
            if not ok:
                rej[why] += 1
                continue
            rec.update(label="EDIT", slang=sl, edit=fix_punct(s, ed), judge=dict(OK), keep=True,
                       also_ok=[names[x.lower()] for x in o.get("also", []) if x.lower() in names][:4])
            a = o.get("alt")
            if isinstance(a, dict):
                a_ed = a["edit"].strip()
                a_sl = detect(s, a_ed, names)
                if a_sl and a_sl != sl and validate_edit(s, a_ed, a_sl, allnames)[0]:
                    rec["alt"] = {"slang": a_sl, "edit": fix_punct(s, a_ed), "judge": dict(OK), "keep": True}
                else:
                    rej["alt:" + (validate_edit(s, a_ed, a_sl, allnames)[1] if a_sl else "no-slang")] += 1
            rows.append(rec)
    write_jsonl(D + "judged_hand.jsonl", rows)
    c = Counter(r["label"] for r in rows)
    n_alt = sum(1 for r in rows if r.get("alt"))
    print(f"hand pairs: {dict(c)} | alts kept {n_alt} | rejected {dict(rej)}")
    use = Counter(r["slang"] for r in rows if r["label"] == "EDIT")
    for r in rows:
        if r.get("alt"):
            use[r["alt"]["slang"]] += 1
    print("distinct slangs:", len(use), "| rows:", sum(use.values()))
    print("least covered:", [(s, use.get(s, 0)) for s in sorted(names.values(), key=lambda x: use.get(x, 0))[:18]])


if __name__ == "__main__":
    main()
