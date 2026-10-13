"""
data_prep/extract_real_v3.py
Collect the REAL Agent-3 sentences the editor actually receives (peer speakers only) from the five
benchmark traces. These are NEVER used for training; they form the held-out "real" evaluation set,
so the reported improvement is synthetic-training -> real-pipeline transfer.

  python -m data_prep.extract_real_v3
"""
import json, os, re
from data_prep.v3_common import candidate_units, write_jsonl, norm

import glob
TRACES = {os.path.basename(p).replace("_trace.json", ""): p for p in sorted(glob.glob("outputs/*_trace.json"))}
# elders / formal speakers are never edited by the pipeline -> excluded from the eval set
SKIP_TAG = re.compile(r"\b(dadi|daadi|dada|nani|naani|grandmother|grandfather|committee|panel|jury|judge|"
                      r"judges|board|announcer|host|principal|teacher|professor|doctor|officer|official|"
                      r"mother|father|maa|papa|mummy|uncle|aunty|narrator|narration)\b", re.I)
# SPEAKER/UNKNOWN-tagged lines are skipped by default at runtime, but the editor's *sentence quality*
# can still be measured on them, so they are kept and flagged.
UNATTR = re.compile(r"\b(speaker|unknown|voice)\b", re.I)


def main(out="data/felix_dataset/v3/real_lines.jsonl"):
    rows, seen = [], set()
    for story, path in TRACES.items():
        if not os.path.exists(path):
            print("missing", path); continue
        try:
            t = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        for sc in t.get("neutral_script") or []:
            if "[generation failed" in sc.get("dialogue", ""):
                continue
            for ln in sc["dialogue"].split("\n"):
                if ":" not in ln:
                    continue
                spk, utt = [x.strip() for x in ln.split(":", 1)]
                utt = utt.strip().strip('"').strip()
                if not utt or SKIP_TAG.search(spk):
                    continue
                for u in candidate_units(utt, 4, 24):
                    if norm(u) in seen:
                        continue
                    seen.add(norm(u))
                    rows.append({"scene_id": f"real_{story}", "speaker": spk, "line": utt, "story": story, "unattributed": bool(UNATTR.search(spk))})
    write_jsonl(out, rows)
    from collections import Counter
    print(len(rows), "real sentences", dict(Counter(r["story"] for r in rows)))


if __name__ == "__main__":
    main()
