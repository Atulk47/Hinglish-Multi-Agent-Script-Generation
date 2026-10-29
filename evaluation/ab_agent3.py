"""
evaluation/ab_agent3.py
A/B test of the Agent-3 Pass-1 style ("default" vs "slotrich") through the WHOLE pipeline (models loaded once).
For each story x style it reports what actually matters for the slang editor and for validity:
  * Pass-1 stays NEUTRAL (no slang/filler words leaked in)  and  nothing failed
  * share of peer sentences that contain a word a slang can naturally replace ("slot")
  * how many edits the trained editor made, how many distinct slangs, mood-slang vs filler
  * MuRIL event recall (the dialogue must still cover the story's key events)

    python -u -m evaluation.ab_agent3 --stories data/raw/demo_science_fair.txt data/raw/test_story_1.txt
"""
import argparse, json, re, time
from collections import Counter
import config
from orchestrator.pipeline import HinglishPipeline
from agents.edit_gate import has_slot, FILLERS
from data_prep.slang_bank_v3 import BANK

SLANG_WORDS = {k for k in BANK if " " not in k} | set(FILLERS)
ELDER = re.compile(r"\b(dadi|daadi|nani|grandmother|sharm|teacher|professor|committee|judge)\b", re.I)


def measure(pl, story, style):
    config.AGENT3_STYLE = style
    t = time.time()
    pl.run(open(story, encoding="utf-8").read())
    secs = time.time() - t
    neutral = pl.scratchpad.read("neutral_script") or []
    slang = pl.scratchpad.read("slang_script") or []
    diag = pl.scratchpad.read("scene_diagnoses") or {}
    sents, slots, leaked, words, failed = 0, 0, 0, 0, 0
    sample = []
    for sc in neutral:
        if "[generation failed" in sc["dialogue"]:
            failed += 1
        for ln in sc["dialogue"].split("\n"):
            if ":" not in ln:
                continue
            spk, utt = [x.strip() for x in ln.split(":", 1)]
            if ELDER.search(spk):
                continue
            for u in re.split(r"(?<=[.!?])\s+", utt.strip().strip('"')):
                if len(u.split()) < 4:
                    continue
                sents += 1
                words += len(u.split())
                slots += has_slot(u)
                leaked += bool(set(re.findall(r"[a-z]+", u.lower())) & SLANG_WORDS)
            if len(sample) < 4:
                sample.append(f"{spk}: {utt.strip()[:130]}")
    edits = [d for sc in slang for d in sc["decisions"] if d.get("applied")]
    used = Counter(d.get("slang") for d in edits)
    nfill = sum(v for k, v in used.items() if k in FILLERS)
    rec = [v.get("event_recall", 1.0) for v in diag.values()] or [1.0]
    return {"story": story.split("/")[-1], "style": style, "secs": round(secs), "peer_sentences": sents,
            "mean_words": round(words / max(sents, 1), 1), "slot_share": round(slots / max(sents, 1), 2),
            "leaked_slang_words": leaked, "failed_scenes": failed, "edits": len(edits),
            "distinct_slangs": len(used), "mood_edits": len(edits) - nfill, "filler_edits": nfill,
            "slangs": dict(used), "mean_event_recall": round(sum(rec) / len(rec), 2), "sample": sample}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stories", nargs="+", default=["data/raw/demo_science_fair.txt"])
    ap.add_argument("--out", default="outputs/ab_agent3.json")
    args = ap.parse_args()
    pl = HinglishPipeline("ab_agent3", verbose=False)
    res = []
    for story in args.stories:
        for style in ("default", "slotrich"):
            r = measure(pl, story, style)
            res.append(r)
            print(json.dumps({k: v for k, v in r.items() if k != "sample"}, ensure_ascii=False), flush=True)
            for ln in r["sample"]:
                print("      ", ln, flush=True)
            json.dump(res, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
