"""
evaluation/eval_segmenter.py - how well are scenes recovered from a story whose paragraph breaks have been REMOVED?

Every story in data/raw/ that the author wrote with blank lines is flattened (paragraph breaks deleted), segmented again, and compared with the
author's paragraphs (the gold scene starts).   python -m evaluation.eval_segmenter [--llm]

Metrics: boundary precision / recall / F1 (exact, and within +-1 sentence), and WindowDiff (lower is better).
Systems: OLD = the previous ProseChunker fallback on the romanized text, SIGNALS = new segmenter without the LLM, SIGNALS+LLM.
"""
from __future__ import annotations
import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.getcwd())
from dotenv import load_dotenv
load_dotenv()
from preprocessing.segmenter import SceneSegmenter, split_sentences

GOLD_STORIES = ["demo_cricket_tournament", "demo_science_fair", "test_story_1", "test_story_2", "test_story_3", "test_story_long"]


def gold_starts(paras):
    starts, n = [], 0
    for p in paras:
        starts.append(n)
        n += len(split_sentences(p))
    return starts, n


def prf(pred, gold, tol=0):
    p, g = set(pred) - {0}, set(gold) - {0}
    if not g and not p:
        return 1.0, 1.0, 1.0
    tp_p = sum(any(abs(x - y) <= tol for y in g) for x in p)
    tp_r = sum(any(abs(x - y) <= tol for x in p) for y in g)
    prec = tp_p / len(p) if p else 0.0
    rec = tp_r / len(g) if g else 0.0
    f = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f


def windowdiff(pred, gold, n):
    gs = sorted(set(gold) | {0})
    k = max(2, round(n / (2 * len(gs))))
    pb, gb = set(pred) - {0}, set(gold) - {0}
    wrong = 0
    for i in range(n - k + 1):
        if sum(1 for b in pb if i < b <= i + k - 1) != sum(1 for b in gb if i < b <= i + k - 1):
            wrong += 1
    return wrong / max(1, n - k + 1)


def old_chunker_starts(flat_text, n_gold_sents):
    """The previous behaviour on a one-block story: transliterate, then ProseChunker's sentence-grouping fallback."""
    from preprocessing.transliterate import HindiTransliterator
    from preprocessing.chunker import ProseChunker
    rom = HindiTransliterator().transliterate_sentence(flat_text)
    chunks = ProseChunker().chunk(rom)
    starts, n = [], 0
    for c in chunks:
        starts.append(n)
        n += len([s for s in re.split(r"(?<=[\.\?\!])\s+", c["text"].strip()) if s.strip()])
    return starts if n == n_gold_sents else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true")
    a = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    sig = SceneSegmenter(use_llm=False)
    llm = SceneSegmenter(use_llm=True) if a.llm else None
    agg = {"OLD": [], "SIGNALS": [], "SIGNALS+LLM": []}
    print(f"{'story':26s} {'sent':>4s} {'gold':>4s} | {'system':12s} {'scenes':>6s} {'P':>5s} {'R':>5s} {'F1':>5s} {'F1±1':>5s} {'WD':>5s}")
    for name in GOLD_STORIES:
        path = f"data/raw/{name}.txt"
        if not os.path.exists(path):
            continue
        txt = open(path, encoding="utf-8").read()
        paras = [p.strip() for p in re.split(r"\n\s*\n", txt) if p.strip()]
        gold, n = gold_starts(paras)
        flat = " ".join(paras)
        systems = {"OLD": old_chunker_starts(flat, n), "SIGNALS": sig.segment(flat).starts}
        if llm:
            seg_l = llm.segment(flat)
            systems["SIGNALS+LLM"] = seg_l.starts
            if "+llm" not in seg_l.method:
                print(f"   (LLM referee not used for {name}: {seg_l.notes})")
        for sysname, st in systems.items():
            if st is None:
                print(f"{name:26s} {n:4d} {len(gold):4d} | {sysname:12s}  (sentence counts differ - skipped)")
                continue
            p, r, f = prf(st, gold)
            f1 = prf(st, gold, 1)[2]
            wd = windowdiff(st, gold, n)
            agg[sysname].append((p, r, f, f1, wd, len(st)))
            print(f"{name:26s} {n:4d} {len(gold):4d} | {sysname:12s} {len(st):6d} {p:5.2f} {r:5.2f} {f:5.2f} {f1:5.2f} {wd:5.2f}")
    print("\nMEAN over stories:")
    for sysname, rows in agg.items():
        if rows:
            m = [sum(x[i] for x in rows) / len(rows) for i in range(5)]
            print(f"  {sysname:12s} P {m[0]:.2f}  R {m[1]:.2f}  F1 {m[2]:.2f}  F1(+-1) {m[3]:.2f}  WindowDiff {m[4]:.2f}   ({len(rows)} stories)")


if __name__ == "__main__":
    main()
