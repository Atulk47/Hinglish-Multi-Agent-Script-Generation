"""
evaluation/eval_voice_input.py - end-to-end test of VOICE INPUT with narrated recordings whose words and scene breaks are known.

    python -m evaluation.eval_voice_input [--story sciencefair|cricket] [--backend cloud|local|auto] [--local-model small]

Reports: transcription character error rate, how well the measured pauses separate scene breaks from ordinary sentence gaps, and the
scene-boundary F1 of the Scene Segmenter with and without the pause evidence.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.getcwd())
from dotenv import load_dotenv
load_dotenv()
from audio.asr import transcribe
from evaluation.eval_segmenter import prf, windowdiff
from preprocessing.segmenter import SceneSegmenter, split_sentences


def cer(ref: str, hyp: str) -> float:
    norm = lambda s: re.sub(r"[^ऀ-ॿa-z0-9]", "", s.lower())
    a, b = norm(ref), norm(hyp)
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (a[i - 1] != b[j - 1]))
    return d[len(b)] / max(len(a), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--story", default="sciencefair")
    ap.add_argument("--backend", default="cloud")
    ap.add_argument("--local-model", default="small")
    ap.add_argument("--no-llm", action="store_true")
    a = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    stem = f"outputs/audio_samples/{a.story}_narrated"
    truth = json.load(open(stem + "_truth.json", encoding="utf-8"))
    ref_text = " ".join(truth["sentences"])
    t0 = time.time()
    res = transcribe(stem + ".wav", backend=a.backend, local_model=a.local_model)
    print(f"backend {res['backend']} | audio {res['duration']}s | transcribed in {res['seconds']}s | notes {res['notes']}")
    print(f"CER vs the true story text: {cer(ref_text, res['text']):.3f}")
    print("first 160 chars:", res["text"][:160])
    sents = res["sentences"]
    print(f"sentences: transcript {len(sents)} vs true {len(truth['sentences'])}")
    # how well do the pauses separate scene breaks?
    gold_gaps = set(truth["scene_starts"][1:])
    if len(sents) == len(truth["sentences"]):
        scene_p = [res["pauses"][i - 1] for i in gold_gaps]
        other_p = [p for i, p in enumerate(res["pauses"], 1) if i not in gold_gaps]
        print(f"measured pause at scene breaks: mean {sum(scene_p) / len(scene_p):.2f}s | at ordinary gaps: mean {sum(other_p) / len(other_p):.2f}s")
    seg = SceneSegmenter(use_llm=not a.no_llm)
    # the transcript may split sentences slightly differently from the original, so boundaries are compared by their POSITION in the text
    # (character fraction), a boundary counting as found when it lies within 2.5% of the story of a true one
    tot_ref = sum(len(x) for x in truth["sentences"])
    gold_frac, acc = [], 0
    for i, snt in enumerate(truth["sentences"]):
        if i in set(truth["scene_starts"][1:]):
            gold_frac.append(acc / tot_ref)
        acc += len(snt)
    for label, pz in (("without pauses", None), ("with pauses", res["pauses"])):
        s = seg.segment(res["text"], pauses=pz, trust_paragraphs=False)
        tot = sum(len(x) for x in s.sentences)
        pred, acc = [], 0
        for i, snt in enumerate(s.sentences):
            if i in set(s.starts[1:]):
                pred.append(acc / tot)
            acc += len(snt)
        tol = 0.025
        tp_p = sum(any(abs(x - y) <= tol for y in gold_frac) for x in pred)
        tp_r = sum(any(abs(x - y) <= tol for x in pred) for y in gold_frac)
        pr = tp_p / len(pred) if pred else 0.0
        rc = tp_r / len(gold_frac) if gold_frac else 0.0
        f = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        print(f"[{label:15s}] {len(s.starts):2d} scenes (true {len(truth['scene_starts'])}) | P {pr:.2f} R {rc:.2f} F1 {f:.2f} | {s.method}")


if __name__ == "__main__":
    main()
