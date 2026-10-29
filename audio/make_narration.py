"""
audio/make_narration.py - turn a Devanagari story into a narrated recording (a stand-in for a person telling the story), with a longer
silence between paragraphs than between sentences. Used (1) as a repeatable test of the speech-input path - the true words and the
true scene breaks are known - and (2) to create a sample voice file for the demo UI.

    python -m audio.make_narration data/raw/demo_cricket_tournament.txt outputs/audio_samples/cricket_narrated
"""
from __future__ import annotations
import json
import os
import re
import sys

import numpy as np
import soundfile as sf

from audio import mix
from audio.tts import EdgeBackend, load_mono
from preprocessing.segmenter import split_sentences


def narrate(text: str, out_noext: str, voice: str = "hi-IN-MadhurNeural", sent_gap: float = 0.35, para_gap: float = 1.7,
            rate: str = "+0%", backend=None) -> dict:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    sents, para_start = [], []
    for p in paras:
        para_start.append(len(sents))
        sents += split_sentences(p)
    backend = backend or EdgeBackend()
    paths = backend.synth_many([{"text": s, "voice": voice, "rate": rate, "pitch": "+0Hz", "volume": "+0%"} for s in sents])
    chunks, t, times = [], 0.0, []
    gap_set = set(para_start[1:])
    for i, pth in enumerate(paths):
        x, sr = load_mono(pth)
        x = mix._resample(x, sr)
        x = mix.fade(mix.level(mix.trim_silence(x, keep_ms=40)))
        if i > 0:
            g = para_gap if i in gap_set else sent_gap
            chunks.append(np.zeros(int(g * mix.SR), np.float32))
            t += g
        times.append((round(t, 2), round(t + len(x) / mix.SR, 2)))
        chunks.append(x)
        t += len(x) / mix.SR
    audio = np.concatenate(chunks)
    os.makedirs(os.path.dirname(out_noext) or ".", exist_ok=True)
    sf.write(out_noext + ".wav", audio, mix.SR)
    try:
        sf.write(out_noext + ".mp3", audio, mix.SR, format="MP3")
    except Exception:
        pass
    meta = {"sentences": sents, "scene_starts": para_start, "sentence_times": times, "duration": round(len(audio) / mix.SR, 1),
            "voice": voice}
    json.dump(meta, open(out_noext + "_truth.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return meta


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    src, out = sys.argv[1], sys.argv[2]
    m = narrate(open(src, encoding="utf-8").read(), out)
    print(f"narrated {len(m['sentences'])} sentences, {m['duration']}s -> {out}.wav/.mp3 (+_truth.json)")
