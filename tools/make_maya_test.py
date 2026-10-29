"""
tools/make_maya_test.py  -  prepares the Maya 2 Native listening test.

  1. picks ~13 representative dialogue lines from an existing audio-drama manifest (varied speakers / emotions / heavy-Hinglish lines),
  2. restores them to Devanagari (cached script_restore), maps every character to a Maya voice,
  3. cuts the SAME lines out of the existing Edge / Studio renderings, for side-by-side listening,
  4. writes  notebooks/test_maya2_tts_kaggle.py  (one file to upload to Kaggle, the lines are embedded in it).

    python -m tools.make_maya_test
"""
from __future__ import annotations
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import soundfile as sf

BASE = "outputs/prof2_audio_standard"
STUDIO = "outputs/prof2_audio_studio"
OUT = "outputs/maya_test"
NOTEBOOK = "notebooks/test_maya2_tts_kaggle.py"

# (scene, index of the speaker's n-th line within the scene) -> chosen to cover emotions, short/long lines and English words inside Hindi
PICK = [("scene_01", "Ananya", 0), ("scene_01", "Vikram", 1), ("scene_02", "Vikram", 0), ("scene_02", "Dukaandaar", 0),
        ("scene_02", "Vikram", 1), ("scene_03", "Ananya", 0), ("scene_03", "Vikram", 0), ("scene_03", "Ananya", 2),
        ("scene_04", "Priya", 0), ("scene_04", "Ananya", 2), ("scene_05", "Nani", 0), ("scene_05", "Nani", 2), ("scene_05", "Vikram", 0)]

# character -> Maya voice (gender / age matched; the "conversational" Hindi voices first)
VOICE = {"Ananya": "Vrinda", "Vikram": "Shaunak", "Nani": "Malini", "Priya": "Nivedita", "Dukaandaar": "Akshat", "Narrator": "Anirban"}
TAG = {"excited": "[excited]", "happy": "[excited]", "sad": "[sad]", "scared": "[whispers]", "affectionate": "", "frustrated": "", "determined": "",
       "amused": "[laughs]", "neutral": ""}


def cut(wav, sr, t):
    return wav[int(t["start"] * sr):int(t["end"] * sr)]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    from audio import script_restore
    man = json.load(open(f"{BASE}/slang_manifest.json", encoding="utf-8"))
    studio = None
    if os.path.exists(f"{STUDIO}/slang_manifest.json"):
        studio = json.load(open(f"{STUDIO}/slang_manifest.json", encoding="utf-8"))
    chosen = []
    for sid, who, n in PICK:
        rows = [t for t in man["timeline"] if t["kind"] == "dialogue" and t["scene_id"] == sid and t["speaker"] == who]
        if n < len(rows):
            chosen.append(rows[n])
    deva = script_restore.restore([c["text"] for c in chosen])
    os.makedirs(f"{OUT}/edge", exist_ok=True)
    os.makedirs(f"{OUT}/studio", exist_ok=True)
    w_edge, sr = sf.read(f"{BASE}/slang.wav")
    w_studio = sf.read(f"{STUDIO}/slang.wav")[0] if studio else None
    lines = []
    for i, (c, d) in enumerate(zip(chosen, deva), 1):
        lid = f"{i:02d}_{c['scene_id'][-2:]}_{c['speaker']}_{c['emotion']}"
        sf.write(f"{OUT}/edge/{lid}.wav", cut(w_edge, sr, c), sr)
        if studio:
            match = next((t for t in studio["timeline"] if t["kind"] == "dialogue" and t["text"] == c["text"] and t["speaker"] == c["speaker"]), None)
            if match is not None:
                sf.write(f"{OUT}/studio/{lid}.wav", cut(w_studio, sr, match), sr)
        lines.append({"id": lid, "speaker": c["speaker"], "emotion": c["emotion"], "voice": VOICE.get(c["speaker"], "Anirban"),
                      "text_roman": c["text"], "text_deva": d, "tag": TAG.get(c["emotion"], "")})
    json.dump(lines, open(f"{OUT}/lines.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    tpl = open("notebooks/_maya_test_template.py", encoding="utf-8").read()
    open(NOTEBOOK, "w", encoding="utf-8").write(tpl.replace("__LINES_JSON__", json.dumps(lines, ensure_ascii=False, indent=1)))
    print(f"{len(lines)} lines -> {OUT}/lines.json ; reference clips cut ; notebook -> {NOTEBOOK}")
    for l in lines:
        print(f"  {l['id']:34s} {l['voice']:9s} {l['tag']:11s} {l['text_deva']}")


if __name__ == "__main__":
    main()
