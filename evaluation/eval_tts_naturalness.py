"""
evaluation/eval_tts_naturalness.py - which voice engine sounds most natural and expressive?

    python -m evaluation.eval_tts_naturalness

Renders the SAME six character lines with (a) the free neural voices (prosody from the emotion rules), (b) gpt-audio-mini and (c) gpt-audio with
acting instructions, then has an audio-capable model (Gemini 2.5 Flash, a different model family from the generators) LISTEN to every clip blind
and score it 1-5 for naturalness (human vs robotic), emotion match and Hindi clarity, and transcribe it (for a character error rate).
An LLM listener is a proxy, not a replacement for a human ear - but it is repeatable and unbiased towards any one engine.
"""
from __future__ import annotations
import base64
import json
import os
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.getcwd())
from dotenv import load_dotenv
load_dotenv(os.path.join(os.getcwd(), ".env"))
from audio import asr, prosody, voices
from audio.tts import EdgeBackend, GPTAudioBackend
from evaluation.eval_voice_input import cer

LINES = [  # (speaker, gender, age, role, emotion, intensity, listener, devanagari)
    ("Ananya", "female", "teen", "peer", "scared", 2, "Vikram", "भैया, सजावट का सामान सब भीग जाएगा! मुझे बहुत डर लग रहा है।"),
    ("Vikram", "male", "young_adult", "peer", "affectionate", 2, "Ananya", "चिंता मत कर, हम सब कुछ मिलकर ठीक कर लेंगे।"),
    ("Nani", "female", "elder", "elder", "affectionate", 3, "Ananya", "मेरे बच्चों... आप सबने मेरे लिए यह सब किया? मेरी आँखों में आँसू आ गए हैं।"),
    ("Priya", "female", "young_adult", "peer", "excited", 2, "Ananya", "देखना, नानी यह सब देखकर बहुत खुश होंगी!"),
    ("Dukaandaar", "male", "adult", "peer", "neutral", 1, "Vikram", "बेटा, इससे सस्ता केक भी मिल जाएगा।"),
    ("Vikram", "male", "young_adult", "peer", "anxious", 2, "Dukaandaar", "मुझे बहुत चिंता हो रही है, पर नानी के लिए मैं अपनी सारी बचत लगा दूँगा।"),
]
JUDGE = """You are an expert listener evaluating a text-to-speech clip for a Hindi film dubbing project.
The speaker is meant to be: {who}. Intended emotion: {emotion}. Intended text: {text}
Listen and rate each from 1 to 5 (be strict; use the whole scale):
- natural: 5 = indistinguishable from a real human actor, 1 = clearly robotic / reading voice
- emotion: how well the delivery matches the intended emotion and character (5 = perfect, 1 = flat or wrong)
- clarity: Hindi pronunciation and intelligibility (5 = native-like, 1 = hard to understand)
Also transcribe exactly what is said, in Devanagari.
Return ONLY JSON: {{"natural": n, "emotion": n, "clarity": n, "transcript": "..."}}"""


def listen(path: str, who: str, emotion: str, text: str) -> dict:
    x = asr.load_audio(path)
    body = {"model": "google/gemini-2.5-flash", "temperature": 0.0, "max_tokens": 600,
            "messages": [{"role": "user", "content": [{"type": "text", "text": JUDGE.format(who=who, emotion=emotion, text=text)},
                                                      {"type": "input_audio", "input_audio": {"data": base64.b64encode(asr._wav_bytes(x)).decode(), "format": "wav"}}]}]}
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"], "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.loads(r.read().decode())
    txt = out["choices"][0]["message"]["content"]
    return json.loads(re.search(r"\{.*\}", txt, re.DOTALL).group())


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    profiles = {n: {"gender": g, "age_group": a, "social_role": r, "traits": [], "turn_count": 5 - i}
                for i, (n, g, a, r, *_x) in enumerate({l[0]: l for l in LINES}.values())}
    cast = voices.assign_gpt_voices(voices.cast_voices(profiles))
    os.makedirs("outputs/_audio_try/nat", exist_ok=True)
    systems = {"edge (free neural voices)": EdgeBackend(), "gpt-audio-mini": GPTAudioBackend("openai/gpt-audio-mini"),
               "gpt-audio (full)": GPTAudioBackend("openai/gpt-audio")}
    results = {k: [] for k in systems}
    for name, be in systems.items():
        jobs = []
        for (spk, g, a, role, emo, inten, listener, deva) in LINES:
            base = cast[spk]
            st = prosody.combine(base, prosody.emotion_prosody(emo, inten, deva))
            jobs.append({"text": deva, "voice": base["voice"], "gpt_voice": base["gpt_voice"],
                         "instructions": voices.acting_instructions(spk, profiles[spk], base, emo, inten, listener, None, "", deva), **prosody.edge_args(st)})
        t0 = time.time()
        paths = be.synth_many(jobs)
        print(f"{name}: rendered {len(paths)} lines in {time.time() - t0:.0f}s | stats {getattr(be, 'stats', {})}", flush=True)
        for (spk, g, a, role, emo, inten, listener, deva), pth, job in zip(LINES, paths, jobs):
            if isinstance(pth, Exception):
                print("   FAILED:", pth)
                continue
            who = f"{spk}, a {a.replace('_', ' ')} {g}"
            try:
                j = listen(pth, who, emo, deva)
                j["cer"] = cer(deva, j.get("transcript", ""))
                results[name].append(j)
            except Exception as e:
                print("   judge failed:", str(e)[:100])
    print("\nMEAN SCORES (1-5, higher is better; CER lower is better) - judged blind by Gemini 2.5 Flash:")
    print(f"{'engine':28s} {'natural':>8s} {'emotion':>8s} {'clarity':>8s} {'CER':>6s}  n")
    for k, rows in results.items():
        if rows:
            m = lambda f: sum(float(r[f]) for r in rows) / len(rows)
            print(f"{k:28s} {m('natural'):8.2f} {m('emotion'):8.2f} {m('clarity'):8.2f} {m('cer'):6.3f}  {len(rows)}")
    json.dump(results, open("outputs/eval_tts_naturalness.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
