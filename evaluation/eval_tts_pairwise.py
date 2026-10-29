"""
evaluation/eval_tts_pairwise.py - blind A/B/C preference: for every line the listener model hears the three renderings in RANDOM order and picks the one
that sounds most like a real human actor performing the intended emotion. More sensitive than absolute scores. Re-uses the clips rendered by
evaluation.eval_tts_naturalness (cached).    python -m evaluation.eval_tts_pairwise
"""
from __future__ import annotations
import base64
import json
import os
import random
import re
import sys
import urllib.request

sys.path.insert(0, os.getcwd())
from dotenv import load_dotenv
load_dotenv(os.path.join(os.getcwd(), ".env"))
from audio import asr, prosody, voices
from audio.tts import EdgeBackend, GPTAudioBackend
from evaluation.eval_tts_naturalness import LINES

PROMPT = """You are an expert Hindi film dubbing director. Three clips (labelled A, B, C) all say the same line.
Speaker: {who}. Intended emotion: {emotion}. Line: {text}
Which clip sounds most like a REAL HUMAN actor naturally performing this line with this emotion - natural rhythm, believable emotion, native-sounding Hindi,
no robotic or reading quality? Answer ONLY JSON: {{"best": "A|B|C", "worst": "A|B|C", "reason": "one short sentence"}}"""


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    profiles = {n: {"gender": g, "age_group": a, "social_role": r, "traits": [], "turn_count": 5 - i}
                for i, (n, g, a, r, *_x) in enumerate({l[0]: l for l in LINES}.values())}
    cast = voices.assign_gpt_voices(voices.cast_voices(profiles))
    systems = {"edge": EdgeBackend(), "gpt-mini": GPTAudioBackend("openai/gpt-audio-mini"), "gpt-full": GPTAudioBackend("openai/gpt-audio")}
    clips = {k: [] for k in systems}
    for name, be in systems.items():
        jobs = []
        for (spk, g, a, role, emo, inten, listener, deva) in LINES:
            base = cast[spk]
            st = prosody.combine(base, prosody.emotion_prosody(emo, inten, deva))
            jobs.append({"text": deva, "voice": base["voice"], "gpt_voice": base["gpt_voice"],
                         "instructions": voices.acting_instructions(spk, profiles[spk], base, emo, inten, listener, None, "", deva), **prosody.edge_args(st)})
        clips[name] = be.synth_many(jobs)
    wins = {k: 0 for k in systems}
    losses = {k: 0 for k in systems}
    rng = random.Random(11)
    for rep in range(2):                                           # two shuffles per line to cancel position bias
        for i, (spk, g, a, role, emo, inten, listener, deva) in enumerate(LINES):
            order = list(systems)
            rng.shuffle(order)
            parts = [{"type": "text", "text": PROMPT.format(who=f"{spk}, a {a.replace('_', ' ')} {g}", emotion=emo, text=deva)}]
            for lab, sysname in zip("ABC", order):
                parts.append({"type": "text", "text": f"Clip {lab}:"})
                x = asr.load_audio(clips[sysname][i])
                parts.append({"type": "input_audio", "input_audio": {"data": base64.b64encode(asr._wav_bytes(x)).decode(), "format": "wav"}})
            body = {"model": "google/gemini-2.5-flash", "temperature": 0.0, "max_tokens": 400, "messages": [{"role": "user", "content": parts}]}
            req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"], "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    txt = json.loads(r.read().decode())["choices"][0]["message"]["content"]
                j = json.loads(re.search(r"\{.*\}", txt, re.DOTALL).group())
                best, worst = order["ABC".index(j["best"])], order["ABC".index(j["worst"])]
                wins[best] += 1
                losses[worst] += 1
                print(f"line {i + 1} ({spk}, {emo}): best {best:8s} worst {worst:8s} | {j.get('reason', '')[:90]}", flush=True)
            except Exception as e:
                print("judge failed:", str(e)[:100])
    print("\nBLIND PREFERENCE (12 judgements: 6 lines x 2 shuffles):")
    for k in systems:
        print(f"  {k:9s} chosen best {wins[k]:2d}x | chosen worst {losses[k]:2d}x")


if __name__ == "__main__":
    main()
