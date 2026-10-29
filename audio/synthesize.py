"""
audio/synthesize.py  -  screenplay (from a saved pipeline trace) -> multi-voice audio drama + subtitles + synced HTML player.

    python -m audio.synthesize --trace outputs/cricket5_trace.json --out outputs/cricket5_audio --variant both

Steps:  Voice Casting Agent (voices.py)  ->  per-line emotion (Persona Agent's arc)  ->  prosody (prosody.py)  ->  [optional]
Script Restorer (mixed Devanagari / English text for the Hindi voice)  ->  TTS (tts.py)  ->  levelling + timeline (mix.py)
->  WAV / MP3, SRT, manifest.json, player.html.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv()                                   # OPENROUTER_API_KEY for the expressive voices / script restorer when run as a CLI
from agents.agent2b_persona import emotion_at
from agents.name_utils import match_profile
from audio import mix, prosody, voices
from audio.tts import EdgeBackend, FakeBackend

DEFAULT_TEXT_MODE = os.environ.get("AUDIO_TEXT_MODE", "auto")


def default_backend(kind: str | None = None):
    """auto = expressive audio-LLM voices when an OpenRouter key exists (free neural voice as the per-line fallback), else the free voices."""
    import config
    kind = (kind or getattr(config, "AUDIO_TTS", "auto")).lower()
    if kind == "edge" or (kind == "auto" and not os.environ.get("OPENROUTER_API_KEY")):
        return EdgeBackend(concurrency=6)
    from audio.tts import GPTAudioBackend, HybridBackend, MayaBackend
    if kind == "maya":                                     # pre-rendered clips / live GPU server first, the free voices if neither is there
        return HybridBackend(MayaBackend(), EdgeBackend(concurrency=6))
    return HybridBackend(GPTAudioBackend(), EdgeBackend(concurrency=6))


def parse_dialogue(dialogue: str) -> List[tuple]:
    out = []
    for ln in (dialogue or "").split("\n"):
        if ":" not in ln:
            continue
        spk, utt = ln.split(":", 1)
        spk, utt = spk.strip().strip("*#_ "), utt.strip()
        if spk and utt and len(spk.split()) <= 3:
            out.append((spk, utt))
    return out


def _scene_number(sid: str) -> int:
    m = re.search(r"(\d+)", sid)
    return int(m.group(1)) if m else 0


def build_plan(trace: dict, variant: str, narrator: bool = True) -> List[dict]:
    """The ordered list of things to say: [{kind, scene_id, speaker, text, emotion, intensity, voice settings ...}]."""
    profiles = trace.get("character_profiles") or {}
    import config
    cast = voices.assign_maya_voices(voices.assign_gpt_voices(voices.cast_voices(profiles)))
    use_tags = getattr(config, "AUDIO_MAYA_TAGS", False)
    extras: Dict[str, str] = {}
    script = trace["slang_script"] if variant == "slang" else trace["neutral_script"]
    events = {s["scene_id"]: (s.get("events") or {}) for s in (trace.get("neutral_script") or [])}
    events.update({s["scene_id"]: s.get("events") or events.get(s["scene_id"], {}) for s in (trace.get("slang_script") or [])})
    plan = []
    for sc in script:
        sid = sc["scene_id"]
        if narrator:
            loc = (events.get(sid) or {}).get("location") or ""
            line = f"Scene {_scene_number(sid)}." + (f" {loc.strip().rstrip('.')}." if loc else "")
            n = voices.NARRATOR
            plan.append({"kind": "narration", "scene_id": sid, "speaker": "Narrator", "text": line, "tts_text": line,
                         "emotion": "neutral", "intensity": 1, "voice": n["voice"], "gpt_voice": voices.GPT_NARRATOR["voice"],
                         "instructions": "You are a calm storyteller introducing a scene of a film. Speak clearly, warmly and unhurried, in English with an Indian accent.",
                         "rate_pct": n["rate_pct"], "pitch_hz": n["pitch_hz"], "volume_pct": n["volume_pct"],
                         "maya_voice": voices.MAYA_NARRATOR["voice"], "maya_lang": voices.MAYA_NARRATOR["lang"],
                         "pause_before_ms": 0, "pause_after_ms": 150})
        prev_speaker = ""
        for spk, utt in parse_dialogue(sc["dialogue"]):
            prof = match_profile(profiles, spk) or {}
            emo = emotion_at(prof, sid) if prof else {"emotion": "neutral", "intensity": 1}
            base = voices.voice_for(cast, spk, profiles)
            if base.get("why") == "fallback":              # a speaker with no profile: its own Maya voice, never one of the main cast's
                base = dict(base, maya_voice=extras.setdefault(spk.lower(), voices.spare_maya_voice(cast, extras.values())))
            delta = prosody.emotion_prosody(emo["emotion"], emo["intensity"], utt)
            st = prosody.combine(base, delta)
            relation = (prof.get("relationship") or {}).get(prev_speaker.title()) if prev_speaker else None
            if not relation and prev_speaker:
                relation = next((r for o, r in (prof.get("relationship") or {}).items() if o.lower() == prev_speaker.lower()), None)
            loc = (events.get(sid) or {}).get("location") or ""
            plan.append({"kind": "dialogue", "scene_id": sid, "speaker": spk, "text": utt, "tts_text": utt,
                         "emotion": emo["emotion"], "intensity": emo["intensity"], "voice": base["voice"],
                         "gpt_voice": base.get("gpt_voice", "alloy"), "maya_voice": base.get("maya_voice", "Satvik"), "maya_lang": "hi",
                         "maya_tag": voices.MAYA_TAGS.get(emo["emotion"], "") if use_tags else "",
                         "instructions": voices.acting_instructions(spk.title(), prof, base, emo["emotion"], emo["intensity"],
                                                                    prev_speaker.title() if prev_speaker and prev_speaker.lower() != spk.lower() else "",
                                                                    relation, loc, utt), **st})
            prev_speaker = spk
    return plan


def _use_deva(text_mode: str) -> bool:
    return text_mode == "deva"


def spoken(text: str) -> str:
    """What the voice should actually say: no stage directions ("(muskurate hue)", "[laughs]", "*sigh*"), no markdown."""
    t = re.sub(r"\([^)]*\)|\[[^\]]*\]|\*[^*]*\*", " ", text)
    return re.sub(r"\s{2,}", " ", t).strip(" -") or text


def scene_effects(trace: dict, plan: list) -> dict:
    from audio import ambience
    events = {s["scene_id"]: (s.get("events") or {}) for s in (trace.get("neutral_script") or [])}
    fx = {}
    for sid in dict.fromkeys(p["scene_id"] for p in plan):
        ev = events.get(sid) or {}
        text = " ".join(p["text"] for p in plan if p["scene_id"] == sid and p["kind"] == "dialogue")
        fx[sid] = ambience.detect(ev.get("location") or "", " ".join(ev.get("key_events") or []), text)
    return fx


def prepare_plan(trace: dict, variant: str, narrator: bool = True, text_mode: str = DEFAULT_TEXT_MODE, restore_client=None):
    """-> (plan, resolved text_mode): the lines to say, stage directions removed, Hindi words restored to Devanagari for the voice."""
    plan = build_plan(trace, variant, narrator)
    for p in plan:
        if p["kind"] == "dialogue":
            p["tts_text"] = spoken(p["text"])
    if text_mode == "auto":                    # measured: Devanagari text -> 38% fewer ASR round-trip errors than romanized text
        text_mode = "deva"
    if _use_deva(text_mode):
        from audio import script_restore
        idx = [i for i, p in enumerate(plan) if p["kind"] == "dialogue"]
        restored = script_restore.restore([plan[i]["tts_text"] for i in idx], client=restore_client)
        for i, r in zip(idx, restored):
            plan[i]["tts_text"] = r
    return plan, text_mode


def plan_jobs(plan: List[dict]) -> List[dict]:
    """One job per line, carrying the settings of ALL engines: Maya voice, the expressive voice (instructions + gpt_voice) and the free neural
    voice (prosody)."""
    return [{"text": p["tts_text"], "voice": p["voice"], "edge_voice": p["voice"], "gpt_voice": p["gpt_voice"], "instructions": p["instructions"],
             "maya_voice": p.get("maya_voice"), "maya_lang": p.get("maya_lang", "hi"), "maya_tag": p.get("maya_tag", ""), **prosody.edge_args(p)}
            for p in plan]


def synthesize_variant(trace: dict, out_dir: str, variant: str = "slang", backend=None, narrator: bool = True,
                       text_mode: str = DEFAULT_TEXT_MODE, restore_client=None, ambience: bool = True) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    backend = backend or default_backend()
    plan, text_mode = prepare_plan(trace, variant, narrator, text_mode, restore_client)
    jobs = plan_jobs(plan)
    paths = backend.synth_many(jobs)
    items = [{**p, "path": path} for p, path in zip(plan, paths)]
    fx = scene_effects(trace, plan) if ambience else None
    audio, timeline = mix.assemble(items, fx)
    stem = os.path.join(out_dir, variant)
    files = mix.save_audio(audio, stem)
    srt_path = stem + ".srt"
    open(srt_path, "w", encoding="utf-8").write(mix.to_srt(timeline))
    manifest = {"variant": variant, "backend": getattr(backend, "name", "?"), "text_mode": text_mode,
                "duration_s": round(len(audio) / mix.SR, 2), "scene_fx": fx or {}, "files": {k: os.path.basename(v) for k, v in files.items()},
                "tts_stats": getattr(backend, "stats", {}), "timeline": timeline}
    json.dump(manifest, open(stem + "_manifest.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return manifest


def synthesize_trace(trace_path: str, out_dir: str, variant: str = "both", backend=None, narrator: bool = True,
                     text_mode: str = DEFAULT_TEXT_MODE, player: bool = True, ambience: bool = True) -> Dict[str, dict]:
    trace = json.load(open(trace_path, encoding="utf-8")) if isinstance(trace_path, str) else trace_path
    variants = ["slang", "neutral"] if variant == "both" else [variant]
    backend = backend or default_backend()
    out = {}
    cast = voices.assign_maya_voices(voices.assign_gpt_voices(voices.cast_voices(trace.get("character_profiles") or {})))
    for v in variants:
        out[v] = synthesize_variant(trace, out_dir, v, backend, narrator, text_mode, ambience=ambience)
        print(f"  [{v}] {out[v]['duration_s']:.0f}s of audio, {len(out[v]['timeline'])} clips | tts {out[v]['tts_stats']}")
    json.dump({n: {k: c[k] for k in ("voice", "gpt_voice", "maya_voice", "gender", "age_group", "rate_pct", "pitch_hz", "volume_pct", "why") if k in c}
               for n, c in cast.items()}, open(os.path.join(out_dir, "cast.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    if player:
        from audio.player import write_player
        out["player"] = write_player(out_dir, out, cast)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--variant", choices=["slang", "neutral", "both"], default="both")
    ap.add_argument("--text-mode", choices=["auto", "roman", "deva"], default=DEFAULT_TEXT_MODE)
    ap.add_argument("--no-narrator", action="store_true")
    ap.add_argument("--no-ambience", action="store_true", help="dry voices: no room reverb and no background beds")
    ap.add_argument("--fake", action="store_true", help="offline tones instead of speech (tests)")
    ap.add_argument("--tts", choices=["auto", "edge", "gpt", "maya"], default=None,
                    help="voice engine: gpt = expressive audio model, edge = free neural voices, maya = Maya 2 Native (pre-rendered clips / GPU server)")
    a = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    out_dir = a.out or a.trace.replace("_trace.json", "_audio")
    res = synthesize_trace(a.trace, out_dir, a.variant, FakeBackend() if a.fake else default_backend(a.tts), not a.no_narrator, a.text_mode,
                           ambience=not a.no_ambience)
    print("\nDone ->", out_dir)
    if "player" in res:
        print("Open:", res["player"])


if __name__ == "__main__":
    main()
