"""Offline tests for the audio output: casting, prosody, parsing, mixing/timeline, subtitles, script restorer, end-to-end with a fake TTS."""
import json
import os
import re
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import soundfile as sf

from audio import mix, prosody, script_restore, voices
from audio.synthesize import build_plan, parse_dialogue, synthesize_trace
from audio.tts import FakeBackend


def prof(gender, age, role="peer", traits=(), turns=5):
    return {"gender": gender, "age_group": age, "social_role": role, "traits": list(traits), "turn_count": turns}


PROFILES = {"Kabir": prof("male", "young_adult", traits=["leader"], turns=11), "Rohan": prof("male", "teen", traits=["energetic"], turns=8),
            "Aryan": prof("male", "teen", turns=5), "Coach Varma": prof("male", "adult", "authority", ["strict"], 3),
            "Dadaji": prof("male", "elder", "elder", ["wise"], 6), "Papa": prof("male", "adult", "elder", turns=1),
            "Mira": prof("female", "young_adult", turns=6), "Sneha": prof("female", "child", turns=6),
            "Mummy": prof("female", "adult", "elder", turns=4)}


def test_casting_gives_every_character_a_distinct_sound():
    cast = voices.cast_voices(PROFILES)
    sig = [(c["voice"], c["pitch_hz"]) for c in cast.values()]
    assert len(set(sig)) == len(sig), "two characters sound identical"
    assert cast["Mira"]["voice"].startswith("hi-IN-Swara") and cast["Kabir"]["voice"].startswith("hi-IN-Madhur")
    assert cast["Dadaji"]["pitch_hz"] < cast["Rohan"]["pitch_hz"] and cast["Dadaji"]["rate_pct"] < cast["Rohan"]["rate_pct"]
    assert cast["Sneha"]["pitch_hz"] > cast["Mummy"]["pitch_hz"]                     # child higher than adult woman
    assert cast["Coach Varma"]["volume_pct"] > cast["Kabir"]["volume_pct"]           # authority a bit louder
    assert voices.cast_voices(PROFILES) == cast                                      # deterministic


def test_unknown_gender_is_guessed_deterministically_and_flagged():
    c1 = voices.cast_voices({"Zed": {"gender": "unknown"}})
    c2 = voices.cast_voices({"Zed": {"gender": "unknown"}})
    assert c1 == c2 and "guessed" in c1["Zed"]["why"]


def test_voice_for_is_case_and_variant_tolerant():
    cast = voices.cast_voices(PROFILES)
    assert voices.voice_for(cast, "KABIR", PROFILES)["voice"] == cast["Kabir"]["voice"]
    assert voices.voice_for(cast, "Coach Varma", PROFILES)["voice"] == cast["Coach Varma"]["voice"]
    assert voices.voice_for(cast, "Nobody", PROFILES)["why"] == "fallback"


def test_prosody_follows_arousal_and_valence_and_is_clamped():
    p = {e: prosody.emotion_prosody(e, 2) for e in ("neutral", "excited", "sad", "scared", "angry", "proud")}
    assert p["excited"]["rate_pct"] > p["neutral"]["rate_pct"] > p["sad"]["rate_pct"]
    assert p["scared"]["pitch_hz"] > p["sad"]["pitch_hz"]
    assert p["sad"]["volume_pct"] < 0 < p["angry"]["volume_pct"]
    assert prosody.emotion_prosody("excited", 3)["rate_pct"] >= prosody.emotion_prosody("excited", 1)["rate_pct"]
    for e in prosody.EMOTIONS:
        for k in (1, 2, 3):
            for key, (lo, hi) in prosody.LIMITS.items():
                assert lo <= prosody.emotion_prosody(e, k)[key] <= hi
    assert prosody.emotion_prosody("neutral", 1, "Kya hua?")["pitch_hz"] > prosody.emotion_prosody("neutral", 1, "Kya hua")["pitch_hz"]
    assert prosody.emotion_prosody("neutral", 1, "Soch raha hoon...")["pause_after_ms"] > 0


def test_edge_args_match_edge_tts_regexes():
    a = prosody.edge_args({"rate_pct": -5, "pitch_hz": 0, "volume_pct": 12})
    assert re.fullmatch(r"[+-]\d+%", a["rate"]) and re.fullmatch(r"[+-]\d+Hz", a["pitch"]) and re.fullmatch(r"[+-]\d+%", a["volume"])
    assert a == {"rate": "-5%", "pitch": "+0Hz", "volume": "+12%"}


def test_parse_dialogue():
    d = parse_dialogue("Kabir: Rohan, tum thik ho?\n\n**Mira**: Haan.\nnot a line\nCoach Varma: Bachchon, suno: dhyan se.")
    assert d == [("Kabir", "Rohan, tum thik ho?"), ("Mira", "Haan."), ("Coach Varma", "Bachchon, suno: dhyan se.")]


def test_gap_logic():
    a = {"scene_id": "s1", "speaker": "A", "text": "Hi."}
    assert mix.gap_ms(None, a) == 250
    assert mix.gap_ms(a, {"scene_id": "s2", "speaker": "A", "text": "x"}) > 1000                          # scene break
    q = {"scene_id": "s1", "speaker": "A", "text": "Kya?"}
    assert mix.gap_ms(q, {"scene_id": "s1", "speaker": "B", "text": "x"}) > mix.gap_ms(a, {"scene_id": "s1", "speaker": "B", "text": "x"})
    assert mix.gap_ms(a, {"scene_id": "s1", "speaker": "A", "text": "x"}) < mix.gap_ms(a, {"scene_id": "s1", "speaker": "B", "text": "x"})


def test_assemble_timeline_levels_and_srt(tmp_path):
    be = FakeBackend(str(tmp_path / "c"))
    jobs = [{"text": t, "voice": v, "rate": "+0%", "pitch": "+0Hz", "volume": "+0%"} for t, v in
            [("Rohan, tum thik ho?", "a"), ("Haan main theek hoon.", "b"), ("Chalo shuru karte hain.", "a")]]
    paths = be.synth_many(jobs)
    items = [{"path": p, "scene_id": "scene_01" if i < 2 else "scene_02", "speaker": s, "text": j["text"], "kind": "dialogue",
              "pause_before_ms": 0, "pause_after_ms": 0} for i, (p, j, s) in enumerate(zip(paths, jobs, ["Kabir", "Mira", "Kabir"]))]
    audio, tl = mix.assemble(items)
    assert audio.dtype == np.float32 and np.abs(audio).max() <= 0.91
    for a, b in zip(tl, tl[1:]):
        assert a["end"] <= b["start"] and a["start"] < a["end"]                                          # ordered, no overlap
    assert tl[2]["start"] - tl[1]["end"] > tl[1]["start"] - tl[0]["end"]                                 # longer pause between scenes
    assert abs(tl[-1]["end"] - len(audio) / mix.SR) < 0.6
    srt = mix.to_srt(tl)
    assert re.search(r"1\n00:00:\d\d,\d{3} --> 00:00:\d\d,\d{3}\nKabir: Rohan, tum thik ho\?", srt)
    out = mix.save_audio(audio, str(tmp_path / "x"))
    d, sr = sf.read(out["wav"])
    assert sr == mix.SR and len(d) == len(audio)


def test_script_restore_validates_and_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(script_restore, "CACHE", str(tmp_path / "cache.json"))

    def client(reply):
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **k: SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply))]))))
    lines = ["Mujhe dar lag raha hai.", "Kya hua?"]
    good = json.dumps(["मुझे डर लग रहा है।", "क्या हुआ?"], ensure_ascii=False)
    assert script_restore.restore(lines, client(good)) == ["मुझे डर लग रहा है।", "क्या हुआ?"]
    assert script_restore.restore(lines, client("nonsense")) == ["मुझे डर लग रहा है।", "क्या हुआ?"]              # served from the cache
    monkeypatch.setattr(script_restore, "CACHE", str(tmp_path / "other.json"))
    assert script_restore.restore(lines, client("nonsense")) == lines                                    # invalid answer -> original text
    bad_q = json.dumps(["मुझे डर लग रहा है।", "क्या हुआ"], ensure_ascii=False)                              # lost the '?'
    assert script_restore.restore(lines, client(bad_q))[1] == "Kya hua?"


MINI = {
    "character_profiles": {n: {**p, "emotion_arc": [{"scene_id": "scene_01", "emotion": "excited" if n == "Rohan" else "sad", "intensity": 2},
                                                     {"scene_id": "scene_02", "emotion": "happy", "intensity": 3}]} for n, p in PROFILES.items()},
    "neutral_script": [{"scene_id": "scene_01", "dialogue": "Kabir: Rohan, tum thik ho?\nRohan: Haan yaar, main theek hoon!\nCoach Varma: Bachchon, suno.",
                        "events": {"location": "School cricket field"}},
                       {"scene_id": "scene_02", "dialogue": "Mira: Hum jeet gaye!\nDadaji: Shabaash, beta.", "events": {"location": "Hospital room"}}],
}
MINI["slang_script"] = [dict(MINI["neutral_script"][0], dialogue=MINI["neutral_script"][0]["dialogue"].replace("thik", "ekdum thik")),
                        MINI["neutral_script"][1]]


def test_plan_has_narration_emotion_and_distinct_settings():
    plan = build_plan(MINI, "slang")
    kinds = [p["kind"] for p in plan]
    assert kinds == ["narration", "dialogue", "dialogue", "dialogue", "narration", "dialogue", "dialogue"]
    assert plan[0]["text"] == "Scene 1. School cricket field."
    rohan = next(p for p in plan if p["speaker"] == "Rohan")
    kabir = next(p for p in plan if p["speaker"] == "Kabir")
    assert rohan["emotion"] == "excited" and kabir["emotion"] == "sad" and rohan["rate_pct"] > kabir["rate_pct"]
    assert "ekdum thik" in plan[1]["text"]
    assert all(p["voice"].startswith(("hi-IN", "en-IN")) for p in plan)
    assert not [p for p in build_plan(MINI, "neutral", narrator=False) if p["kind"] == "narration"]


def test_end_to_end_with_fake_tts(tmp_path):
    res = synthesize_trace(MINI, str(tmp_path), "both", backend=FakeBackend(str(tmp_path / "c")), text_mode="roman")
    for v in ("slang", "neutral"):
        m = res[v]
        assert m["duration_s"] > 3 and len(m["timeline"]) == 7
        for f in ("wav", "srt", "manifest.json"):
            assert any(n.startswith(v) and n.endswith(f) for n in os.listdir(tmp_path)), f
    html = open(res["player"], encoding="utf-8").read()
    assert "Hinglish Audio Screenplay" in html and "ekdum thik" in html and "Dadaji" in html
    cast = json.load(open(tmp_path / "cast.json", encoding="utf-8"))
    assert cast["Mira"]["voice"].startswith("hi-IN-Swara")
