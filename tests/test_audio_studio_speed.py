"""Offline tests for: expressive-voice casting + acting directions, the hybrid TTS fallback, stage-direction stripping, room acoustics / ambience,
the concurrent Agent 1 extraction and the overlapped Pass 1 -> editor pipeline stage."""
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from audio import ambience, mix, voices
from audio.synthesize import scene_effects, spoken
from audio.tts import FakeBackend, HybridBackend


def test_expressive_voices_are_distinct_and_fit_gender_and_age():
    profiles = {"Ananya": {"gender": "female", "age_group": "teen", "turn_count": 9}, "Priya": {"gender": "female", "age_group": "young_adult", "turn_count": 5},
                "Nani": {"gender": "female", "age_group": "elder", "social_role": "elder", "turn_count": 4},
                "Vikram": {"gender": "male", "age_group": "young_adult", "turn_count": 8}, "Papa": {"gender": "male", "age_group": "adult", "turn_count": 2},
                "Dadaji": {"gender": "male", "age_group": "elder", "social_role": "elder", "turn_count": 3}}
    cast = voices.assign_gpt_voices(voices.cast_voices(profiles))
    females = [cast[n]["gpt_voice"] for n in ("Ananya", "Priya", "Nani")]
    males = [cast[n]["gpt_voice"] for n in ("Vikram", "Papa", "Dadaji")]
    assert len(set(females)) == 3 and len(set(males)) == 3
    assert all(v in voices.GPT_VOICES["female"] for v in females) and all(v in voices.GPT_VOICES["male"] for v in males)
    assert cast["Nani"]["gpt_voice"] in ("sage", "shimmer") and cast["Dadaji"]["gpt_voice"] in ("onyx", "fable", "alloy")


def test_acting_instructions_carry_character_emotion_and_listener():
    base = {"gender": "female", "age_group": "teen"}
    t = voices.acting_instructions("Ananya", {"traits": ["anxious", "caring"]}, base, "scared", 3, "Vikram",
                                   {"register": "respectful", "relation": "elder brother"}, "a dark house in the rain", "Mujhe dar lag raha hai!")
    for needle in ("Ananya", "teenager", "woman or girl", "anxious, caring", "Speaking to Vikram", "elder brother", "respectfully", "very strongly scared",
                   "shaky", "exclamation", "Never add, drop or change a word"):
        assert needle in t, needle
    q = voices.acting_instructions("Rohan", {}, {"gender": "male", "age_group": "adult"}, "neutral", 1, "", None, "", "Kya hua?")
    assert "question" in q and "Speaking to" not in q


def test_stage_directions_are_never_spoken():
    assert spoken("(muskurate hue) Haan, main cake launga.") == "Haan, main cake launga."
    assert spoken("Chalo [hansta hai] shuru karte hain *sigh*.") == "Chalo shuru karte hain ."
    assert spoken("Bas yahi hai.") == "Bas yahi hai."
    assert spoken("(sirf direction)") == "(sirf direction)"                      # never return an empty line


class _Boom:
    name = "boom"
    stats = {}

    def synth_many(self, jobs):
        return [RuntimeError("no credit") if i % 2 == 0 else None for i, _ in enumerate(jobs)]


def test_hybrid_backend_repairs_every_failed_line_with_the_free_voice(tmp_path):
    fb = FakeBackend(str(tmp_path))
    jobs = [{"text": f"line number {i}", "voice": "hi-IN-MadhurNeural", "edge_voice": "hi-IN-SwaraNeural"} for i in range(4)]
    out = HybridBackend(_Boom(), fb).synth_many(jobs)
    assert len(out) == 4 and all(isinstance(o, str) and os.path.exists(o) for o in out)


def test_ambience_detection_reverb_and_beds():
    assert ambience.detect("Practice field", "Kabir plays", "match tomorrow")["bed"] == "stadium"
    assert ambience.detect("Inside the house", "heavy rain, power cut", "")["bed"] == "rain"
    assert ambience.detect("School market", "", "")["bed"] in ("market", "room")
    assert ambience.detect("Hospital room", "", "")["bed"] == "room" and ambience.detect("Hospital room", "", "")["reverb"] > 0.08
    assert ambience.detect("Open field", "", "")["reverb"] < 0.06
    x = (0.1 * np.sin(2 * np.pi * 200 * np.arange(24000) / 24000)).astype(np.float32)
    y = ambience.reverb(x, wet=0.1)
    assert len(y) == len(x) and 0.5 < np.sqrt(np.mean(y ** 2)) / np.sqrt(np.mean(x ** 2)) < 1.5
    assert np.array_equal(ambience.reverb(x, wet=0.0), x)
    for kind in ("rain", "stadium", "market", "room"):
        b = ambience.bed(kind, 3.0, "s")
        assert len(b) == 72000 and 0.03 < np.sqrt(np.mean(b[24000:48000] ** 2)) < 0.08 and abs(b[0]) < 0.01      # faded in
        assert np.array_equal(b, ambience.bed(kind, 3.0, "s"))                                                      # deterministic


def test_scene_beds_are_quieter_than_speech_and_mix_assembles_with_fx(tmp_path):
    fb = FakeBackend(str(tmp_path / "c"))
    jobs = [{"text": "yeh ek lambi si line hai jisme kuch bol raha hoon", "voice": "a"}, {"text": "aur yeh doosri line hai", "voice": "b"}]
    paths = fb.synth_many(jobs)
    items = [{"path": p, "scene_id": "scene_01", "speaker": s, "text": j["text"], "kind": "dialogue", "pause_before_ms": 0, "pause_after_ms": 0}
             for p, j, s in zip(paths, jobs, ["A", "B"])]
    dry, _ = mix.assemble(items)
    wet, tl = mix.assemble(items, {"scene_01": {"reverb": 0.1, "bed": "rain"}})
    assert abs(len(wet) - len(dry)) < 24000 and np.abs(wet).max() <= 0.91
    quiet = wet[:int(tl[0]["start"] * mix.SR * 0.5)]                      # before the first word: only the bed
    assert np.sqrt(np.mean(quiet ** 2)) < 0.5 * np.sqrt(np.mean(wet[int(tl[0]["start"] * mix.SR):int(tl[0]["end"] * mix.SR)] ** 2))


def test_scene_effects_from_trace():
    trace = {"neutral_script": [{"scene_id": "scene_01", "events": {"location": "Home", "key_events": ["heavy rain starts"]}},
                                {"scene_id": "scene_02", "events": {"location": "Cricket ground", "key_events": ["the match begins"]}}]}
    plan = [{"scene_id": "scene_01", "kind": "dialogue", "text": "Baarish bahut tez hai."},
            {"scene_id": "scene_02", "kind": "dialogue", "text": "Match shuru ho gaya."}]
    fx = scene_effects(trace, plan)
    assert fx["scene_01"]["bed"] == "rain" and fx["scene_02"]["bed"] == "stadium"


def test_agent1_extracts_scenes_concurrently_and_links_them():
    from agents.agent1_narrative import NarrativeAnalyst
    a = NarrativeAnalyst.__new__(NarrativeAnalyst)

    def fake_llm(prompt):
        sid = [w for w in prompt.split('"') if w.startswith("scene_")][0]
        return json.dumps([{"scene_id": sid, "key_events": ["e"], "characters": ["Ananya"], "scene_goal": f"goal of {sid}",
                            "location": "home", "narrative_link": "Following x, Ananya now does y", "pronoun_map": {}}])
    a._call_llm = fake_llm
    chunks = [{"scene_id": f"scene_0{i}", "text": f"ananya scene {i}"} for i in range(1, 6)]
    ev = a.extract_event_chain(chunks)
    assert [e["scene_id"] for e in ev] == [c["scene_id"] for c in chunks]
    assert "goal of scene_02" in ev[2]["narrative_link"] and "goal of scene_01" not in ev[0]["narrative_link"]


def test_overlapped_stage_edits_scenes_while_pass1_is_written_and_stage5_reuses_them():
    from orchestrator.pipeline import HinglishPipeline
    calls = []
    p = HinglishPipeline.__new__(HinglishPipeline)
    p.verbose = False
    p.agent3 = SimpleNamespace(generate_pass1=lambda chunk, prof, ev: f"Ananya: line of {ev['scene_id']}.")
    p.critic = None
    p.transliterator = SimpleNamespace(transliterate_sentence=lambda s: s)
    p.agent4 = SimpleNamespace(reset=lambda: calls.append("reset"), intent_clf=SimpleNamespace(classify=lambda u: calls.append("intent") or "CASUAL_ASSERTION"),
                               rewrite_script=lambda sc: calls.append("edit:" + sc[0]["scene_id"]) or [{"scene_id": sc[0]["scene_id"], "dialogue": sc[0]["dialogue"] + "!", "decisions": []}])
    p.scratchpad = SimpleNamespace(read=lambda k: {"persona": {"scene_cast": {}, "secrets": []}, "deva_scenes": ["a", "b", "c"]}.get(k),
                                   write=lambda k, v: None)
    state = {"scene_chunks": [{"scene_id": f"scene_0{i}", "text": "t"} for i in (1, 2, 3)],
             "event_chain": [{"scene_id": f"scene_0{i}", "characters": ["Ananya"], "key_events": ["e"]} for i in (1, 2, 3)],
             "character_profiles": {}, "neutral_script": []}
    state = p._generate_pass1(state)
    assert calls.count("reset") == 1 and [c for c in calls if c.startswith("edit")] == ["edit:scene_01", "edit:scene_02", "edit:scene_03"]
    assert calls.count("intent") >= 3                                                  # intents were classified ahead of the editor
    p._log = lambda *a, **k: None
    state.update({"rewrite_count": 0, "failed_scenes": [], "scene_diagnoses": {}})
    out = p._inject_slang(state)
    assert [s["dialogue"] for s in out["slang_script"]] == ["Ananya: line of scene_01.!", "Ananya: line of scene_02.!", "Ananya: line of scene_03.!"]
    assert calls.count("reset") == 1                                                   # stage 5 reused the overlapped edits: no second pass
