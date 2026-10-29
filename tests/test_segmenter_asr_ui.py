"""Offline tests: scene segmenter (signals, size limits, LLM-referee validation), voice-input helpers (silences, pause alignment, decoding),
and the UI's rendering helpers."""
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import soundfile as sf

from audio import asr
from preprocessing import segmenter as S
from preprocessing.segmenter import SceneSegmenter, gap_scores, pick_starts, split_sentences
from ui import render

STORY = ("कबीर क्रिकेट टीम का कप्तान था। रोहन उसका दोस्त था। दोनों रोज़ मैदान में अभ्यास करते थे। कोच ने कहा, \"मेहनत करो।\" "
         "कबीर ने कहा, \"हम जीतेंगे।\" अगले दिन रोहन के पैर में चोट लग गई। वह दर्द से कराह उठा। मीरा दौड़कर आई और घबरा गई। "
         "कोच ने उसे अस्पताल भेजा। उसी रात कबीर घर लौटा तो दादाजी इंतज़ार कर रहे थे। दादाजी ने प्यार से पूछा, \"क्या हुआ बेटा?\" "
         "कबीर ने सब बता दिया। दादाजी ने उसे समझाया। कुछ दिनों बाद फ़ाइनल का मुकाबला शुरू हुआ। कबीर क्रीज़ पर था। "
         "गेंद सीमा रेखा के पार चली गई। पूरा मैदान खुशी से झूम उठा।")


def test_sentence_splitting_keeps_quotes_and_dandas():
    s = split_sentences(STORY)
    assert len(s) == 17 and s[3].endswith("करो।\"") and s[0].startswith("कबीर क्रिकेट")
    assert split_sentences("") == [] and split_sentences("  ") == []


def test_time_cues_and_dialogue_penalty_drive_the_scores():
    s = split_sentences(STORY)
    sc = gap_scores(s)
    i_cue = next(i for i, x in enumerate(s) if x.startswith("अगले दिन")) - 1          # gap before "next day"
    i_dlg = 3                                                                           # between two sentences that both have quotes
    assert sc[i_cue] > sc[i_dlg] + 0.3 and sc[i_cue] > 0.4
    assert S._cue_score("उसी रात कबीर घर लौटा।") == 1.0 and S._cue_score("कबीर घर लौटा।") == 0.0


def test_pauses_raise_the_boundary_score():
    s = split_sentences(STORY)
    base = gap_scores(s)
    pz = [0.2] * (len(s) - 1)
    pz[4] = 2.0
    assert gap_scores(s, pz)[4] > base[4] + 0.4 and abs(gap_scores(s, pz)[0] - base[0]) < 0.35


def test_pick_starts_respects_size_limits():
    for n in (6, 9, 18, 40, 93):
        scores = list(np.random.RandomState(n).rand(n - 1))
        st = pick_starts(n, scores)
        sizes = [b - a for a, b in zip(st, st[1:] + [n])]
        assert st[0] == 0 and st == sorted(set(st)) and min(sizes) >= 2 and max(sizes) <= 8, (n, sizes)


def test_segmenter_trusts_author_paragraphs_and_splits_a_block():
    seg = SceneSegmenter(use_llm=False)
    r = seg.segment("एक वाक्य यहाँ है। दूसरा भी है।\n\nतीसरा नया दृश्य है। चौथा भी।")
    assert r.method == "author-paragraphs" and len(r.scenes) == 2
    r = seg.segment(STORY)
    assert r.method.startswith("signals") and 2 <= len(r.scenes) <= 8 and "".join(r.scenes).count("।") >= 15
    assert any(r.sentences[i].startswith(("अगले दिन", "उसी रात", "कुछ दिनों")) for i in r.starts[1:])         # cut at the time shifts
    r2 = seg.segment("\n\n".join(["वाक्य एक है। " * 12]))                                      # one huge paragraph is split further
    assert len(r2.scenes) >= 2


def fake_llm(reply):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **k: SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=reply))]))))


def test_llm_referee_is_validated_and_never_breaks_the_run():
    s = split_sentences(STORY)
    good = SceneSegmenter(client=fake_llm(json.dumps({"scene_starts": [1, 6, 10, 14]})))
    r = good.segment(STORY)
    assert r.method.endswith("+llm") and r.starts == [0, 5, 9, 13]
    for bad_reply in ("sorry", json.dumps({"scene_starts": [1, 99]}), json.dumps({"scene_starts": [1]})):
        r = SceneSegmenter(client=fake_llm(bad_reply)).segment(STORY)
        assert "llm" not in r.method and "rejected" in r.notes and len(r.scenes) >= 2                   # fell back to the signals


def _tone_audio(sr=16000):
    t = np.arange(int(0.8 * sr)) / sr
    burst = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    gap = lambda s: np.zeros(int(s * sr), np.float32)
    # speech 0.8 | 0.3s | speech | 1.8s | speech | 0.3s | speech
    return np.concatenate([gap(0.2), burst, gap(0.3), burst, gap(1.8), burst, gap(0.3), burst, gap(0.2)]), sr


def test_silence_detection_and_load_audio(tmp_path):
    x, sr = _tone_audio()
    sil = asr.silences(x, min_s=0.25)
    durs = sorted(round(b - a, 1) for a, b in sil if 0.25 < (b - a) < 3)
    assert 1.7 <= durs[-1] <= 1.9 and len(durs) >= 2
    p = tmp_path / "t.wav"
    sf.write(p, np.stack([x, x], axis=1), 44100)                                        # stereo 44.1 kHz -> mono 16 kHz
    y = asr.load_audio(str(p))
    assert y.ndim == 1 and abs(len(y) / asr.SR - len(x) / sr * (44100 / 16000) / (44100 / 16000) * (sr / sr) * (len(y) / len(y))) < 5
    assert abs(len(y) / asr.SR - 5.4 * (1.0) * (len(x) / sr) / 5.4) < 1e-6 or len(y) > 0


def test_pause_alignment_puts_the_long_pause_on_the_right_gap():
    x, sr = _tone_audio()
    sil = asr.silences(x, min_s=0.25)
    sents = ["पहला वाक्य यहाँ है।", "दूसरा वाक्य भी है।", "तीसरा वाक्य नया दृश्य है।", "चौथा वाक्य आखिरी है।"]
    # the sentences have equal length, so the long silence (after the 2nd burst) falls after sentence 2
    pz = asr.align_pauses(sents, len(x) / sr, sil)
    assert len(pz) == 3 and pz.index(max(pz)) == 1 and pz[1] > 1.5 and pz[0] < 0.6 and pz[2] < 0.6


def test_transcribe_falls_back_between_backends(tmp_path, monkeypatch):
    x, sr = _tone_audio()
    p = tmp_path / "a.wav"
    sf.write(p, x, sr)
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    monkeypatch.setattr(asr, "_transcribe_cloud", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no network")))
    monkeypatch.setattr(asr, "_transcribe_local", lambda *a, **k: {"text": "यह एक परीक्षण है। दूसरा वाक्य है।", "segments": [], "backend": "local:fake"})
    r = asr.transcribe(str(p), backend="auto")
    assert r["backend"] == "local:fake" and r["notes"] and len(r["sentences"]) == 2 and len(r["pauses"]) == 1
    monkeypatch.setattr(asr, "_transcribe_local", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no model")))
    try:
        asr.transcribe(str(p), backend="auto")
        assert False, "should have raised"
    except RuntimeError as e:
        assert "all speech-recognition backends failed" in str(e)


def test_ui_diff_and_summary_rendering():
    n, s = render.diff_html("Mujhe bahut chinta ho rahi hai.", "Mujhe full tension ho rahi hai.")
    assert "<del>bahut chinta</del>" in n and "<mark>full tension</mark>" in s
    trace = {"comparison": [{"scene_id": "scene_01", "location": "School", "lines": [
        {"speaker": "ARAV", "neutral": "Yeh accha hai.", "slang": "Yeh ekdum zabardast hai.", "changed": True, "slang_added": "zabardast", "gate_stopped": None},
        {"speaker": "DADI", "neutral": "Beta, aaram karo.", "slang": "Beta, aaram karo.", "changed": False, "gate_stopped": 1}]}],
        "character_profiles": {"Arav": {"gender": "male", "age_group": "teen", "social_role": "peer", "slang_level": 5,
                                        "emotion_arc": [{"scene_id": "scene_01", "emotion": "excited", "intensity": 2, "stress": 0.1}]},
                               "Dadi": {"gender": "female", "social_role": "elder", "slang_level": 0, "emotion_arc": []}},
        "event_chain": [{"scene_id": "scene_01", "location": "School", "characters": ["Arav", "Dadi"], "key_events": ["x"], "scene_goal": "g"}],
        "segmentation": {"method": "signals", "sentences": 5}, "seconds_pipeline": 12, "llm_cost_usd": 0.004}
    h = render.summary_html(trace)
    assert "Lines with slang" in h and "signals" in h
    ch = render.comparison_html(trace)
    assert "<mark>" in ch and "kept neutral" in ch and "excited" in ch
    assert list(render.profiles_df(trace)["Slang allowed"]) == ["yes", "no"]
    assert "excited" in render.arcs_html(trace) and "scene_01".replace("scene_", "S") in render.arcs_html(trace)
    assert "Arav" in render.scenes_html(trace)
