"""D1: the Director sees what the editor can swap and cannot nominate a line with nothing swappable.  W1: the writer is asked to voice each PEER's analysed feeling.  W3: feelings go
between peers, and an elder says tum / tu to a child."""
import json

import pytest

import config
from agents import edit_gate as eg
from agents import slang_director as sd
from agents.agent3_synthesizer import EMOTION_WORDS, feelings_block
from tests.test_fidelity import EVENTS, NAMES, _agent3


@pytest.fixture(autouse=True)
def _setting():
    eg.set_setting(None)
    yield
    eg.set_setting(None)


def test_d1_every_line_carries_the_editor_note_and_the_prompt_explains_it():
    ctx = sd.build_context("s1", [("Priya", "Dekhna, Nani yeh sab dekhkar bahut khush hongi."), ("Vikram", "Chinta mat kar, hum sab theek kar lenge."),
                                  ("Ananya", "Nani, Janmdin mubarak ho!")], {}, None, [], 12)
    assert "khush hongi.  [swappable: khush=joy]" in ctx and ctx.count("[nothing the editor can swap]") == 2
    assert "[nothing the editor can swap]" in sd.SYSTEM and "FILLER-ONLY" in sd.SYSTEM and sd.PROMPT_VERSION == "v6"


def test_d1_a_slang_nomination_for_a_line_with_nothing_swappable_is_demoted():
    dialogue = "Priya: Dekhna, Nani yeh sab dekhkar bahut khush hongi.\nVikram: Chinta mat kar, hum sab theek kar lenge.\nAnanya: Chalo, jaldi shuru karte hain!"
    answer = json.dumps({"scene_read": "x", "lines": [
        {"n": 1, "why": "w", "slang": True, "register": "hype", "strength": 2, "filler": False},
        {"n": 2, "why": "w", "slang": True, "register": "stress", "strength": 2, "filler": False},          # nothing swappable: cannot be carried out
        {"n": 3, "why": "w", "slang": False, "register": "none", "strength": 0, "filler": True}]})
    d = sd.SlangDirector(scratchpad=type("P", (), {"read": lambda self, k: {}})(), cache_path="", caller=lambda m: answer)
    plan = d.plan_scene("s1", dialogue)
    assert plan["lines"][0]["slang"] is True
    assert plan["lines"][1]["slang"] is False and "nothing the editor can swap" in plan["lines"][1]["why"] and d.stats["demoted"] == 1


def _profiles(emotion="scared", role="peer"):
    arc = [{"scene_id": "scene_01", "emotion": emotion, "intensity": 2}]
    return {"Ananya": {"social_role": role, "emotion_arc": arc}, "Nani": {"social_role": "elder", "emotion_arc": [{"scene_id": "scene_01", "emotion": "happy", "intensity": 2}]}}


def test_w1_feelings_are_voiced_for_peers_only_and_only_the_listed_ones():
    b = feelings_block(["Ananya", "Nani"], _profiles(), "scene_01")
    assert "Ananya feels scared" in b and "darr lag raha hai" in b and "Nani" not in b.split("between peers")[0].replace("elder", "")
    assert feelings_block(["Ananya"], _profiles("determined"), "scene_01") == ""                    # no plain word for it: nothing is asked
    assert feelings_block(["Ananya"], _profiles("scared", "elder"), "scene_01") == "" and feelings_block([], {}, "x") == ""
    assert set(EMOTION_WORDS) <= {"happy", "excited", "proud", "amused", "relieved", "affectionate", "scared", "anxious", "angry", "frustrated", "sad", "ashamed"}


def test_w1_w3_the_prompt_has_the_feelings_block_and_the_elder_rule_only_in_the_padded_style(monkeypatch):
    ev = dict(EVENTS[0], scene_id="scene_01", scene_cast={"present": ["Ananya", "Nani"], "mentioned_only": [], "extra_speakers": []})
    out = {}
    for style in ("slotrich_pad", "slotrich"):
        monkeypatch.setattr(config, "AGENT3_STYLE", style)
        a = _agent3()
        a.generate_pass1({"scene_id": "scene_01", "text": "x", "deva_text": "अनन्या और नानी।"}, _profiles(), ev)
        out[style] = a.groq_client.calls[0]["messages"][0]["content"]
    assert "FEELINGS TO VOICE" in out["slotrich_pad"] and "Ananya feels scared" in out["slotrich_pad"]
    assert "FEELINGS TO VOICE" not in out["slotrich"]
    for p in out.values():
        assert "never 'aap'" in p                                       # the elder rule applies in every style (a register fact, not a style)
    assert "not in a line to an elder" in out["slotrich_pad"]           # W3 bullet of the padded rules
