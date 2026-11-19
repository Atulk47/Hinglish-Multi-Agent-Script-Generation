"""Slang Director: parsing, caps, fallback, caching, and how the rewriter uses a plan (Gate 1 must still veto)."""
import json
from types import SimpleNamespace

import pytest

import config
from agents import slang_director as sd
from agents.agent4_slang_rewriter import SlangRewriter

PROFILES = {
    "Rohan": {"name": "Rohan", "gender": "male", "age_group": "young_adult", "social_role": "peer", "speech_style": "casual",
              "traits": ["anxious"], "slang_level": 5, "current_emotion": "anxious",
              "emotion_arc": [{"scene_id": "scene_01", "emotion": "anxious", "intensity": 2}],
              "relationship": {"Aryan": {"relation": "roommates", "i_address_them": "Aryan", "register": "casual"},
                               "Professor Mehata": {"relation": "student / teacher", "i_address_them": "Sir", "register": "respectful"}}},
    "Aryan": {"name": "Aryan", "gender": "male", "age_group": "young_adult", "social_role": "peer", "speech_style": "casual",
              "traits": [], "slang_level": 5, "current_emotion": "neutral", "relationship": {}},
    "Professor Mehata": {"name": "Professor Mehata", "gender": "male", "age_group": "adult", "social_role": "authority",
                         "speech_style": "formal", "traits": ["strict"], "slang_level": 0, "current_emotion": "neutral", "relationship": {}},
}
DIALOGUE = ("ROHAN: Mujhe bahut chinta ho rahi hai.\n"
            "ARYAN: Chal, abhi kaam shuru karte hain.\n"
            "PROFESSOR MEHATA: Project kal subah nau baje tak jama karna hai.\n"
            "ROHAN: Ji Sir, main jama kar dunga.")


class FakePad:
    def read(self, key):
        return PROFILES if key == "character_profiles" else None


def answer(*rows):
    return json.dumps({"scene_read": "tense hostel night", "lines": [
        {"n": i + 1, "why": "x", "slang": r[0], "register": r[1], "strength": r[2]} for i, r in enumerate(rows)]})


def test_dialogue_items_matches_rewriter_walk():
    items = sd.dialogue_items("A: hi\nstage direction\nB:   \nC: yo")
    assert items == [("A", "hi"), ("C", "yo")]


def test_parse_plan_and_register_filter():
    plan = sd.parse_plan(answer((True, "stress", 2), (False, "none", 0), (True, "banter", 1), (True, "nonsense", 3)), 4, 1.0)
    assert plan["lines"][0] == {"slang": True, "register": "stress", "strength": 2, "why": "x", "filler": False}
    assert plan["lines"][3]["slang"] is False                 # unknown register is not a nomination
    assert plan["scene_read"]


def test_parse_plan_fenced_json_and_garbage():
    fenced = "```json\n" + answer((True, "hype", 3)) + "\n```"
    assert sd.parse_plan(fenced, 1, 1.0)["lines"][0]["slang"] is True
    assert sd.parse_plan("I think line one is fine", 1, 1.0) is None
    assert sd.parse_plan('{"lines": "nope"}', 1, 1.0) is None


def test_missing_lines_default_to_no_slang_and_cap_keeps_strongest():
    plan = sd.parse_plan(answer((True, "hype", 1), (True, "hype", 3), (True, "banter", 2), (True, "stress", 3)), 6, 0.5)  # cap = 3 of 6
    kept = [i for i, v in plan["lines"].items() if v["slang"]]
    assert len(kept) == 3 and 1 in kept and 3 in kept       # the two strength-3 lines survive
    assert plan["lines"][4]["slang"] is False and plan["lines"][5]["slang"] is False   # lines the model never listed


def test_context_carries_addressee_relevant_facts():
    items = sd.dialogue_items(DIALOGUE)
    ctx = sd.build_context("scene_01", items, PROFILES, {"scene_goal": "deadline tension", "key_events": ["Sir announces the deadline"]}, ["full tension"], 4)
    assert "respectful register" in ctx and "Professor Mehata" in ctx          # who Rohan must not joke with
    assert "anxious (intensity 2/3)" in ctx and "deadline tension" in ctx and "full tension" in ctx
    assert "1. ROHAN:" in ctx and "4. ROHAN:" in ctx


def test_plan_scene_caches_and_falls_back(tmp_path):
    calls = []
    def caller(messages):
        calls.append(messages)
        return answer((True, "stress", 2), (True, "banter", 1), (False, "none", 0), (False, "none", 0))
    d = sd.SlangDirector(FakePad(), cache_path=str(tmp_path / "c.json"), caller=caller)
    p1 = d.plan_scene("scene_01", DIALOGUE)
    p2 = d.plan_scene("scene_01", DIALOGUE)
    assert len(calls) == 1 and d.stats["cache_hits"] == 1 and p1["lines"] == p2["lines"]
    d2 = sd.SlangDirector(FakePad(), cache_path=str(tmp_path / "c.json"), caller=lambda m: 1 / 0)    # fresh process, same disk cache
    assert d2.plan_scene("scene_01", DIALOGUE)["lines"] == p1["lines"] and d2.stats["failures"] == 0

    broken = sd.SlangDirector(FakePad(), cache_path="", caller=lambda m: (_ for _ in ()).throw(RuntimeError("down")))
    assert broken.plan_scene("scene_01", DIALOGUE) is None and broken.stats["failures"] == 1
    junk = sd.SlangDirector(FakePad(), cache_path="", caller=lambda m: "not json")
    assert junk.plan_scene("scene_01", DIALOGUE) is None


class _Gate:
    def __init__(self, g1=True, permissive=False):
        self._permissive, self._g1, self.classic = permissive, g1, 0

    def gate1_speaker_eligible(self, speaker):
        return self._g1

    def run_all_gates(self, *a, **k):
        self.classic += 1
        return {"go": False, "gate_stopped": 2, "intent": "INFORM_NEUTRAL", "reason": "classic"}


def _rewriter(**gate_kw):
    rw = SlangRewriter.__new__(SlangRewriter)
    rw.gate = _Gate(**gate_kw)
    rw.intent_clf = SimpleNamespace(classify=lambda t: "INFORM_NEUTRAL")
    return rw


def test_decide_uses_plan_and_maps_register_to_intent():
    rw = _rewriter()
    d = rw._decide("ROHAN", "x", "scene_01", {"slang": True, "register": "stress", "strength": 2, "why": "worried"})
    assert d["go"] and d["intent"] == "EXPRESS_STRESS" and d["gate_stopped"] is None and rw.gate.classic == 0
    d = rw._decide("ROHAN", "x", "scene_01", {"slang": False, "register": "none", "strength": 0, "why": "talking to sir"})
    assert not d["go"] and d["gate_stopped"] == "director" and "talking to sir" in d["reason"]


def test_gate1_still_vetoes_a_nominated_line():
    rw = _rewriter(g1=False)
    d = rw._decide("PROFESSOR MEHATA", "x", "scene_01", {"slang": True, "register": "hype", "strength": 3, "why": "model was wrong"})
    assert not d["go"] and d["gate_stopped"] == 1


def test_no_plan_or_permissive_retry_uses_the_classic_gates():
    rw = _rewriter()
    assert rw._decide("ROHAN", "x", "s", None)["reason"] == "classic"
    rw = _rewriter(permissive=True)
    assert rw._decide("ROHAN", "x", "s", {"slang": True, "register": "hype", "strength": 1, "why": ""})["reason"] == "classic"


def test_attach_director_toggles_flag():
    rw = SlangRewriter.__new__(SlangRewriter)
    rw.director, rw.use_director = None, False
    rw.attach_director(object())
    assert rw.use_director is True
    rw.attach_director(None)
    assert rw.use_director is False


def test_agentic_mode_skips_gate1_and_filler_cap_but_classic_does_not():
    rw = _rewriter(g1=False)                                   # Gate 1 would veto this speaker
    plan = {"slang": True, "register": "banter", "strength": 2, "why": "friends"}
    rw._agentic_line = True                                    # the Director owns this line
    assert rw._decide("ROHAN", "x", "s", plan)["go"] is True and rw.gate.classic == 0
    rw._agentic_line = False                                   # classic / director-assisted mode
    assert rw._decide("ROHAN", "x", "s", plan)["gate_stopped"] == 1
    rw._filler_n, rw._scene_n = 99, 1
    assert rw._filler_allowed() is False
    rw._agentic_line = True
    assert rw._filler_allowed() is True


def test_attach_director_owns_follows_config():
    rw = SlangRewriter.__new__(SlangRewriter)
    rw.attach_director(object())
    assert rw.director_owns is config.SLANG_AGENTIC
    rw.attach_director(object(), owns=False)
    assert rw.director_owns is False


def test_config_defaults():
    assert config.SLANG_DIRECTOR is True and config.SLANG_AGENTIC is True      # the slang decision is agentic by default
    assert 0 < config.SLANG_DIRECTOR_MAX_SHARE <= 1
    assert "director" in config.OPENROUTER_MODELS


# ------------------------------------------------------------------------------------------------ review loop
def review_answer(*rows):
    return json.dumps({"verdicts": [{"n": n, "why": w, "keep": k} for n, w, k in rows]})


def test_parse_review_keeps_unlisted_and_ignores_unexpected():
    v = sd.parse_review(review_answer((1, "garbled", False), (9, "not an edited line", False)), {1, 3})
    assert v[1] == {"keep": False, "why": "garbled"}
    assert v[3]["keep"] is True and 9 not in v                      # unlisted edited line is KEPT; stray numbers are ignored
    assert sd.parse_review("no json here", {1}) is None


def test_review_scene_fails_open_and_caches(tmp_path):
    neutral = [("ROHAN", "Mujhe bahut chinta ho rahi hai."), ("ARYAN", "Chal kaam shuru karte hain.")]
    edited = [("ROHAN", "Mujhe full tension ho rahi hai."), ("ARYAN", "Chal kaam shuru karte hain.")]
    calls = []
    def caller(messages):
        calls.append(messages)
        return review_answer((1, "natural", True))
    d = sd.SlangDirector(FakePad(), cache_path=str(tmp_path / "r.json"), caller=caller)
    assert d.review_scene("scene_01", neutral, edited) == {0: {"keep": True, "why": "natural"}}
    d.review_scene("scene_01", neutral, edited)
    assert len(calls) == 1                                          # second call served from the cache
    assert "[EDITED]" in calls[0][1]["content"] and "neutral original: Mujhe bahut chinta" in calls[0][1]["content"]
    assert d.review_scene("scene_01", neutral, neutral) == {}       # nothing edited -> no model call, nothing to review
    down = sd.SlangDirector(FakePad(), cache_path="", caller=lambda m: (_ for _ in ()).throw(RuntimeError("down")))
    assert down.review_scene("scene_01", neutral, edited) is None   # unreachable reviewer -> None (caller keeps every edit)


def _review_fixture(review_verdicts):
    rw = SlangRewriter.__new__(SlangRewriter)
    rw.review_on = True
    rw.director = SimpleNamespace(review_scene=lambda sid, n, e, ev: review_verdicts)
    out_lines = ["ROHAN: Mujhe full tension ho rahi hai.", "ARYAN: Chal yaar kaam shuru karte hain.", "PRIYA: Theek hai."]
    decisions = [{"speaker": "ROHAN", "applied": True, "slang": "full tension"}, {"speaker": "ARYAN", "applied": True, "slang": "yaar"},
                 {"speaker": "PRIYA", "applied": False}]
    neutral = ["Mujhe bahut chinta ho rahi hai.", "Chal kaam shuru karte hain.", "Theek hai."]
    return rw, out_lines, decisions, [0, 1, 2], neutral


def test_apply_review_reverts_only_rejected_edits():
    rw, out, dec, slots, neutral = _review_fixture({0: {"keep": True, "why": "fine"}, 1: {"keep": False, "why": "filler interrupts"}})
    n = rw._apply_review("scene_01", out, dec, slots, neutral, None)
    assert n == 1
    assert out[0] == "ROHAN: Mujhe full tension ho rahi hai."       # kept
    assert out[1] == "ARYAN: Chal kaam shuru karte hain."            # reverted to the neutral line
    assert dec[1]["applied"] is False and dec[1]["slang"] is None and dec[1]["reverted_slang"] == "yaar"
    assert dec[1]["reason"].startswith("director review") and dec[0]["applied"] is True


def test_apply_review_is_off_by_default_and_fails_open():
    rw, out, dec, slots, neutral = _review_fixture({0: {"keep": False, "why": "x"}})
    rw.review_on = False
    assert rw._apply_review("s", out, dec, slots, neutral, None) == 0 and out[0].endswith("full tension ho rahi hai.")
    rw, out, dec, slots, neutral = _review_fixture(None)            # reviewer down -> None -> keep everything
    assert rw._apply_review("s", out, dec, slots, neutral, None) == 0 and dec[0]["applied"] is True


def test_review_config_defaults():
    assert "reviewer" in config.OPENROUTER_MODELS
    fam = lambda m: m.split("/")[0]
    judges = {"anthropic", "google"}                                 # the evaluation judges (Claude Haiku, Gemini Flash)
    assert fam(config.OPENROUTER_MODELS["reviewer"]) not in judges   # reviewer must not share a family with the scoring judges
