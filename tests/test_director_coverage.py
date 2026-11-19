"""Coverage without loosening the rules: the Director no longer invents sub-quotas, and may nominate a FILLER-ONLY line (casual peer, no feeling word)
that goes through the editor's existing, gated filler route and nothing else."""
import json
from types import SimpleNamespace

from agents import slang_director as sd
from agents.agent4_slang_rewriter import SlangRewriter


def ans(rows):
    return json.dumps({"scene_read": "x", "lines": [dict({"n": i + 1, "why": "w", "slang": False, "filler": False, "register": "none", "strength": 0}, **r)
                                                   for i, r in enumerate(rows)]})


def test_prompt_no_longer_asks_for_spreading_and_states_the_only_limit():
    assert "Spread nominations" not in sd.SYSTEM
    assert "ONLY limit" in sd.SYSTEM and "FILLER-ONLY" in sd.SYSTEM and sd.PROMPT_VERSION == "v6"
    ctx = sd.build_context("s1", [("A", "x")] * 4, {}, None, [], 12)
    assert "cap: 2 slang line(s)" in ctx


def test_filler_only_line_is_kept_but_limited_to_one_in_four_and_the_cap():
    rows = [{"filler": True}] * 8
    plan = sd.parse_plan(ans(rows), 8, 0.5, filler_share=0.25)
    assert sum(v["filler"] for v in plan["lines"].values()) == 2          # 25% of 8
    assert sum(v["filler"] for v in sd.parse_plan(ans(rows), 8, 0.5)["lines"].values()) == 1      # the shipped default share is 15%: 1 of 8
    assert all(not v["slang"] for v in plan["lines"].values())
    # slang lines use the cap first: cap 2 of 4, both slang -> no room for a filler-only line
    rows = [{"slang": True, "register": "hype", "strength": 2}, {"slang": True, "register": "hype", "strength": 2}, {"filler": True}, {"filler": True}]
    plan = sd.parse_plan(ans(rows), 4, 0.5)
    assert sum(v["slang"] for v in plan["lines"].values()) == 2 and sum(v["filler"] for v in plan["lines"].values()) == 0


def test_filler_flag_is_ignored_on_a_slang_line_and_for_non_true_values():
    plan = sd.parse_plan(ans([{"slang": True, "register": "hype", "strength": 1, "filler": True}, {"filler": "yes"}, {"filler": 1}]), 3, 1.0)
    assert plan["lines"][0]["filler"] is False and plan["lines"][1]["filler"] is False and plan["lines"][2]["filler"] is False


class _Gate:
    _permissive = False

    def gate1_speaker_eligible(self, s):
        return True


def _rw():
    rw = SlangRewriter.__new__(SlangRewriter)
    rw.gate = _Gate()
    return rw


def test_decide_turns_a_filler_only_nomination_into_a_filler_only_go():
    rw = _rw()
    plan = {"slang": False, "filler": True, "register": "none", "strength": 0, "why": "friendly reply"}
    rw._agentic_line = True
    d = rw._decide("RAJU", "Mohan, chalo chalte hain!", "s", plan)
    assert d["go"] and d["filler_only"] and d["intent"] == "CASUAL_ASSERTION"
    rw._agentic_line = False                                   # never outside the Director's ownership
    d = rw._decide("RAJU", "x", "s", plan)
    assert not d["go"]


def test_a_filler_only_line_only_ever_tries_the_filler_route_and_never_a_mood_swap():
    rw = SlangRewriter.__new__(SlangRewriter)
    rw._filler_only = True
    seen = []
    rw._has_filler = lambda c: False
    rw._clause_polarity = lambda c: 1
    rw._slot_shortlist = lambda c, extra_only=False: (["ekdum kamaal"], 1)           # a mood shortlist exists, but must not be used
    rw._filler_allowed = lambda: True
    rw._filler_shortlist = lambda c, pol: seen.append("filler") or ["yaar"]
    rw._qwen_edit = lambda c, sl: seen.append(tuple(sl)) or "Mohan yaar, chalo chalte hain!"
    rw._accept = lambda core, cand, sl, hint: (True, "yaar", "ok")
    rw._split_vocative = lambda c: ("", c, "")
    import config
    cand, used, eng, notes, rej, route = rw._edit_sentence("Mohan, chalo chalte hain!", "CASUAL_ASSERTION", "neutral")
    assert route == "filler" and used == "yaar" and seen == ["filler", ("yaar",)]


def test_a_filler_only_line_with_no_natural_filler_stays_neutral():
    rw = SlangRewriter.__new__(SlangRewriter)
    rw._filler_only = True
    rw._has_filler = lambda c: False
    rw._clause_polarity = lambda c: 0
    rw._slot_shortlist = lambda c, extra_only=False: ([], 0)
    rw._filler_allowed = lambda: True
    rw._filler_shortlist = lambda c, pol: []                    # e.g. a line to an elder, or no natural slot for a filler
    rw._split_vocative = lambda c: ("", c, "")
    cand, used, eng, notes, rej, route = rw._edit_sentence("Station kab aayega.", "CASUAL_ASSERTION", "neutral")
    assert cand is None and "no-natural-filler" in notes


def test_demo_target_and_per_scene_ceiling(monkeypatch):
    import config
    from agents import edit_gate
    edit_gate.set_setting(None)
    ctx = sd.build_context("s1", [("A", "x")] * 8, {}, None, [], 12)
    assert "cap: 3 slang line(s)" in ctx and "aim for 2 slang line(s)" in ctx            # 8 lines would allow 4 by share, the ceiling is 3
    short = sd.build_context("s1", [("A", "x")] * 3, {}, None, [], 12)
    assert "TARGET" not in short                                                       # a 3-line scene gets no target
    edit_gate.set_setting({"era": "historical", "milieu": "rural", "formality": "traditional"})
    assert "TARGET" not in sd.build_context("s1", [("A", "x")] * 8, {}, None, [], 12)   # never for a traditional story
    edit_gate.set_setting(None)
    plan = sd.parse_plan(ans([{"slang": True, "register": "hype", "strength": 3 - (i % 3)} for i in range(8)]), 8, 0.5)
    assert sum(v["slang"] for v in plan["lines"].values()) == 3                         # the ceiling is enforced on the plan itself
    monkeypatch.setattr(config, "SLANG_DIRECTOR_TARGET", 0)
    assert "TARGET" not in sd.build_context("s1", [("A", "x")] * 8, {}, None, [], 12)   # 0 switches the target off
