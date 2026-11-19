"""An audit-clean idiom swap must not trigger a MEANING_DRIFT re-do just because the centred cosine is lowish (found on a real voice-input run)."""
from evaluation.muril_validator import MuRILValidator

NEUTRAL = {"scene_id": "scene_04", "events": {"key_events": []},
           "dialogue": "Priya: Main bhi aa gayi, Ananya! Chalo, ghar bahut accha lagega.\n"
                       "Ananya: Priya, tum aa gayi, mujhe bahut accha laga. Yeh decoration dekhkar mujhe bahut khushi ho rahi hai."}
SLANG = {"scene_id": "scene_04", "events": {"key_events": []},
         "dialogue": "Priya: Main bhi aa gayi, Ananya! Chalo, ghar bahut accha lagega.\n"
                     "Ananya: Priya, tum aa gayi, mujhe bahut accha laga. Yeh decoration dekhkar mere mann mein laddu phoot rahe hain.",
         "decisions": [{"speaker": "Priya", "applied": False, "go": False},
                       {"speaker": "Ananya", "applied": True, "go": True, "slang": "mann mein laddu phoot rahe hain", "intent": "ASSERT_POSITIVE"}]}


def _validator(sim):
    v = object.__new__(MuRILValidator)
    v.compute_similarity = lambda a, b: sim
    v._event_coverage = lambda d, e: {"recall": 1.0, "covered": 0, "total": 0, "method": "none"}
    v.line_sim_threshold = v.drift_threshold = 0.6655
    v._center = True
    v.fluency, v.fluency_delta_max = None, None
    v.min_slang_coverage, v.max_repetition_ratio, v.event_threshold = 0.0, 1.0, 0.0
    return v


def test_lowish_cosine_on_an_audit_clean_idiom_is_only_a_note():
    d = _validator(0.531).diagnose_scene(NEUTRAL, SLANG, {})
    assert d["failure_mode"] == "OK" and d["passed"]
    assert "diagnostic" in d["hint"]


def test_a_very_low_cosine_still_triggers_the_re_do():
    d = _validator(0.20).diagnose_scene(NEUTRAL, SLANG, {})
    assert d["failure_mode"] == "MEANING_DRIFT"


def _drifting(sim_bad):
    v = _validator(1.0)
    v.compute_similarity = lambda a, b: sim_bad if "laddu" in (b or "") else 1.0         # only the idiom line is far from its neutral text
    v.fluency = None
    return v


def test_a_drifted_line_is_reverted_not_re_run():
    fixed, diag = _drifting(0.20).validate_and_repair(NEUTRAL, SLANG, {})
    assert "laddu" not in fixed["dialogue"]                                            # that line is back to its neutral text
    assert "mujhe bahut khushi ho rahi hai" in fixed["dialogue"]
    # its only edit is gone, so the scene is neutral: the pipeline accepts that ("kept neutral by design - not re-done") and never re-runs the editor
    assert diag["failure_mode"] == "NO_SLANG"
    assert any("meaning-drift" in " ".join(r["why"]) for r in diag["reverted_edits"])
    assert [d["applied"] for d in fixed["decisions"]] == [False, False]


def test_other_edits_in_the_scene_survive_the_drift_revert():
    two = dict(SLANG)
    two["dialogue"] = SLANG["dialogue"].replace("ghar bahut accha lagega", "ghar ekdum solid lagega")
    two["decisions"] = [{"speaker": "Priya", "applied": True, "go": True, "slang": "solid", "intent": "ASSERT_POSITIVE"}, SLANG["decisions"][1]]
    fixed, diag = _drifting(0.20).validate_and_repair(NEUTRAL, two, {})
    assert "ekdum solid" in fixed["dialogue"] and "laddu" not in fixed["dialogue"]


def test_a_moderately_lower_cosine_keeps_the_edit():
    fixed, diag = _drifting(0.531).validate_and_repair(NEUTRAL, SLANG, {})
    assert "laddu" in fixed["dialogue"] and diag["failure_mode"] == "OK" and not diag["reverted_edits"]
