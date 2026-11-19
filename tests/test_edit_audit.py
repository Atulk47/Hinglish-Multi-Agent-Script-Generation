"""Offline tests for the edit auditor and the validator's repair step (no model: embeddings are stubbed)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest

from evaluation.edit_audit import audit_edit, audit_scene, repair_scene


@pytest.mark.parametrize("neutral,edited,slang,code", [
    ("Kabir: Kya? Papa, aap thik ho? Mujhe bahut ghabrahat ho rahi hai.", "Kabir: Kya? Papa, aap thik ho? Mujhe jaan nikal gayi.", "jaan nikal gayi", "respect-context"),
    ("Kabir: Yeh bahut bura hua. Humara khiladi ghayal ho gaya.", "Kabir: Yeh ekdum wahiyat hua. Humara khiladi ghayal ho gaya.", "wahiyat", "serious-context"),
    ("Kabir: Yeh bahut accha idea hai.", "Kabir: Yeh bahut bekaar idea hai.", "bekaar", "polarity-flip"),
    ("Priya: Yeh kaam nahi hoga.", "Priya: Yeh kaam hoga.", None, "negation-changed"),
    ("Rohan: Mujhe 5 baje tak project dena hai.", "Rohan: Mujhe 7 baje tak project dena hai yaar.", "yaar", "entity-changed"),
    ("Mira: Kya tum aaoge?", "Mira: Tum aaoge.", None, "question-changed"),
    ("Mira: Main scorecard bahut dhyan se rakhungi, par mujhe thodi chinta hai.", "Mira: Main scorecard bahut dhyan se rakhungi, par mujhe thodi dimaag fry hai.", "dimaag fry", "rule-gate"),
    ("Aryan: Chalo, kaam karte hain.", "Aryan: Chalo, kaam karte hain, aur ek aur baat yeh hai ki kal match hai.", None, "non-local"),
])
def test_unsafe_edits_are_blocked_with_the_right_reason(neutral, edited, slang, code):
    a = audit_edit(neutral, edited, slang)
    assert not a["ok"] and code in [f["code"] for f in a["flags"] if f["blocking"]], a["flags"]


@pytest.mark.parametrize("neutral,edited,slang", [
    ("Sneha: Haan, lekin woh sach mein kaafi accha hai.", "Sneha: Haan, lekin woh sach mein ekdum ek number hai.", "ek number"),
    ("Aryan: Chalo, abhi baith ke saath mein kaam karte hain.", "Aryan: Chalo yaar, abhi baith ke saath mein kaam karte hain.", "yaar"),
    ("Kabir: Yeh bahut bura hua, mujhe bahut kharab lag raha hai.", "Kabir: Yeh ekdum wahiyat hua, mujhe bahut kharab lag raha hai.", "wahiyat"),
])
def test_good_edits_pass(neutral, edited, slang):
    a = audit_edit(neutral, edited, slang)
    assert a["ok"], a["flags"]


def test_unchanged_line_is_fine_and_elder_speaker_is_blocked():
    assert audit_edit("Nani: Beta, khana kha lo.", "Nani: Beta, khana kha lo.")["changed"] is False
    a = audit_edit("Beta, yeh accha hai.", "Beta, yeh zabardast hai.", "zabardast", role="elder")
    assert not a["ok"] and "elder-speaker" in [f["code"] for f in a["flags"]]


def test_scene_audit_and_repair_revert_only_the_unsafe_lines():
    neutral = {"scene_id": "s1", "dialogue": "Arav: Yeh bahut accha idea hai.\nPriya: Papa ki tabiyat kharab hai, bahut chinta ho rahi hai.\nArav: Chalo, kaam karte hain."}
    slang = {"scene_id": "s1", "dialogue": "Arav: Yeh zabardast idea hai.\nPriya: Papa ki tabiyat kharab hai, full tension ho rahi hai.\nArav: Chalo yaar, kaam karte hain.",
             "decisions": [{"slang": "zabardast", "applied": True}, {"slang": "full tension", "applied": True}, {"slang": "yaar", "applied": True}]}
    audits = audit_scene(neutral, slang, {})
    assert [a["ok"] for a in audits] == [True, False, True]
    fixed, reverted = repair_scene(neutral, slang, audits)
    assert len(reverted) == 1
    lines = fixed["dialogue"].split("\n")
    assert lines[0] == "Arav: Yeh zabardast idea hai." and lines[2] == "Arav: Chalo yaar, kaam karte hain."      # good edits stay
    assert lines[1] == "Priya: Papa ki tabiyat kharab hai, bahut chinta ho rahi hai."                             # the unsafe one is neutral again
    assert fixed["decisions"][1]["applied"] is False and "serious-context" in fixed["decisions"][1]["reverted_by_validator"]
    assert fixed["decisions"][0]["applied"] is True


def test_validator_validate_and_repair_diagnoses_the_repaired_scene():
    from evaluation.muril_validator import MuRILValidator
    v = MuRILValidator.__new__(MuRILValidator)
    v.__dict__.update(dict(sim_threshold=0.5, event_threshold=0.5, line_sim_threshold=0.5, drift_threshold=0.5, min_slang_coverage=0.15,
                           max_repetition_ratio=0.6, event_sim_min=0.0, fluency_delta_max=None, fluency=None, _center=None))
    v.compute_similarity = lambda a, b: 1.0 if a == b else 0.9
    neutral = {"scene_id": "s1", "dialogue": "Priya: Papa ki tabiyat kharab hai, bahut chinta ho rahi hai.", "events": {"key_events": []}}
    slang = {"scene_id": "s1", "dialogue": "Priya: Papa ki tabiyat kharab hai, full tension ho rahi hai.", "events": {},
             "decisions": [{"slang": "full tension", "applied": True, "go": True}]}
    fixed, diag = v.validate_and_repair(neutral, slang, {})
    assert fixed["dialogue"] == neutral["dialogue"]
    assert diag["reverted_edits"][0]["why"][0].startswith(("serious-context", "respect-context", "rule-gate"))
    assert diag["failure_mode"] == "NO_SLANG" and diag["slang_lines"] == 0            # nothing unsafe is left; the scene is neutral by design


@pytest.mark.parametrize("neutral,edited,slang,ok", [
    ("Rohan: Mujhe bahut dar lag raha tha.", "Rohan: Mujhe full tension ho rahi tha.", "full tension", False),      # idiom is feminine, the auxiliary 'tha' is not
    ("Rohan: Mujhe bahut dar lag raha tha.", "Rohan: Mujhe full tension ho rahi thi.", "full tension", True),
    ("Rohan: Mujhe pata tha ki yeh problem hogi.", "Rohan: Mujhe pata tha ki yeh jhamela hogi.", "jhamela", False),   # masculine noun, feminine future
    ("Rohan: Yeh film boring thi.", "Rohan: Yeh film ekdum flop thi.", "flop", True),                                # predicate noun agrees with the film
])
def test_gender_agreement_covers_the_whole_verb_chain(neutral, edited, slang, ok):
    assert audit_edit(neutral, edited, slang)["ok"] is ok


@pytest.mark.parametrize("neutral,edited,slang,ok", [
    # an elder / teacher only MENTIONED is not addressed: harmless edits stay
    ("Maine Dadaji ko yaad kiya. Mujhe bahut accha laga.", "Maine Dadaji ko yaad kiya. Mujhe ekdum zabardast laga.", "zabardast", True),
    ("Lekin Aryan Sir ki team mein nahin hai. Yeh bahut accha lag raha hai.", "Lekin Aryan Sir ki team mein nahin hai. Yeh ekdum kamaal lag raha hai.", "kamaal", True),
    # ... but one who is ADDRESSED (vocative) or 'aap' is protected
    ("Dadaji, Rohan ko chot lag gayi hai. Mujhe bahut chinta ho rahi hai.", "Dadaji, Rohan ko chot lag gayi hai. Mujhe full tension ho rahi hai.", "full tension", False),
    # an injury mentioned in passing ("despite the injury") is not a health line
    ("Rohan, tum chot ke bawajood yorker dal diya! Sach mein bahut accha laga.", "Rohan yaar, tum chot ke bawajood yorker dal diya! Sach mein bahut accha laga.", "yaar", True),
    # a crude word is never the speaker's FEELING
    ("Mujhe bahut bura lag raha hai ki late ho gaya.", "Mujhe ekdum jhand lag raha hai ki late ho gaya.", "jhand", False),
    ("Yeh film bahut bura lag raha hai.", "Yeh film ekdum jhand lag rahi hai.", "jhand", True),
])
def test_addressed_vs_mentioned_elders_concessive_health_and_crude_feelings(neutral, edited, slang, ok):
    assert audit_edit(neutral, edited, slang)["ok"] is ok, audit_edit(neutral, edited, slang)["flags"]


def test_optional_judge_second_opinion_reverts_rejected_edits_and_fails_open(monkeypatch):
    from evaluation import judge
    from evaluation.muril_validator import MuRILValidator
    import config
    v = MuRILValidator.__new__(MuRILValidator)
    v.__dict__.update(dict(sim_threshold=0.5, event_threshold=0.5, line_sim_threshold=0.5, drift_threshold=0.5, min_slang_coverage=0.15,
                           max_repetition_ratio=0.6, event_sim_min=0.0, fluency_delta_max=None, fluency=None, _center=None))
    v.compute_similarity = lambda a, b: 1.0 if a == b else 0.9
    neutral = {"scene_id": "s1", "dialogue": "Arav: Yeh bahut accha idea hai.\nRohan: Chalo, kaam karte hain.", "events": {"key_events": []}}
    slang = {"scene_id": "s1", "dialogue": "Arav: Yeh zabardast idea hai.\nRohan: Chalo yaar, kaam karte hain.", "events": {},
             "decisions": [{"slang": "zabardast", "applied": True, "go": True}, {"slang": "yaar", "applied": True, "go": True}]}
    monkeypatch.setattr(config, "VALIDATOR_JUDGE", True, raising=False)
    ok = {"meaning_preserved": True, "grammar_ok": True, "natural": 5, "fits_situation": True, "problem": "none"}
    no = {**ok, "natural": 1, "problem": "forced"}

    def fake(pairs, models=None, workers=8, contexts=None):
        return [{"accept": "zabardast" not in b, "models": {"m1": no if "zabardast" in b else ok}, "problems": ["forced"] if "zabardast" in b else []} for a, b in pairs]
    monkeypatch.setattr(judge, "judge_pairs", fake)
    fixed, diag = v.validate_and_repair(neutral, slang, {})
    assert fixed["dialogue"].split("\n") == ["Arav: Yeh bahut accha idea hai.", "Rohan: Chalo yaar, kaam karte hain."]
    assert any("judge-reject" in r["why"][0] for r in diag["reverted_edits"])
    # the judge is unreachable (no verdict at all): nothing is reverted
    monkeypatch.setattr(judge, "judge_pairs", lambda pairs, **k: [{"accept": False, "models": {"m1": None}, "problems": []} for _ in pairs])
    fixed, diag = v.validate_and_repair(neutral, slang, {})
    assert fixed["dialogue"] == slang["dialogue"]


# ---- an EMPTY "Narrator:" turn in the neutral script (Agent 4 skips it) must not shift every later pair
EMPTY_TURN_NEUTRAL = {"scene_id": "s1", "dialogue": "Arav: Yeh dekho, diary hai.\nNarrator:\nArav: Mujhe bahut accha laga.\nRohan: Chalo kaam karte hain.", "events": {"key_events": []}}
EMPTY_TURN_SLANG = {"scene_id": "s1", "dialogue": "Arav: Yeh dekho, diary hai.\nArav: Mujhe ekdum bindaas laga.\nRohan: Chalo kaam karte hain.",
                    "decisions": [{"speaker": "Arav", "applied": False}, {"speaker": "Arav", "applied": True, "slang": "bindaas"}, {"speaker": "Rohan", "applied": False}]}


def test_empty_neutral_turn_does_not_shift_the_pairs():
    from evaluation import edit_audit as ea
    from evaluation.compare import build_comparison
    audits = ea.audit_scene(EMPTY_TURN_NEUTRAL, EMPTY_TURN_SLANG, {})
    assert [a["changed"] for a in audits] == [False, True, False]
    assert audits[1]["neutral"] == "Mujhe bahut accha laga." and audits[1]["edited"] == "Mujhe ekdum bindaas laga."
    lines = build_comparison([EMPTY_TURN_NEUTRAL], [EMPTY_TURN_SLANG])[0]["lines"]
    assert [l["changed"] for l in lines] == [False, True, False] and lines[1]["neutral"] == "Mujhe bahut accha laga." and lines[1]["slang_added"] == "bindaas"


def test_repair_reverts_the_right_line_when_the_neutral_has_an_empty_turn():
    from evaluation import edit_audit as ea
    bad = dict(EMPTY_TURN_SLANG, dialogue="Arav: Yeh dekho, diary hai.\nArav: Mujhe bahut accha nahi laga.\nRohan: Chalo kaam karte hain.")      # negation flip = blocking
    audits = ea.audit_scene(EMPTY_TURN_NEUTRAL, bad, {})
    assert not audits[1]["ok"]
    fixed, reverted = ea.repair_scene(EMPTY_TURN_NEUTRAL, bad, audits)
    assert fixed["dialogue"].split("\n") == ["Arav: Yeh dekho, diary hai.", "Arav: Mujhe bahut accha laga.", "Rohan: Chalo kaam karte hain."]     # NOT 'Narrator:' / the next line's text
    assert len(reverted) == 1


def test_ablation_pairs_skip_empty_turns():
    from evaluation.aggregate_ablation import _pairs
    lines, changed = _pairs({"neutral_script": [EMPTY_TURN_NEUTRAL], "slang_script": [EMPTY_TURN_SLANG]})
    assert lines == 3 and changed == [("Mujhe bahut accha laga.", "Mujhe ekdum bindaas laga.")]


def test_comma_may_only_be_added_directly_after_an_inserted_filler():
    assert not audit_edit("Accha hai, lekin mujhe darr hai.", "Arre accha, hai, lekin mujhe darr hai.", "arre")["ok"]            # a comma inside "accha hai"
    assert audit_edit("Haan, mujhe bhi thoda kam laga.", "Arre haan, mujhe bhi thoda kam laga.", "arre")["ok"]
    assert audit_edit("Rohan, tum kaise ho?", "Rohan yaar, tum kaise ho?", "yaar")["ok"]
    assert audit_edit("Theek hai, chalo.", "Arre, chalo.", "arre")["ok"]                                                          # "Arre," is a normal opener
