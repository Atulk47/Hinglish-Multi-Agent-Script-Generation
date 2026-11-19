"""The invariant checker must itself be right: it flags every known failure shape and stays silent on a good run."""
import pytest

from evaluation import invariants as inv

DEVA = ["हामिद और अमीना घर में थे। हामिद ने कहा कि वह जल्दी आएगा।", "अमीना ने सिर हिलाया।"]
ROMAN = ["hamid aur amina ghar men the. hamid ne kaha ki vah jaldi aega.", "amina ne sir hilaya."]


def result(dialogues, extras=None, location=None, loc_ev=None, narration_only=()):
    neutral = [{"scene_id": f"scene_0{i+1}", "dialogue": d, "fidelity": {"final_issues": [], "narration_only": (i in narration_only)}} for i, d in enumerate(dialogues)]
    chain = [{"scene_id": f"scene_0{i+1}", "characters": ["Hamid", "Amina"], "location": (location or "Unknown") if i == 0 else "Unknown",
              "location_evidence": (loc_ev or "") if i == 0 else ""} for i in range(len(dialogues))]
    persona = {"characters": {"Hamid": {}, "Amina": {}}, "scene_cast": {f"scene_0{i+1}": {"extra_speakers": extras or []} for i in range(len(dialogues))}, "secrets": []}
    return {"event_chain": chain, "persona": persona, "neutral": neutral}


def codes(violations, level="error"):
    return {v["code"] for v in violations if v["level"] == level}


GOOD = ["Hamid: Amma, main jaldi aa jaunga.\nAmina: Theek hai beta.", ""]


def test_a_good_run_has_no_error():
    V = inv.check_structure(" ".join(DEVA), DEVA, result(GOOD)["event_chain"], result(GOOD)["neutral"]) + inv.check_pass1(DEVA, ROMAN, result(GOOD, narration_only=(1,)))
    assert codes(V) == set(), V


@pytest.mark.parametrize("dialogue, expected", [
    ("Narrator: Sab chup the.", "meta-speaker"),
    ("Wait, that violates the rule: narrator cannot speak.\nHamid: Haan.", "meta-speaker"),
    ("Hamid: हाँ अम्मा।", "devanagari-in-output"),
    ("Hamid: Dekho sajī hui hain.", "non-ascii-artefact"),
    ("Hamid: Haan.\nBas yun hi ek line", "unlabelled-line"),
    ("Hamid: Ji Amma, main khushi se haan karta hoon.", "gesture-as-speech"),
    ("Hamid: Yeh kamaal hai, yaar.", "slang-leaked-into-neutral"),
    ("Hamid: Haan.\nHamid: Haan.", "duplicate-line"),
    ("Hamid:", "empty-line"),
    ("Asif: Main bhi aa gaya.", "speaker-not-grounded"),
    ("Hamid: The sole speaker of this scene is the listener of the story, which is not a character.", "english-prose-line"),
    ("Hamid: Haan * *", "artefact-in-dialogue"),
])
def test_each_known_failure_shape_is_flagged(dialogue, expected):
    V = inv.check_pass1(DEVA, ROMAN, result([dialogue, "Amina: Theek hai."]))
    assert expected in codes(V), (expected, V)


def test_spoken_words_that_vanish_are_an_error_but_narration_only_scenes_are_fine():
    assert "spoken-words-lost" in codes(inv.check_pass1(DEVA, ROMAN, result(["", ""])))
    assert "spoken-words-lost" not in codes(inv.check_pass1(DEVA, ROMAN, result(["", ""], narration_only=(0, 1))))


def test_a_location_must_be_quoted_from_the_source():
    ok = inv.check_pass1(DEVA, ROMAN, result(GOOD, location="home", loc_ev="hamid aur amina ghar men the"))
    bad = inv.check_pass1(DEVA, ROMAN, result(GOOD, location="the market", loc_ev="a busy bazaar of the old city"))
    assert "location-not-grounded" not in codes(ok) and "location-not-grounded" in codes(bad)


def test_a_role_speaker_translated_by_the_writer_is_not_flagged_but_an_invented_name_is():
    deva = ["दुकानदार ने कहा कि छः पैसे लगेंगे। हामिद चुप रहा।"]
    roman = ["dukandar ne kaha ki chhah paise lagenge. hamid chup raha."]
    res = result(["Dukaandaar: Chhah paise.\nHamid: Theek hai."], extras=["Dukaandaar"])
    assert "speaker-not-grounded" not in codes(inv.check_pass1(deva, roman, res))
    assert "speaker-not-grounded" in codes(inv.check_pass1(deva, roman, result(["Zoravar: Chhah paise."])))


def test_interjections_are_not_reported_as_invented_names_but_real_ones_are_warned():
    V = inv.check_pass1(DEVA, ROMAN, result(["Hamid: Haan, main aa raha hoon.\nAmina: Beta, dhyan se.", "Hamid: Zoravar, suno."]))
    assert not any(v["code"] == "name-not-in-source" and v["detail"] in ("Haan", "Beta") for v in V)
    assert any(v["code"] == "name-not-in-source" and v["detail"] == "Zoravar" for v in V)


def test_symbols_are_not_artefacts_but_stray_letters_are():
    assert "non-ascii-artefact" not in codes(inv.check_pass1(DEVA, ROMAN, result(["Hamid: Yeh ₹500 ka hai."])))
    assert "non-ascii-artefact" in codes(inv.check_pass1(DEVA, ROMAN, result(["Hamid: Yeh sajī hai."])))


def test_structure_checks():
    assert "too-many-scenes" in codes(inv.check_structure("a", ["a"] * 16))
    assert "text-not-preserved" in codes(inv.check_structure("पूरा पाठ यहाँ है", ["पूरा पाठ"]))
    assert "no-scenes" in codes(inv.check_structure("x", []))
    assert "event-chain-misaligned" in codes(inv.check_structure("a b", ["a", "b"], event_chain=[{}], neutral=[{}, {}]))
    assert codes(inv.check_structure("a b", ["a", "b"], event_chain=[{}, {}], neutral=[{}, {}])) == set()


NEUTRAL = [{"scene_id": "scene_01", "dialogue": "Riya: Main bahut pareshan thi, isliye gussa aa gaya.\nSiya: Chalo, shuru karte hain.\nHamid: Kal ka exam mujhe aasaani se milega."}]


def _slang(lines, decisions, setting=None, gender="female"):
    sc = [{"scene_id": "scene_01", "dialogue": "\n".join(lines), "decisions": decisions}]
    profiles = {"Riya": {"gender": gender}, "Siya": {"gender": "female"}, "Hamid": {"gender": "male"}}
    return inv.check_slang(NEUTRAL, sc, profiles, setting)


def test_slang_invariants_flag_the_known_slang_defects():
    base = NEUTRAL[0]["dialogue"].split("\n")
    d = [{"slang": "aag-babula"}, {}, {}]
    assert "gender-disagreement" in codes(_slang(["Riya: Main bahut pareshan thi, isliye aag-babula ho gaya.", base[1], base[2]], d))
    neutral = [{"scene_id": "scene_01", "dialogue": "Riya: Siya, mujhe samajh nahi aa raha."}]
    sc = [{"scene_id": "scene_01", "dialogue": "Riya: Siya bro, mujhe samajh nahi aa raha.", "decisions": [{"slang": "bro"}]}]
    assert "male-address-for-a-girl" in codes(inv.check_slang(neutral, sc, {"Riya": {"gender": "female"}, "Siya": {"gender": "female"}}, None))
    assert "predicate-idiom-without-copula" in codes(_slang([base[0], base[1], "Hamid: Kal ka exam mujhe bacchon ka khel milega."], [{}, {}, {"slang": "bacchon ka khel"}]))
    assert "urban-slang-in-traditional-story" in codes(_slang([base[0], base[1], "Hamid: Kal ka exam mujhe dope laga."], [{}, {}, {"slang": "dope"}], setting={"traditional": True}))
    assert "urban-slang-in-traditional-story" not in codes(_slang([base[0], base[1], "Hamid: Kal ka exam mujhe dope laga."], [{}, {}, {"slang": "dope"}], setting={"traditional": False}))


def test_a_clean_slang_run_has_no_error():
    base = NEUTRAL[0]["dialogue"].split("\n")
    V = _slang(["Riya: Main bahut pareshan thi, isliye aag-babula ho gayi.", "Siya yaar: Chalo, shuru karte hain.", base[2]], [{"slang": "aag-babula"}, {"slang": "yaar"}, {}])
    assert "gender-disagreement" not in codes(V) and "male-address-for-a-girl" not in codes(V)


def test_the_checker_finds_the_real_historical_defects_in_saved_runs_when_they_exist():
    """Replay (skipped if the old traces are absent): the Godaan run had 17 Narrator lines; the Idgah run had a gesture spoken as speech."""
    import json
    import os
    f = os.path.join("outputs", "ui_20261116_234646_trace.json")
    if not os.path.exists(f):
        pytest.skip("saved trace not present")
    t = json.load(open(f, encoding="utf-8"))
    V = inv.check_pass1(t["deva_scenes"], [c.get("text", "") for c in t.get("scene_chunks", [])], {"event_chain": t["event_chain"], "persona": t["persona"], "neutral": t["neutral_script"]})
    assert sum(1 for v in V if v["code"] == "meta-speaker") >= 17
