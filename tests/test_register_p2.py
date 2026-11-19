"""P2: slang REGISTER follows the story setting (era / milieu / formality), for any story."""
import json
import os

import pytest

from agents import edit_gate as eg


@pytest.fixture(autouse=True)
def _reset_setting():
    eg.set_setting(None)
    yield
    eg.set_setting(None)


TRAD = {"era": "historical", "milieu": "rural", "formality": "traditional"}


def test_every_pool_phrase_has_a_register_tag():
    pool = json.load(open(os.path.join("data", "slang", "slang_pool.json"), encoding="utf-8"))
    assert all(e.get("register") in ("urban_youth", "general") for e in pool["entries"])
    reg = {e["phrase"]: e["register"] for e in pool["entries"]}
    for p in ("dope", "cringe", "full tension", "red flag", "panic", "anxiety", "trash", "flop", "pakau"):
        assert reg[p] == "urban_youth", p
    for p in ("jaan nikal gayi", "dil baith raha hai", "paara chadh gaya", "sone pe suhaga", "bekaar", "badiya", "fikar", "zabardast"):
        assert reg[p] == "general", p


def test_default_setting_changes_nothing():
    assert not eg.setting_is_traditional()
    assert eg.register_ok("dope") and eg.register_ok("bro")
    assert eg.rule_gate("Mujhe bahut accha laga.", "Mujhe ekdum dope laga.", "dope")[0]


def test_traditional_setting_blocks_urban_youth_but_keeps_hindi_idioms():
    eg.set_setting(TRAD)
    assert eg.setting_is_traditional()
    assert not eg.register_ok("dope") and not eg.register_ok("panic") and not eg.register_ok("bro")
    assert eg.register_ok("jaan nikal gayi") and eg.register_ok("yaar")
    assert eg.rule_gate("Mujhe bahut accha laga.", "Mujhe ekdum dope laga.", "dope") == (False, "register-mismatch")
    assert eg.rule_gate("Mujhe bahut chinta ho rahi hai.", "Meri jaan nikal gayi.", "jaan nikal gayi")[0]


def test_slot_candidates_drop_urban_slang_in_a_traditional_story_and_keep_the_rest():
    modern = {s for c in eg.slot_candidates("Mujhe bahut chinta ho rahi hai.") for s in c["slangs"]}
    assert "full tension" in modern or "panic" in modern
    eg.set_setting(TRAD)
    trad = {s for c in eg.slot_candidates("Mujhe bahut chinta ho rahi hai.") for s in c["slangs"]}
    assert trad and not {s for s in trad if eg.REGISTER.get(s.lower()) == "urban_youth"}
    assert "dil baith raha hai" in trad or "jaan nikal gayi" in trad


def test_setting_from_the_persona_dict_and_reset():
    eg.set_setting({"era": "contemporary", "milieu": "campus", "formality": "casual", "traditional": False})
    assert not eg.setting_is_traditional()
    eg.set_setting({"era": "contemporary", "milieu": "rural", "formality": "neutral"})       # rural alone is enough
    assert eg.setting_is_traditional()
    eg.set_setting(None)
    assert not eg.setting_is_traditional()


def test_the_director_hears_about_a_traditional_setting_only():
    from agents.slang_director import build_context
    items = [("Hori", "Mujhe bahut chinta ho rahi hai."), ("Dhaniya", "Theek hai.")]
    plain = build_context("scene_01", items, {}, {"location": "Unknown"}, [], 3)
    assert "STORY SETTING" not in plain                                                       # modern stories: prompt (and its cache key) unchanged
    eg.set_setting(TRAD)
    trad = build_context("scene_01", items, {}, {"location": "Unknown"}, [], 3)
    assert "STORY SETTING" in trad and "historical" in trad


def test_fillers_are_register_filtered_and_blocked_between_family_in_a_traditional_story():
    from agents.agent4_slang_rewriter import SlangRewriter

    class Stub(SlangRewriter):                      # no models: only the filler decision is exercised
        def __init__(self):
            from collections import Counter
            self._used = Counter()
            self._cur_speaker = "Hori"
            self._all_profiles = {"Hori": {"relationship": {"Dhaniya": {"relation": "husband of Dhaniya"}, "Bhola": {"relation": "neighbour and friend"}}}}

    s = Stub()
    eg.set_setting(TRAD)
    assert s._filler_shortlist("Dhaniya, chalo ghar chalte hain!", 0) == []                    # spouse: no "yaar"
    assert "yaar" in s._filler_shortlist("Bhola, chalo ghar chalte hain!", 0)                  # a friend is fine
    assert "bro" not in s._filler_shortlist("Bhola, chalo ghar chalte hain!", 0)               # but not "bro"
    eg.set_setting(None)
    assert "bro" in s._filler_shortlist("Bhola, chalo ghar chalte hain!", 0) or "yaar" in s._filler_shortlist("Dhaniya, chalo ghar chalte hain!", 0)


def test_per_story_name_sets_do_not_leak_between_stories():
    assert eg._BASE_FEMALE_NAMES <= eg.FEMALE_NAMES and eg._BASE_RESPECT_ADDR <= eg.RESPECT_ADDR


# ---- sentence-frame rule for praise adjectives (found on a real run: "mujhe bahut lajawaab lag raha hai") -------------------------------------------
def test_a_thing_adjective_cannot_fill_the_feeling_frame_but_fits_a_named_thing():
    ok = lambda line, out, sl: eg.rule_gate(line, out, sl)[0]
    assert not ok("Mujhe bahut accha lag raha hai.", "Mujhe bahut lajawaab lag raha hai.", "lajawaab")
    assert eg.rule_gate("Mujhe bahut accha lag raha hai.", "Mujhe bahut lajawaab lag raha hai.", "lajawaab")[1] == "experiencer-frame"
    assert ok("Mujhe yeh bahut accha laga.", "Mujhe yeh bahut lajawaab laga.", "lajawaab")
    assert ok("Mujhe bahut accha laga.", "Mujhe ekdum zabardast laga.", "zabardast")                       # past "laga": "I found it awesome" is fine
    assert ok("Mujhe aapka idea bahut accha laga.", "Mujhe aapka idea ekdum zabardast laga.", "zabardast")
    assert ok("Yeh bahut accha hai.", "Yeh ekdum lajawaab hai.", "lajawaab")
    assert ok("Mujhe bahut accha lag raha hai.", "Mujhe bahut mast lag raha hai.", "mast")                 # feel-good words fit the feeling frame


def test_the_shortlist_for_a_feeling_frame_leaves_out_thing_adjectives():
    feeling = {s for c in eg.slot_candidates("Mujhe bahut accha lag raha hai.") for s in c["slangs"]}
    thing = {s for c in eg.slot_candidates("Mujhe yeh bahut accha laga.") for s in c["slangs"]}
    assert "lajawaab" not in feeling and "mast" in feeling
    assert "lajawaab" in thing


def test_every_pool_phrase_flagged_as_thing_only_is_a_praise_or_criticism_phrase():
    pool = json.load(open(os.path.join("data", "slang", "slang_pool.json"), encoding="utf-8"))
    flagged = [e for e in pool["entries"] if e.get("experiencer_ok") is False]
    assert flagged and all(("praise" in e["slots"] or "criticism" in e["slots"]) for e in flagged)
    assert all(e["kind"] == "slang_word" for e in flagged if "praise" in e["slots"])
    assert not any(e["phrase"] in ("mast", "bindaas", "badiya", "cool", "bore", "cringe", "wahiyat") for e in flagged)       # real feeling words


def test_a_predicate_idiom_needs_a_linking_verb_and_cannot_replace_an_adverb():
    assert eg.rule_gate("Kal ka exam mujhe aasaani se milega.", "Kal ka exam mujhe bacchon ka khel milega.", "bacchon ka khel") == (False, "needs-copula")
    assert eg.copula_ok("Yeh exam mere liye bacchon ka khel hai.", "bacchon ka khel") and eg.copula_ok("Bacchon ka khel!", "bacchon ka khel")
    assert not eg.copula_ok("Mujhe bacchon ka khel milega.", "bacchon ka khel")
    assert eg.copula_ok("Mujhe chutki mein milega.", "chutki mein")                                   # adverbial idioms are untouched


def test_a_person_subject_idiom_agrees_with_the_speaker_gender():
    bad = "Haan, shayad. Main sach mein samay ko lekar bahut pareshan thi, isliye aag-babula ho gaya."
    neutral = "Haan, shayad. Main sach mein samay ko lekar bahut pareshan thi, isliye gussa aa gaya."
    assert eg.rule_gate(neutral, bad, "aag-babula") == (False, "person-agreement")
    assert eg.repair_person_agreement(bad, ["aag-babula"], "f").endswith("aag-babula ho gayi.")
    assert eg.repair_person_agreement(bad, ["aag-babula"], None).endswith("aag-babula ho gayi.")          # the line itself says "thi"
    assert eg.repair_person_agreement("Main kaam se aag-babula ho gayi thi.", ["aag-babula"], "m").endswith("ho gaya thi.")
    assert eg.repair_person_agreement("Main aag-babula ho gaya.", ["aag-babula"], "m") == "Main aag-babula ho gaya."
    assert eg.repair_person_agreement("Yeh film jhol ho gayi.", ["jhol"], "f") == "Yeh film jhol ho gayi."      # only person-subject idioms are touched
    assert eg.rule_gate("Main bahut pareshan thi, isliye gussa aa gaya.", "Main bahut pareshan thi, isliye aag-babula ho gayi.", "aag-babula")[0]


def test_male_address_words_are_dropped_for_a_girl():
    assert eg.drop_male_address_for_female_names(["yaar", "bro", "bhai"], "Sneha, chalo!") == ["yaar"]          # (names of the current story are added from the Persona profiles)
    assert eg.drop_male_address_for_female_names(["yaar", "bro", "bhai"], "Rohan, chalo!") == ["yaar", "bro", "bhai"]
