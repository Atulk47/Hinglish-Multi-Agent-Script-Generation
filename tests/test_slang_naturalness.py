"""Defects found by reading every saved slang edit (52 distinct edits of the full-pipeline runs): a thing-judging criticism word after "mujhe ... lag raha hai" changed
"I feel bad / hurt" into a verdict, and "sis" / "didi" got a second address word ("Sis yaar", "Chalo yaar, Sis")."""
import pytest

from agents import edit_gate as eg
from agents.agent4_slang_rewriter import SlangRewriter


@pytest.fixture(autouse=True)
def _pool():
    eg.set_setting(None)
    yield
    eg.set_setting(None)


@pytest.mark.parametrize("out,slang,ok", [
    ("Mujhe ekdum flop lag raha hai, Sis.", "flop", False),                                   # I feel bad  ->  "I find it a flop"
    ("Mujhe ekdum bakwaas lag raha hai ki tum aise le gaye.", "bakwaas", False),              # hurt  ->  "nonsense"
    ("Mujhe bahut ghatiya lag raha hai.", "ghatiya", False),
    ("Yeh ekdum bakwaas lag raha hai mujhe.", "bakwaas", True),                               # the thing is named / the experiencer follows: fine
    ("Yeh plan flop lag raha hai.", "flop", True),
    ("Yeh bahut pathetic shot tha, Gita!", "pathetic", True),
    ("Mujhe bore lag raha hai.", "bore", True),                                               # real FEELING words stay allowed
    ("Mujhe cringe lag raha hai.", "cringe", True),
    ("Mujhe wahiyat lag raha hai.", "wahiyat", True),
])
def test_thing_criticism_words_cannot_fill_the_i_feel_frame(out, slang, ok):
    assert eg.experiencer_frame_ok(out, slang) is ok


def test_the_rule_gate_rejects_the_two_saved_defective_edits_and_keeps_good_ones():
    ok, why = eg.rule_gate("Mujhe bahut bura lag raha hai ki tum aise le gaye.", "Mujhe ekdum bakwaas lag raha hai ki tum aise le gaye.", "bakwaas")
    assert not ok and why == "experiencer-frame"
    assert eg.rule_gate("Yeh idea bahut accha hai.", "Yeh idea ekdum solid hai.", "solid")[0]


@pytest.mark.parametrize("line,has", [("Chalo yaar, Sis, ab hum share karenge.", True), ("Sis, yeh remote mera hai.", True), ("Didi, chalo.", True),
                                      ("Chalo, ab hum share karenge.", False)])
def test_sis_and_didi_count_as_an_address_word_already_present(line, has):
    assert SlangRewriter._has_filler(line) is has


def test_pool_flags_are_in_the_data_file():
    import json
    pool = {e["phrase"].lower(): e for e in json.load(open("data/slang/slang_pool.json", encoding="utf-8"))["entries"]}
    for p in ("flop", "bakwaas", "pathetic", "ghatiya", "ek mazaak"):
        assert pool[p].get("experiencer_ok") is False
    for p in ("bore", "cringe", "wahiyat"):
        assert pool[p].get("experiencer_ok") is not False


@pytest.mark.parametrize("line,out,slang,ok", [
    ("Haan, mujhe bhi bahut bura laga.", "Haan, mujhe bhi bahut bore laga.", "bore", False),                       # hurt -> bored
    ("Mujhe bahut bura lag raha hai.", "Mujhe bahut cringe lag raha hai.", "cringe", False),
    ("Yeh film bahut bura lag raha hai.", "Yeh film ekdum jhand lag rahi hai.", "jhand", True),                    # a verdict on a THING may take a verdict word
    ("Mujhe bahut bura laga.", "Mujhe bahut wahiyat laga.", "wahiyat", True),                                      # "awful" keeps the meaning
    ("Haan, mujhe bahut bura laga.", "Haan yaar, mujhe bahut bura laga.", "yaar", True),                           # a filler replaces nothing
    ("Yeh plan bahut kharab hai.", "Yeh plan bahut bore hai.", "bore", True),                                      # no "bura lag" idiom involved
])
def test_bura_laga_keeps_its_meaning(line, out, slang, ok):
    assert eg.bura_meaning_ok(line, out, slang) is ok


@pytest.mark.parametrize("out,slang,ok", [
    ("Sumit, mujhe jhakkas laga ki Papa khush hain.", "jhakkas", False),                                           # thing-praise before "laga ki <clause>"
    ("Mujhe bahut awesome lag raha hai ki tum aaye.", "awesome", False),
    ("Yeh dekh kar bahut awesome laga.", "awesome", True),                                                         # no clause: fine
    ("Yeh idea bahut awesome hai ki sab khush hain.", "awesome", True),                                            # "hai ki", not a lag-form
    ("Sumit, mujhe dil garden-garden laga ki Papa khush hain.", "dil garden-garden ho gaya", True),                # feeling idiom is not a thing-word
])
def test_thing_praise_cannot_stand_before_laga_ki(out, slang, ok):
    assert eg.clause_frame_ok(out, slang) is ok


def test_the_demo_run_defects_are_rejected_by_the_full_gate():
    assert eg.rule_gate("Sumit, mujhe bahut accha laga ki Papa khush hain.", "Sumit, mujhe jhakkas laga ki Papa khush hain.", "jhakkas") == (False, "clause-frame")
    assert eg.rule_gate("Haan, mujhe bhi bahut bura laga.", "Haan, mujhe bhi bahut bore laga.", "bore") == (False, "bura-meaning")
    assert eg.rule_gate("Wah, yeh bahut accha idea hai, Munna!", "Wah, yeh bahut awesome idea hai, Munna!", "awesome")[0]


# ---- 2026-11-19 demo: profiles dropped by the romaniser's spelling, hurt slang, filler caps ------------------------------------------------
def test_a_character_whose_romanised_spelling_differs_keeps_their_profile():
    from agents.agent2_character import name_in_story
    roman = "adity, neha, karan aur simaran ek hi injiniyaring college men padhate the. rohit bhaiya ne kaha."
    assert all(name_in_story(n, roman) for n in ("aditya", "simran", "neha", "karan", "rohit"))
    assert not any(name_in_story(n, roman) for n in ("riya", "arjun", "priyanka"))                 # invented names are still rejected


@pytest.mark.parametrize("out,slang", [
    ("Mera mood off ho gaya hai, Neha.", "mood off ho gaya"), ("Mera dil toot gaya hai, Neha.", "dil toot gaya"),
    ("Mera mann kharab ho gaya hai, Neha.", "mann kharab ho gaya"), ("Mujhe bahut low lag raha hai, Neha.", "low"),
])
def test_hurt_slang_can_replace_bura_lag_raha_hai(out, slang):
    assert eg.rule_gate("Mujhe bahut bura lag raha hai, Neha.", out, slang)[0] is True
    assert eg.rule_gate("Mujhe bahut bura lag raha hai, Neha.", "Mujhe bahut bore lag raha hai, Neha.", "bore")[0] is False      # other feelings still cannot


def test_the_hurt_slot_only_fires_in_the_bura_lag_frame():
    eg.set_setting(None)
    assert any(c["type"] == "sad" for c in eg.slot_candidates("Mujhe bahut bura lag raha hai, Neha."))
    assert not any(c["type"] == "sad" for c in eg.slot_candidates("Yeh idea bahut bura hai."))
    assert not any(c["type"] == "sad" for c in eg.slot_candidates("Usne bura maan liya."))


def test_fillers_are_capped_per_scene_and_per_word():
    import config
    rw = SlangRewriter.__new__(SlangRewriter)
    rw._agentic_line, rw._filler_n, rw._scene_n, rw._edit_n, rw._scene_fillers = True, 0, 1, 0, 0
    from collections import defaultdict
    rw._used = defaultdict(int)
    assert rw._filler_allowed() is True
    rw._mark("yaar", "filler")
    assert rw._scene_fillers == 1 and rw._filler_allowed() is False                       # one filler per scene
    rw._scene_fillers = 0
    rw._mark("yaar", "filler")
    rw._scene_fillers = 0
    assert rw._used["yaar"] == 2
    rw._has_filler = lambda c: False
    rw._filler_kind = lambda c: "opener"
    rw._addressee_is_not_a_peer = lambda c: False
    assert "yaar" not in rw._filler_shortlist("Haan, theek hai.", 0) and "arre" in rw._filler_shortlist("Haan, theek hai.", 0)       # a word used twice is out
    assert config.SLANG_MAX_FILLERS_PER_SCENE == 1 and config.SLANG_FILLER_REPEAT_MAX == 2 and config.SLANG_DIRECTOR_FILLER_SHARE == 0.15


def test_excitement_has_a_slot_and_slang_but_not_in_a_traditional_story():
    eg.set_setting(None)
    L = "Main bahut excited hoon, chalo shuru karte hain."
    assert any(c["type"] == "excite" for c in eg.slot_candidates(L))
    assert eg.rule_gate(L, "Main bahut hyped hoon, chalo shuru karte hain.", "hyped")[0] is True
    assert eg.rule_gate(L, "Main bahut bore hoon, chalo shuru karte hain.", "bore")[0] is False
    eg.set_setting({"era": "historical", "milieu": "rural", "formality": "traditional"})
    assert not any(c["type"] == "excite" for c in eg.slot_candidates(L))                      # urban_youth slang never in a village / period story
    eg.set_setting(None)
