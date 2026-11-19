"""Slang pool as data + grammatical frames for subject-bound idioms (found by the review loop: "Mujhe dil baith raha hai" is ungrammatical)."""
import json

import pytest

from agents import edit_gate as eg

LEGACY_POS = ["bindaas", "paisa vasool", "zabardast", "kamaal", "mast", "kadak", "solid", "lajawaab", "jhakaas", "gazab", "faadu", "top class",
              "on point", "ek number", "dhamaal"]
LEGACY_NEG = ["ghatiya", "bakwaas", "kachra", "flop", "jhand", "bekaar", "faltu", "wahiyat", "third class"]
LEGACY_FEEL = {"full tension", "dil baith raha hai", "jaan nikal gayi", "fatti padi hai", "dimaag fry", "pasine chhoot gaye", "haath-pair phool gaye"}
LEGACY_PROB = {"panga", "locha", "lafda", "ghapla", "jhol", "gadbad", "chakkar", "jhamela"}
LEGACY_EASE = {"chutki mein", "bacchon ka khel", "left hand ka khel"}


def test_pool_keeps_every_legacy_phrase_in_order():
    assert eg.POS_SLANG[:len(LEGACY_POS)] == LEGACY_POS                      # same order: seeded shortlists stay reproducible
    assert eg.NEG_SLANG[:len(LEGACY_NEG)] == LEGACY_NEG
    assert LEGACY_FEEL <= eg.FEEL_SLANG and LEGACY_PROB <= eg.PROB_SLANG and LEGACY_EASE <= eg.EASE_SLANG
    assert eg.PREDICATE_SLANG[:8] == LEGACY_POS[:8] and "sorted" in eg.PREDICATE_SLANG
    assert eg.INTENSITY["kamaal"] == 2 and eg.SLANG_GENDER["full tension"] == "f" and eg.NOUN_GENDER["gadbad"] == "f"
    assert isinstance(eg.FEEL_SLANG, set) and isinstance(eg.PROB_SLANG, set)    # edit_audit does set algebra on these


def test_candidates_are_not_used_until_promoted(tmp_path):
    p = tmp_path / "pool.json"
    p.write_text(json.dumps({"entries": [{"phrase": "kamaal", "slots": ["praise"], "status": "core"},
                                         {"phrase": "newword", "slots": ["praise"], "status": "candidate"}]}), encoding="utf-8")
    core = eg.load_pool(str(p))
    assert [e["phrase"] for e in core] == ["kamaal"]
    assert eg.load_pool(str(tmp_path / "missing.json")) is None


@pytest.mark.parametrize("raw,fixed,slang", [
    ("Mujhe dil baith raha hai.", "Mera dil baith raha hai.", "dil baith raha hai"),
    ("Mujhe jaan nikal gayi.", "Meri jaan nikal gayi.", "jaan nikal gayi"),
    ("Mujhe pasine chhoot gaye.", "Mere pasine chhoot gaye.", "pasine chhoot gaye"),
    ("Humein dil baith raha hai.", "Hamara dil baith raha hai.", "dil baith raha hai"),
    ("Aaj mujhe jaan nikal gayi thi.", "Aaj meri jaan nikal gayi thi.", "jaan nikal gayi"),
])
def test_repair_frame_makes_the_experiencer_possessive(raw, fixed, slang):
    assert eg.repair_frame(raw, [slang]) == fixed
    assert eg.frame_ok(fixed, slang) and not eg.frame_ok(raw, slang)


def test_unfixable_frames_are_rejected_not_guessed():
    out = "Mujhe ekdum dil baith raha hai."                      # an intensifier in between: do not guess, leave it for the gate
    assert eg.repair_frame(out, ["dil baith raha hai"]) == out
    assert eg.rule_gate("Mujhe bahut dar lag raha hai.", out, "dil baith raha hai") == (False, "idiom-frame")


def test_rule_gate_accepts_the_repaired_swap_and_rejects_the_raw_one():
    line = "Mujhe bahut dar lag raha hai."
    assert eg.rule_gate(line, "Mujhe dil baith raha hai.", "dil baith raha hai") == (False, "idiom-frame")
    ok, why = eg.rule_gate(line, "Mera dil baith raha hai.", "dil baith raha hai")
    assert ok and why == "phrase-swap"


def test_non_framed_slang_is_untouched():
    assert eg.repair_frame("Mujhe full tension ho rahi hai.", ["full tension"]) == "Mujhe full tension ho rahi hai."
    assert eg.rule_gate("Mujhe bahut chinta ho rahi hai.", "Mujhe full tension ho rahi hai.", "full tension")[0] is True


def test_situational_phrase_needs_its_topic():
    has = lambda s: any("paisa vasool" in c["slangs"] for c in eg.slot_candidates(s))
    assert not has("Yeh baat bahut accha lagi.")
    assert has("Yeh film bahut accha thi.") and has("Ticket ke paise ke hisaab se bahut accha tha.")


def test_audit_signs_follow_the_pool():
    from evaluation import edit_audit
    assert "kamaal" in edit_audit.POS_SIGN and "dil baith raha hai" in edit_audit.NEG_SIGN


def test_apply_review_promotes_keeps_and_disables(tmp_path, monkeypatch):
    from tools import apply_slang_review as ar
    pool = {"entries": [{"phrase": "kamaal", "slots": ["praise"], "sign": 1, "intensity": 2, "status": "core"},
                        {"phrase": "oldword", "slots": ["praise"], "sign": 1, "intensity": 1, "status": "core"}]}
    cands = {"entries": [{"phrase": "killer", "slots": ["praise"], "sign": 1, "intensity": 3, "kind": "slang_word", "gender": None, "frame": None,
                          "context": None, "do_not_use_when": "serious moments", "risk": "safe", "status": "candidate"},
                         {"phrase": "mid", "slots": ["criticism"], "sign": -1, "intensity": 1, "status": "candidate"},
                         {"phrase": "scene", "slots": ["trouble"], "sign": -1, "intensity": 1, "status": "candidate"}]}
    (tmp_path / "pool.json").write_text(json.dumps(pool), encoding="utf-8")
    (tmp_path / "cands.json").write_text(json.dumps(cands), encoding="utf-8")
    monkeypatch.setattr(ar, "POOL", str(tmp_path / "pool.json"))
    monkeypatch.setattr(ar, "CANDS", str(tmp_path / "cands.json"))
    answers = [{"phrase": "killer", "section": "B. New", "decision": "keep", "note": ""},
               {"phrase": "mid", "section": "B. New", "decision": "drop", "note": ""},
               {"phrase": "scene", "section": "B. New", "decision": "fix", "note": "use 'bada scene'"},
               {"phrase": "oldword", "section": "A. Existing", "decision": "drop", "note": ""},
               {"phrase": "kamaal", "section": "A. Existing", "decision": "", "note": ""}]
    rep = ar.apply(answers)
    assert rep["promoted"] == ["killer"] and rep["rejected"] == ["mid"] and rep["disabled"] == ["oldword"]
    assert rep["fix_new"] == [("scene", "use 'bada scene'")] and rep["unanswered"] == ["kamaal"]
    saved = json.loads((tmp_path / "pool.json").read_text(encoding="utf-8"))
    by = {e["phrase"]: e for e in saved["entries"]}
    assert by["killer"]["status"] == "core" and by["killer"]["controls"] == ["ASSERT_POSITIVE", "ASSERT_EXCITED"] and by["oldword"]["status"] == "disabled"
    core = eg.load_pool(str(tmp_path / "pool.json"))
    assert [e["phrase"] for e in core] == ["kamaal", "killer"]             # disabled / rejected / fix are not used at runtime
    assert json.loads((tmp_path / "cands.json").read_text(encoding="utf-8"))["entries"][1]["status"] == "rejected"


# ------------------------------------------------------------------ native-speaker rules + extra emotion slots (round 1 review)
def test_on_point_is_fine_as_a_predicate_but_never_before_laga():
    assert eg.rule_gate("Yeh sab bahut badhiya hai.", "Yeh sab ekdum on point hai.", "on point")[0] is True
    assert eg.rule_gate("Aaj bahut accha laga.", "Aaj ekdum on point laga.", "on point") == (False, "slang-usage")


def test_dimaag_fry_needs_mera_and_never_a_genitive_noun():
    assert eg.rule_gate("Mujhe bahut dar lag raha hai.", "Mera dimaag fry ho raha hai.", "dimaag fry")[0] is True
    assert eg.repair_frame("Mujhe dimaag fry ho raha hai.", ["dimaag fry"]) == "Mera dimaag fry ho raha hai."
    assert eg.rule_gate("Ab sab ki chinta kam ho gayi hai.", "Ab sab ki dimaag fry kam ho gayi hai.", "dimaag fry") == (False, "idiom-frame")


def test_review_promoted_phrases_are_live():
    for p in ("cringe", "ekdum smooth", "panic", "kaand"):
        assert any(e["phrase"] == p for e in eg.POOL_ENTRIES), p
    assert not any(e["phrase"] in ("bawaal", "default mein", "setting kharab") for e in eg.POOL_ENTRIES)       # dropped by the reviewer


def test_extra_emotion_slots_work_through_the_gate_and_the_audit():
    core = eg.load_pool()
    joy = {"phrase": "dil garden-garden ho gaya", "slots": ["joy"], "sign": 1, "intensity": 2, "kind": "idiom", "gender": None,
           "frame": {"possessive_before": "dil", "gender": "m", "number": "sg"}, "context": None, "controls": [], "status": "core"}
    try:
        eg._apply_pool(core + [joy], {"joy": {"sign": 1, "triggers": ["khushi", "maza"]}})
        assert "joy" in eg.EXTRA_SLOTS
        line = "Mujhe bahut khushi ho rahi hai."
        assert eg.slot_candidates(line)[0]["type"] == "joy"
        out = eg.repair_frame("Mujhe dil garden-garden ho gaya.", ["dil garden-garden ho gaya"])
        assert out == "Mera dil garden-garden ho gaya."
        assert eg.rule_gate(line, out, joy["phrase"])[0] is True
        assert eg.rule_gate(line, "Mujhe kamaal ho rahi hai.", "kamaal")[0] is False                  # a praise slang may not replace a joy word
    finally:
        eg._apply_pool(core)
    want = sorted(e["phrase"].lower() for e in core if "joy" in (e.get("slots") or []))                  # restored state = the pool file, nothing leaked from the test
    assert sorted(eg.EXTRA_SLOTS.get("joy", {}).get("slangs", [])) == want


def test_emotion_idiom_swaps_pass_with_the_right_possessive_and_aaya_verbs():
    cases = [("Rohan ne phir se ghalat rasta le liya, mujhe bahut gussa aaya.", "Rohan ne phir se ghalat rasta le liya, mujhe pitta chadh gaya.", "pitta chadh gaya"),
             ("Anjali ne chai itni achhi banayi ki mujhe bahut maza aa raha hai.", "Anjali ne chai itni achhi banayi ki mujhe mann mein laddu phoot rahe hain.", "mann mein laddu phoot rahe hain")]
    for line, out, slang in cases:
        fixed = eg.repair_frame(out, [slang])
        assert eg.rule_gate(line, fixed, slang)[0] is True, fixed
    assert eg.repair_frame("Mujhe neend udd gayi.", ["neend udd gayi"]) == "Meri neend udd gayi."            # feminine noun
    assert eg.repair_frame("Mujhe mann mein laddu phoot rahe hain.", ["mann mein laddu phoot rahe hain"]) == "Mere mann mein laddu phoot rahe hain."   # oblique


def test_native_speaker_usage_rules_from_the_v6_review():
    # aag-babula needs a nominative subject; ghazab ka / kamaal ka are noun modifiers; masculine "ka" idioms never precede feminine nouns
    line = "Mujhe bahut gussa aa raha hai, yeh bahut kharab hua hai."
    assert eg.usage_ok("Mujhe bahut aag-babula ho raha hai.", "aag-babula") is False
    assert eg.usage_ok("Main bahut aag-babula ho raha hoon.", "aag-babula") is True
    assert eg.usage_ok("Mujhe dekh ke woh aag-babula ho gaya.", "aag-babula") is False          # 'mujhe' within the same clause
    assert eg.usage_ok("Rohan ko bura laga, woh aag-babula ho gaya.", "aag-babula") is True
    assert eg.usage_ok("Mujhe ghazab ka laga.", "ghazab ka") is False and eg.usage_ok("Yeh idea ghazab ka hai.", "ghazab ka") is True
    assert eg.usage_ok("Mujhe yeh kamaal ka lag raha hai.", "kamaal ka") is False
    assert eg.usage_ok("Yeh bina sar-pair ka baat hai.", "bina sar-pair ka") is False and eg.usage_ok("Yeh bina sar-pair ka plan hai.", "bina sar-pair ka") is True
    assert eg.rule_gate(line, "Mujhe bahut aag-babula ho raha hai, yeh bahut kharab hua hai.", "aag-babula")[0] is False


def test_emotion_slot_gets_its_own_shortlist_when_the_line_also_has_a_praise_word():
    from types import SimpleNamespace
    from collections import defaultdict
    from agents.agent4_slang_rewriter import SlangRewriter
    import random
    me = SimpleNamespace(_serious=False, _line_low="", _used=defaultdict(int), _rng=random.Random(7), _target_intensity=None)
    line = "Haan, yeh idea bahut accha hai. Mujhe bahut khushi ho rahi hai."
    merged, _ = SlangRewriter._slot_shortlist(me, line)
    only, sign = SlangRewriter._slot_shortlist(me, line, extra_only=True)
    joy = set(eg.EXTRA_SLOTS["joy"]["slangs"])
    assert only and {x for x in only} <= joy and sign == 1
    assert SlangRewriter._slot_shortlist(me, "Yeh film bahut accha thi.", extra_only=True) == ([], 0)       # no emotion word: nothing special


def test_emotion_idioms_may_swap_a_longer_span_but_nothing_else_loosens():
    from agents.agent4_slang_rewriter import SlangRewriter
    line = "Haan, yeh idea bahut accha hai. Hum dono milkar kar sakte hain, mujhe bhi bahut khushi ho rahi hai."
    out = eg.repair_frame("Haan, yeh idea bahut accha hai. Hum dono milkar kar sakte hain, mere dil ko thandak mil gayi.", ["dil ko thandak mil gayi"])
    assert eg.span_limit("dil ko thandak mil gayi") == 7 and eg.span_limit("dil baith raha hai") == 7 and eg.span_limit("jhakaas") == 5
    assert eg.rule_gate(line, out, "dil ko thandak mil gayi")[0] is True                      # 6-word span: fine for an emotion idiom
    assert SlangRewriter._keeps_meaning(line, out, span_max=7) and not SlangRewriter._keeps_meaning(line, out)
    # still rejected: a content word lost, a negation in the sentence
    assert eg.rule_gate("Rohan ne meri chai gira di, mujhe bahut gussa aa raha hai aur tension bhi hai.", "Rohan ne meri chai gira di, pitta chadh gaya aur tension bhi hai.", "pitta chadh gaya")[0] is False
    assert eg.rule_gate("Mujhe nahi lag raha ki main bahut khush hoon aaj.", "Mujhe nahi lag raha ki mera dil garden-garden ho gaya aaj.", "dil garden-garden ho gaya")[0] is False


def test_a_sentence_with_an_emotion_word_is_tried_before_the_praise_sentence():
    from types import SimpleNamespace
    from agents.agent4_slang_rewriter import SlangRewriter as R
    me = SimpleNamespace(_SENT_RE=R._SENT_RE, _line_polarity=lambda core: 1, _has_emotion_slot=R._has_emotion_slot)
    line = "Haan, yeh idea bahut accha hai. Hum dono milkar kar sakte hain, mujhe bhi bahut khushi ho rahi hai."
    spans = R._pick_sentences(me, line, 2)
    assert "khushi" in line[spans[0][0]:spans[0][1]] and len(spans) == 2                       # emotion sentence first, praise sentence kept as the fallback


def test_negation_only_blocks_a_swap_in_its_own_sentence():
    other = "Rohan, tumne phir se apna hissa nahi kiya? Mujhe bahut gussa aa raha hai, yeh bahut kharab baat hai."
    assert eg.rule_gate(other, other.replace("Mujhe bahut gussa aa raha", "Mera paara chadh gaya"), "paara chadh gaya")[0] is True        # 'nahi' is in the OTHER sentence
    same = "Mujhe nahi lag raha ki main bahut khush hoon aaj."
    assert eg.rule_gate(same, "Mujhe nahi lag raha ki mera dil garden-garden ho gaya aaj.", "dil garden-garden ho gaya")[0] is False   # 'nahi' in the same sentence


def test_audit_does_not_mistake_sentence_starters_or_idiom_words_for_entities_or_negations():
    from evaluation import edit_audit as ea
    codes = lambda a, b, s: {f["code"] for f in ea.audit_edit(a, b, s)["flags"] if f["blocking"]}
    assert "entity-changed" not in codes("Haan Priya, bahut accha laga. Mujhe bahut khushi ho rahi hai.", "Haan Priya, bahut accha laga. Mera dil garden-garden ho gaya.", "dil garden-garden ho gaya")
    assert "entity-changed" not in codes("Nauvi class ka hai. Bahut accha laga sunkar.", "Nauvi class ka hai. Ekdum bindaas laga sunkar.", "bindaas")
    assert "negation-changed" not in codes("Stadium gunj utha, mujhe bahut khushi hui!", "Stadium gunj utha, mere pair zameen par nahi hain!", "pair zameen par nahi hain")
    assert "non-local" not in codes("Hum dono milkar kar sakte hain, mujhe bhi bahut khushi ho rahi hai.", "Hum dono milkar kar sakte hain, mere dil ko thandak mil gayi.", "dil ko thandak mil gayi")
    # real entities and real negations are still caught
    assert "entity-changed" in codes("Kal Rohan aayega.", "Kal Aryan aayega.", None)
    assert "negation-changed" in codes("Mujhe yeh pasand nahi hai.", "Mujhe yeh pasand hai.", None)


def test_audit_infers_a_hyphenated_idiom_and_pride_nouns_never_take_laga():
    from evaluation import edit_audit as ea
    a = ea.audit_edit("Haan Priya, bahut accha laga. Mujhe bahut khushi ho rahi hai.", "Haan Priya, bahut accha laga. Mera dil garden-garden ho gaya.", None)
    assert a["slang"] == "dil garden-garden ho gaya" and a["ok"]
    assert eg.usage_ok("Mujhe bahut naaz laga yeh jaanke.", "naaz") is False and eg.usage_ok("Mujhe tum par bahut naaz hai.", "naaz") is True


# --- vocative / mention handling and the "darr" spelling (2026-11-16, found on a real voice-input run) ---
def _slot_words(s):
    from agents import edit_gate as eg
    return [c["word"] for c in eg.slot_candidates(s)]


def test_darr_spelling_is_a_worry_slot():
    assert _slot_words("Mujhe bahut darr lag raha hai.") == ["darr"]
    assert _slot_words("Mujhe bahut dar lag raha hai.") == ["dar"]
    assert _slot_words("Wahan darr ka mahaul hai.") == []                    # "darr" is a slot only before lag/lagta


def test_mentioning_an_elder_does_not_block_a_feeling_slot():
    assert _slot_words("Nani ka janamdin hai, mujhe bahut khushi ho rahi hai!") == ["khushi"]
    assert _slot_words("Bhaiya, nani ka janamdin agle hafte hai, mujhe bahut khushi ho rahi hai!") == ["khushi"]
    assert _slot_words("Sir ne bola ki yeh idea bahut accha hai.") == ["accha"]


def test_addressing_an_elder_still_blocks_slang():
    for s in ("Nani, mujhe bahut khushi ho rahi hai!", "Mujhe bahut khushi ho rahi hai, Nani.", "Aap ka idea bahut accha hai.",
              "Sir, yeh idea bahut accha hai!", "Papa, mujhe bahut darr lag raha hai."):
        assert _slot_words(s) == [], s


def test_a_vocative_like_bhaiya_blocks_only_the_filler_route():
    from agents.agent4_slang_rewriter import SlangRewriter
    assert SlangRewriter._has_filler("Bhaiya, mujhe bahut khushi ho rahi hai!")          # still true: no SECOND filler on this line
    assert _slot_words("Bhaiya, mujhe bahut khushi ho rahi hai!") == ["khushi"]           # but its feeling word can be swapped


def test_split_vocative_prefix_and_suffix():
    from agents.agent4_slang_rewriter import SlangRewriter as R
    assert R._split_vocative("Bhaiya, mujhe bahut khushi ho rahi hai!") == ("Bhaiya, ", "Mujhe bahut khushi ho rahi hai!", "")
    assert R._split_vocative("Lekin Bhaiya, ye saamaan bahut kharab ho jayega toh kya hoga?")[0] == "Lekin Bhaiya, "
    assert R._split_vocative("Bahut accha plan hai, Bhaiya!") == ("", "Bahut accha plan hai!", ", Bhaiya!")
    assert R._split_vocative("Bhaiya, chalo.") == ("", "Bhaiya, chalo.", "")           # too short a remainder to edit
    assert R._split_vocative("Mujhe bahut khushi ho rahi hai.") == ("", "Mujhe bahut khushi ho rahi hai.", "")


def test_filler_is_not_placed_after_a_sentence_initial_word():
    from agents.edit_gate import rule_gate
    assert not rule_gate("Accha hai, tu itni excited hai toh mujhe bhi bahut accha laga.",
                         "Accha yaar, hai, tu itni excited hai toh mujhe bhi bahut accha laga.", "yaar")[0]
    assert rule_gate("Rohan, chalo shuru karte hain.", "Rohan yaar, chalo shuru karte hain.", "yaar")[0]
    assert rule_gate("Chinta mat kar, Anu. Hum sab theek kar lenge.", "Chinta mat kar yaar, Anu. Hum sab theek kar lenge.", "yaar")[0]


def test_filler_new_comma_allowed_at_sentence_opener_and_before_a_name():
    from agents.edit_gate import rule_gate
    assert rule_gate("Chinta mat karo Mira, bas thodi si moch hai.", "Chinta mat karo yaar, Mira, bas thodi si moch hai.", "yaar")[0]
    assert rule_gate("Kabir, tum ro rahe ho? Bahut khushi ki baat hai.", "Kabir, tum ro rahe ho? Arre, bahut khushi ki baat hai.", "arre")[0]
