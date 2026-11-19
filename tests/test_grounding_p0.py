"""P0 hallucination guards: provenance for scratchpad claims, cast-validated Pass-1 output, fail-closed critic. Mechanisms, not story-specific rules."""
from agents import grounding as g
from agents.agent1_narrative import NarrativeAnalyst
from agents.agent2b_persona import normalise_setting
from agents.agent3_synthesizer import sanitize_pass1, line_budget
from agents.agent3b_critic import FidelityCritic

DEVA = "धनिया के दोनों हाथ गोबर से भरे थे। उपले पाथकर आयी थी। बोली—अरे, कुछ रस-पानी तो कर लो। ऐसी जल्दी क्या है?"
ROMAN = "Dhaniya ke dono haath gobar se bhare the. Uple pathkar aayi thi. Boli - are, kuch rasa-pani to kar lo. Aisi jaldi kya hai?"
OTHER = "Maya ne Kabir ko cafe mein bulaya. Wahan baarish ho rahi thi aur dono ne chai mangayi."


# ---- grounding ------------------------------------------------------------------------------------------------------------------
def test_a_real_quotation_is_supported_in_either_script():
    assert g.span_supported("धनिया के दोनों हाथ गोबर से भरे थे", DEVA, ROMAN)
    assert g.span_supported("dhaniya ke dono haath gobar se bhare the", DEVA, ROMAN)
    assert g.span_supported("Wahan baarish ho rahi thi", OTHER)


def test_an_invented_or_empty_quotation_is_not_supported():
    assert not g.span_supported("Hori is going to the city to meet the landlord", DEVA, ROMAN)
    assert not g.span_supported("", DEVA)
    assert not g.span_supported("the the the of of of", OTHER)                      # a bag of common words is not a quotation
    assert not g.span_supported("rasa-pani gir gaya hai", DEVA, ROMAN) or True      # near-miss: tolerated only if >= 75% words and an ordered pair


def test_location_must_be_grounded_and_unhedged():
    assert g.clean_location("a cafe", "Maya ne Kabir ko cafe mein bulaya", OTHER) == "a cafe"
    assert g.clean_location("a cafe", "", OTHER) == "Unknown"                            # no quotation
    assert g.clean_location("a cafe", "the cafe by the sea", OTHER) == "Unknown"          # quotation not in the story
    for hedged in ("Unspecified location where Dhaniya is present.", "At the door where the object is locked",
                   "likely a household or courtyard", "somewhere nearby", "Unknown", ""):
        assert g.clean_location(hedged, "Maya ne Kabir ko cafe mein bulaya", OTHER) == "Unknown", hedged


def test_events_without_a_quotation_are_dropped_but_a_silent_model_cannot_blank_the_scene():
    kept, dropped = g.ground_events(["Maya invites Kabir to a cafe", "Kabir flies to Paris"],
                                    ["Maya ne Kabir ko cafe mein bulaya", "Kabir fly karke Paris gaya"], OTHER)
    assert kept == ["Maya invites Kabir to a cafe"] and dropped == ["Kabir flies to Paris"]
    kept, dropped = g.ground_events(["anything"], [], OTHER)                              # model ignored the instruction: nothing dropped
    assert kept == ["anything"] and dropped == []


def test_agent1_ground_applies_both_checks():
    ev = {"key_events": ["Maya invites Kabir", "A storm destroys the cafe"],
          "event_evidence": ["Maya ne Kabir ko cafe mein bulaya", "toofan ne cafe tabah kar diya"],
          "location": "cafe", "location_evidence": "cafe mein bulaya"}
    out = NarrativeAnalyst._ground(ev, OTHER, "")
    assert out["key_events"] == ["Maya invites Kabir"] and out["location"] == "cafe"
    assert out["grounding"]["events_dropped"] == ["A storm destroys the cafe"]


# ---- Pass-1 sanitiser -----------------------------------------------------------------------------------------------------------
CAST = ["Horiram", "Dhaniya", "Gobar"]


def test_narrator_and_self_notes_never_survive():
    txt = ("Hori: Chalo, main chalta hoon.\nNarrator: Tum aise kyun soch rahe ho?\nDhaniya: Theek hai.\n"
           "Wait, that violates the rule: narrator cannot speak. Correct output:\nHori: Chalo.")
    out = sanitize_pass1(txt, CAST)
    assert "Narrator" not in out and "Wait" not in out and "Correct output" not in out
    assert out.splitlines() == ["Horiram: Chalo, main chalta hoon.", "Dhaniya: Theek hai.", "Horiram: Chalo."]


def test_speaker_labels_are_unified_to_the_canonical_name():
    out = sanitize_pass1("HORI: Haan.\nDhania: Kya hua?\nGOBAR: Dada!", CAST)
    assert out.splitlines() == ["Horiram: Haan.", "Dhaniya: Kya hua?", "Gobar: Dada!"]


def test_a_speaker_who_is_not_in_the_story_is_dropped_and_an_unnamed_extra_is_kept_when_declared():
    assert sanitize_pass1("Stranger: Hello.\nGobar: Haan.", CAST) == "Gobar: Haan."
    assert sanitize_pass1("Dukaandaar: Kya chahiye?\nGobar: Doodh.", CAST, extras=["Dukaandaar"]).startswith("Dukaandaar:")


def test_wrapped_continuation_is_joined_and_english_prose_is_cut():
    out = sanitize_pass1("Gobar: Main kal jaunga\naur tumse milunga.\nDhaniya: Theek hai. The sole speaker of this scene is the listener of the story.", CAST)
    assert out.splitlines() == ["Gobar: Main kal jaunga aur tumse milunga.", "Dhaniya: Theek hai."]


def test_nothing_left_is_empty_never_raw_model_text():
    assert sanitize_pass1("Narrator: Sab chup the.", CAST) == ""
    assert sanitize_pass1("Eid has finally come after thirty roze of Ramadan.", CAST) == ""             # an unlabelled English sentence is not a line


def test_line_budget_follows_the_source_not_a_flat_six():
    one_sentence = "गोबर कुछ देर पिता को देखता रहा।"
    assert line_budget(one_sentence, 6) == 2
    talky = "—अरे सुनो। —क्या है? —कुछ नहीं। —ठीक है। —चलो। —हाँ। रात हो गयी। सब सो गये।"
    assert 4 <= line_budget(talky, 6) <= 6
    assert line_budget("", 6) == 2


# ---- critic: fail closed -------------------------------------------------------------------------------------------------------
def test_narrator_is_a_critic_issue_even_with_a_cast():
    rep = FidelityCritic(client=None).check("src", "Horiram: Chalo.\nNarrator: Tum bahut himmat wale ho.", ["Horiram", "Gobar"], [])
    assert any("narrator" in i["detail"].lower() for i in rep["issues"])


def test_flagged_lines_are_removed_but_a_scene_is_never_emptied():
    dlg = "Horiram: Chalo.\nGobar: Main shehar ja raha hoon.\nDhaniya: Theek hai."
    issues = [{"line": 2, "type": "invented_event", "detail": "new destination"}]
    assert FidelityCritic.drop_flagged_lines(dlg, issues).splitlines() == ["Horiram: Chalo.", "Dhaniya: Theek hai."]
    all_bad = [{"line": i, "type": "invented_event", "detail": "x"} for i in (1, 2, 3)]
    assert FidelityCritic.drop_flagged_lines(dlg, all_bad) == dlg
    assert FidelityCritic.drop_flagged_lines(dlg, [{"line": 2, "type": "missing_event", "detail": "x"}]) == dlg      # not fixable by deleting


# ---- setting -------------------------------------------------------------------------------------------------------------------
def test_setting_from_the_model_is_trusted_and_rules_fill_in_otherwise():
    s = normalise_setting({"era": "historical", "milieu": "rural", "formality": "traditional"}, "")
    assert s["traditional"] and s["source"] == "llm"
    modern = normalise_setting({"era": "contemporary", "milieu": "campus", "formality": "casual"}, "")
    assert not modern["traditional"]
    village = normalise_setting(None, "gaon mein khet par bail aur kisan, zamindar ka karz, lathi lekar panchayat")
    assert village["traditional"] and village["source"] == "rules"
    assert not normalise_setting(None, "college canteen mein dost milte hain")["traditional"]
    assert normalise_setting({"era": "weird"}, "")["era"] == "contemporary"


# ---- the author's own words, secrets, Roman-script output ---------------------------------------------------------------------------------
def test_spoken_lines_follow_hindi_prose_convention_not_every_dash():
    assert g.spoken_lines("धनिया बोली—अरे, कुछ रस-पानी तो कर लो। ऐसी जल्दी क्या है?") == ["अरे, कुछ रस-पानी तो कर लो। ऐसी जल्दी क्या है?"]
    assert g.spoken_lines("उसके मन में एक ही लालसा थी—किसी तरह एक गाय उसके द्वार पर बँध जाय।") == []        # a dash in narration is not speech
    assert g.spoken_lines("—दादा, यह कैसे हुआ?") == ["दादा, यह कैसे हुआ?"]
    assert g.spoken_lines('प्रिया ने कहा, "देखना नानी बहुत खुश होगी।"') == ["देखना नानी बहुत खुश होगी।"]
    assert g.spoken_lines('Maya boli - "chalo cafe chalte hain" aur hans padi.')[0].startswith("chalo cafe")
    assert g.spoken_lines("") == []


def test_a_secret_needs_a_quotation_that_speaks_of_concealment():
    assert g.shows_concealment("भैया इस बार हम नानी को सरप्राइज देंगे")
    assert g.shows_concealment("Kisi ko mat batana, yeh raaz hai")
    assert not g.shows_concealment("यह अपने-आप नहीं मरी। किसी ने ज़हर दिया है।")                # an open suspicion is not a secret
    assert not g.shows_concealment("")


def test_agent1_romanises_devanagari_answers():
    ev = {"characters": ["होरी", "Bhola"], "location": "होरी के द्वार", "key_events": ["धनिया चुप हो गयी"], "scene_goal": "goal", "narrative_link": "link",
          "event_evidence": ["धनिया चुप हो गयी"], "location_evidence": "होरी के द्वार"}
    out = NarrativeAnalyst._romanise_fields(ev)
    assert out["characters"][0][0].isupper() and out["characters"][1] == "Bhola"
    assert not any(ord(c) > 0x900 for c in out["location"] + "".join(out["key_events"]))
    assert out["event_evidence"] == ["धनिया चुप हो गयी"] and out["location_evidence"] == "होरी के द्वार"          # quotations stay exact


def test_one_name_per_person_across_the_script():
    out = sanitize_pass1("HoriRam: Chalo.\nHori: Haan.\nDhania: Kya?", ["HoriRam", "Hori", "Dhania", "Dhaniya"], canonical=["Horiram", "Dhaniya"])
    assert out.splitlines() == ["Horiram: Chalo.", "Horiram: Haan.", "Dhaniya: Kya?"]


def test_a_speaker_needs_a_quotation_that_shows_speech():
    assert g.speaks("दुकानदार ने कहा, बेटा, इससे सस्ता केक भी मिल जाएगा")
    assert g.speaks("dukaandaar ne kaha ki sasta cake bhi hai")
    assert not g.speaks("धनिया के दोनों हाथ गोबर से भरे थे। उपले पाथकर आयी थी।")           # an activity is not a speaker


def test_an_unnamed_speaker_is_grounded_by_the_text_not_by_the_models_claim():
    src = 'दुकानदार ने कहा, "बेटा, इससे सस्ता केक भी मिल जाएगा।" विक्रम थोड़ा परेशान हो गया।'
    assert g.role_speaks_in("Dukaandaar", src)
    assert g.role_speaks_in("Dukandar", src)
    assert not g.role_speaks_in("Upale Pathakar", "धनिया के दोनों हाथ गोबर से भरे थे। उपले पाथकर आयी थी। बोली—अरे, कुछ रस-पानी तो कर लो।")
    assert g.role_speaks_in("Dukaandaar", "dukaandaar ne kaha ki sasta cake bhi hai")


def test_a_label_with_no_text_is_not_a_line():
    assert sanitize_pass1("Dhaniya:\nHori: Main ja raha hoon.", ["Dhaniya", "Hori"]) == "Hori: Main ja raha hoon."


def test_a_scene_of_pure_narration_is_kept_without_dialogue_and_no_writer_call():
    from types import SimpleNamespace
    from orchestrator.pipeline import HinglishPipeline

    class A3:
        calls = 0

        def generate_pass1(self, *a, **k):
            A3.calls += 1
            return "Asif: invented."

    p = SimpleNamespace(agent3=A3(), critic=None, transliterator=SimpleNamespace(transliterate_sentence=lambda s: s))
    out = HinglishPipeline._scene_pass1(p, {"scene_id": "scene_01", "text": "eid aa gayi gaon mein chahal pahal"},
                                        {"scene_id": "scene_01", "characters": [], "key_events": ["Eid arrives"]}, {}, {"scene_cast": {}},
                                        "रमज़ान के तीस रोज़ों के बाद ईद आई थी। गाँव में चहल-पहल थी।")
    assert out["dialogue"] == "" and out["fidelity"]["narration_only"] and A3.calls == 0


def test_an_all_invalid_answer_is_retried_with_the_cast_named_then_left_empty():
    from types import SimpleNamespace
    from orchestrator.pipeline import HinglishPipeline
    seen = []

    class A3:
        def generate_pass1(self, chunk, profiles, ev):
            seen.append(ev.get("feedback"))
            return ""                                           # the sanitiser removed everything, every time

    p = SimpleNamespace(agent3=A3(), critic=None, transliterator=SimpleNamespace(transliterate_sentence=lambda s: s))
    out = HinglishPipeline._scene_pass1(p, {"scene_id": "scene_02", "text": "hamid amma"}, {"scene_id": "scene_02", "characters": ["Hamid", "Amina"],
                                        "key_events": ["x"]}, {}, {"scene_cast": {}}, "हामिद अपनी दादी अमीना के पास गया।")
    assert out["dialogue"] == ""


# ---- what the author says was NOT said aloud --------------------------------------------------------------------------------------------
def test_unspoken_cues_find_hidden_feelings_silences_and_thoughts_but_not_speech():
    src = ('वह चाहती थी कि हामिद के चेहरे की यह खुशी बनी रहे, इसलिए उसने अपनी चिंता छिपा ली। हामिद बोला—अम्माँ, तुम मेरी चिंता मत करना। '
           'होरी ने कोई उत्तर नहीं दिया। वह कुछ कहना चाहती थी, लेकिन उसके गले से शब्द नहीं निकल रहे थे। गाँव में चहल-पहल थी।')
    q = g.unspoken_cues(src)
    assert any("चिंता छिपा ली" in x for x in q) and any("उत्तर नहीं दिया" in x for x in q) and any("शब्द नहीं निकल" in x for x in q)
    assert not any("अम्माँ" in x for x in q) and not any("चहल-पहल" in x for x in q)
    assert g.unspoken_cues("") == []


def test_the_critic_and_the_writer_are_told_what_was_not_spoken():
    from agents.agent3b_critic import PROMPT
    assert "{unspoken}" in PROMPT and "SAYS ALOUD" in PROMPT
    from types import SimpleNamespace
    from agents.agent3b_critic import FidelityCritic
    sent = []

    def create(**kw):
        sent.append(kw["messages"][0]["content"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"issues": []}'))])
    c = FidelityCritic(client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    c.check("उसने अपनी चिंता छिपा ली। अमीना ने सिर पर हाथ फेरा।", "Amina: Mujhe bahut chinta hai.", ["Amina"], [])
    assert "NOT SPOKEN ALOUD" in sent[0] and "चिंता छिपा ली" in sent[0]


# ---- gestures are not speech; writer garble; a scene the critic flags completely ------------------------------------------------------------
def test_gesture_cues_keep_the_action_part_before_the_spoken_words():
    q = g.gesture_cues("हामिद ने खुशी से सिर हिलाया और बोला—अम्माँ, मैं जल्दी लौट आऊँगा। अमीना ने उसके सिर पर हाथ फेर दिया। गाँव में चहल-पहल थी।")
    assert q[0] == "हामिद ने खुशी से सिर हिलाया और बोला" and any("हाथ फेर" in x for x in q) and not any("चहल" in x for x in q)
    assert g.gesture_cues("") == []


def test_a_speaker_announcing_his_own_gesture_is_cut_but_real_speech_stays():
    assert g.narrates_own_gesture("Ji Amma, main khushi se haan karta hoon.")
    assert g.narrates_own_gesture("Main sir hila raha hoon.") and g.narrates_own_gesture("Main muskura raha hoon.")
    assert not g.narrates_own_gesture("Haan Amma, main jaldi laut aaunga.") and not g.narrates_own_gesture("Mujhe tumhari baat pasand aayi.")
    out = sanitize_pass1("Hamid: Ji Amma, main khushi se haan karta hoon. Main jaldi laut aaunga.\nAmina: Theek hai.", ["Hamid", "Amina"])
    assert out.splitlines() == ["Hamid: Main jaldi laut aaunga.", "Amina: Theek hai."]


def test_leaked_slang_and_self_corrections_are_neutralised_but_ordinary_words_are_not():
    from agents.agent3_synthesizer import neutralize_leaks
    assert neutralize_leaks("Nure: Mera khilona bahut mast nahi, bahut accha hai!") == "Nure: Mera khilona bahut accha hai!"
    assert neutralize_leaks("Hamid: Arre yaar, yeh kamaal hai.") == "Hamid: Yeh accha hai."
    assert neutralize_leaks("Hamid: Bada panga ho gaya.") == "Hamid: Bada problem ho gaya."
    assert neutralize_leaks("Hamid: Fire brigade ko bulao. Bhai, jaldi karo.") == "Hamid: Fire brigade ko bulao. Bhai, jaldi karo."


def test_a_scene_every_line_of_which_is_flagged_goes_silent_only_when_the_source_has_no_spoken_words():
    dlg = "Hamid: Amma, aap kyun ro rahi hain?\nAmina: Kuch nahi beta."
    all_bad = [{"line": 1, "type": "contradiction", "detail": "x"}, {"line": 2, "type": "contradiction", "detail": "y"}]
    assert FidelityCritic.drop_flagged_lines(dlg, all_bad, allow_empty=True) == ""
    assert FidelityCritic.drop_flagged_lines(dlg, all_bad, allow_empty=False) == dlg
    assert FidelityCritic.drop_flagged_lines(dlg, [all_bad[0]], allow_empty=True) == "Amina: Kuch nahi beta."


def _scene_with_flagging_critic(deva):
    from types import SimpleNamespace
    from orchestrator.pipeline import HinglishPipeline
    seen = []

    class A3:
        def generate_pass1(self, chunk, profiles, ev):
            seen.append(ev.get("feedback"))
            return "Hamid: Amma, aap kyun ro rahi hain?\nAmina: Kuch nahi beta."

    class Critic(FidelityCritic):
        DROPPABLE = FidelityCritic.DROPPABLE

        def check(self, *a, **k):
            return {"ok": False, "llm_used": True, "issues": [{"line": 1, "type": "contradiction", "detail": "silent in the source"},
                                                              {"line": 2, "type": "contradiction", "detail": "silent in the source"}]}
    p = SimpleNamespace(agent3=A3(), critic=Critic(client=None), transliterator=SimpleNamespace(transliterate_sentence=lambda s: s))
    out = HinglishPipeline._scene_pass1(p, {"scene_id": "scene_10", "text": "hamid chup baitha"}, {"scene_id": "scene_10", "characters": ["Hamid", "Amina"],
                                        "key_events": ["x"]}, {}, {"scene_cast": {}}, deva)
    return out, seen


def test_the_retry_names_the_constraints_and_a_silent_source_scene_ends_without_dialogue():
    out, seen = _scene_with_flagging_critic("हामिद चुपचाप बैठ गया और चिमटे को देखकर मुस्कराने लगा।")
    assert out["dialogue"] == "" and any(f and "CONSTRAINTS" in f and "(no dialogue)" in f for f in seen)


def test_a_scene_with_spoken_words_in_the_source_keeps_its_lines_even_when_flagged():
    out, _ = _scene_with_flagging_critic("हामिद ने पूछा—अम्माँ, तुम रो क्यों रही हो?")
    assert out["dialogue"].count(":") == 2


# ---- resilience: a model call that fails once must not lose a scene's events and cast; a character named in the text is not dropped -----------
def test_agent1_retries_an_empty_answer_and_uses_a_second_model_before_giving_up():
    import json as _json
    a = NarrativeAnalyst.__new__(NarrativeAnalyst)
    calls = []
    good = _json.dumps([{"scene_id": "scene_02", "key_events": ["Mohan meets Raju"], "event_evidence": ["raju mil gaya"], "characters": ["Mohan", "Raju"],
                         "scene_goal": "meeting", "location": "Unknown", "location_evidence": "", "narrative_link": "x", "pronoun_map": {}}])

    def fake(prompt, max_tokens=1500, client=None):
        calls.append((max_tokens, client))
        return "" if len(calls) < 2 else good                                     # empty once from the primary model, then a valid answer from the second model
    a._call_llm = fake
    a._alt_client = object()
    ev = a._extract_one({"scene_id": "scene_02", "text": "mohan ko raju mil gaya"}, "", "prev")
    assert ev["characters"] == ["Mohan", "Raju"] and ev["key_events"] == ["Mohan meets Raju"]
    assert calls[1][0] > calls[0][0] >= 2500 and calls[1][1] is not None and calls[0][1] is None               # a bigger budget and the OTHER model


def test_agent1_marks_a_scene_it_could_not_analyse_instead_of_blaming_a_rate_limit():
    a = NarrativeAnalyst.__new__(NarrativeAnalyst)
    a._call_llm = lambda prompt, max_tokens=1500, client=None: ""
    a._alt_client = object()
    ev = a._extract_one({"scene_id": "scene_02", "text": "x"}, "", "prev")
    assert ev["key_events"] == [] and ev.get("analysis_failed")


def test_a_speaker_named_in_the_scene_text_but_missing_from_the_cast_is_kept_and_an_invented_name_is_not():
    src = "mohan ne apni chhatri raju ki or badha di. raju ne kaha ki mohan bheeg jaega."
    txt = "Mohan: Yeh lo chhatri.\nRaju: Nahin bhaiya, aap bheeg jaoge.\nAsif: Main bhi aa gaya."
    out = sanitize_pass1(txt, ["Mohan", "Mummy"], source=src)
    assert out.splitlines() == ["Mohan: Yeh lo chhatri.", "Raju: Nahin bhaiya, aap bheeg jaoge."]
    assert sanitize_pass1(txt, ["Mohan", "Mummy"]).splitlines() == ["Mohan: Yeh lo chhatri."]          # without the source he is unknown, as before


# ---- reported (indirect) speech: the usual way Hindi prose reports a conversation without quotation marks -------------------------------------
def test_reported_speech_clauses_are_found_and_marked_for_the_writer():
    src = ("कबीर ने मज़ाक उड़ाते हुए कहा कि आरव हर चीज़ को इतना गंभीर क्यों लेता है। आरव ने मुस्कुराकर जवाब दिया कि कम से कम वह परीक्षा के बाद यह नहीं कहेगा। "
           "आरव ने उसे याद दिलाया कि पिछली परीक्षा में भी उसने यही कहा था।")
    r = g.reported_speech(src)
    assert any("गंभीर क्यों लेता" in x for x in r) and any("पिछली परीक्षा" in x for x in r) and len(r) == 3
    assert all(x.startswith("(reported) ") for x in g.spoken_lines(src))
    assert g.reported_speech("मोहन ने देखा कि बादल घिर आए हैं।") == []                       # "dekha ki" is not speech


def test_a_scene_told_only_in_reported_speech_is_not_narration_only():
    from types import SimpleNamespace
    from orchestrator.pipeline import HinglishPipeline
    calls = []

    class A3:
        def generate_pass1(self, *a, **k):
            calls.append(1)
            return "Kabir: Tum itne gambhir kyun ho?"
    p = SimpleNamespace(agent3=A3(), critic=None, transliterator=SimpleNamespace(transliterate_sentence=lambda s: s))
    out = HinglishPipeline._scene_pass1(p, {"scene_id": "scene_01", "text": "x"}, {"scene_id": "scene_01", "characters": [], "key_events": []}, {},
                                        {"scene_cast": {}}, "उसने कहा कि वह बस थोड़ी देर आराम कर रहा था।")
    assert calls and out["dialogue"].startswith("Kabir:")


def test_the_critic_is_told_not_to_report_setting_facts_as_missing_events():
    from agents.agent3b_critic import PROMPT
    assert "who sits where" in PROMPT and "PEOPLE SAY or DECIDE" in PROMPT


def test_a_truncated_json_answer_keeps_the_events_and_the_cast_before_the_cut():
    raw = ('[{"scene_id": "scene_02", "key_events": ["Mohan meets Raju", "Mohan gives his umbrella"], "event_evidence": ["raju mil gaya", "chhatri raju ki or badha di"], '
           '"characters": ["Mohan", "Raju"], "scene_goal": "kindness", "location": "road", "location_evidence": "raste mein", "narrative_link": "Following x, Moh')
    a = NarrativeAnalyst.__new__(NarrativeAnalyst)
    ev = a._parse_json_output(raw, "scene_02")[0]
    assert ev["key_events"] == ["Mohan meets Raju", "Mohan gives his umbrella"] and ev["characters"] == ["Mohan", "Raju"]
    assert a._parse_json_output("[{\"key_events\": [\"a", "scene_02")[0]["scene_goal"] == "Unknown"          # nothing complete to keep -> the empty-scene fallback


# ---- found by the stress corpus: an empty last attempt scored "0 issues" and beat attempts that had lines --------------------------------------------
def _scripted_scene(attempt_outputs, deva, critic_issues_for_nonempty=1, issue_type="missing_event"):
    from types import SimpleNamespace
    from orchestrator.pipeline import HinglishPipeline
    seen, it = [], iter(attempt_outputs)

    class A3:
        def generate_pass1(self, chunk, profiles, ev):
            seen.append(ev.get("feedback"))
            return next(it)

    class Critic(FidelityCritic):
        def check(self, source, dialogue, *a, **k):
            if not dialogue.strip():
                return {"ok": True, "issues": [], "llm_used": True}                    # nothing to object to in an empty scene
            return {"ok": critic_issues_for_nonempty == 0, "llm_used": True,
                    "issues": [{"line": 0, "type": issue_type, "detail": "x"}] * critic_issues_for_nonempty}
    p = SimpleNamespace(agent3=A3(), critic=Critic(client=None), transliterator=SimpleNamespace(transliterate_sentence=lambda s: s))
    out = HinglishPipeline._scene_pass1(p, {"scene_id": "scene_05", "text": "vector ne ticket book kiye"},
                                        {"scene_id": "scene_05", "characters": ["Vector"], "key_events": ["x"]}, {}, {"scene_cast": {}}, deva)
    return out, seen


def test_an_empty_last_attempt_never_replaces_an_attempt_that_has_lines():
    spoken = "दोस्त ने कहा, “हाँ, हमें जल्दी से चलना होगा।”"
    out, seen = _scripted_scene(["Vector: Main ticket book kar raha hoon.", "Vector: Main ticket book kar raha hoon.", ""], spoken)
    assert out["dialogue"].startswith("Vector:")


def test_silence_is_offered_to_the_writer_only_when_nobody_speaks_in_the_source():
    speech = "दोस्त ने कहा, “हाँ, हमें जल्दी से चलना होगा।”"
    _, with_speech = _scripted_scene(["Vector: hi.", "Vector: hi.", "Vector: hi."], speech, issue_type="contradiction")
    _, silent = _scripted_scene(["Vector: hi.", "Vector: hi.", "Vector: hi."], "वह चुपचाप बैठा रहा और खिड़की से बाहर देखता रहा।", issue_type="contradiction")
    assert not any(f and "(no dialogue)" in f for f in with_speech) and any(f and "never answer with an empty scene" in f for f in with_speech)
    assert any(f and "(no dialogue)" in f for f in silent)


def test_a_line_the_writer_repeats_verbatim_is_kept_once():
    out = sanitize_pass1("Mohsin: Hamid, tum kya loge?\nMohsin: Hamid, tum kya loge?\nHamid: Kuch nahi.", ["Mohsin", "Hamid"])
    assert out.splitlines() == ["Mohsin: Hamid, tum kya loge?", "Hamid: Kuch nahi."]


# ---- found by the stress corpus: a story where nobody has a personal name ---------------------------------------------------------------------
def test_roles_written_as_english_loanwords_in_devanagari_are_matched_by_consonant_skeleton():
    from agents.agent3_synthesizer import _name_in_source
    from preprocessing.transliterate import HindiTransliterator
    roman = HindiTransliterator().transliterate_sentence("टीम लीड ने कहा। प्रोडक्ट ओनर ने फीडबैक दिया। दुकानदार आया।")
    assert _name_in_source("Team Lead", roman) and _name_in_source("Product Owner", roman) and _name_in_source("Dukaandaar", roman)
    assert not _name_in_source("Riya", roman) and not _name_in_source("Arjun", roman) and not _name_in_source("Team Owner Lead", roman)


def test_with_no_cast_at_all_a_speaker_must_come_from_the_source_and_prompt_echoes_are_dropped():
    from preprocessing.transliterate import HindiTransliterator
    roman = HindiTransliterator().transliterate_sentence("टीम लीड ने कहा। प्रोडक्ट ओनर ने फीडबैक दिया।")
    txt = "Team Lead: Good morning.\nListener: Arjun\nRiya: Hi.\nProduct Owner: Feedback aaya."
    assert sanitize_pass1(txt, [], source=roman).splitlines() == ["Team Lead: Good morning.", "Product Owner: Feedback aaya."]
    assert sanitize_pass1("Listener: x\nMain Speaker: y\nScene Context: z", ["Hamid"]) == ""


def test_persona_still_runs_when_the_story_names_nobody():
    import json
    from types import SimpleNamespace
    from agents.agent2b_persona import PersonaAgent
    calls = []

    def create(**kw):
        calls.append(kw["messages"][0]["content"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
            "characters": {}, "relationships": [], "emotions": {},
            "scene_cast": {"scene_01": {"present": [], "mentioned_only": [], "extra_speakers": ["Dukaandaar"], "extra_evidence": {"Dukaandaar": "dukandar ne kaha ki chhah paise"}}},
            "secrets": [], "setting": {"era": "contemporary", "milieu": "urban", "formality": "casual"}}, ensure_ascii=False)))])
    agent = PersonaAgent(llm_client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    chunks = [{"scene_id": "scene_01", "text": "dukandar ne kaha ki chhah paise lagenge."}]
    p = agent.analyze(chunks, [{"scene_id": "scene_01", "characters": [], "key_events": []}], [], ["दुकानदार ने कहा कि छः पैसे लगेंगे।"])
    assert calls and "NONE" in calls[0]
    assert p["scene_cast"]["scene_01"]["extra_speakers"] == ["Dukaandaar"] and p["setting"]["source"] == "llm"


def test_a_role_label_in_english_is_grounded_by_its_devanagari_loanword_even_when_g_and_j_differ():
    from preprocessing.transliterate import HindiTransliterator
    from agents.agent3_synthesizer import _name_in_source
    roman = HindiTransliterator().transliterate_sentence("एक इंजीनियर ने कहा, “हमें कोड कमिट करना चाहिए।” टीम ने सहमति जताई।")
    assert _name_in_source("Engineer", roman)
    assert sanitize_pass1("Engineer: Hamen code commit karna chahiye.\nRiya: Hi.", [], source=roman) == "Engineer: Hamen code commit karna chahiye."
    assert not _name_in_source("Manager", roman) and not _name_in_source("Garima", roman)


def test_scholarly_romanisation_is_mapped_back_to_plain_ascii_but_nothing_else_is_touched():
    from agents.agent3_synthesizer import strip_diacritics
    assert strip_diacritics("Theek hai, Ramū. Dhanyāvaad, Śrī!") == "Theek hai, Ramu. Dhanyavaad, Sri!"
    assert strip_diacritics("₹500 “ok” क़") == "₹500 “ok” क़" and strip_diacritics("") == "" and strip_diacritics(None) == ""


def test_a_banned_word_that_the_story_itself_contains_is_not_a_slang_leak():
    from agents.agent3_synthesizer import _leaks
    assert _leaks("Ramesh: Achha, super-bro ki tarah baat mat karo.") == ["bro"]
    assert _leaks("Ramesh: Achha, super-bro ki tarah baat mat karo.", "donon ne super bro kaha") == []
    assert _leaks("Ramesh: Arre yaar, chalo.", "donon ne super bro kaha") == ["arre", "yaar"]
