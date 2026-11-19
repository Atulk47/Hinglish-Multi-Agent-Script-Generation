"""Defects found by reading the demo screenplays: 'riyayat' (leniency) misread as 'rearrangement', a glued 'Schoolmein', 'sab bakwaas ja raha hai'."""
from agents import edit_gate as eg
from agents import grounding as g
from agents.agent3_synthesizer import fix_glued_words


def test_glossary_only_returns_words_that_occur_whole_in_the_text():
    assert g.gloss_lines("agali bar koi riyayat nahin milegi") == [next(x for x in g.gloss_lines("riyayat")) ]
    assert "rearrangement" in g.gloss_lines("koi riyayat nahin")[0]
    assert g.gloss_lines("riyayatdar aadmi") == [] and g.gloss_lines("") == [] and g.gloss_lines(None) == []
    assert len(g.gloss_lines("riyayat mohalat sifarish jurmana saza ijazat shikayat namumakin", limit=3)) == 3


def test_glued_english_noun_and_postposition_are_split_but_real_words_are_not():
    assert fix_glued_words("Schoolmein Vigyan Mela hai. Teamne jeet li. Hostelse aaya.") == "School mein Vigyan Mela hai. Team ne jeet li. Hostel se aaya."
    assert fix_glued_words("Main school mein hoon. Classical music, planet, gamer, teamwork, matching") == "Main school mein hoon. Classical music, planet, gamer, teamwork, matching"
    assert fix_glued_words("") == "" and fix_glued_words(None) == ""


def test_verdict_words_cannot_be_followed_by_ja_raha_hai():
    assert not eg.usage_ok("Maine itni mehnat ki hai, aur sab ekdum bakwaas ja raha hai.", "bakwaas")
    assert eg.usage_ok("Yeh film ekdum bakwaas hai.", "bakwaas")
    assert eg.rule_gate("Aur sab bekar ja raha hai.", "Aur sab bakwaas ja raha hai.", "bakwaas")[0] is False


def test_the_writer_prompt_carries_the_glossary_when_a_word_occurs():
    from tests.test_fidelity import _agent3, EVENTS, NAMES
    a = _agent3()
    ev = dict(EVENTS[0], scene_cast={"present": ["Ananya", "Vikram"], "mentioned_only": [], "extra_speakers": []})
    profiles = {n: {"social_role": "peer", "gender": "female" if n != "Vikram" else "male"} for n in NAMES}
    a.generate_pass1({"scene_id": "scene_01", "text": "agali bar koi riyayat nahin milegi", "deva_text": "अगली बार कोई रियायत नहीं मिलेगी।"}, profiles, ev)
    prompt = a.groq_client.calls[0]["messages"][0]["content"]
    assert "WORD MEANINGS" in prompt and "riyayat = leniency" in prompt


def test_markdown_asterisks_never_reach_the_screenplay_or_the_subtitles():
    from agents.agent3_synthesizer import clean_markup, sanitize_pass1
    t = "Sharma: * * Tumhari mehnat saaf dikhayi deti hai, Arav.\n**Arav**: Maine kaha tha na.\n- Riya: Haan.\nRiya: Mujhe 5 * 7 nahi chahiye *sigh* theek hai."
    out = clean_markup(t).split("\n")
    assert out[0] == "Sharma: Tumhari mehnat saaf dikhayi deti hai, Arav." and out[1] == "Arav: Maine kaha tha na." and out[2] == "Riya: Haan."
    assert "*" not in out[3] and clean_markup("") == "" and clean_markup(None) == ""
    assert sanitize_pass1(clean_markup("Sharma: * * Achha.\nArav: Haan."), ["Sharma", "Arav"]) == "Sharma: Achha.\nArav: Haan."


def test_the_narrator_never_announces_an_unknown_location():
    import glob
    import json
    from audio import synthesize as S
    trace = {"character_profiles": {}, "neutral_script": [], "slang_script": [
        {"scene_id": "scene_01", "dialogue": "A: Hi.", "events": {"location": "Unknown"}},
        {"scene_id": "scene_02", "dialogue": "A: Hi.", "events": {"location": "School"}},
        {"scene_id": "scene_03", "dialogue": "A: Hi.", "events": {"location": "n/a"}},
        {"scene_id": "scene_04", "dialogue": "A: Hi.", "events": {}}]}
    lines = [p["text"] for p in S.build_plan(trace, "slang") if p["kind"] == "narration"]
    assert lines == ["Scene 1.", "Scene 2. School.", "Scene 3.", "Scene 4."]
