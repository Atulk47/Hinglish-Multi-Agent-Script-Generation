"""One person known by two names (Rahul / Rocky) must become ONE character - but only on the story's own word for it; two people who merely appear together
must never be merged. Also: an English kin word next to the Hindi kin name of the same person (Grandma / Dadiji) is the same person."""
import json
from types import SimpleNamespace

from agents import grounding as g
from agents.agent1_narrative import NarrativeAnalyst
from agents.agent2b_persona import PersonaAgent

CASES = [("राहुल, जो अपने दोस्तों के बीच रॉकी के नाम से जाना जाता था", "Rahul", "Rocky", True),
         ("साक्षी (जिन्हें सैक के नाम से भी पुकारते थे)", "Sakshi", "Sak", True),
         ("Rahul, also known as Rocky, came", "Rahul", "Rocky", True),
         ("राहुल और रॉकी साथ खेल रहे थे", "Rahul", "Rocky", False),
         ("राहुल ने नाम बताया, मोहन आया", "Rahul", "Mohan", False),
         ("राहुल ने मोहन को रॉकी के नाम से पुकारा", "Rahul", "Mohan", False),
         ("rahul ke nam se jana jata tha", "Rahul", "Rahul", False),
         ("", "Rahul", "Rocky", False)]


def test_names_one_person_needs_both_names_and_a_nickname_cue_next_to_one_of_them():
    for q, a, b, want in CASES:
        assert g.names_one_person(q, a, b) is want, q


SRC = "राहुल, जो अपने दोस्तों के बीच रॉकी के नाम से जाना जाता था, गेट से बाहर निकला। मोहन और गीता भी आए।"


def _persona(alias_rows):
    def create(**kw):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
            "characters": {"Rahul": {"gender": "male"}, "Rocky": {"gender": "male"}, "Mohan": {"gender": "male"}},
            "relationships": [{"a": "Rahul", "b": "Rocky", "relation": "x"}, {"a": "Rocky", "b": "Mohan", "relation": "friend"}],
            "emotions": {"scene_01": {"Rahul": {"emotion": "happy", "intensity": 2}, "Rocky": {"emotion": "happy", "intensity": 1}}},
            "scene_cast": {"scene_01": {"present": ["Rocky", "Mohan"], "mentioned_only": []}},
            "aliases": alias_rows, "secrets": [], "setting": {}}, ensure_ascii=False)))])
    return PersonaAgent(llm_client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))


def _run(alias_rows):
    chain = [{"scene_id": "scene_01", "characters": ["Rahul", "Rocky", "Mohan"], "pronoun_map": {"he": "Rocky"}},
             {"scene_id": "scene_02", "characters": ["Rahul", "Mohan"]}]
    chunks = [{"scene_id": "scene_01", "text": "rahul rki ke nam se jana jata tha"}, {"scene_id": "scene_02", "text": "mohan aya"}]
    p = _persona(alias_rows).analyze(chunks, chain, [], [SRC, "मोहन आया"])
    return p, chain


def test_a_quoted_alias_merges_the_two_characters_everywhere():
    rows = [{"alias": "Rocky", "same_as": "Rahul", "evidence": "राहुल, जो अपने दोस्तों के बीच रॉकी के नाम से जाना जाता था"}]
    p, chain = _run(rows)
    assert "Rocky" not in p["characters"] and "Rahul" in p["characters"] and p["aliases"] == [{"alias": "Rocky", "canonical": "Rahul"}]
    assert chain[0]["characters"] == ["Rahul", "Mohan"] and chain[0]["pronoun_map"] == {"he": "Rahul"}
    assert p["scene_cast"]["scene_01"]["present"] == ["Rahul", "Mohan"]
    assert all(r["a"] != r["b"] for r in p["relationships"])


def test_an_alias_claim_without_a_real_quotation_or_cue_is_ignored():
    for rows in ([{"alias": "Rocky", "same_as": "Rahul", "evidence": ""}],
                 [{"alias": "Rocky", "same_as": "Rahul", "evidence": "राहुल और रॉकी साथ खेल रहे थे"}],                       # not a statement of identity
                 [{"alias": "Mohan", "same_as": "Rahul", "evidence": "राहुल, जो अपने दोस्तों के बीच रॉकी के नाम से जाना जाता था"}],   # wrong pair
                 "garbage", [None, 3]):
        p, chain = _run(rows)
        assert "Rocky" in p["characters"] and p["aliases"] == [] and chain[0]["characters"] == ["Rahul", "Rocky", "Mohan"]


def test_english_kin_word_next_to_the_hindi_kin_name_is_one_person():
    ev = [{"scene_id": "s1", "characters": ["Dadiji", "Grandma", "Rahul"], "pronoun_map": {"she": "Grandma"}},
          {"scene_id": "s2", "characters": ["Grandma", "Rahul"]}]
    NarrativeAnalyst._canonicalize(ev)
    assert ev[0]["characters"] == ["Dadiji", "Rahul"] and ev[1]["characters"] == ["Dadiji", "Rahul"] and ev[0]["pronoun_map"] == {"she": "Dadiji"}
    ev = [{"scene_id": "s1", "characters": ["Grandma", "Dadi", "Nani"]}]            # two possible Hindi words: ambiguous, left alone
    NarrativeAnalyst._canonicalize(ev)
    assert "Grandma" in ev[0]["characters"]


def test_kin_rule_also_runs_after_spelling_variants_are_merged_and_one_letter_variants_merge():
    ev = [{"scene_id": "s1", "characters": ["Grandma", "Dadiiji", "Rahul"]}, {"scene_id": "s2", "characters": ["Dadiji", "Rahul"]}]
    NarrativeAnalyst._canonicalize(ev)
    assert ev[0]["characters"] == ["Dadiji", "Rahul"] and ev[1]["characters"] == ["Dadiji", "Rahul"]
    ev = [{"scene_id": "s1", "characters": ["Rohan", "Rohit", "Riya", "Siya", "Kavita", "Savita"]}]            # different people stay different
    NarrativeAnalyst._canonicalize(ev)
    assert len(ev[0]["characters"]) == 6


def test_a_speaker_who_names_himself_or_his_nickname_as_the_subject_says_maine():
    from agents.agent3_synthesizer import fix_self_reference
    al = [{"alias": "Rocky", "canonical": "Rahul"}]
    t = "Rahul: Wah, Rocky ne Gita ki bal rok di!\nRahul: Rahul ne bhi dekha.\nSakshi: Rocky ne bahut accha khela.\nGita: Rocky, tum kamaal ho. Rocky ne kya kiya?"
    out = fix_self_reference(t, al).split(chr(10))
    assert out[0] == "Rahul: Wah, maine Gita ki bal rok di!"
    assert out[1] == "Rahul: Maine bhi dekha."
    assert out[2] == "Sakshi: Rocky ne bahut accha khela."                 # another person talking ABOUT him: unchanged
    assert out[3] == "Gita: Rocky, tum kamaal ho. Rocky ne kya kiya?"          # not Gita's own name
    assert fix_self_reference("Rahul: Rocky ne khela.", None) == "Rahul: Rocky ne khela."           # no alias known: only the speaker's own name counts
    assert fix_self_reference("", al) == "" and fix_self_reference(None, al) == ""


def test_a_character_the_story_never_names_is_dropped_but_spelling_variants_survive():
    from preprocessing.transliterate import HindiTransliterator
    roman = HindiTransliterator().transliterate_sentence("राहुल, जो अपने दोस्तों के बीच रॉकी के नाम से जाना जाता था, गेट से बाहर निकला। गीता और मोहन भी आए। एक इंजीनियर ने कहा।")
    ev = {"scene_id": "s1", "key_events": [], "characters": ["Rahul", "Rocky", "Geeta", "Mohan", "Sara", "Arjun", "Engineer"], "pronoun_map": {"he": "Rahul", "she": "Sara"}}
    out = NarrativeAnalyst._ground(ev, roman, "")
    assert out["characters"] == ["Rahul", "Rocky", "Geeta", "Mohan", "Engineer"]
    assert out["grounding"]["characters_dropped"] == ["Sara", "Arjun"] and out["pronoun_map"] == {"he": "Rahul"}
    short = NarrativeAnalyst._ground({"scene_id": "s", "key_events": [], "characters": ["Ananya"]}, "x", "")           # too short a text to judge: nothing is dropped
    assert short["characters"] == ["Ananya"]


def test_name_grounding_ignores_spacing_and_still_rejects_names_the_story_lacks():
    from agents import grounding as g
    roman = "dada ji ne kaha ki damad sahab aaye hain aur ramchandra bhi ghar par hai"
    assert g.name_grounded("DadaJi", roman) and g.name_grounded("DamadSahab", roman) and g.name_grounded("Ramchandra", roman)
    assert not g.name_grounded("Priyanka", roman) and not g.name_grounded("Sundari", roman)


def test_spacing_variants_of_a_name_are_one_name_and_the_writer_may_not_narrate_scenery():
    ev = [{"scene_id": "s1", "characters": ["Damad Sahab", "Ramchandra"]}, {"scene_id": "s2", "characters": ["DamadSahab", "Ramchandra"]}]
    NarrativeAnalyst._canonicalize(ev)
    assert ev[0]["characters"] == ev[1]["characters"] == ["Damad Sahab", "Ramchandra"]
