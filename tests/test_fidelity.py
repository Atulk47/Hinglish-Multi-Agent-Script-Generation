"""Offline tests for story fidelity: the story bible from the Persona Agent (who is present / mentioned only / secrets), Agent 3's use of it,
the Fidelity Critic (deterministic + LLM layers) and the per-scene fix loop in the pipeline."""
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.agent2b_persona import PersonaAgent
from agents.agent3_synthesizer import DialogueSynthesizer
from agents.agent3b_critic import FidelityCritic, parse_lines


def fake(reply_fn):
    calls = []

    def create(**kw):
        calls.append(kw)
        r = reply_fn(kw) if callable(reply_fn) else reply_fn
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=r))])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), calls=calls)


def test_critic_deterministic_layer_flags_absent_speakers_without_any_llm():
    c = FidelityCritic(client=None)
    c.client = None
    dlg = "Ananya: Bhaiya, Nani ka janamdin hai.\nNani: Mujhe sab pata hai.\nVikram: Chup raho.\nNarrator: sab hans diye.\nDukaandaar: Ji beta."
    rep = c.check("src", dlg, ["Ananya", "Vikram"], ["Nani"], extras=["Dukaandaar"])
    # line 2: Nani is absent; line 4: a "Narrator" is never a character (it used to be whitelisted, which let invented narrator dialogue through)
    assert not rep["ok"] and [i["line"] for i in rep["issues"]] == [2, 4] and all(i["type"] == "absent_speaker" for i in rep["issues"])
    assert "narrator" in rep["issues"][1]["detail"].lower()
    assert FidelityCritic(client=None).check("src", "Ananya: hi\nVikram: hello", ["Ananya", "Vikram"], [])["ok"] in (True, False)


def test_drop_absent_lines_keeps_everyone_else():
    dlg = "ANANYA: Bhaiya!\nNANI: Beta, yeh kya hai?\nVIKRAM: Kuch nahi.\nBACHCHE: Surprise!"
    out = FidelityCritic.drop_absent_lines(dlg, ["Ananya", "Vikram"])
    assert "NANI" not in out and "ANANYA" in out and "VIKRAM" in out and "BACHCHE" in out


def test_critic_llm_layer_parses_filters_and_survives_garbage():
    reply = json.dumps({"issues": [{"line": 3, "type": "secret_leak", "detail": "Nani is told about the surprise"},
                                   {"line": 1, "type": "bad_type", "detail": "ignored"},
                                   {"line": "x", "type": "invented_event", "detail": "a new accident"}]})
    c = FidelityCritic(client=fake(reply))
    rep = c.check("src", "Ananya: a\nVikram: b\nNani: c", ["Ananya", "Vikram", "Nani"], [],
                  secrets=[{"what": "birthday surprise", "hidden_from": ["Nani"]}])
    types = sorted(i["type"] for i in rep["issues"])
    assert types == ["invented_event", "secret_leak"] and rep["llm_used"]
    assert "birthday surprise" in c.client.calls[0]["messages"][0]["content"]
    bad = FidelityCritic(client=fake("not json")).check("src", "Ananya: a\nVikram: b", ["Ananya", "Vikram"], [])
    assert bad["ok"] and not bad["llm_used"]


NAMES = ["Ananya", "Vikram", "Nani"]
CHUNKS = [{"scene_id": "scene_01", "text": "ananya aur vikram nani ke liye surprise plan karte hain"},
          {"scene_id": "scene_02", "text": "nani ghar lauti"}]
EVENTS = [{"scene_id": "scene_01", "characters": ["Ananya", "Vikram", "Nani"], "key_events": ["plan"]},
          {"scene_id": "scene_02", "characters": ["Nani", "Ananya"], "key_events": ["return"]}]


def test_persona_story_bible_scene_cast_and_secrets():
    chunks = [dict(CHUNKS[0], text=CHUNKS[0]["text"] + ". dukaandaar ne kaha ji beta"), CHUNKS[1]]       # the shopkeeper really speaks in the story
    llm = {"characters": {}, "relationships": [], "emotions": {},
           "scene_cast": {"scene_01": {"present": ["Ananya", "Vikram", "Ghost"], "mentioned_only": ["Nani"], "extra_speakers": ["Dukaandaar!!", "x"]},
                          "scene_02": {"present": ["Nani", "Ananya"], "mentioned_only": []}},
           "secrets": [{"what": "birthday surprise for Nani", "hidden_from": ["Nani", "Nobody"], "scenes": ["scene_01", "scene_99"],
                        "evidence": "nani ke liye surprise plan karte hain"},                      # quoted from the story and about concealment
                       {"what": "the cow was bought with a shady loan", "hidden_from": ["Nani"], "evidence": "aisi koi baat kahi nahi gayi"},       # invented: dropped
                       {"what": "", "hidden_from": ["Nani"]}, {"what": "x", "hidden_from": []}]}
    p = PersonaAgent(llm_client=fake(json.dumps(llm))).analyze(chunks, EVENTS, NAMES)
    cast = p["scene_cast"]
    assert cast["scene_01"]["present"] == ["Ananya", "Vikram"] and cast["scene_01"]["mentioned_only"] == ["Nani"]
    assert cast["scene_01"]["extra_speakers"] == ["Dukaandaar"]
    assert p["secrets"] == [{"what": "birthday surprise for Nani", "hidden_from": ["Nani"], "scenes": ["scene_01"]}]
    # LLM silent -> Agent 1's list, flagged as a fallback
    p2 = PersonaAgent(llm_client=fake("garbage")).analyze(CHUNKS, EVENTS, NAMES)
    assert p2["scene_cast"]["scene_01"]["present"] == ["Ananya", "Vikram", "Nani"] and p2["scene_cast"]["scene_01"]["fallback"]
    assert p2["secrets"] == []


def _agent3(reply="Ananya: Bhaiya, chalo.\nVikram: Haan."):
    a = DialogueSynthesizer.__new__(DialogueSynthesizer)
    a.groq_client = fake(reply)
    return a


def test_agent3_prompt_contains_the_story_bible():
    a = _agent3()
    ev = dict(EVENTS[0], scene_cast={"present": ["Ananya", "Vikram"], "mentioned_only": ["Nani"], "extra_speakers": ["Dukaandaar"]},
              secrets=[{"what": "birthday surprise for Nani", "hidden_from": ["Nani"], "scenes": ["scene_01"]}],
              feedback="line 2: absent_speaker - Nani is not in this scene")
    profiles = {n: {"social_role": "peer", "gender": "female" if n != "Vikram" else "male"} for n in NAMES}
    a.generate_pass1({"scene_id": "scene_01", "text": "x", "deva_text": "अनन्या और विक्रम योजना बनाते हैं।"}, profiles, ev)
    prompt = a.groq_client.calls[0]["messages"][0]["content"]
    assert "NOT PRESENT in this scene" in prompt and "Nani" in prompt.split("NOT PRESENT")[1][:80]
    assert "Characters present in this scene: Ananya" in prompt and "Vikram" in prompt.split("Characters present")[1].split(".")[0]
    assert "Nani (" not in prompt.split("Characters present")[1].split("NOT PRESENT")[0]          # absent character is not in the cast list
    assert "SECRET that must stay hidden: birthday surprise for Nani" in prompt and "Dukaandaar" in prompt
    assert "REVIEWER REJECTED" in prompt and "ORIGINAL SCENE (Hindi" in prompt and "STAY ON THE STORY" in prompt


def test_scene_loop_regenerates_with_feedback_and_removes_leftover_absent_lines():
    from orchestrator.pipeline import HinglishPipeline
    first = "Ananya: Bhaiya!\nNani: Mujhe sab pata hai.\nVikram: Chup."
    replies = iter([first, first, "Ananya: Bhaiya!\nVikram: Chup.\nNani: Phir se main hoon."])
    seen = []

    class A3:
        def generate_pass1(self, chunk, profiles, ev):
            seen.append(ev.get("feedback"))
            return next(replies)

    crit = FidelityCritic(client=None)
    crit.client = None
    p = SimpleNamespace(agent3=A3(), critic=crit, transliterator=SimpleNamespace(transliterate_sentence=lambda s: s))
    persona = {"scene_cast": {"scene_01": {"present": ["Ananya", "Vikram"], "mentioned_only": ["Nani"], "extra_speakers": []}},
               "secrets": [{"what": "surprise", "hidden_from": ["Nani"], "scenes": ["scene_01"]}]}
    out = HinglishPipeline._scene_pass1(p, CHUNKS[0], EVENTS[0], {}, persona, "deva")
    assert seen[0] is None and "absent_speaker" in seen[1]                       # attempt 2 got the reviewer's feedback
    assert out["fidelity"]["attempts"] == 3
    assert "Nani" not in out["dialogue"] and "Ananya" in out["dialogue"]
    assert out["fidelity"]["lines_removed"] + out["fidelity"]["flagged_lines_removed"] >= 1       # fail-closed removal (or the older absent-speaker cleanup)
    assert "\n\n" not in out["dialogue"]


def test_agent3_prompt_tells_the_writer_two_names_are_one_person():
    a = _agent3()
    ev = dict(EVENTS[0], scene_cast={"present": ["Ananya", "Vikram"], "mentioned_only": [], "extra_speakers": []},
              aliases=[{"alias": "Vicky", "canonical": "Vikram"}])
    profiles = {n: {"social_role": "peer", "gender": "female" if n != "Vikram" else "male"} for n in NAMES}
    a.generate_pass1({"scene_id": "scene_01", "text": "x", "deva_text": "अनन्या और विक्रम योजना बनाते हैं।"}, profiles, ev)
    prompt = a.groq_client.calls[0]["messages"][0]["content"]
    assert 'SAME PERSON: "Vicky" is only another name for Vikram' in prompt


def test_writer_no_longer_forces_a_feeling_into_every_scene_and_asks_for_spoken_hinglish(monkeypatch):
    import config
    monkeypatch.setattr(config, "AGENT3_STYLE", "slotrich")                  # the faithful style (the shipped default is slotrich_pad)
    from agents.agent3_synthesizer import PEER_STYLE_RULES, line_budget
    assert "at least two lines" not in PEER_STYLE_RULES and "Do NOT add a feeling" in PEER_STYLE_RULES
    a = _agent3()
    ev = dict(EVENTS[0], scene_cast={"present": ["Ananya", "Vikram"], "mentioned_only": [], "extra_speakers": []})
    profiles = {n: {"social_role": "peer", "gender": "female" if n != "Vikram" else "male"} for n in NAMES}
    a.generate_pass1({"scene_id": "scene_01", "text": "x", "deva_text": "अनन्या और विक्रम योजना बनाते हैं।"}, profiles, ev)
    prompt = a.groq_client.calls[0]["messages"][0]["content"]
    assert "the way people really TALK" in prompt and "Never tack" in prompt and "at least two lines" not in prompt
    assert line_budget("। ".join(["वाक्य"] * 8), 12) == 5                       # 8 sentences, no quotes: 1 + 8 // 2


def test_the_shipped_default_is_the_padded_writer_and_the_demo_slang_target_is_two_to_three():
    import config
    assert config.AGENT3_STYLE == "slotrich_pad" and config.SLANG_DIRECTOR_TARGET == 2 and config.SLANG_DIRECTOR_MAX_PER_SCENE == 3


def test_the_line_budget_leaves_room_for_every_key_event():
    a = _agent3()
    ev = dict(EVENTS[0], key_events=[f"event {i}" for i in range(7)], scene_cast={"present": ["Ananya", "Vikram"], "mentioned_only": [], "extra_speakers": []})
    profiles = {n: {"social_role": "peer", "gender": "female" if n != "Vikram" else "male"} for n in NAMES}
    a.generate_pass1({"scene_id": "scene_01", "text": "x", "deva_text": "अनन्या और विक्रम योजना बनाते हैं।"}, profiles, ev)
    assert "At most 8 dialogue lines" in a.groq_client.calls[0]["messages"][0]["content"]


def test_writer_prompt_forbids_speaking_scenery_and_added_thanks_or_promises(monkeypatch):
    import config
    ev = dict(EVENTS[0], scene_cast={"present": ["Ananya", "Vikram"], "mentioned_only": [], "extra_speakers": []})
    profiles = {n: {"social_role": "peer", "gender": "female" if n != "Vikram" else "male"} for n in NAMES}
    prompts = {}
    for style in ("slotrich", "slotrich_pad"):
        monkeypatch.setattr(config, "AGENT3_STYLE", style)
        a = _agent3()
        a.generate_pass1({"scene_id": "scene_01", "text": "x", "deva_text": "अनन्या और विक्रम योजना बनाते हैं।"}, profiles, ev)
        prompts[style] = a.groq_client.calls[0]["messages"][0]["content"]
    p = prompts["slotrich"]                                   # the faithful style
    assert "DESCRIPTION IS NOT SPEECH" in p and "do NOT add thanks, promises, opinions" in p and "never pad a short scene" in p
    q = prompts["slotrich_pad"]                               # the padded style no longer contradicts itself: feelings are allowed, scenery is still not speech
    assert "DESCRIPTION IS NOT SPEECH" in q and "DO let a character say how they FEEL" in q and "opinions, jokes" not in q and "never pad a short scene" not in q
    assert "at least two lines" in q


def test_style_switch_selects_the_padded_legacy_rules_or_the_faithful_ones(monkeypatch):
    import config
    ev = dict(EVENTS[0], scene_cast={"present": ["Ananya", "Vikram"], "mentioned_only": [], "extra_speakers": []})
    profiles = {n: {"social_role": "peer", "gender": "female" if n != "Vikram" else "male"} for n in NAMES}
    got = {}
    for style in ("slotrich", "slotrich_pad", "default"):
        monkeypatch.setattr(config, "AGENT3_STYLE", style)
        a = _agent3()
        a.generate_pass1({"scene_id": "scene_01", "text": "x", "deva_text": "अनन्या और विक्रम योजना बनाते हैं।"}, profiles, ev)
        got[style] = a.groq_client.calls[0]["messages"][0]["content"]
    assert "at least two lines" in got["slotrich_pad"] and "at least two lines" not in got["slotrich"]
    assert "Do NOT add a feeling" in got["slotrich"] and "Peer characters" not in got["default"]
