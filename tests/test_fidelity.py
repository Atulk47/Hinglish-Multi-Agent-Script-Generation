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
    assert not rep["ok"] and [i["line"] for i in rep["issues"]] == [2] and rep["issues"][0]["type"] == "absent_speaker"
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
    llm = {"characters": {}, "relationships": [], "emotions": {},
           "scene_cast": {"scene_01": {"present": ["Ananya", "Vikram", "Ghost"], "mentioned_only": ["Nani"], "extra_speakers": ["Dukaandaar!!", "x"]},
                          "scene_02": {"present": ["Nani", "Ananya"], "mentioned_only": []}},
           "secrets": [{"what": "birthday surprise for Nani", "hidden_from": ["Nani", "Nobody"], "scenes": ["scene_01", "scene_99"]},
                       {"what": "", "hidden_from": ["Nani"]}, {"what": "x", "hidden_from": []}]}
    p = PersonaAgent(llm_client=fake(json.dumps(llm))).analyze(CHUNKS, EVENTS, NAMES)
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
    assert "Nani" not in out["dialogue"] and "Ananya" in out["dialogue"] and out["fidelity"]["lines_removed"] >= 1
    assert "\n\n" not in out["dialogue"]
