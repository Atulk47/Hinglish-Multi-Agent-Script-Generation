"""
Property-based / fault-injection tests: what must hold for ANY input, not for the stories somebody happened to try.
Deterministic (seeded) so a failure is reproducible; each property runs over many random inputs.
"""
import random
import re
import time

import pytest

from agents import grounding as g
from agents.agent1_narrative import NarrativeAnalyst
from agents.agent3_synthesizer import neutralize_leaks, sanitize_pass1, strip_meta, line_budget
from preprocessing.segmenter import SceneSegmenter, split_sentences

HINDI = ["राम ने कहा कि वह कल आएगा।", "सीता बाज़ार गई और फल लायी।", "अगले दिन सुबह सब स्कूल पहुँचे।", "— तुम कहाँ थे? — मैं घर पर था।", "उसने सिर हिलाया और चुप रहा।",
         "\"क्या तुम आओगे?\" उसने पूछा।", "रात को बारिश तेज़ हो गई।", "दादी ने उसके सिर पर हाथ फेर दिया।", "वह मन ही मन सोचता रहा।", "घर पहुँचकर माँ ने दरवाज़ा खोला।",
         "ok", "…", "।", "!!!", "मोबाइल पर मीटिंग का लिंक आया।", "Ravi ne kaha ki main aa raha hoon.", "x" * 400, "बहुत " * 150 + "।"]


def _story(rng):
    paras = []
    for _ in range(rng.randint(0, 25)):
        k = rng.randint(0, 9)
        paras.append(" ".join(rng.choice(HINDI) for _ in range(k)))
    sep = rng.choice(["\n\n", "\n\n\n", "\n \n", "\n"])
    return sep.join(paras)


# ------------------------------------------------------------------------------------------------------------------------ segmenter
@pytest.mark.parametrize("seed", range(40))
def test_segmenter_never_crashes_keeps_the_text_and_respects_the_cap(seed):
    rng = random.Random(seed)
    text = _story(rng)
    cap = rng.choice([3, 8, 15])
    r = SceneSegmenter(use_llm=False, max_scenes=cap).segment(text)
    norm = lambda t: re.sub(r"\s+", "", t)
    if text.strip():
        assert 1 <= len(r.scenes) <= max(cap, 1)
        # sentence splitting normalises spacing only: no word is lost or invented
        assert norm(" ".join(r.scenes)) == norm(text) or sorted(norm(" ".join(r.scenes))) == sorted(norm(text))
        assert r.starts[0] == 0 and r.starts == sorted(set(r.starts))
    else:
        assert r.scenes == []


def test_segmenter_handles_degenerate_inputs():
    for t in ["", "   ", "\n\n\n", "।", "!!!", "a", "x " * 5000, "।" * 200, "राम " * 3000, "\n\n".join(["।"] * 30), "A. B. C. D."]:
        r = SceneSegmenter(use_llm=False).segment(t)
        assert len(r.scenes) <= 15


# -------------------------------------------------------------------------------------------------------------------------- regexes
@pytest.mark.parametrize("fn", [g.spoken_lines, g.reported_speech, g.unspoken_cues, g.gesture_cues, g.narrates_own_gesture, g.shows_concealment, g.speech_subjects])
def test_text_helpers_are_fast_on_adversarial_text(fn):
    """No catastrophic regex backtracking: a pathological 60 000-character input must finish quickly."""
    for text in ["कहा कि " * 8000, "कह" * 20000, "—" * 20000, "मन में " * 9000, "\"" * 20000, "बोला " * 12000 + "कि", ("चुप " * 5000) + "।" + ("सिर हिला " * 3000)]:
        t0 = time.time()
        fn(text)
        assert time.time() - t0 < 2.0, (fn.__name__, text[:20])


def test_grounding_helpers_survive_random_unicode():
    rng = random.Random(1)
    pool = "अआइईउऊएऐओऔकखगघचछजझटठडढतथदधनपफबभमयरलवशषसह़ािीुूेैोौ्ंःँ।!?.,;:\"'—…abcXYZ 0123\n\t​‍"
    for _ in range(200):
        s = "".join(rng.choice(pool) for _ in range(rng.randint(0, 300)))
        g.norm(s), g.tokens(s), g.span_supported(s, s[: len(s) // 2]), g.clean_location(s, s[:20], s), g.spoken_lines(s), g.unspoken_cues(s), g.gesture_cues(s)
        g.ground_events([s[:30]], [s[:30]], s), g.is_unknown_location(s), g.role_speaks_in(s[:12] or "x", s)


# ---------------------------------------------------------------------------------------------------------------------- the sanitiser
LABELS = ["Hamid", "Amina", "Narrator", "NARRATOR", "Wait", "Note", "Stranger", "hamid", "HAMID", "Amina ji", "Dukaandaar", "(Note", "A B C D E", "x", "123", "Correct output"]
BODIES = ["Chalo ghar chalte hain.", "", "   ", "The sole speaker of this scene is the listener of the story, which is not a character.", "Main khushi se haan karta hoon.", "Yeh kamaal hai, yaar!",
          "राम ने कहा", "okay", "mast nahi, bahut accha hai", "A" * 700, "(Note: I kept it short.)"]


def _noisy_output(rng):
    out = []
    for _ in range(rng.randint(0, 9)):
        r = rng.random()
        if r < 0.7:
            out.append(f"{rng.choice(LABELS)}: {rng.choice(BODIES)}")
        elif r < 0.85:
            out.append(rng.choice(BODIES))
        else:
            out.append(rng.choice(["", "  ", "---", "* *", "```", "{\"a\": 1}"]))
    return "\n".join(out)


@pytest.mark.parametrize("seed", range(60))
def test_sanitiser_output_is_always_clean_and_idempotent(seed):
    rng = random.Random(seed)
    cast = ["Hamid", "Amina"]
    txt = _noisy_output(rng)
    out = sanitize_pass1(strip_meta(txt), cast, extras=["Dukaandaar"])
    assert sanitize_pass1(out, cast, extras=["Dukaandaar"]) == out                       # idempotent
    for ln in out.split("\n") if out else []:
        m = re.match(r"^([^:]+):\s*(\S.*)$", ln)
        assert m, f"unlabelled / empty line survived: {ln!r}"
        assert m.group(1) in cast + ["Dukaandaar"], f"unknown speaker survived: {ln!r}"
        assert m.group(1).lower() not in ("narrator", "wait", "note", "correct output")
    assert "narrates" not in out


@pytest.mark.parametrize("seed", range(30))
def test_neutralize_leaks_is_idempotent_and_never_raises(seed):
    rng = random.Random(seed)
    txt = _noisy_output(rng) + "\nHamid: " + rng.choice(["Arre yaar, yeh kamaal hai!", "bahut mast nahi, bahut accha hai", "Fire brigade aa gayi.", ""])
    once = neutralize_leaks(txt)
    assert neutralize_leaks(once) == once
    assert not re.search(r"\b(kamaal|zabardast|panga|yaar)\b", once, re.I)


def test_line_budget_is_always_within_bounds():
    rng = random.Random(3)
    for _ in range(200):
        s = "".join(rng.choice(["क", "ख", "।", " ", "—", "\"", "!"]) for _ in range(rng.randint(0, 500)))
        assert 2 <= line_budget(s, rng.choice([6, 12])) <= 12


# ------------------------------------------------------------------------------------------------- fault injection: a model may return anything
GARBAGE = ["", None, "I'm sorry, I can't help with that.", "[", "{", "[{", "```json\n[{\"scene_id\": \"scene_01\", \"key_events\": [\"a\"", "null", "[]", "[1, 2, 3]", "{\"key_events\": 5}",
           "[{\"scene_id\": \"scene_01\", \"key_events\": \"not a list\", \"characters\": \"Ram\", \"location\": 7}]", "\u0000\u0001", "x" * 50000]


def test_agent1_parser_never_raises_on_garbage():
    a = NarrativeAnalyst.__new__(NarrativeAnalyst)
    for raw in GARBAGE:
        out = a._parse_json_output(raw or "", "scene_01")
        assert isinstance(out, list) and out and isinstance(out[0], dict)


def test_agent1_survives_a_model_that_always_fails_or_raises():
    a = NarrativeAnalyst.__new__(NarrativeAnalyst)
    a._alt_client = object()
    for behaviour in (lambda *x, **k: "", lambda *x, **k: None, lambda *x, **k: "garbage", lambda *x, **k: (_ for _ in ()).throw(RuntimeError("provider down"))):
        a._call_llm = behaviour
        ev = a._extract_one({"scene_id": "scene_01", "text": "ram ghar gaya"}, "", "prev")
        assert ev["scene_id"] == "scene_01" and isinstance(ev["key_events"], list)
        assert ev.get("analysis_failed")


def _pipeline_scene(writer_output, critic=None):
    from types import SimpleNamespace
    from orchestrator.pipeline import HinglishPipeline

    class A3:
        def generate_pass1(self, *a, **k):
            if isinstance(writer_output, Exception):
                raise writer_output
            return writer_output
    p = SimpleNamespace(agent3=A3(), critic=critic, transliterator=SimpleNamespace(transliterate_sentence=lambda s: s))
    return HinglishPipeline._scene_pass1(p, {"scene_id": "scene_01", "text": "hamid aur amina ghar mein"},
                                         {"scene_id": "scene_01", "characters": ["Hamid", "Amina"], "key_events": ["x"]}, {"Hamid": {}, "Amina": {}},
                                         {"scene_cast": {"scene_01": {"present": ["Hamid", "Amina"], "mentioned_only": [], "extra_speakers": []}}},
                                         "हामिद और अमीना घर में थे। हामिद ने कहा कि वह आएगा।")


def _real_writer(reply):
    """The REAL DialogueSynthesizer.generate_pass1 (prompt building, banned-word retry, sanitiser) with a model that answers `reply`."""
    from types import SimpleNamespace
    from agents.agent3_synthesizer import DialogueSynthesizer
    a = DialogueSynthesizer.__new__(DialogueSynthesizer)
    a.groq_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kw: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))]))))
    return a


@pytest.mark.parametrize("out", ["", "   ", None, "Narrator: Sab chup the.", "Wait, that violates the rule: narrator cannot speak.",
                                 "The sole speaker is the listener of the story, so no dialogue.", "```" + chr(10) + "Hamid: hi" + chr(10) + "```", "Hamid: " + "bahut " * 400,
                                 "हामिद: नमस्ते", "Hamid: Arre yaar, yeh kamaal hai!" + chr(10) + "Amina: bahut mast nahi, bahut accha hai",
                                 "Stranger: Hello." + chr(10) + "Hamid: Haan."])
def test_a_bad_writer_answer_never_reaches_the_script(out):
    ev = {"scene_id": "scene_01", "characters": ["Hamid", "Amina"], "key_events": ["x"],
          "scene_cast": {"present": ["Hamid", "Amina"], "mentioned_only": [], "extra_speakers": []}}
    d = _real_writer(out).generate_pass1({"scene_id": "scene_01", "text": "hamid aur amina ghar mein", "deva_text": "हामिद और अमीना घर में थे।"},
                                         {"Hamid": {}, "Amina": {}}, ev)
    assert isinstance(d, str)
    assert not re.search(r"narrator|violates the rule|sole speaker|Stranger", d, re.I)
    assert not re.search(r"\b(kamaal|mast|yaar)\b", d, re.I)                       # leaked slang is neutralised
    for ln in [l for l in d.splitlines() if l.strip()]:
        assert re.match(r"^(Hamid|Amina):\s*\S", ln) or "[generation failed" in ln, ln


def test_a_critic_that_crashes_or_returns_nonsense_does_not_crash_the_scene():
    from agents.agent3b_critic import FidelityCritic

    class Boom(FidelityCritic):
        def check(self, *a, **k):
            raise RuntimeError("critic down")
    try:
        r = _pipeline_scene("Hamid: Amma, main aa gaya.", critic=Boom(client=None))
    except RuntimeError:
        pytest.fail("a crashing critic took the whole scene down")
    assert r["dialogue"].startswith("Hamid:")


def test_a_writer_that_raises_is_reported_not_fatal():
    try:
        r = _pipeline_scene(RuntimeError("provider down"))
    except RuntimeError:
        pytest.fail("a failing writer took the whole run down")
    assert isinstance(r["dialogue"], str)
