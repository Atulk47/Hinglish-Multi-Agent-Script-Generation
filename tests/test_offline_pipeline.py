"""
tests/test_offline_pipeline.py
End-to-end smoke test of the WHOLE pipeline with NO network: the three Groq-backed pieces (Agent 1, Agent 3, the intent
classifier, and Agent 4's tone/rerank calls) are replaced by stubs fed from a saved trace. Everything else is real:
preprocessing, NER + MuRIL, the Qwen LoRA editor, the gates, the rule gate, MuRIL validation, routing and assembly.

It proves the wiring works (and keeps working) independently of Groq quota.
    python -m pytest tests/test_offline_pipeline.py -q        (~4-6 min on CPU)
"""
import json
import os
import re

STORY = "data/raw/demo_science_fair.txt"
TRACE = "outputs/final_a_v3_trace.json"       # saved Pass-1 dialogue + event chain from a real run


class _Msg:
    def __init__(self, text):
        self.content = text


class _Choice:
    def __init__(self, text):
        self.message = _Msg(text)
        self.finish_reason = "stop"


class _Resp:
    def __init__(self, text):
        self.choices = [_Choice(text)]


class FakeGroq:
    """Answers the editor's two Groq prompts (tone, rerank) deterministically."""
    class _Completions:
        def create(self, **kw):
            prompt = kw["messages"][-1]["content"]
            if "Answer with exactly one word" in prompt:                     # tone
                line = prompt.split("Line:")[-1].lower()
                neg = any(w in line for w in ("darr", "nakam", "fail", "bekaar", "galat", "dukh"))
                pos = any(w in line for w in ("accha", "achha", "jeet", "badhai", "mast", "zabardast", "safal"))
                return _Resp("negative" if neg else ("positive" if pos else "neutral"))
            if "Choose up to 6" in prompt:                                    # rerank
                opts = re.findall(r"^- (.+)$", prompt, flags=re.M)
                return _Resp(json.dumps({"fit": opts[:5]}))
            return _Resp("{}")

    def __init__(self):
        self.chat = type("Chat", (), {"completions": FakeGroq._Completions()})()


class _StubClf:
    def classify(self, text):
        return "CASUAL_ASSERTION"


def test_whole_pipeline_runs_offline():
    from orchestrator.pipeline import HinglishPipeline
    t = json.load(open(TRACE, encoding="utf-8"))
    by_scene = {n["scene_id"]: n["dialogue"] for n in t["neutral_script"]}

    pl = HinglishPipeline("offline_smoke", verbose=True)
    # --- stub every network piece ---
    pl.agent1.extract_event_chain = lambda chunks, *a, **k: t["event_chain"][: len(chunks)]
    pl.agent3.generate_pass1 = lambda chunk, profiles, ev: by_scene[ev["scene_id"]]
    pl.intent_clf = _StubClf()
    pl.agent4.intent_clf = pl.intent_clf
    pl.agent4.groq_client = FakeGroq()
    pl.persona_agent.client = None            # offline: the Persona Agent falls back to its rule-based mode
    pl.segmenter.use_llm = False              # offline: signal-based scene segmentation only
    pl.critic = None                          # offline: no fidelity-critic LLM call

    story = open(STORY, encoding="utf-8").read()
    final = pl.run(story)

    assert isinstance(final, str) and final.strip(), "pipeline returned an empty script"
    assert "[generation failed" not in final
    assert "# HINGLISH SCREENPLAY" in final
    slang = pl.scratchpad.read("slang_script")
    assert slang and len(slang) == len(t["neutral_script"])
    # Dadi / formal speakers must never be edited, and no mood-loaded slang may sit on the fear line
    for sc in slang:
        for d in sc["decisions"]:
            if d["speaker"].upper() in ("DADI", "SHARM"):
                assert not d.get("applied"), f"elder/authority speaker was edited: {d}"
    assert "nakam ho jaunga, fire" not in final
    diag = pl.scratchpad.read("scene_diagnoses") or {}
    assert diag, "validation did not run"
    print("\nOFFLINE PIPELINE OK:", sum(1 for sc in slang for d in sc["decisions"] if d.get("applied")), "edits applied")
