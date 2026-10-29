"""
evaluation/eval_fidelity.py - does the generated dialogue stay on the author's story?

    python -m evaluation.eval_fidelity data/raw/voice_test_transcript.txt [--show]

Runs the text front half of the pipeline (segmenter -> Agent 1 -> Persona Agent) and then writes the dialogue TWICE:
  BASELINE  Agent 3 as before: no story bible (everyone Agent 1 mentions is a speaker), no secrets, no critic
  NEW       Agent 3 with the story bible (who is present / only mentioned / secrets) + the Fidelity Critic's fix loop
Both are then judged by an INDEPENDENT checker (a different, stronger model than the critic) against the original scene, so the new system does
not grade its own homework. Reports problems per scene and per type.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.getcwd())
from dotenv import load_dotenv

load_dotenv()
import config
from agents.agent1_narrative import NarrativeAnalyst
from agents.agent2b_persona import PersonaAgent, apply_persona
from agents.agent3_synthesizer import DialogueSynthesizer
from agents.agent3b_critic import FidelityCritic
from llm_client import cost_summary, make_client
from preprocessing.chunker import ProseChunker
from preprocessing.segmenter import SceneSegmenter
from preprocessing.transliterate import HindiTransliterator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("story")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    text = open(a.story, encoding="utf-8").read()
    seg = SceneSegmenter().segment(text)
    print(f"segmenter: {seg.method} -> {len(seg.scenes)} scenes")
    tr = HindiTransliterator()
    rom = "\n\n".join(tr.transliterate_sentence(s) for s in seg.scenes)
    chunks = ProseChunker().chunk(rom)
    events = NarrativeAnalyst(llm_backend="groq").extract_event_chain(chunks)
    names = sorted({c for e in events for c in e.get("characters", [])})
    persona = PersonaAgent().analyze(chunks, events, names, seg.scenes)
    profiles = {n: {"name": n, "social_role": "peer", "slang_level": 5, "current_emotion": "neutral", "stress_index": 0.0} for n in names}
    apply_persona(profiles, persona, [c["scene_id"] for c in chunks])
    print("secrets found:", persona.get("secrets"))
    for sid, c in persona["scene_cast"].items():
        print(f"  {sid}: present {c['present']} | mentioned only {c['mentioned_only']} | extra speakers {c['extra_speakers']}")

    agent3 = DialogueSynthesizer()
    judge = FidelityCritic(client=make_client("dialogue"))            # independent, stronger judge (Haiku), not the critic's own model
    from orchestrator.pipeline import HinglishPipeline
    results = {"BASELINE": [], "NEW": []}

    class _P:                                                           # the scene loop only needs these members of the pipeline
        pass
    p = _P()
    p.agent3, p.critic, p.transliterator = agent3, FidelityCritic(), tr
    for i, (chunk, ev) in enumerate(zip(chunks, events)):
        sid = ev["scene_id"]
        deva = seg.scenes[i]
        cast = persona["scene_cast"].get(sid, {})
        present = cast.get("present") or ev.get("characters", [])
        absent, extras = cast.get("mentioned_only", []), cast.get("extra_speakers", [])
        secrets = [s for s in persona.get("secrets", []) if sid in s.get("scenes", [sid])]
        # BASELINE: the old behaviour
        base_ev = dict(ev)
        base = agent3.generate_pass1({**chunk}, profiles, base_ev)
        # NEW: the pipeline's own per-scene procedure
        new = HinglishPipeline._scene_pass1(p, chunk, ev, profiles, persona, deva)
        for label, dlg in (("BASELINE", base), ("NEW", new["dialogue"])):
            rep = judge.check(deva, dlg, present, absent, secrets, ev.get("key_events"), extras)
            results[label].append({"scene": sid, "issues": rep["issues"], "dialogue": dlg})
        if a.show:
            print(f"\n=== {sid} | present {present} | absent {absent}")
            print("-- baseline:\n" + base + "\n-- new (attempts " + str(new["fidelity"]["attempts"]) + "):\n" + new["dialogue"])
    print("\nJUDGED BY AN INDEPENDENT CHECKER (problems per system):")
    for label, rows in results.items():
        n_issue = sum(len(r["issues"]) for r in rows)
        scenes_bad = sum(1 for r in rows if r["issues"])
        by = {}
        for r in rows:
            for it in r["issues"]:
                by[it["type"]] = by.get(it["type"], 0) + 1
        print(f"  {label:9s} {n_issue:2d} problems in {scenes_bad}/{len(rows)} scenes | {by}")
        for r in rows:
            for it in r["issues"]:
                print(f"       {r['scene']} line {it['line']}: {it['type']} - {it['detail'][:110]}")
    print(cost_summary())
    json.dump(results, open("outputs/eval_fidelity.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
