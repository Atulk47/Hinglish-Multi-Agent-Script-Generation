"""
evaluation/stress_run.py  -  run the pipeline (up to the neutral dialogue; no local models) on a corpus of very different stories and check the invariants.

    python -m evaluation.stress_run                       # data/stress/*.txt + the held-out real stories
    python -m evaluation.stress_run --only s03 s15        # a subset
    python -m evaluation.stress_run --full --only s01 s20 # the WHOLE pipeline incl. the slang stage (close the UI first: needs ~3 GB RAM)
    python -m evaluation.stress_run --workers 2 --out outputs/stress

Stages exercised (live LLM calls): segmenter -> transliteration -> Agent 1 (events, cast, location) -> Persona (setting, secrets, cast) -> Agent 3 (writer) +
Fidelity Critic. Every story gets: structure checks, Pass-1 checks (evaluation/invariants.py), the number of LLM calls and the time. A crash is a failure too.
The slang stage is checked separately (it needs the local models): evaluation.invariants.check_slang on saved traces.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

REAL = ["data/raw/heldout_classical_village_excerpt.txt", "data/raw/heldout_mohan_umbrella.txt", "data/raw/heldout_math_exam_bench.txt",
        "data/raw/heldout_science_model.txt", "data/raw/heldout_cafe_story.txt", "data/raw/demo_hostel_project_night.txt", "outputs/_eidgah_story.txt",
        "outputs/_birthday_story.txt"]


def run_story(text: str, sid: str) -> dict:
    """-> {"scenes": [...], "result": {...}, "seconds": float}; raises on a crash (the caller records it)."""
    from memory.scratchpad import Scratchpad
    from preprocessing.transliterate import HindiTransliterator
    from preprocessing.chunker import ProseChunker
    from preprocessing.segmenter import SceneSegmenter
    from agents.agent1_narrative import NarrativeAnalyst
    from agents.agent2b_persona import PersonaAgent, apply_persona
    from agents.agent3_synthesizer import DialogueSynthesizer
    from agents.agent3b_critic import FidelityCritic
    from orchestrator.pipeline import HinglishPipeline
    t0 = time.time()
    pl = HinglishPipeline.__new__(HinglishPipeline)
    pl.session_id, pl.verbose, pl.scratchpad = "stress_" + sid, False, Scratchpad("stress_" + sid + "_" + str(int(t0)))
    tr = HindiTransliterator()
    pl.transliterator, pl.chunker, pl.segmenter, pl._pauses = tr, ProseChunker(), SceneSegmenter(), None
    pl.agent1, pl.persona_agent = NarrativeAnalyst(llm_backend="groq"), PersonaAgent()
    pl.agent3, pl.critic = DialogueSynthesizer(llm_backend="groq"), FidelityCritic()
    seg = pl.segmenter.segment(text, pauses=None)
    scenes = [s for s in seg.scenes if s.strip()]
    roman = [tr.transliterate_sentence(p) for p in scenes]
    chunks = pl.chunker.from_scenes(roman)
    chain = pl.agent1.extract_event_chain(chunks, deva_scenes=scenes)
    persona = pl.persona_agent.analyze(chunks, chain, [], scenes)
    names = list(persona["characters"])
    profiles = apply_persona({n: {"name": n, "social_role": "peer"} for n in names}, persona, [c["scene_id"] for c in chunks])
    neutral = []
    for i, (c, e) in enumerate(zip(chunks, chain)):
        neutral.append(pl._scene_pass1(c, e, profiles, persona, scenes[i] if i < len(scenes) else ""))
    return {"original": text, "scenes": scenes, "roman": roman, "segmentation": seg.method,
            "result": {"event_chain": chain, "persona": persona, "neutral": neutral}, "seconds": round(time.time() - t0, 1)}


def run_story_full(pipeline, text: str, sid: str) -> dict:
    """The WHOLE pipeline (needs the local models, ~3 GB RAM): Pass 1, the Slang Director, the local editor, the validator. Returns the same record shape as run_story
    plus the slang script, so the slang invariants (evaluation.invariants.check_slang) can be applied too."""
    t0 = time.time()
    pipeline.run(text)
    rd = pipeline.scratchpad.read
    chunks = rd("scene_chunks") or []
    return {"original": text, "scenes": rd("deva_scenes") or [], "roman": [c.get("text", "") for c in chunks], "segmentation": (rd("segmentation") or {}).get("method"),
            "result": {"event_chain": rd("event_chain") or [], "persona": rd("persona") or {}, "neutral": rd("neutral_script") or []},
            "slang_script": rd("slang_script") or [], "profiles": rd("character_profiles") or {}, "seconds": round(time.time() - t0, 1)}


def check(run: dict) -> list:
    from evaluation import invariants as inv
    V = inv.check_structure(run["original"], run["scenes"], run["result"]["event_chain"], run["result"]["neutral"])
    V += inv.check_pass1(run["scenes"], run["roman"], run["result"])
    if run.get("slang_script"):
        V += inv.check_slang(run["result"]["neutral"], run["slang_script"], run.get("profiles"), (run["result"]["persona"] or {}).get("setting"))
    return V


def one(path: str, out_dir: str, pipeline=None) -> dict:
    sid = os.path.splitext(os.path.basename(path))[0]
    text = open(path, encoding="utf-8").read()
    rec = {"id": sid, "path": path, "chars": len(text)}
    try:
        run = run_story_full(pipeline, text, sid) if pipeline is not None else run_story(text, sid)
        V = check(run)
        rec.update(scenes=len(run["scenes"]), lines=sum(len([l for l in s["dialogue"].split("\n") if l.strip()]) for s in run["result"]["neutral"]),
                   seconds=run["seconds"], violations=V, crashed=False)
        json.dump({k: v for k, v in run.items() if k != "original"} | {"violations": V}, open(os.path.join(out_dir, sid + ".json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    except Exception as e:                                                                    # a crash is a failure of the system, not of the harness
        rec.update(crashed=True, violations=[{"code": "crash", "level": "error", "scene": None, "detail": f"{type(e).__name__}: {str(e)[:160]}"}],
                   trace=traceback.format_exc()[-900:])
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=os.path.join("outputs", "stress"))
    ap.add_argument("--no-real", action="store_true")
    ap.add_argument("--full", action="store_true", help="run the WHOLE pipeline incl. the local editor + slang stage (needs ~3 GB free RAM; one story at a time)")
    a = ap.parse_args()
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    import config
    config.LLM_COST_CAP_USD = float(os.getenv("STRESS_COST_CAP", "3.0"))
    os.makedirs(a.out, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(ROOT, "data", "stress", "s*.txt")))
    if not a.no_real:
        paths += [os.path.join(ROOT, p) for p in REAL if os.path.exists(os.path.join(ROOT, p))]
    if a.only:
        paths = [p for p in paths if any(o in os.path.basename(p) for o in a.only)]
    print(f"{len(paths)} stories, {1 if a.full else a.workers} workers{' (FULL pipeline)' if a.full else ''}", flush=True)
    if a.full:
        from orchestrator.pipeline import HinglishPipeline
        pl = HinglishPipeline("stress_full", verbose=False)
        recs = [one(p, a.out, pl) for p in paths]
    else:
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            recs = list(ex.map(lambda p: one(p, a.out), paths))
    from evaluation import invariants as inv
    allv = [v for r in recs for v in r["violations"]]
    summary = {"stories": len(recs), "crashed": sum(r["crashed"] for r in recs),
               "stories_with_errors": sum(1 for r in recs if any(v["level"] == "error" for v in r["violations"])),
               "by_code": inv.summarize(allv),
               "per_story": [{"id": r["id"], "scenes": r.get("scenes"), "lines": r.get("lines"), "seconds": r.get("seconds"),
                              "errors": sum(v["level"] == "error" for v in r["violations"]), "warns": sum(v["level"] == "warn" for v in r["violations"]),
                              "error_codes": sorted({v["code"] for v in r["violations"] if v["level"] == "error"})} for r in recs]}
    json.dump({"summary": summary, "records": recs}, open(os.path.join(a.out, "report.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nSTORIES {summary['stories']} | with errors {summary['stories_with_errors']} | crashed {summary['crashed']}")
    for r in summary["per_story"]:
        print(f"  {r['id']:32s} scenes={r['scenes']} lines={r['lines']} {r['seconds']}s errors={r['errors']} warns={r['warns']} {r['error_codes']}")
    print("\nBY CODE:")
    for k, v in summary["by_code"].items():
        print(f"  {v:3d}  {k}")
    try:
        import llm_client
        print("\n" + llm_client.cost_summary())
    except Exception:
        pass


if __name__ == "__main__":
    main()
