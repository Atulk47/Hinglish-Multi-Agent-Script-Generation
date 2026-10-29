"""
orchestrator/pipeline.py
Layer 6 — LangGraph orchestration.

Wires the full pipeline as a stateful graph with one feedback edge:

  preprocess → plan_narrative → build_characters → generate_pass1
     → inject_slang → validate → (rewrite ↺ inject_slang | assemble) → END

"Groq-first" configuration (LLM model = config.GROQ_MODEL, currently openai/gpt-oss-20b;
llama-3.1-8b-instant was retired by Groq):
  Agent 1 (narrative)  : Groq (config.GROQ_MODEL)
  Agent 2 (character)  : NER (mBERT) + MuRIL
  Agent 3 (Pass 1)     : Groq (config.GROQ_MODEL)
  Agent 4 (Pass 2)     : Three-Gate controller + slang editor (mT5 full model,
                         else Groq fallback if the editor can't load/generate)
  Validation           : MuRIL cosine + event-coverage recall
"""
from __future__ import annotations
from typing import Literal
import re
import time

from langgraph.graph import StateGraph, END

import config
from memory.scratchpad import PipelineState, Scratchpad
from preprocessing.transliterate import HindiTransliterator
from preprocessing.chunker import ProseChunker
from preprocessing.segmenter import SceneSegmenter
from agents.agent1_narrative import NarrativeAnalyst
from agents.agent2_character import CharacterAnalyst
from agents.agent2b_persona import PersonaAgent, apply_persona
from agents.agent3_synthesizer import DialogueSynthesizer
from agents.agent3b_critic import FidelityCritic
from agents.agent4_slang_rewriter import SlangRewriter
from agents.intent_classifier import GroqIntentClassifier
from evaluation.muril_validator import MuRILValidator

MAX_REWRITE_ATTEMPTS = getattr(config, "MAX_REWRITE_ATTEMPTS", 3)


class HinglishPipeline:
    def __init__(self, session_id: str, verbose: bool = True):
        self.session_id = session_id
        self.verbose = verbose
        self.scratchpad = Scratchpad(session_id)

        t0 = time.time()
        print("Initializing pipeline components...")
        self.transliterator = HindiTransliterator()
        self.chunker = ProseChunker()
        self.segmenter = SceneSegmenter()                         # signals + pauses (voice input) + LLM referee
        self._pauses = None                                       # per sentence gap, set by run(...) for voice input
        self.agent1 = NarrativeAnalyst(llm_backend="groq")
        self.agent2 = CharacterAnalyst(self.scratchpad)          # loads NER + MuRIL
        self.persona_agent = PersonaAgent()                       # gender, relationships, per-scene emotion
        self.agent3 = DialogueSynthesizer(llm_backend="groq")
        self.critic = FidelityCritic()                            # checks Agent 3 against the author's scene + story bible
        self.intent_clf = GroqIntentClassifier()                  # Groq-first (no BART-MNLI)
        self.agent4 = SlangRewriter(self.scratchpad, self.intent_clf)
        # Reuse Agent 2's MuRIL weights for the validator (saves ~900MB RAM)
        self.validator = MuRILValidator(
            shared_muril=(self.agent2.muril_tokenizer, self.agent2.muril_model))
        print(f"All components ready in {time.time()-t0:.1f}s.\n")

        self.graph = self._build_graph()

    # ── Graph wiring ───────────────────────────────────────────────────
    def _build_graph(self):
        g = StateGraph(PipelineState)
        g.add_node("preprocess", self._preprocess)
        g.add_node("plan_narrative", self._plan_narrative)
        g.add_node("build_characters", self._build_characters)
        g.add_node("generate_pass1", self._generate_pass1)
        g.add_node("inject_slang", self._inject_slang)
        g.add_node("validate", self._validate)
        g.add_node("assemble_output", self._assemble_output)

        g.set_entry_point("preprocess")
        g.add_edge("preprocess", "plan_narrative")
        g.add_edge("plan_narrative", "build_characters")
        g.add_edge("build_characters", "generate_pass1")
        g.add_edge("generate_pass1", "inject_slang")
        g.add_edge("inject_slang", "validate")
        g.add_conditional_edges("validate", self._route_after_validate, {
            "rewrite": "inject_slang",
            "assemble": "assemble_output",
        })
        g.add_edge("assemble_output", END)
        return g.compile()

    def _log(self, *a):
        if self.verbose:
            # elapsed-time stamp on stage headers, so slow stages are visible in every log
            if a and isinstance(a[0], str) and a[0].startswith("──"):
                a = (f"[+{time.time() - getattr(self, '_t_start', time.time()):.0f}s] " + a[0],) + a[1:]
            print(*a)

    # ── Nodes ──────────────────────────────────────────────────────────
    def _preprocess(self, state: PipelineState) -> PipelineState:
        self._log("── [1/6] Preprocess: segment scenes + transliterate ──")
        try:
            seg = self.segmenter.segment(state["raw_hindi_text"], pauses=self._pauses)
            paragraphs = [s for s in seg.scenes if s.strip()]
            self.scratchpad.write("segmentation", {"method": seg.method, "starts": seg.starts, "sentences": len(seg.sentences),
                                                   "notes": seg.notes})
            self._log(f"   scene segmenter: {seg.method} -> {len(paragraphs)} scene(s) from {len(seg.sentences)} sentences"
                      + (f" ({seg.notes})" if seg.notes else ""))
            if not paragraphs:
                raise ValueError("segmenter returned no scenes")
        except Exception as e:                                    # never lose the run over segmentation: use the paragraphs
            self._log(f"   scene segmenter failed ({str(e)[:80]}); falling back to blank-line paragraphs")
            paragraphs = [p.strip() for p in state["raw_hindi_text"].split("\n\n") if p.strip()] or [state["raw_hindi_text"]]
        self.scratchpad.write("deva_scenes", paragraphs)
        romanized = "\n\n".join(
            self.transliterator.transliterate_sentence(p) for p in paragraphs)
        chunks = self.chunker.chunk(romanized)
        self.scratchpad.write("romanized_text", romanized)
        self.scratchpad.write("scene_chunks", chunks)
        self._log(f"   {len(chunks)} scene(s) detected.")
        state["romanized_text"] = romanized
        state["scene_chunks"] = chunks
        return state

    def _plan_narrative(self, state: PipelineState) -> PipelineState:
        self._log("── [2/6] Agent 1: narrative event chain (Groq) ──")
        chain = self.agent1.extract_event_chain(state["scene_chunks"])
        self.scratchpad.write("event_chain", chain)
        for e in chain:
            self._log(f"   {e['scene_id']}: chars={e.get('characters')} "
                      f"| {e.get('scene_goal','')[:50]}")
        state["event_chain"] = chain
        return state

    def _build_characters(self, state: PipelineState) -> PipelineState:
        self._log("── [3/6] Agent 2: character profiles (NER + MuRIL) ──")
        # Persona Agent: gender, age group, relationships, per-scene emotions, story bible. Its LLM call and Agent 2's NER are independent, so they run
        # at the same time.
        paragraphs = self.scratchpad.read("deva_scenes") or [p.strip() for p in state["raw_hindi_text"].split("\n\n") if p.strip()]
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(self.persona_agent.analyze, state["scene_chunks"], state["event_chain"], [], paragraphs)
            profiles = self.agent2.build_profiles(state["scene_chunks"], state["event_chain"])
            persona = fut.result()
        # drop ghost names the LLM saw in Agent 1's list but Agent 2's name checks rejected (e.g. a misread word)
        from agents.name_utils import same_person
        keep = lambda names: [n for n in names if any(same_person(n, p) for p in profiles)]
        for sc in (persona.get("scene_cast") or {}).values():
            sc["present"], sc["mentioned_only"] = keep(sc.get("present", [])), keep(sc.get("mentioned_only", []))
        scene_ids = [c["scene_id"] for c in state["scene_chunks"]]
        profiles = apply_persona(profiles, persona, scene_ids)
        self.scratchpad.write("character_profiles", profiles)
        self.scratchpad.write("persona", persona)
        self._log(f"   persona source: {persona.get('source')} | checks: {persona.get('checks')}")
        for n, p in profiles.items():
            arc = " > ".join(f"{e['emotion']}{e['intensity']}" for e in p.get("emotion_arc", []))
            self._log(f"   {n:12s} {p.get('gender','?'):7s} {p.get('social_role','?'):9s} arc: {arc}")
        self._log(f"   profiles: {list(profiles.keys())}")
        state["character_profiles"] = profiles
        return state

    def _prefetch_intents(self, dialogue: str):
        """Classify every line's intent NOW (concurrently) so the slang editor, which needs it for each line, finds it already cached."""
        clf = getattr(self.agent4, "intent_clf", None)
        utts = [ln.split(":", 1)[1].strip() for ln in dialogue.split("\n") if ":" in ln and ln.split(":", 1)[1].strip()]
        if clf is None or not utts:
            return
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(8, len(utts))) as ex:
            list(ex.map(lambda u: clf.classify(u), utts))

    def _edit_scene(self, scene):
        out = self.agent4.rewrite_script([{"scene_id": scene["scene_id"], "dialogue": scene["dialogue"]}])[0]
        out["events"] = scene["events"]
        return out

    def _scene_pass1(self, chunk, events, profiles, persona, deva_text):
        """One scene: write the dialogue from the story bible, have the Fidelity Critic check it against the author's scene, and regenerate
        with the reviewer's feedback while there are clear problems. Keeps the attempt with the fewest problems."""
        sid = events["scene_id"]
        ev = dict(events)
        cast = (persona or {}).get("scene_cast", {}).get(sid) or {}
        ev["scene_cast"] = cast
        ev["secrets"] = [s for s in (persona or {}).get("secrets", []) if sid in s.get("scenes", [sid])]
        ch = dict(chunk)
        ch["deva_text"] = deva_text
        present = list(cast.get("present") or events.get("characters") or [])
        absent, extras = list(cast.get("mentioned_only") or []), list(cast.get("extra_speakers") or [])
        use_critic = bool(getattr(config, "AGENT3_CRITIC", True)) and self.critic is not None
        tries = 1 + (int(getattr(config, "AGENT3_FIX_ATTEMPTS", 2)) if use_critic else 0)
        best, history, feedback = None, [], None
        for attempt in range(tries):
            ev["feedback"] = feedback
            dialogue = self.agent3.generate_pass1(ch, profiles, ev)
            # Agent 3 sometimes slips a Devanagari word into the dialogue (seen: "धन्यवाद, दादी!").
            # Everything downstream (register guard, editor, validator) expects Roman script, so
            # romanize any leak with the same rule-based transliterator used on the input.
            if re.search(r"[ऀ-ॿ]", dialogue):
                dialogue = re.sub(r"[ऀ-ॿ]+(?:[\sऀ-ॿ]*[ऀ-ॿ])?",
                                  lambda m: self.transliterator.transliterate_sentence(m.group(0)).strip(), dialogue)
            if not use_critic or "[generation failed" in dialogue:
                best = (dialogue, {"ok": True, "issues": [], "llm_used": False})
                history.append(0)
                break
            rep = self.critic.check(deva_text or chunk.get("text", ""), dialogue, present, absent, ev["secrets"],
                                    events.get("key_events"), extras)
            history.append(len(rep["issues"]))
            if best is None or len(rep["issues"]) < len(best[1]["issues"]):
                best = (dialogue, rep)
            if rep["ok"]:
                break
            feedback = "; ".join(f"line {i['line']}: {i['type']} - {i['detail']}" for i in rep["issues"][:6])
        dialogue, rep = best
        dialogue = re.sub(r"\n\s*\n", "\n", dialogue).strip()          # the model sometimes double-spaces the turns
        dialogue = "\n".join(re.sub(r"\s*\([^)]*\)\s*", " ", ln).replace(" :", ":").strip() if ":" in ln else ln for ln in dialogue.split("\n"))   # no (stage directions)
        removed = 0
        if use_critic and any(i["type"] == "absent_speaker" for i in rep["issues"]):
            cleaned = self.critic.drop_absent_lines(dialogue, present, extras)
            removed = len(dialogue.split("\n")) - len(cleaned.split("\n"))
            dialogue = cleaned
        fid = {"attempts": len(history), "issues_per_attempt": history, "final_issues": rep["issues"], "lines_removed": removed}
        try:
            self._prefetch_intents(dialogue)
        except Exception:
            pass
        return {"scene_id": sid, "dialogue": dialogue, "events": events, "fidelity": fid}

    def _generate_pass1(self, state: PipelineState) -> PipelineState:
        self._log("── [4/6] Agent 3: Pass 1 neutral dialogue + fidelity check ──")
        persona = self.scratchpad.read("persona") or {}
        deva = self.scratchpad.read("deva_scenes") or []
        jobs = list(enumerate(zip(state["scene_chunks"], state["event_chain"])))
        workers = max(1, int(getattr(config, "AGENT3_PARALLEL", 4)))
        results = {}
        if workers == 1:
            for i, (chunk, events) in jobs:
                results[i] = self._scene_pass1(chunk, events, state["character_profiles"], persona, deva[i] if i < len(deva) else "")
        else:
            from concurrent.futures import ThreadPoolExecutor
            overlap = bool(getattr(config, "PIPELINE_OVERLAP", True))
            self._prefetched = {}
            if overlap:
                self.agent4.reset()                       # fresh density budget; the first editing attempt happens while Pass 1 is still being written
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {i: ex.submit(self._scene_pass1, chunk, events, state["character_profiles"], persona, deva[i] if i < len(deva) else "")
                        for i, (chunk, events) in jobs}
                for i, _ in jobs:                         # in story order: as soon as scene i is written, the editor starts on it
                    results[i] = futs[i].result()
                    if overlap:
                        try:
                            self._prefetched[results[i]["scene_id"]] = self._edit_scene(results[i])
                        except Exception as e:            # never lose the run: the normal editing stage will redo this scene
                            self._log(f"   (overlapped editing of {results[i]['scene_id']} failed: {str(e)[:60]})")
                            self._prefetched = {}
                            overlap = False
        neutral = [results[i] for i, _ in jobs]
        for n in neutral:
            f = n["fidelity"]
            self._log(f"   {n['scene_id']}: attempts {f['attempts']} issues {f['issues_per_attempt']}"
                      + (f" | removed {f['lines_removed']} absent-speaker line(s)" if f["lines_removed"] else "")
                      + (f" | remaining: {[i['type'] for i in f['final_issues']]}" if f["final_issues"] else ""))
        self.scratchpad.write("neutral_script", neutral)
        state["neutral_script"] = neutral
        # Fail LOUDLY if upstream LLM calls failed (almost always a Groq rate/daily-token cap).
        # Otherwise the run completes and looks plausible but the screenplay is invalid.
        failed_p1 = [n["scene_id"] for n in neutral if "[generation failed" in n["dialogue"]]
        empty_ev = [e["scene_id"] for e in state["event_chain"] if not e.get("key_events")]
        if failed_p1 or empty_ev:
            print("\n" + "!" * 68)
            if failed_p1:
                print(f"  WARNING: Pass-1 dialogue FAILED for {len(failed_p1)} scene(s): {failed_p1}")
            if empty_ev:
                print(f"  WARNING: Agent 1 returned NO events for {len(empty_ev)} scene(s): {empty_ev}")
            print("  Likely a Groq rate / daily-token limit. These results are NOT valid.")
            print("  Retry with the other model pool, e.g. PowerShell:")
            print('      $env:GROQ_MODEL="openai/gpt-oss-120b"')
            print("!" * 68 + "\n")
        return state

    def _inject_slang(self, state: PipelineState) -> PipelineState:
        attempt = state.get("rewrite_count", 0) + 1
        self._log(f"── [5/6] Agent 4: Pass 2 slang injection (attempt {attempt}) ──")
        pre = getattr(self, "_prefetched", None)
        if attempt == 1 and pre and all(s["scene_id"] in pre for s in state["neutral_script"]):
            slang_script = [pre[s["scene_id"]] for s in state["neutral_script"]]
            self._prefetched = None
            self._log("   (editing was overlapped with Pass 1 - done)")
            self.scratchpad.write("slang_script", slang_script)
            state["slang_script"] = slang_script
            return state
        failed = set(state.get("failed_scenes", []))
        prev = {s["scene_id"]: s for s in (state.get("slang_script") or [])}

        # Anika: on retry, use each failed scene's diagnosis to pick a
        # *targeted* fix strategy instead of blindly re-running Agent 4.
        diagnoses = state.get("scene_diagnoses") or {}
        base_threshold = self.agent4.gate.density_threshold
        low_mult = getattr(config, "DENSITY_MULTIPLIER_LOW_COVERAGE", 2.0)
        drift_mult = getattr(config, "DENSITY_MULTIPLIER_MEANING_DRIFT", 0.5)

        self.agent4.reset()  # fresh density budget for this pass
        slang_script = []
        for scene in state["neutral_script"]:
            sid = scene["scene_id"]
            # carry over already-passing scenes on a retry
            if prev and sid not in failed and sid in prev:
                slang_script.append(prev[sid])
                continue

            mode = (diagnoses.get(sid, {}) or {}).get("failure_mode", "OK")
            self._apply_retry_strategy(mode, base_threshold, low_mult, drift_mult)
            if mode != "OK":
                self._log(f"     -> retry {sid} in mode={mode} "
                          f"(density_threshold={self.agent4.gate.density_threshold:.2f}, "
                          f"permissive={self.agent4.gate._permissive})")
            try:
                rewritten = self.agent4.rewrite_script([{"scene_id": sid,
                                                         "dialogue": scene["dialogue"]}])[0]
            finally:
                self.agent4.gate.set_permissive(False)
                self.agent4.gate.density_threshold = base_threshold
            rewritten["events"] = scene["events"]
            slang_script.append(rewritten)

        self.scratchpad.write("slang_script", slang_script)
        state["slang_script"] = slang_script
        return state

    def _apply_retry_strategy(self, mode: str, base_threshold: float,
                              low_mult: float, drift_mult: float) -> None:
        """Configure the shared gate for one scene's retry, based on why it failed.

          NO_SLANG      -> permissive mode: force something through gates 2+3
          LOW_COVERAGE  -> loosen density budget (multiply threshold)
          MEANING_DRIFT -> tighten density budget (fewer, safer insertions)
          EVENT_LOSS    -> tighten density (Agent 4 changed too much text)
          MONOTONOUS    -> reset variety counters so Agent 4 re-diversifies
          OK / default  -> leave gates at defaults
        """
        gate = self.agent4.gate
        gate.set_permissive(False)
        gate.density_threshold = base_threshold

        if mode == "NO_SLANG":
            gate.set_permissive(True)
        elif mode == "LOW_COVERAGE":
            gate.density_threshold = min(1.0, base_threshold * low_mult)
        elif mode in ("MEANING_DRIFT", "EVENT_LOSS"):
            gate.density_threshold = max(0.05, base_threshold * drift_mult)
        elif mode == "MONOTONOUS":
            # variety only: keep gates as-is but wipe Agent 4's slang history
            # so repeat-avoidance kicks in from a clean slate for this scene.
            if hasattr(self.agent4, "_used"):
                self.agent4._used.clear()
            if hasattr(self.agent4, "_last"):
                self.agent4._last = None

    def _validate(self, state: PipelineState) -> PipelineState:
        self._log("── [6/6] MuRIL validation (strict) ──")
        failed, scores, diagnoses = [], [], {}
        profiles = state.get("character_profiles") or {}
        for neutral, slang in zip(state["neutral_script"], state["slang_script"]):
            sid = slang["scene_id"]
            diag = self.validator.diagnose_scene(neutral, slang, profiles)
            diagnoses[sid] = diag
            scores.append(diag["similarity"])
            flag = ("PASS" if diag["passed"]
                    else "PASS (neutral by design)" if diag.get("failure_mode") == "NO_SLANG"
                    else f"RE-DO[{diag['failure_mode']}]")
            self._log(f"   {sid}: sim={diag['similarity']:.3f} "
                      f"line_min={diag['line_min_similarity']:.3f} "
                      f"recall={diag['event_recall']:.2f} "
                      f"cov={diag['slang_coverage']:.0%} "
                      f"var={diag['slang_variety']:.2f} -> {flag}")
            if not diag["passed"] and diag.get("failure_mode") == "NO_SLANG":
                # The editor found nothing that fits this scene. Forcing slang here is what produced
                # "fire" on a fearful line, so a scene that stays neutral is ACCEPTED, not re-done.
                self._log("       kept neutral by design (no fitting slang) - not re-done")
            elif not diag["passed"]:
                self._log(f"       reason: {diag['hint']}")
                failed.append(sid)

        state["validation_scores"] = scores
        state["failed_scenes"] = failed
        state["scene_diagnoses"] = diagnoses
        state["rewrite_count"] = state.get("rewrite_count", 0) + 1
        # persist for post-run inspection / evaluation harness
        try:
            self.scratchpad.write("scene_diagnoses", diagnoses)
        except Exception:
            pass
        return state

    def _route_after_validate(self, state: PipelineState) -> Literal["rewrite", "assemble"]:
        if not state["failed_scenes"]:
            return "assemble"
        if state["rewrite_count"] >= MAX_REWRITE_ATTEMPTS:
            self._log(f"   max attempts ({MAX_REWRITE_ATTEMPTS}) reached — accepting best effort.")
            return "assemble"
        modes = [
            (state.get("scene_diagnoses") or {}).get(sid, {}).get("failure_mode", "?")
            for sid in state["failed_scenes"]
        ]
        self._log(f"   {len(state['failed_scenes'])} scene(s) failed "
                  f"({', '.join(modes)}) — re-injecting with targeted fixes.")
        return "rewrite"

    def _assemble_output(self, state: PipelineState) -> PipelineState:
        lines = ["# HINGLISH SCREENPLAY", ""]
        for scene in state["slang_script"]:
            ev = scene.get("events", {})
            lines.append(f"## {scene['scene_id'].upper()}")
            lines.append(f"*Location: {ev.get('location', 'unknown')}*")
            lines.append("")
            lines.append(scene["dialogue"])
            lines.append("\n---\n")
        final = "\n".join(lines)
        self.scratchpad.write("final_script", final)
        state["final_script"] = final
        return state

    # ── Public entry ───────────────────────────────────────────────────
    def run(self, hindi_story: str, pauses=None) -> str:
        """hindi_story: Devanagari text (typed, or the transcript of a spoken story). pauses: seconds of silence after each sentence,
        from audio.asr.transcribe(...)['pauses'] when the input was a recording - the scene segmenter uses them as evidence."""
        self._pauses = pauses
        self._t_start = time.time()
        init: PipelineState = {
            "raw_hindi_text": hindi_story, "romanized_text": "", "scene_chunks": [],
            "event_chain": [], "character_profiles": {}, "neutral_script": [],
            "slang_script": [], "validation_scores": [], "failed_scenes": [],
            "rewrite_count": 0, "final_script": "",
        }
        init["scene_diagnoses"] = {}
        final_state = self.graph.invoke(init, {"recursion_limit": 25})
        try:
            import llm_client
            self._log("\n" + llm_client.cost_summary())
        except Exception:
            pass
        return final_state["final_script"]
