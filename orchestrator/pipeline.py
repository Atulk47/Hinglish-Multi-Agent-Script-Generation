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
from agents.agent1_narrative import NarrativeAnalyst
from agents.agent2_character import CharacterAnalyst
from agents.agent3_synthesizer import DialogueSynthesizer
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
        self.agent1 = NarrativeAnalyst(llm_backend="groq")
        self.agent2 = CharacterAnalyst(self.scratchpad)          # loads NER + MuRIL
        self.agent3 = DialogueSynthesizer(llm_backend="groq")
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
        self._log("── [1/6] Preprocess: transliterate + chunk ──")
        paragraphs = [p.strip() for p in state["raw_hindi_text"].split("\n\n") if p.strip()]
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
        profiles = self.agent2.build_profiles(state["scene_chunks"], state["event_chain"])
        # advance emotion/stress state scene by scene
        for events in state["event_chain"]:
            for char in events.get("characters", []):
                self.agent2.update_state(char, events.get("key_events", []),
                                         events.get("scene_goal", ""))
        profiles = self.scratchpad.read("character_profiles") or profiles
        self._log(f"   profiles: {list(profiles.keys())}")
        state["character_profiles"] = profiles
        return state

    def _generate_pass1(self, state: PipelineState) -> PipelineState:
        self._log("── [4/6] Agent 3: Pass 1 neutral dialogue (Groq) ──")
        neutral = []
        for chunk, events in zip(state["scene_chunks"], state["event_chain"]):
            dialogue = self.agent3.generate_pass1(chunk, state["character_profiles"], events)
            # Agent 3 sometimes slips a Devanagari word into the dialogue (seen: "धन्यवाद, दादी!").
            # Everything downstream (register guard, editor, validator) expects Roman script, so
            # romanize any leak with the same rule-based transliterator used on the input.
            if re.search(r"[ऀ-ॿ]", dialogue):
                dialogue = re.sub(r"[ऀ-ॿ]+(?:[\sऀ-ॿ]*[ऀ-ॿ])?",
                                  lambda m: self.transliterator.transliterate_sentence(m.group(0)).strip(),
                                  dialogue)
            neutral.append({"scene_id": events["scene_id"], "dialogue": dialogue,
                            "events": events})
            time.sleep(0.3)
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
    def run(self, hindi_story: str) -> str:
        self._t_start = time.time()
        init: PipelineState = {
            "raw_hindi_text": hindi_story, "romanized_text": "", "scene_chunks": [],
            "event_chain": [], "character_profiles": {}, "neutral_script": [],
            "slang_script": [], "validation_scores": [], "failed_scenes": [],
            "rewrite_count": 0, "final_script": "",
        }
        init["scene_diagnoses"] = {}
        final_state = self.graph.invoke(init, {"recursion_limit": 25})
        return final_state["final_script"]
