"""
evaluation/muril_validator.py
Layer 5 — MuRIL Validation agent  (Anika, Pending #3).

Checks that Pass-2 (slang) dialogue stays semantically faithful to Pass-1
(neutral) dialogue AND that the slang injection actually did something useful,
then emits a **structured, actionable diagnosis** the orchestrator's redo loop
consumes to fix the *specific* problem.

Two public entry points:

  * `validate_scene(pass1_dialogue, pass2_dialogue, key_events)`
        Backward-compatible scalar check (kept for older callers / tests).
        Returns {passed, similarity, event_recall, event_coverage, action}.

  * `diagnose_scene(neutral_scene, slang_scene, character_profiles=None)`
        Rich per-scene diagnosis used by the redo loop.
        Returns:
          {
            passed: bool,
            action: "accept" | "rewrite",
            failure_mode: "OK" | "NO_SLANG" | "LOW_COVERAGE"
                        | "MEANING_DRIFT" | "EVENT_LOSS" | "MONOTONOUS",
            similarity: float,           # whole-scene MuRIL cosine
            line_min_similarity: float,  # worst individual line
            event_recall: float,
            event_coverage: {covered, total, recall},
            slang_coverage: float,       # applied / (applied + gate_blocked)
            slang_lines: int,
            eligible_lines: int,
            total_lines: int,
            slang_variety: float,        # unique_slangs / applied_slangs
            tone_signal: str,            # "OK" | "TOO_SUBTLE" | "TOO_HEAVY"
            hint: str,                   # human-readable summary
            per_line: [ {speaker, neutral, slang, similarity, applied,
                         slang_added, gate_stopped} ],
          }

Only needs google/muril-base-cased (already cached locally).
"""
from __future__ import annotations
import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

import config
from evaluation.compare import build_comparison

_STOPWORDS = {
    "the", "a", "an", "is", "was", "in", "of", "and", "to", "ka", "ki", "ke",
    "ko", "se", "par", "hai", "tha", "thi", "ne", "aur", "woh", "yeh", "ek",
    "mein", "hai.", "kar", "raha", "rahi", "rahe",
}

# Emotions where slang should feel expressive (a nearly-identical rewrite is a
# missed opportunity). Kept small and local so we don't couple to agent 4.
_INTENSE_EMOTIONS = {"angry", "happy", "anxious", "excited", "urgent",
                     "frustrated", "sad"}


class MuRILValidator:
    def __init__(self, similarity_threshold: float = None,
                 event_recall_threshold: float = None, shared_muril=None):
        self.sim_threshold = (similarity_threshold if similarity_threshold is not None
                              else config.MURIL_SIMILARITY_THRESHOLD)
        self.event_threshold = (event_recall_threshold if event_recall_threshold is not None
                                else getattr(config, "EVENT_COVERAGE_RECALL_THRESHOLD",
                                             getattr(config, "EVENT_COVERAGE_F1_THRESHOLD", 0.5)))
        # Anika's stricter knobs (safe defaults if config didn't declare them)
        self.line_sim_threshold = getattr(config, "MURIL_LINE_SIM_THRESHOLD", 0.72)
        self.drift_threshold = getattr(config, "MURIL_MEANING_DRIFT_THRESHOLD", 0.78)
        self.min_slang_coverage = getattr(config, "MIN_SLANG_COVERAGE_PER_SCENE", 0.15)
        self.max_repetition_ratio = getattr(config, "MAX_SLANG_REPETITION_RATIO", 0.60)

        if shared_muril is not None:
            self.tokenizer, self.model = shared_muril
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(config.MURIL_MODEL_PATH)
            self.model = AutoModel.from_pretrained(config.MURIL_MODEL_PATH)
        self.model.eval()
        print("MuRILValidator initialized (google/muril-base-cased, strict mode).")

    # ── Embedding helpers ──────────────────────────────────────────────
    def _embed(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            text = "."
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=256, padding=True)
        with torch.no_grad():
            out = self.model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        summed = (out.last_hidden_state * mask).sum(1)
        counts = mask.sum(1).clamp(min=1e-9)
        return (summed / counts)[0].numpy()

    def compute_similarity(self, text1: str, text2: str) -> float:
        a, b = self._embed(text1), self._embed(text2)
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
        return float(np.dot(a, b) / denom)

    # ── Event-coverage (unchanged, kept for backward compat) ───────────
    def _event_coverage(self, dialogue: str, events: list) -> dict:
        dl = (dialogue or "").lower()
        covered = 0
        for ev in events or []:
            words = {w for w in re.findall(r"[a-z]+", ev.lower()) if w not in _STOPWORDS}
            if words and any(w in dl for w in words):
                covered += 1
        if not events:
            # Nothing to check against (e.g. Agent 1 was rate-limited and returned no events
            # for this scene). Recall is undefined, not 0 - reporting 0 made the redo loop
            # fire EVENT_LOSS and re-run the slang editor for nothing (retrying can't fix it).
            return {"recall": 1.0, "covered": 0, "total": 0}
        total = len(events)
        return {"recall": covered / total, "covered": covered, "total": total}

    @staticmethod
    def _count_eligible(decisions: list) -> int:
        """Lines that passed ALL gates and were not deliberately skipped (grateful line,
        failed upstream generation). Falls back to the old rule (everything except Gate-1
        blocks) for traces written before decisions carried a "go" flag."""
        if not any("go" in d for d in decisions):
            return max(len(decisions) - sum(1 for d in decisions
                                            if d.get("gate_stopped") == 1), 0)
        return sum(1 for d in decisions
                   if d.get("go") and "skipped" not in (d.get("reason") or "").lower())

    # ── Backward-compatible scalar check ───────────────────────────────
    def validate_scene(self, pass1_dialogue: str, pass2_dialogue: str,
                       key_events: list) -> dict:
        """Original coarse check — kept so old callers (tests, __main__) still work."""
        sim = self.compute_similarity(pass1_dialogue, pass2_dialogue)
        cov = self._event_coverage(pass2_dialogue, key_events or [])
        passed = (sim >= self.sim_threshold) and (cov["recall"] >= self.event_threshold)
        return {
            "passed": passed,
            "similarity": round(sim, 4),
            "event_recall": round(cov["recall"], 4),
            "event_coverage": cov,
            "action": "accept" if passed else "rewrite",
        }

    # ── Rich per-scene diagnosis (used by the redo loop) ───────────────
    def diagnose_scene(self, neutral_scene: dict, slang_scene: dict,
                       character_profiles: dict = None) -> dict:
        """
        Produce a structured diagnosis for one scene.

        `neutral_scene` : {scene_id, dialogue, events}
        `slang_scene`   : {scene_id, dialogue, decisions, events}
                          `decisions` comes from Agent 4 (per-line meta).
        `character_profiles` : optional; used for the tone-alignment signal.
        """
        neutral_dialogue = neutral_scene.get("dialogue", "") or ""
        slang_dialogue = slang_scene.get("dialogue", "") or ""
        events = (neutral_scene.get("events") or {}).get("key_events", [])
        decisions = slang_scene.get("decisions", []) or []
        profiles = character_profiles or {}

        # Whole-scene semantic similarity + event recall (same as before).
        scene_sim = self.compute_similarity(neutral_dialogue, slang_dialogue)
        cov = self._event_coverage(slang_dialogue, events)

        # Pair up lines using the compare helper (already reused across the codebase).
        paired = build_comparison([neutral_scene], [slang_scene])
        pair_lines = paired[0]["lines"] if paired else []

        # Per-line similarity — the "single corrupted line" catcher.
        per_line, line_sims = [], []
        for ln in pair_lines:
            n_txt = (ln.get("neutral") or "").strip()
            s_txt = (ln.get("slang") or "").strip()
            if not n_txt or not s_txt:
                line_sim = 1.0
            elif n_txt.lower() == s_txt.lower():
                line_sim = 1.0
            else:
                line_sim = self.compute_similarity(n_txt, s_txt)
            line_sims.append(line_sim)
            per_line.append({
                "speaker": ln.get("speaker"),
                "neutral": n_txt,
                "slang": s_txt,
                "similarity": round(float(line_sim), 4),
                "applied": bool(ln.get("slang_applied")),
                "slang_added": ln.get("slang_added"),
                "gate_stopped": ln.get("gate_stopped"),
                "intent": ln.get("intent"),
            })
        line_min_sim = min(line_sims) if line_sims else 1.0

        # Slang-coverage statistics from Agent 4's decisions.
        total_lines = len(decisions)
        applied = [d for d in decisions if d.get("applied")]
        # "Eligible" = lines that genuinely COULD take slang. Previously only Gate-1 blocks
        # were excluded, so a scene whose only peer line was blocked at Gate 2 (formal /
        # informational intent) or deliberately skipped (grateful line) still counted as
        # "eligible", got flagged NO_SLANG, and the redo loop then FORCED slang into a line
        # the gates had correctly protected - at ~30 s per line with the local editor.
        eligible = self._count_eligible(decisions)
        slang_lines = len(applied)
        slang_coverage = (slang_lines / eligible) if eligible > 0 else 0.0

        # Slang variety — catches "every line became 'yaar'".
        applied_slangs = [d.get("slang") for d in applied if d.get("slang")]
        unique_slangs = len(set(s.lower() for s in applied_slangs))
        slang_variety = (unique_slangs / len(applied_slangs)) if applied_slangs else 1.0
        top_slang_share = 0.0
        if applied_slangs:
            from collections import Counter
            counts = Counter(s.lower() for s in applied_slangs)
            top_slang_share = counts.most_common(1)[0][1] / len(applied_slangs)

        # Tone-alignment signal — a soft hint, not a hard fail on its own.
        tone_signal = self._tone_signal(per_line, profiles)

        # Failure-mode classification (ordered by severity: worst first).
        failure_mode = "OK"
        hint_parts = []

        if eligible > 0 and slang_lines == 0:
            failure_mode = "NO_SLANG"
            hint_parts.append("scene got 0 slang despite having eligible lines")

        elif scene_sim < self.drift_threshold:
            failure_mode = "MEANING_DRIFT"
            hint_parts.append(f"whole-scene similarity dropped to {scene_sim:.3f} "
                              f"(< {self.drift_threshold:.2f})")

        elif line_min_sim < self.line_sim_threshold:
            failure_mode = "MEANING_DRIFT"
            hint_parts.append(f"a line's similarity dropped to {line_min_sim:.3f} "
                              f"(< {self.line_sim_threshold:.2f})")

        elif cov["recall"] < self.event_threshold:
            failure_mode = "EVENT_LOSS"
            hint_parts.append(f"event recall {cov['recall']:.2f} "
                              f"(< {self.event_threshold:.2f})")

        elif eligible > 0 and slang_coverage < self.min_slang_coverage:
            failure_mode = "LOW_COVERAGE"
            hint_parts.append(f"slang coverage {slang_coverage:.0%} "
                              f"(< {self.min_slang_coverage:.0%})")

        elif (len(applied_slangs) >= 3
              and top_slang_share > self.max_repetition_ratio):
            # Only meaningful when at least 3 slangs were applied — with 1 or 2
            # applied slangs the "share" is always 50–100% by definition and
            # is not evidence of monotony.
            failure_mode = "MONOTONOUS"
            hint_parts.append(f"one slang token was reused "
                              f"{top_slang_share:.0%} of the time "
                              f"({len(applied_slangs)} applied)")

        # Tone signal is a soft add-on — attached to the hint even when passing.
        if tone_signal != "OK":
            hint_parts.append(f"tone: {tone_signal.lower().replace('_', ' ')}")

        passed = (failure_mode == "OK")
        action = "accept" if passed else "rewrite"
        hint = "; ".join(hint_parts) if hint_parts else "all checks passed"

        return {
            "passed": passed,
            "action": action,
            "failure_mode": failure_mode,
            "similarity": round(float(scene_sim), 4),
            "line_min_similarity": round(float(line_min_sim), 4),
            "event_recall": round(cov["recall"], 4),
            "event_coverage": cov,
            "slang_coverage": round(slang_coverage, 4),
            "slang_lines": slang_lines,
            "eligible_lines": eligible,
            "total_lines": total_lines,
            "slang_variety": round(slang_variety, 4),
            "top_slang_share": round(top_slang_share, 4),
            "tone_signal": tone_signal,
            "hint": hint,
            "per_line": per_line,
        }

    # ── Tone-alignment signal ──────────────────────────────────────────
    def _tone_signal(self, per_line: list, profiles: dict) -> str:
        """
        Coarse per-line check: for each applied-slang line, look at how
        different the slang line is from the neutral line.

          - very similar (>0.97) AND speaker's emotion is intense
                → the slang was too subtle to carry that emotion  → TOO_SUBTLE
          - very different (<0.70)
                → the slang overshot the meaning                  → TOO_HEAVY

        We pick the most severe signal in the scene; ties break TOO_HEAVY
        (meaning drift is worse than "not spicy enough").
        """
        if not per_line or not profiles:
            return "OK"

        from agents.name_utils import match_profile
        signal = "OK"
        subtle_hits = 0
        heavy_hits = 0

        for ln in per_line:
            if not ln.get("applied"):
                continue
            sim = ln.get("similarity", 1.0)
            speaker = ln.get("speaker", "") or ""
            profile = match_profile(profiles, speaker) or {}
            emotion = (profile.get("current_emotion") or "neutral").lower()

            if sim < 0.70:
                heavy_hits += 1
            elif sim > 0.97 and emotion in _INTENSE_EMOTIONS:
                subtle_hits += 1

        if heavy_hits > 0:
            signal = "TOO_HEAVY"
        elif subtle_hits >= 2:
            signal = "TOO_SUBTLE"
        return signal


if __name__ == "__main__":
    v = MuRILValidator()
    # 1) Old scalar API still works
    p1 = "ARAV: Mujhe ek purani diary mili hai library mein."
    p2 = "ARAV: Yaar mujhe ek purani diary mili hai library mein!"
    print("scalar:", v.validate_scene(p1, p2,
                                       ["Arav finds a diary", "in the library"]))

    # 2) New structured diagnosis
    neutral_scene = {
        "scene_id": "scene_01",
        "dialogue": "ARAV: Mujhe ek purani diary mili hai library mein.\n"
                    "ARAV: Iska mujhe aur pata karna hai.",
        "events": {"key_events": ["Arav finds a diary", "in the library"]},
    }
    slang_scene = {
        "scene_id": "scene_01",
        "dialogue": "ARAV: Yaar mujhe ek purani diary mili hai library mein!\n"
                    "ARAV: Iska mujhe aur pata karna hai.",
        "decisions": [
            {"speaker": "ARAV", "applied": True, "slang": "yaar",
             "intent": "ASSERT_POSITIVE", "gate_stopped": None},
            {"speaker": "ARAV", "applied": False, "slang": None,
             "intent": "INFORM_NEUTRAL", "gate_stopped": 2},
        ],
        "events": {"key_events": ["Arav finds a diary", "in the library"]},
    }
    profiles = {"Arav": {"current_emotion": "happy", "social_role": "peer",
                          "slang_level": 5}}
    diag = v.diagnose_scene(neutral_scene, slang_scene, profiles)
    print("diagnosis:")
    for k in ("passed", "failure_mode", "similarity", "line_min_similarity",
              "slang_coverage", "slang_variety", "tone_signal", "hint"):
        print(f"  {k}: {diag[k]}")
