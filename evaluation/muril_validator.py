"""
evaluation/muril_validator.py  -  Layer 5: the VALIDATOR.

Checks that Pass-2 (slang) dialogue stays faithful to Pass-1 (neutral) AND that the slang actually helps, then emits a structured diagnosis
that the orchestrator's redo loop consumes.  Version 2 (this file) replaces the raw-cosine design, which could not discriminate anything:

  1. CENTRED MuRIL.  Raw MuRIL cosine is ~0.999 for any two similar-looking Hinglish lines (anisotropy).  Embeddings are now centred on the project's
     own corpus and their top principal directions removed (tools/calibrate_validator.py), so unrelated lines score ~0.2-0.5 and a faithful edit ~0.9.
  2. EDIT AUDIT (evaluation/edit_audit.py).  Every changed line is re-verified from the text alone: locality, names / numbers / negation / questions
     preserved, slang mood vs the replaced word, health / elder / "aap" lines, the deterministic rule gate.  A line that fails a BLOCKING check is
     REPAIRED, i.e. reverted to its neutral Pass-1 text (`validate_and_repair`): always safe, no model call, no retry loop.
  3. SEMANTIC EVENT COVERAGE.  Key events (English) are matched to dialogue lines (Hinglish) by centred MuRIL similarity instead of "any shared word".
     Event recall is REPORTED (and compared with the neutral text) but is not a re-do trigger: audited edits are small and local, a real editor-caused loss
     is implausible, the check is noisy, and events Pass 1 never contained are the Fidelity Critic's business (re-running the editor cannot fix them).
  4. Optional LM FLUENCY check (evaluation/fluency.py, config.VALIDATOR_FLUENCY).

Thresholds come from evaluation/eval_validator.py (data/validator/thresholds.json); without the calibration files the validator falls back to the old
raw-cosine behaviour and says so.

Public entry points (unchanged signatures):
  validate_scene(pass1, pass2, key_events)                      -> coarse scalar check (old callers, tests)
  diagnose_scene(neutral_scene, slang_scene, character_profiles) -> {passed, action, failure_mode, similarity, line_min_similarity, event_recall,
                                                                     slang_coverage, ..., bad_edits, per_line, ...}
  validate_and_repair(neutral_scene, slang_scene, profiles)      -> (repaired slang_scene, diagnosis)    <- what the pipeline uses
failure_mode: OK | NO_SLANG | BAD_EDIT | LOW_COVERAGE | MEANING_DRIFT | MONOTONOUS   (event recall is a reported diagnostic)
"""
from __future__ import annotations
import json
import os
import re
from collections import Counter

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

import config
from evaluation import edit_audit
from evaluation.compare import build_comparison

_STOPWORDS = {
    "the", "a", "an", "is", "was", "in", "of", "and", "to", "ka", "ki", "ke",
    "ko", "se", "par", "hai", "tha", "thi", "ne", "aur", "woh", "yeh", "ek",
    "mein", "hai.", "kar", "raha", "rahi", "rahe",
}
_SPEAKER = re.compile(r"^\s*[A-Za-z][A-Za-z_. ]*?\s*:\s*")


def _strip_speaker(line: str) -> str:
    return _SPEAKER.sub("", line or "").strip()


class MuRILValidator:
    def __init__(self, similarity_threshold: float = None,
                 event_recall_threshold: float = None, shared_muril=None):
        self.sim_threshold = (similarity_threshold if similarity_threshold is not None
                              else config.MURIL_SIMILARITY_THRESHOLD)
        self.event_threshold = (event_recall_threshold if event_recall_threshold is not None
                                else getattr(config, "EVENT_COVERAGE_RECALL_THRESHOLD",
                                             getattr(config, "EVENT_COVERAGE_F1_THRESHOLD", 0.5)))
        self.line_sim_threshold = getattr(config, "MURIL_LINE_SIM_THRESHOLD", 0.72)
        self.drift_threshold = getattr(config, "MURIL_MEANING_DRIFT_THRESHOLD", 0.78)
        self.min_slang_coverage = getattr(config, "MIN_SLANG_COVERAGE_PER_SCENE", 0.15)
        self.max_repetition_ratio = getattr(config, "MAX_SLANG_REPETITION_RATIO", 0.60)
        self.event_sim_min = 0.0
        self.fluency_delta_max = None

        if shared_muril is not None:
            self.tokenizer, self.model = shared_muril
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(config.MURIL_MODEL_PATH)
            self.model = AutoModel.from_pretrained(config.MURIL_MODEL_PATH)
        self.model.eval()
        self._center = self._load_centering()
        self._load_thresholds()
        self.fluency = None
        if getattr(config, "VALIDATOR_FLUENCY", False):
            try:
                from evaluation.fluency import FluencyScorer
                self.fluency = FluencyScorer()
            except Exception as e:                         # fluency is optional: never block the pipeline on it
                print(f"MuRILValidator: LM fluency check unavailable ({str(e)[:80]})")
        print(f"MuRILValidator v2 initialized ({'centred' if self._center else 'RAW (no calibration file: run tools.calibrate_validator)'} "
              f"similarity, edit audit{', LM fluency' if self.fluency else ''}).")

    # ── calibration files ──────────────────────────────────────────────
    def _load_centering(self):
        path = getattr(config, "VALIDATOR_CENTERING_PATH", "data/validator/muril_centering.npz")
        try:
            z = np.load(path)
            return z["mu"], z["V"]
        except Exception:
            return None

    def _load_thresholds(self):
        if not self._center:
            return
        path = getattr(config, "VALIDATOR_THRESHOLDS_PATH", "data/validator/thresholds.json")
        try:
            t = json.load(open(path, encoding="utf-8"))
        except Exception:
            t = {}
        # centred cosines live on a different scale from the raw ones the old defaults were set for
        self.line_sim_threshold = t.get("line_sim_min", 0.80)
        self.drift_threshold = t.get("line_sim_min", 0.80)
        self.event_sim_min = t.get("event_sim_min", 0.0)
        self.fluency_delta_max = t.get("fluency_delta_max")

    # ── embeddings ─────────────────────────────────────────────────────
    def _embed(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            text = "."
        cache = self.__dict__.setdefault("_embed_cache", {})
        if text in cache:                                     # unchanged lines are identical in Pass 1 and Pass 2: embed once
            return cache[text]
        cache[text] = v = self._embed_uncached(text)
        return v

    def _embed_uncached(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=256, padding=True)
        with torch.no_grad():
            out = self.model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        summed = (out.last_hidden_state * mask).sum(1)
        counts = mask.sum(1).clamp(min=1e-9)
        return (summed / counts)[0].numpy()

    def _project(self, v: np.ndarray) -> np.ndarray:
        if not self._center:
            return v
        mu, V = self._center
        x = v - mu
        return x - (V @ x) @ V

    @staticmethod
    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))

    def raw_similarity(self, text1: str, text2: str) -> float:
        """The old, uncentred cosine (kept for the comparison in eval_validator)."""
        return self._cos(self._embed(text1), self._embed(text2))

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Centred MuRIL cosine (raw cosine if no calibration file)."""
        return self._cos(self._project(self._embed(text1)), self._project(self._embed(text2)))

    # ── event coverage ─────────────────────────────────────────────────
    @staticmethod
    def _keyword_hit(dialogue: str, event: str) -> bool:
        dl = (dialogue or "").lower()
        words = {w for w in re.findall(r"[a-z]+", (event or "").lower()) if w not in _STOPWORDS}
        return bool(words) and any(w in dl for w in words)

    def event_similarity(self, dialogue: str, event: str) -> float:
        """Best centred similarity between an event description and any single dialogue line of the scene."""
        lines = [_strip_speaker(l) for l in (dialogue or "").split("\n") if l.strip()]
        if not lines:
            return 0.0
        e = self._project(self._embed(event))
        return max(self._cos(e, self._project(self._embed(l))) for l in lines)

    def _event_coverage(self, dialogue: str, events: list) -> dict:
        if not events:
            # Nothing to check against (e.g. Agent 1 was rate-limited and returned no events for this scene). Recall is undefined,
            # not 0 - reporting 0 made the redo loop fire EVENT_LOSS and re-run the slang editor for nothing.
            return {"recall": 1.0, "covered": 0, "total": 0, "method": "none"}
        if getattr(config, "VALIDATOR_JUDGE", False):          # LLM check: 86.9% accuracy vs 73.1% (centred MuRIL) and 64.4% (keywords) - docs/VALIDATOR.md
            try:
                from evaluation import judge
                r = judge.event_recall(dialogue, events)
                if r is not None:
                    return {"recall": r, "covered": round(r * len(events)), "total": len(events), "method": "llm"}
            except Exception:
                pass                                           # fail open to the offline check
        if self._center and self.event_sim_min > 0:
            covered = sum(1 for ev in events if self.event_similarity(dialogue, ev) >= self.event_sim_min)
            method = "semantic"
        else:
            covered = sum(1 for ev in events if self._keyword_hit(dialogue, ev))
            method = "keyword"
        return {"recall": covered / len(events), "covered": covered, "total": len(events), "method": method}

    @staticmethod
    def _count_eligible(decisions: list) -> int:
        """Lines that passed ALL gates and were not deliberately skipped (grateful line, failed upstream generation). Falls back to the old
        rule (everything except Gate-1 blocks) for traces written before decisions carried a "go" flag."""
        if not any("go" in d for d in decisions):
            return max(len(decisions) - sum(1 for d in decisions
                                            if d.get("gate_stopped") == 1), 0)
        return sum(1 for d in decisions
                   if d.get("go") and "skipped" not in (d.get("reason") or "").lower())

    # ── backward-compatible scalar check ───────────────────────────────
    def validate_scene(self, pass1_dialogue: str, pass2_dialogue: str, key_events: list) -> dict:
        """Original coarse check - kept so old callers (tests, __main__) still work."""
        sim = self.compute_similarity(pass1_dialogue, pass2_dialogue)
        cov = self._event_coverage(pass2_dialogue, key_events or [])
        thr = self.line_sim_threshold if self._center else self.sim_threshold
        passed = (sim >= thr) and (cov["recall"] >= self.event_threshold)
        return {"passed": passed, "similarity": round(sim, 4), "event_recall": round(cov["recall"], 4),
                "event_coverage": cov, "action": "accept" if passed else "rewrite"}

    # ── repair: revert unsafe edits to the neutral line ────────────────
    def repair(self, neutral_scene: dict, slang_scene: dict, profiles: dict = None):
        """-> (slang_scene with every blocking-flagged edit reverted to its neutral text, [the audits that were reverted])."""
        audits = edit_audit.audit_scene(neutral_scene, slang_scene, profiles or {})
        return edit_audit.repair_scene(neutral_scene, slang_scene, audits)

    def _judge_repair(self, neutral_scene: dict, slang_scene: dict, profiles: dict = None):
        """Optional second opinion (config.VALIDATOR_JUDGE): the two-family LLM judge (evaluation/judge.py) re-reads every edit that survived the audit and an edit it
        rejects is reverted. Fails OPEN: when no judge answered (no key, outage) nothing is reverted, so the pipeline never loses edits because of the network."""
        from evaluation import judge
        audits = edit_audit.audit_scene(neutral_scene, slang_scene, profiles or {})
        changed = [a for a in audits if a["changed"] and a["ok"]]
        if not changed:
            return slang_scene, []
        res = judge.judge_pairs([(a["neutral"], a["edited"]) for a in changed])
        for a, r in zip(changed, res):
            verdicts = [judge._accept(v) for v in r["models"].values() if v is not None]
            # rule "any" (default): revert only when EVERY judge rejects. Calibrated against a human rater (docs/EVALUATION.md): "all must accept" reverted ~40% of the
            # edits the person would keep (kappa 0.33); "revert only if all reject" agrees with the person at kappa 0.75 and keeps 97% of what they would keep.
            strict = str(getattr(config, "VALIDATOR_JUDGE_RULE", "any")).lower() == "all"
            if verdicts and ((not all(verdicts)) if strict else (not any(verdicts))):
                a["ok"] = False
                a["flags"] = list(a["flags"]) + [{"code": "judge-reject", "blocking": True, "detail": "; ".join(r["problems"])[:80]}]
        return edit_audit.repair_scene(neutral_scene, slang_scene, audits)

    def validate_and_repair(self, neutral_scene: dict, slang_scene: dict, profiles: dict = None):
        """The pipeline's entry: repair first, then diagnose the repaired scene. -> (slang_scene, diagnosis)."""
        fixed, reverted = self.repair(neutral_scene, slang_scene, profiles)
        if getattr(config, "VALIDATOR_JUDGE", False):
            fixed, rej = self._judge_repair(neutral_scene, fixed, profiles)
            reverted = reverted + rej
        diag = self.diagnose_scene(neutral_scene, fixed, profiles)
        # MEANING_DRIFT (a changed line far from its neutral line) is repaired like any other unsafe edit: revert THAT line to its neutral text, then look again.
        # Re-running the whole scene could not help (the Director's plan is cached and the editor is deterministic, so the same edit came back 3 times).
        for _ in range(len(neutral_scene.get("dialogue", "").split("\n")) + 1):
            if diag["failure_mode"] != "MEANING_DRIFT":
                break
            changed = [(r["similarity"], i) for i, r in enumerate(diag.get("per_line", []))
                       if (r.get("slang") or "").strip().lower() != (r.get("neutral") or "").strip().lower()]
            if not changed:
                break
            sim, idx = min(changed)                                  # the changed line that drifted most
            audits = edit_audit.audit_scene(neutral_scene, fixed, profiles)
            if idx >= len(audits) or not audits[idx]["changed"]:
                break
            audits[idx]["ok"] = False
            audits[idx]["flags"] = list(audits[idx]["flags"]) + [{"code": "meaning-drift", "blocking": True, "detail": f"centred similarity {sim:.2f}"}]
            fixed, rev = edit_audit.repair_scene(neutral_scene, fixed, audits)
            if not rev:
                break
            reverted = reverted + rev
            diag = self.diagnose_scene(neutral_scene, fixed, profiles)
        diag["reverted_edits"] = [{"speaker": a["speaker"], "neutral": a["neutral"], "edited": a["edited"], "slang": a["slang"],
                                   "why": [f["code"] + (":" + f["detail"] if f["detail"] else "") for f in a["flags"] if f["blocking"]]}
                                  for a in reverted]
        return fixed, diag

    # ── rich per-scene diagnosis (used by the redo loop) ───────────────
    def diagnose_scene(self, neutral_scene: dict, slang_scene: dict, character_profiles: dict = None) -> dict:
        """`neutral_scene`: {scene_id, dialogue, events}; `slang_scene`: {scene_id, dialogue, decisions, events}."""
        neutral_dialogue = neutral_scene.get("dialogue", "") or ""
        slang_dialogue = slang_scene.get("dialogue", "") or ""
        events = (neutral_scene.get("events") or {}).get("key_events", [])
        decisions = slang_scene.get("decisions", []) or []
        profiles = character_profiles or {}

        scene_sim = self.compute_similarity(neutral_dialogue, slang_dialogue)
        cov = self._event_coverage(slang_dialogue, events)
        cov_neutral = self._event_coverage(neutral_dialogue, events)

        paired = build_comparison([neutral_scene], [slang_scene])
        pair_lines = paired[0]["lines"] if paired else []
        audits = edit_audit.audit_scene(neutral_scene, slang_scene, profiles)

        per_line, line_sims = [], []
        for i, ln in enumerate(pair_lines):
            n_txt, s_txt = (ln.get("neutral") or "").strip(), (ln.get("slang") or "").strip()
            line_sim = 1.0 if (not n_txt or not s_txt or n_txt.lower() == s_txt.lower()) else self.compute_similarity(n_txt, s_txt)
            line_sims.append(line_sim)
            a = audits[i] if i < len(audits) else {}
            row = {"speaker": ln.get("speaker"), "neutral": n_txt, "slang": s_txt, "similarity": round(float(line_sim), 4),
                   "applied": bool(ln.get("slang_applied")), "slang_added": ln.get("slang_added"), "gate_stopped": ln.get("gate_stopped"),
                   "intent": ln.get("intent"),
                   "audit_ok": a.get("ok", True), "audit_flags": [f["code"] for f in a.get("flags", [])]}
            if self.fluency is not None and a.get("changed"):
                row["fluency_delta"] = round(self.fluency.delta(n_txt, s_txt), 3)
            per_line.append(row)
        line_min_sim = min(line_sims) if line_sims else 1.0
        bad_edits = [{"index": a["index"], "speaker": a["speaker"], "codes": [f["code"] for f in a["flags"] if f["blocking"]]}
                     for a in audits if a["changed"] and not a["ok"]]
        unnatural = [r for r in per_line if self.fluency_delta_max is not None and r.get("fluency_delta", -9) > self.fluency_delta_max]

        total_lines = len(decisions)
        applied = [d for d in decisions if d.get("applied")]
        eligible = self._count_eligible(decisions)
        slang_lines = len(applied)
        slang_coverage = (slang_lines / eligible) if eligible > 0 else 0.0
        applied_slangs = [d.get("slang") for d in applied if d.get("slang")]
        unique_slangs = len(set(s.lower() for s in applied_slangs))
        slang_variety = (unique_slangs / len(applied_slangs)) if applied_slangs else 1.0
        top_slang_share = (Counter(s.lower() for s in applied_slangs).most_common(1)[0][1] / len(applied_slangs)) if applied_slangs else 0.0
        tone_signal = self._tone_signal(audits)

        # Failure-mode classification, worst first.
        line_thr = self.line_sim_threshold
        drift_thr = self.drift_threshold
        # event recall vs the neutral text: a DIAGNOSTIC only (see the module docstring)
        event_lost = (cov["recall"] < cov_neutral["recall"] - 0.2) if cov["method"] in ("semantic", "llm") else (cov["recall"] < self.event_threshold)
        failure_mode, hint_parts = "OK", []
        if bad_edits:
            failure_mode = "BAD_EDIT"
            hint_parts.append(f"{len(bad_edits)} edit(s) failed the audit: " + "; ".join(f"{b['speaker']}:{'/'.join(b['codes'])}" for b in bad_edits[:4]))
        elif eligible > 0 and slang_lines == 0:
            failure_mode = "NO_SLANG"
            hint_parts.append("scene got 0 slang despite having eligible lines")
        elif scene_sim < drift_thr / 2:
            failure_mode = "MEANING_DRIFT"
            hint_parts.append(f"whole-scene similarity dropped to {scene_sim:.3f} (< half the calibrated {drift_thr:.2f})")
        elif line_min_sim < line_thr / 2:
            failure_mode = "MEANING_DRIFT"
            hint_parts.append(f"a line's similarity dropped to {line_min_sim:.3f} (< half the calibrated {line_thr:.2f})")
        # Between half the threshold and the threshold the cosine is a DIAGNOSTIC only: every changed line already passed the edit audit (no bad_edits here), and an
        # idiom swap ("mujhe bahut khushi ho rahi hai" -> "mere mann mein laddu phoot rahe hain") changes the words, hence the embedding, a lot while keeping the meaning.
        # Measured: on judged real edits the cosine is at chance (AUC 0.47-0.49) and adds 0.1 point of detection over the audit (97.9% vs 97.8%, docs/VALIDATOR.md);
        # a re-do cannot repair a correct edit (it re-picked the same idiom 3 times, +140 s per scene).
        elif unnatural:
            failure_mode = "BAD_EDIT"
            hint_parts.append(f"{len(unnatural)} edit(s) read unnaturally (LM fluency)")
        elif eligible > 0 and slang_coverage < self.min_slang_coverage:
            failure_mode = "LOW_COVERAGE"
            hint_parts.append(f"slang coverage {slang_coverage:.0%} (< {self.min_slang_coverage:.0%})")
        elif len(applied_slangs) >= 3 and top_slang_share > self.max_repetition_ratio:
            failure_mode = "MONOTONOUS"
            hint_parts.append(f"one slang token was reused {top_slang_share:.0%} of the time ({len(applied_slangs)} applied)")
        if event_lost:                      # diagnostic only: audited edits are small and local, so a real loss is implausible; the LLM / embedding check is noisy
            hint_parts.append(f"note: event recall {cov['recall']:.2f} vs {cov_neutral['recall']:.2f} before the edits (diagnostic, not a re-do trigger)")
        if failure_mode != "MEANING_DRIFT" and not bad_edits and (scene_sim < drift_thr or line_min_sim < line_thr):
            hint_parts.append(f"note: centred similarity {min(scene_sim, line_min_sim):.3f} is below the calibrated {line_thr:.2f}, but the audit cleared "
                              f"every changed line (diagnostic, not a re-do trigger)")
        if tone_signal != "OK":
            hint_parts.append(f"tone: {tone_signal.lower().replace('_', ' ')}")

        passed = failure_mode == "OK"
        return {
            "passed": passed, "action": "accept" if passed else "rewrite", "failure_mode": failure_mode,
            "similarity": round(float(scene_sim), 4), "line_min_similarity": round(float(line_min_sim), 4),
            "similarity_mode": "centred" if self._center else "raw",
            "event_recall": round(cov["recall"], 4), "event_recall_neutral": round(cov_neutral["recall"], 4), "event_coverage": cov,
            "slang_coverage": round(slang_coverage, 4), "slang_lines": slang_lines, "eligible_lines": eligible, "total_lines": total_lines,
            "slang_variety": round(slang_variety, 4), "top_slang_share": round(top_slang_share, 4),
            "tone_signal": tone_signal, "bad_edits": bad_edits,
            "hint": "; ".join(hint_parts) if hint_parts else "all checks passed", "per_line": per_line,
        }

    @staticmethod
    def _tone_signal(audits: list) -> str:
        """TOO_HEAVY when an applied edit carries a mood / register warning (crude slang, slang whose mood does not match the line)."""
        for a in audits:
            if a.get("changed") and any(f["code"] in ("crude", "tone-mismatch", "stacked-filler") for f in a.get("flags", [])):
                return "TOO_HEAVY"
        return "OK"


if __name__ == "__main__":
    v = MuRILValidator()
    neutral_scene = {"scene_id": "scene_01", "dialogue": "ARAV: Yeh bahut accha idea hai.\nPRIYA: Papa ki tabiyat kharab hai, bahut chinta ho rahi hai.",
                     "events": {"key_events": ["Arav likes the idea", "Priya worries about her father's health"]}}
    slang_scene = {"scene_id": "scene_01", "dialogue": "ARAV: Yeh zabardast idea hai.\nPRIYA: Papa ki tabiyat kharab hai, full tension ho rahi hai.",
                   "decisions": [{"speaker": "ARAV", "applied": True, "slang": "zabardast", "go": True},
                                 {"speaker": "PRIYA", "applied": True, "slang": "full tension", "go": True}],
                   "events": neutral_scene["events"]}
    fixed, diag = v.validate_and_repair(neutral_scene, slang_scene, {})
    print(fixed["dialogue"])
    print({k: diag[k] for k in ("failure_mode", "similarity", "similarity_mode", "slang_lines", "reverted_edits", "hint")})
