"""Controlled Agent 4 ablations on a frozen Pass-1 trace.

Every variant receives the same neutral dialogue, character profiles, and events.
The script writes one trace-compatible JSON per variant so each can be scored with
``python -m evaluation.evaluate``.

Variants:
  full                    editor + eligibility gates + acceptance rule gate + validation/redo
  no_eligibility_gates    bypass ThreeGateController only; other safety checks remain
    no_gate1               bypass speaker eligibility only
    no_gate2               bypass pragmatic intent eligibility only
    no_gate3               bypass scene density blocking only (still tracks token totals)
  no_acceptance_rule_gate bypass the deterministic edit_gate rule gate only; lexical
                          meaning/tone checks and repeat limits remain
  no_validation_redo      one editor pass; skip MuRIL validation and retries
  no_editor               neutral Pass-1 dialogue unchanged

This intentionally does not rerun story analysis or Pass-1 generation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv

load_dotenv()

import config

VARIANTS = (
    "full",
    "no_eligibility_gates",
    "no_gate1",
    "no_gate2",
    "no_gate3",
    "no_acceptance_rule_gate",
    "no_validation_redo",
    "no_editor",
)


def _with_scene_events(slang_script: list[dict], neutral_script: list[dict]) -> list[dict]:
    events_by_id = {s["scene_id"]: s.get("events", {}) for s in neutral_script}
    result = copy.deepcopy(slang_script)
    for scene in result:
        scene["events"] = copy.deepcopy(events_by_id.get(scene["scene_id"], {}))
    return result


@contextmanager
def _ablation_controls(editor: Any, variant: str) -> Iterator[None]:
    """Temporarily switch off exactly one named mechanism for an experiment arm."""
    gate = editor.gate
    gate_overrides = {
        name: gate.__dict__.get(name)
        for name in ("run_all_gates", "gate1_speaker_eligible",
                     "gate2_pragmatic_eligible", "gate3_density_ok")
    }
    rule_override = editor.__dict__.get("_judge_ok")
    original_rule_gate = editor._judge_ok
    try:
        if variant == "no_eligibility_gates":
            # Keep the editor's separate register/respectful-line checks intact.
            gate.run_all_gates = lambda *args, **kwargs: {
                "go": True,
                "gate_stopped": None,
                # The caller supplies Agent 4's slot-aware intent wrapper here;
                # retain that classification while removing only gate decisions.
                "intent": args[3].classify(args[1]),
                "reason": "ablation: eligibility gates bypassed",
            }
        elif variant == "no_gate1":
            gate.gate1_speaker_eligible = lambda speaker: True
        elif variant == "no_gate2":
            original_gate2 = gate.gate2_pragmatic_eligible

            def gate2_allow_any_intent(utterance, intent_classifier):
                _, intent = original_gate2(utterance, intent_classifier)
                return True, intent

            gate.gate2_pragmatic_eligible = gate2_allow_any_intent
        elif variant == "no_gate3":
            original_gate3 = gate.gate3_density_ok

            def gate3_track_but_allow(scene_id, utterance_token_count):
                # Preserve the running density counters; ablate only their blocking effect.
                original_gate3(scene_id, utterance_token_count)
                return True

            gate.gate3_density_ok = gate3_track_but_allow
        elif variant == "no_acceptance_rule_gate":
            # _vet_edit still checks for a newly inserted shortlist term, lexical drift,
            # and polarity; only the deterministic naturalness rule gate is ablated.
            editor._judge_ok = lambda neutral, edited, slang: (True, "")
        yield
    finally:
        for name, override in gate_overrides.items():
            if override is None:
                gate.__dict__.pop(name, None)
            else:
                setattr(gate, name, override)
        if rule_override is None:
            editor.__dict__.pop("_judge_ok", None)
        else:
            editor._judge_ok = rule_override


def _diagnose_all(validator: Any, neutral_script: list[dict],
                  slang_script: list[dict], profiles: dict) -> dict[str, dict]:
    slang_by_id = {s["scene_id"]: s for s in slang_script}
    return {
        n["scene_id"]: validator.diagnose_scene(
            n, slang_by_id[n["scene_id"]], profiles
        )
        for n in neutral_script
        if n["scene_id"] in slang_by_id
    }


def _retry_strategy(editor: Any, failure_mode: str, base_threshold: float) -> None:
    """Mirror the production pipeline's targeted retry settings for a failed scene."""
    gate = editor.gate
    gate.set_permissive(False)
    gate.density_threshold = base_threshold
    if failure_mode == "NO_SLANG":
        gate.set_permissive(True)
    elif failure_mode == "LOW_COVERAGE":
        gate.density_threshold = min(
            1.0, base_threshold * getattr(config, "DENSITY_MULTIPLIER_LOW_COVERAGE", 2.0)
        )
    elif failure_mode in ("MEANING_DRIFT", "EVENT_LOSS"):
        gate.density_threshold = max(
            0.05,
            base_threshold * getattr(config, "DENSITY_MULTIPLIER_MEANING_DRIFT", 0.5),
        )
    elif failure_mode == "MONOTONOUS":
        if hasattr(editor, "_used"):
            editor._used.clear()
        if hasattr(editor, "_last"):
            editor._last = None


def run_editor_variant(editor: Any, neutral_script: list[dict], profiles: dict,
                       variant: str, validator: Any = None,
                       max_attempts: int = 1) -> tuple[list[dict], dict[str, dict]]:
    """Run one controlled variant; returns (slang script, final diagnostics)."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; choose from {', '.join(VARIANTS)}")
    if variant == "no_editor":
        unchanged = []
        for scene in neutral_script:
            unchanged.append({
                "scene_id": scene["scene_id"],
                "dialogue": scene.get("dialogue", ""),
                "decisions": [],
                "events": copy.deepcopy(scene.get("events", {})),
            })
        return unchanged, {}

    editor.reset()
    # Reset editor shortlist randomness for a paired, repeatable start per arm.
    if hasattr(editor, "_rng"):
        editor._rng = random.Random(getattr(config, "SLANG_SEED", 7))

    needs_validation = variant != "no_validation_redo"
    if needs_validation and validator is None:
        raise ValueError(f"Variant {variant!r} requires a validator")

    max_attempts = max(1, int(max_attempts)) if needs_validation else 1
    with _ablation_controls(editor, variant):
        current = _with_scene_events(editor.rewrite_script(neutral_script), neutral_script)
        if not needs_validation:
            return current, {}

        diagnoses = _diagnose_all(validator, neutral_script, current, profiles)
        failed = {
            sid for sid, diag in diagnoses.items()
            if not diag.get("passed") and diag.get("failure_mode") != "NO_SLANG"
        }
        attempt = 1
        base_threshold = editor.gate.density_threshold

        while failed and attempt < max_attempts:
            previous = {s["scene_id"]: s for s in current}
            editor.reset()
            next_script = []
            for neutral in neutral_script:
                sid = neutral["scene_id"]
                if sid not in failed and sid in previous:
                    next_script.append(previous[sid])
                    continue

                mode = diagnoses.get(sid, {}).get("failure_mode", "OK")
                _retry_strategy(editor, mode, base_threshold)
                try:
                    rewritten = editor.rewrite_script([neutral])[0]
                finally:
                    editor.gate.set_permissive(False)
                    editor.gate.density_threshold = base_threshold
                rewritten["events"] = copy.deepcopy(neutral.get("events", {}))
                next_script.append(rewritten)

            current = next_script
            diagnoses = _diagnose_all(validator, neutral_script, current, profiles)
            failed = {
                sid for sid, diag in diagnoses.items()
                if not diag.get("passed") and diag.get("failure_mode") != "NO_SLANG"
            }
            attempt += 1

        return current, diagnoses


def _write_variant_trace(source: dict, neutral_script: list[dict],
                         slang_script: list[dict], variant: str,
                         diagnostics: dict[str, dict], destination: Path) -> None:
    out = copy.deepcopy(source)
    out["neutral_script"] = neutral_script
    out["slang_script"] = slang_script
    out["ablation"] = {
        "variant": variant,
        "description": {
            "full": "Agent 4 + eligibility gates + edit acceptance rule gate + MuRIL validation/redo",
            "no_eligibility_gates": "Bypass ThreeGateController eligibility gates only",
            "no_gate1": "Bypass Gate 1 speaker eligibility only",
            "no_gate2": "Bypass Gate 2 pragmatic intent eligibility only",
            "no_gate3": "Bypass Gate 3 density blocking only; continue tracking tokens",
            "no_acceptance_rule_gate": "Bypass deterministic edit_gate naturalness rule only",
            "no_validation_redo": "One Agent 4 pass; no MuRIL validation or retry",
            "no_editor": "Pass-1 dialogue unchanged",
        }[variant],
        "diagnostics": diagnostics,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", help="saved trace containing neutral_script and character_profiles")
    parser.add_argument("--out-dir", default="outputs/ablation",
                        help="directory for per-variant trace JSON files")
    parser.add_argument("--modes", default=",".join(VARIANTS),
                        help="comma-separated variants (default: all five)")
    parser.add_argument("--backend", choices=("auto", "qwen", "groq"), default="auto",
                        help="Agent 4 backend; keep fixed for all experimental arms")
    parser.add_argument("--max-attempts", type=int, default=None,
                        help="full-system validation attempts, including the first pass")
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    trace_path = Path(args.trace)
    source = json.loads(trace_path.read_text(encoding="utf-8"))
    neutral_script = source.get("neutral_script") or []
    if not neutral_script:
        parser.error("trace has no neutral_script; use a pipeline trace with saved Pass-1 dialogue")
    failed = [s.get("scene_id", "?") for s in neutral_script
              if "[generation failed" in s.get("dialogue", "").lower()]
    if failed:
        parser.error("Pass-1 generation failed in scene(s): " + ", ".join(failed))

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = sorted(set(modes) - set(VARIANTS))
    if unknown:
        parser.error("unknown variant(s): " + ", ".join(unknown))
    if not modes:
        parser.error("choose at least one variant with --modes")
    if "no_acceptance_rule_gate" in modes and getattr(config, "QWEN_GATE", "rules") != "rules":
        parser.error("no_acceptance_rule_gate requires QWEN_GATE=rules; set it consistently for every arm")

    profiles = source.get("character_profiles") or {}
    from memory.scratchpad import Scratchpad
    from agents.intent_classifier import GroqIntentClassifier
    from agents.agent4_slang_rewriter import SlangRewriter

    scratchpad = Scratchpad("ablation_" + trace_path.stem)
    scratchpad.write("character_profiles", profiles)
    editor = SlangRewriter(scratchpad, GroqIntentClassifier(), backend=args.backend)
    if editor.backend == "groq" and args.backend == "qwen":
        parser.error("requested Qwen backend could not be loaded; do not silently compare fallback output")

    validation_modes = [m for m in modes if m not in ("no_validation_redo", "no_editor")]
    validator = None
    if validation_modes:
        from evaluation.muril_validator import MuRILValidator
        validator = MuRILValidator()

    attempts = args.max_attempts or getattr(config, "MAX_REWRITE_ATTEMPTS", 3)
    output_dir = Path(args.out_dir)
    manifest = {
        "source_trace": str(trace_path),
        "source_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        "n_scenes": len(neutral_script),
        "backend": editor.backend,
        "adapter": getattr(config, "QWEN_LORA_PATH", None),
        "runtime_slang_source": getattr(config, "SLANG_RUNTIME_SRC", None),
        "acceptance_gate": getattr(config, "QWEN_GATE", "rules"),
        "qwen_judge_enabled": getattr(config, "QWEN_JUDGE", None),
        "groq_fallback_enabled": getattr(config, "QWEN_GROQ_FALLBACK", None),
        "seed": getattr(config, "SLANG_SEED", None),
        "max_validation_attempts": attempts,
        "variants": {},
        "warning": "Run every arm on identical neutral input; inspect API/model errors before using results.",
    }

    for variant in modes:
        print(f"\n=== Ablation: {variant} ===", flush=True)
        errors_before = int(editor.stats.get("groq_errors", 0))
        slang_script, diagnostics = run_editor_variant(
            editor, neutral_script, profiles, variant,
            validator=validator, max_attempts=attempts,
        )
        dest = output_dir / f"{trace_path.stem}_{variant}.json"
        _write_variant_trace(source, neutral_script, slang_script, variant, diagnostics, dest)
        edits = sum(1 for scene in slang_script for d in scene.get("decisions", []) if d.get("applied"))
        item = {
            "trace": str(dest),
            "applied_edits": edits,
            "validated_scenes": len(diagnostics),
            "failed_scenes_after_retries": [
                sid for sid, d in diagnostics.items()
                if not d.get("passed") and d.get("failure_mode") != "NO_SLANG"
            ],
            "editor_groq_errors": int(editor.stats.get("groq_errors", 0)) - errors_before,
            "valid_for_analysis": (int(editor.stats.get("groq_errors", 0)) - errors_before) == 0,
        }
        manifest["variants"][variant] = item
        print(f"Saved {dest} | edits={edits} | validated={len(diagnostics)} | "
              f"editor Groq errors={item['editor_groq_errors']}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{trace_path.stem}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nManifest: {manifest_path}")
    print("Score each variant separately with evaluation.evaluate; use the same --lexicon for every run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
