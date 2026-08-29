# Anika's Part — Summary

_Owner: Anika  ·  Area: Validation & Quality Gate + Redo Loop_

This document summarises exactly what was implemented for Anika's tasks from
`PROJECT_STATUS.md` §5–§6 (Pending items #3 and #4). No teammates' files were
modified beyond the shared coordination touch-points already agreed in the
project plan.

---

## 1. What was pending (before this change)

From `PROJECT_STATUS.md`:

- **Pending #3 — Make validation stricter/smarter.** The old
  `MuRILValidator.validate_scene()` compared the *whole scene* Pass-1 vs Pass-2
  with one cosine similarity plus a keyword event-coverage check. In practice
  it almost never rejected anything: adding a single "yaar" to a long scene
  barely moves whole-scene similarity, so bad slang went through and scenes
  that got **zero** slang trivially "passed" (identical text ⇒ similarity 1.0).
- **Pending #4 — Make the redo loop actually improve scenes.** The old
  `_route_after_validate` sent failed scenes back to `_inject_slang` with the
  same settings, so the retry was just blind repetition.

---

## 2. What was added

### 2.1 `evaluation/muril_validator.py` — a richer, stricter validator

- Kept `validate_scene(pass1, pass2, key_events)` **fully backward-compatible**
  so existing tests and the `__main__` demo still work.
- Added a new entry point:

  ```python
  MuRILValidator.diagnose_scene(neutral_scene, slang_scene,
                                character_profiles=None) -> dict
  ```

  which returns a structured diagnosis with a `failure_mode` field that the
  redo loop consumes. Fields include:

  | field                 | meaning                                                  |
  |-----------------------|----------------------------------------------------------|
  | `passed`              | overall accept/reject                                    |
  | `failure_mode`        | one of `OK`, `NO_SLANG`, `LOW_COVERAGE`, `MEANING_DRIFT`, `EVENT_LOSS`, `MONOTONOUS` |
  | `similarity`          | whole-scene MuRIL cosine (as before)                     |
  | `line_min_similarity` | **new** — worst single line, catches one corrupted line  |
  | `event_recall`        | keyword coverage (as before)                             |
  | `slang_coverage`      | **new** — `applied / gate-eligible lines`                |
  | `slang_variety`       | **new** — `unique_slangs / applied_slangs`               |
  | `top_slang_share`     | **new** — share of the most-used slang token             |
  | `tone_signal`         | **new** — `OK` / `TOO_SUBTLE` / `TOO_HEAVY`              |
  | `hint`                | human-readable reason                                    |
  | `per_line`            | full per-line breakdown (used for offline inspection)    |

- Reuses `evaluation/compare.build_comparison(...)` to pair neutral/slang
  lines. No duplication of parsing logic.
- Uses `agents.name_utils.match_profile` for tolerant speaker→profile lookup
  in the tone-alignment check.

**Failure-mode classification order (worst first):**

1. `NO_SLANG` — scene had eligible lines but Agent 4 produced zero slang.
2. `MEANING_DRIFT` — whole-scene similarity below
   `MURIL_MEANING_DRIFT_THRESHOLD` **or** any single line drops below
   `MURIL_LINE_SIM_THRESHOLD`.
3. `EVENT_LOSS` — key events dropped below `EVENT_COVERAGE_RECALL_THRESHOLD`.
4. `LOW_COVERAGE` — slang coverage below `MIN_SLANG_COVERAGE_PER_SCENE`.
5. `MONOTONOUS` — one slang token used more than
   `MAX_SLANG_REPETITION_RATIO` of the time. Requires **≥3 applied slangs**
   in the scene (with 1–2 slangs the "share" is 50–100% by construction and
   is not real evidence of monotony).
6. `OK` — accept.

The tone signal (`TOO_SUBTLE` / `TOO_HEAVY`) is attached to the hint even when
the scene passes; it doesn't override the failure mode by itself.

### 2.2 `gates/three_gate_controller.py` — permissive-mode toggle

Purely additive. Existing gate 1 / 2 / 3 logic is untouched.

- New attribute `_permissive` (default `False`).
- New method `set_permissive(bool)`.
- Gate 2 and Gate 3 short-circuit to "pass" when permissive is on. Gate 1
  (elders / authority) still applies in permissive mode.

This exists **only** so the redo loop can force slang into a scene that got
zero on the first pass, without diluting normal-run quality.

### 2.3 `orchestrator/pipeline.py` — targeted redo strategy

Only three methods changed; the graph, other nodes, and Anish's carry-over
logic are preserved.

- `_validate` now calls `diagnose_scene(...)` and stores a
  `state["scene_diagnoses"] = {scene_id: diagnosis}` dict, plus a much richer
  per-scene log line. Diagnoses are also persisted to the scratchpad for
  post-run inspection (used by the evaluation harness / demo).
- `_route_after_validate` logs which failure modes are being retried.
- `_inject_slang` reads the per-scene diagnosis and dispatches to a
  targeted-fix strategy (`_apply_retry_strategy`):

  | failure mode      | what the retry does                                                                 |
  |-------------------|--------------------------------------------------------------------------------------|
  | `NO_SLANG`        | `gate.set_permissive(True)` → gates 2 + 3 always pass for this scene                 |
  | `LOW_COVERAGE`    | multiply density threshold by `DENSITY_MULTIPLIER_LOW_COVERAGE` (default 2.0)        |
  | `MEANING_DRIFT`   | multiply density threshold by `DENSITY_MULTIPLIER_MEANING_DRIFT` (default 0.5)       |
  | `EVENT_LOSS`      | same tightening as `MEANING_DRIFT`                                                   |
  | `MONOTONOUS`      | wipe `agent4._used` / `agent4._last` so variety weighting re-diversifies             |
  | `OK` (carry-over) | scene is skipped on retry (already-passing scenes are reused, per Anish's design)    |

  The threshold and permissive flag are always restored in a `finally` block
  before moving to the next scene — no state leaks between scenes.

### 2.4 `config.py` — new tunable thresholds (appended, existing values untouched)

```python
MURIL_LINE_SIM_THRESHOLD          = 0.72   # worst-line floor
MURIL_MEANING_DRIFT_THRESHOLD     = 0.78   # hard drift line
MIN_SLANG_COVERAGE_PER_SCENE      = 0.15   # 15% of eligible lines
MAX_SLANG_REPETITION_RATIO        = 0.60   # one word > 60% ⇒ monotonous
DENSITY_MULTIPLIER_LOW_COVERAGE   = 2.0
DENSITY_MULTIPLIER_MEANING_DRIFT  = 0.5
```

All are read via `getattr(config, ..., default)` in the validator so an older
`config.py` without them still works.

---

## 3. What was intentionally NOT touched

- `agents/*.py` — Anish's / Dia's territory. The redo loop influences Agent 4
  only through **its own public gate object** (`self.agent4.gate`) and its
  public `reset()` / internal `_used` / `_last` counters (which Anish already
  exposes on the class).
- `notebooks/*` and `data_prep/*` — Atul's / Dia's territory.
- `evaluation/metrics.py`, `evaluation/evaluate.py`, `evaluation/compare.py`
  — Atul's evaluation harness / shared helpers. `compare.build_comparison` is
  *used* by the validator, never modified.
- `memory/scratchpad.py` — the `PipelineState` TypedDict is left as-is. The
  new `scene_diagnoses` key is added at runtime (Python `TypedDict` is not
  runtime-enforced) so no cross-owner schema change was required.
- The existing 3-gate logic in `three_gate_controller.py`. Only a small,
  strictly-additive helper (`set_permissive`) was introduced.

---

## 4. How to check that it works

1. Ensure the corrupted NER cache is cleared (see chat) and `.env` has
   `GROQ_API_KEY=...`.
2. Run:
   ```bash
   python run_pipeline.py --story data/raw/test_story_1.txt --session anika_test
   ```
3. In the terminal you should now see richer validation log lines, e.g.:

   ```
   ── [6/6] MuRIL validation (strict) ──
      scene_01: sim=0.912 line_min=0.874 recall=1.00 cov=33% var=1.00 -> PASS
      scene_02: sim=0.868 line_min=0.612 recall=1.00 cov=25% var=1.00 -> RE-DO[MEANING_DRIFT]
          reason: a line's similarity dropped to 0.612 (< 0.72)
      2 scene(s) failed (MEANING_DRIFT) — re-injecting with targeted fixes.
        -> retry scene_02 in mode=MEANING_DRIFT (density_threshold=0.13, permissive=False)
   ```
4. The scratchpad JSON at `data/sessions/anika_test_scratchpad.json` will now
   contain a `scene_diagnoses` key with the full per-scene diagnosis (useful
   for the paper's results section).

Unit-level check: `python -m evaluation.muril_validator` runs a short built-in
demo that exercises both `validate_scene` (backward-compat) and
`diagnose_scene` (new) and prints the diagnosis dict.

---

## 5. What is still open (out of Anika's scope now)

Left explicitly for other owners, per `PROJECT_STATUS.md`:

- Gate **threshold tuning** for the base run (Anish, Pending #5 — the
  infrastructure is now ready; only choosing new numeric defaults is left).
- Character mood/stress dynamics so `tone_signal` fires more often
  (Anish, Pending #6).
- The augmented dataset and mT5 retrain (Dia + Atul, Pending #1 + #2).
- Full evaluation table and paper writeup (everyone, Pending #7 + #8).
