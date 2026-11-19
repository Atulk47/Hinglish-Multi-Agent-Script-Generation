# Testing: how we try to make it work for ANY story

## Why a "new problem with every new story" keeps happening

Short answer: **both**, and they are different kinds of problem.

| Cause | Example found | Fix type |
|---|---|---|
| **Model behaviour (non-deterministic)**: an LLM sometimes returns an empty answer, truncated JSON, a "Narrator:" line, or invents a place/object/speaker | Agent 1 returned nothing for a scene; writer echoed prompt fields ("Listener:") as speakers; invented "Riya"/"Arjun" in a story with no names | Cannot be removed. Must be **detected and handled**: retries with a second model, grounding checks, fail-closed critic, sanitiser |
| **Our code assumed a story shape** | Persona skipped when no names; segmenter over/under-splitting; a regex that took 121 s on long input; `None` JSON crashing a stage | Real bugs. Fixed, and now covered by corpus + fuzz tests |
| **Slang stage rules** that were too generic | urban slang in a village story, male address for girls, experiencer frames | Register/setting tags + rule gates in `edit_gate` |

Every time we only tested the stories we had just seen, the next unseen shape found a new hole. The only way out is to test **shapes**, not stories.

## What "ready for ANY story" can and cannot mean

* It **cannot be proven**: the LLM stages are non-deterministic, so no test can show "never fails".
* It **can be made fail-safe**: a bad model answer must end as a dropped line / retried call / honest warning, never as a crash and never as an invented fact in the output.
* It **can be measured**: a corpus of very different stories, with invariants checked automatically, gives a pass rate we can track before/after each change.

## The four layers

1. **Unit + regression tests** (`tests/`, 446 tests, no network): every defect ever found has a test.
2. **Property / fuzz / fault-injection tests** (`tests/test_fuzz_robustness.py`, 155 seeded cases): random and hostile input (empty, huge, mixed script, emoji, `None`, wrong-shape JSON, an LLM that raises/returns garbage). Found a quadratic regex, a `None`-JSON crash, non-idempotent sanitiser.
3. **Invariants** (`evaluation/invariants.py`, no model needed): rules that must hold for any story: no narrator speaker, no Devanagari/artefacts in Pass 1, no duplicate/empty lines, every speaker grounded in the source, no gesture turned into speech, spoken words not lost, locations grounded, slang stage: no urban slang in traditional setting, gender agreement, experiencer frames, copulas.
4. **Stress corpus** (`data/stress/s01..s24` + held-out real stories, `python -m evaluation.stress_run`): 24 shape-varied stories (dash/quote/indirect speech, narration-only, single block, 2-3 sentences, long wedding, first person, no names at all, play script, fable, Roman-Hindi, ...) + 8 real stories, run through the live pipeline (segmenter -> Agent 1 -> Persona -> Agent 3 + critic) and checked against the invariants.

## Measured results (32 stories, live LLM)

| Run | Clean | Crashes | Notes |
|---|---|---|---|
| baseline | 27 / 32 | 0 | speaker-not-grounded, empty lines, 3 Agent-1 failures |
| run 2 | 28 / 32 | 0 | Agent-1 failures 0 (retries + second model) |
| run 3 | 31 / 32 | 0 | only the no-names startup story failed |
| run 4 | **32 / 32** | 0 | no-names story fixed; 0 errors, 16 critic missing_event warnings (advisory); cost $0.35 |

## Full pipeline (incl. slang stage) - first live result, 2026-11-18

`--full --only s01 s20` (UI closed, models load in ~34 s): 2 stories, 0 crashes, 0 invariant errors (also `check_slang`), 44 LLM calls, $0.08.
Slang coverage: s01 2/18 lines changed (11%), s20 10/27 (37%). Rule gates rejected 4 edits (only-minor-words, slang-usage) and the line was kept neutral, as designed.
Observations that are NOT pipeline bugs: s20's source text is itself odd ("super bro", "girlfriend", a garbled word "kampraij") and the output stays faithful to it, including the garbled word.
Scene count for s20 was 2 in run 4 and 5 here: the LLM referee is non-deterministic (bounded, not eliminated).

## Running it

```bash
python -m pytest -q tests -p no:cacheprovider --ignore=tests/test_offline_pipeline.py
python -m evaluation.stress_run --workers 3 --out outputs/stress4
python -m evaluation.stress_run --only s14 s03          # a subset
python -m evaluation.stress_run --full --only s01 s20   # whole pipeline incl. slang (close the UI first, ~3 GB RAM)
```

## Honest limits

* The Fidelity Critic is itself an LLM; it can miss a subtle invented object.
* The slang stage is checked by invariants only on saved UI traces, or with `--full` (needs free RAM; not run in the last session).
* Scene splitting uses an LLM referee for long stories; counts can vary between runs (bounded window).
* Input is intended to be Devanagari Hindi; Roman Hindi degrades gracefully.
* Native-speaker review of naturalness is still needed for the paper.

## Update 2026-11-18 (after top-up)
* Paired Director test (`evaluation/director_ab.py`, 117 identical scenes, old v3 vs v4 prompt): slang lines 74 -> 83 (17.8% -> 20.0%); 10 scenes up, 106 equal, 1 down; 7 filler-only lines, all sensible peer lines; 0 nominations on authority / slang-level-0 speakers; 0 filler on thanks/questions. Cost $1.61 (226 Sonnet calls; more than first estimated).
* Full pipeline (incl. slang stage) on 7 stories (s01, s20, s02, s10, s13, s19, s22): 0 errors, 0 crashes. Traditional / elder stories (s13, s19) got 0 slang, as they should.
* Found and fixed by reading outputs: nickname aliases (Rahul = Rocky) and "Grandma" next to "Dadiji" were separate characters (see GROUNDING.md round 9). Also a stray-backspace regex in a fuzz test was asserting nothing.
* Regression after the fixes: 459 tests pass; 8-story corpus subset 0 errors.
* Known, not fixed: the writer sometimes appends "mujhe bahut accha laga" to lines that carry no feeling in the source, which then gets nominated for slang (inflates coverage slightly); some test sources are themselves garbled AI-written Hindi and the output stays faithful to them.

## Update 2026-11-18 (evening)
* **Baseline comparison (docs/BASELINE.md): a single Opus call with the rules in the prompt matched or beat the pipeline on every measure.** This changes what the paper can claim.
* 32-story corpus re-run after the alias / kin / name changes: 31 clean; the one error (s14, no named people) exposed two real bugs (Persona-less role labels failing g/j matching; an invented character "Sara"), both fixed; follow-up subsets (12 stories) 0 errors. 462 tests pass.
