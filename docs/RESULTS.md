# Results — 7-story ablation on the real system, judged by validated metrics

System: Qwen2.5-1.5B LoRA v5 (GGUF Q5_K_M) + eligibility gates + rule gate + Validator v2, **final code** (nukta transliteration fix, comma-glitch rule, human-calibrated judge rule).
Harness: `python -m evaluation.ablation` (one frozen Pass-1 text per story, same seed, real local editor, 0 editor errors) → `python -m evaluation.aggregate_ablation` (this table).
Raw outputs: `outputs/ablation_final2/` (final run) and `outputs/ablation_multi/` (an earlier independent run, see "Two runs" below).

**Stories (7, 235 dialogue lines):** hostel project night, cricket tournament, science fair, library mystery, Himalaya village, school science fair, birthday surprise. Pass-1 text came from
fresh runs of the final pipeline (`fin_*` traces). The Himalaya story yields 0 edits (no peer speakers; correct).

**Metrics** (defined and validated in `docs/EVALUATION.md`): *edit rate* = changed lines / dialogue lines · *unsafe edits* = share of edits failing a blocking check of the rule audit
(health / elder / respect / rule gate / polarity / negation / locality: a safety POLICY plus structural quality) · *judge-accepted* = share of edits the two AI-judge families (Claude Haiku 5.5,
Gemini 2.5 Flash) accept on meaning, grammar, naturalness and fit, reported as **(both)** = both accept (strict, a lower bound) and **(either)** = at least one accepts (lenient; the rule that agreed
best with a human rater, κ = 0.75, `docs/EVALUATION.md` §4a) · *useful edits per 100 lines* = edits passing the audit AND the judge rule. Intervals are 95% bootstrap over **stories** (the independent
unit), not over lines.

| Arm | edit rate | unsafe edits | judge-accepted (both) | judge-accepted (either) | useful / 100 lines (both) | useful / 100 lines (either) |
|---|---|---|---|---|---|---|
| no editor (baseline) | 0% | – | – | – | 0 | 0 |
| editor + gates, **no validator repair** (editor alone) | 21% [13–29] | 18% [9–27] | 79% [68–91] | 92% [84–99] | 15.0 [9.0–21.1] | 17.7 [10.2–25.1] |
| **full system** (Validator v2: audit + repair) | 20% [12–28] | **13%** [4–22] | 81% [71–92] | 95% [89–100] | 15.1 [9.0–21.1] | 17.7 [10.3–25.2] |
| full + strict judge filter (`VALIDATOR_JUDGE_RULE=all`, offline estimate) | 16% [10–22] | 8% [0–16] | 100% by construction | 100% | 15.1 | 15.1 |
| no Gate 2 (pragmatic intent) | 22% [13–31] | 22% [12–31] | 75% [66–86] | 90% [79–100] | 15.3 [9.3–21.2] | 17.2 [10.0–24.7] |
| no acceptance rule gate | 21% [13–29] | 21% [10–32] | 78% [67–90] | 92% [84–99] | 14.3 [8.3–20.6] | 17.0 [9.7–24.5] |

> **Re-scored 2026-11-16 (the numbers above are NOT the ones first published).** The edits are the same frozen outputs of the 2026-11 run; what changed is how they are *measured*:
> (1) **pairing bug fixed** — a neutral script with an empty `Narrator:` turn (library story, scene_03) shifted every later line by one, so that story's lines/edits/unsafe were miscounted
> (28 → 27 lines; e.g. editor-alone edits 7 → 4, full 5 → 4); the other six stories are unaffected by this. (2) **the rule audit is now stricter** (idiom frames, gender agreement, usage rules,
> added after this run), so edits the run-time validator let through are now flagged: full-system "unsafe" went from 0% to 13% (5 of 47 edits: 2 hostel, 1 science fair, 1 library, 1 birthday).
> The old table (editor alone 23% edits / 12% unsafe; full 21% / 0%; useful per 100 15.5 both, 19.7 either) is kept in `outputs/ablation_final2/multistory_summary.before_pairing_fix.json`.
> The two effects are not separated here; a clean "full system" number under the *current* validator needs a re-run of the `full` arm on the frozen traces (local, no API cost beyond the judge).

### Clean re-run with the CURRENT code (2026-11-16, `outputs/ablation_final3/`)

The re-scoring above mixes old edits with a new audit. This run regenerates the `no_validator_repair` and `full` arms on the same 7 frozen Pass-1 traces with today's code (v6 editor, current gate, audit and repair), so
the arms are measured by the same validator that produced them. The `full` arm here is the gate pipeline (`SLANG_AGENTIC=0` equivalent); the default agentic arm is in `docs/DIRECTOR.md`.

| Arm | edit rate | unsafe edits | judge-accepted (both) | judge-accepted (either) | useful / 100 lines (both) | useful / 100 lines (either) |
|---|---|---|---|---|---|---|
| editor + gates, no validator repair | 21% [12–29] | 6% [0–12] | 70% [61–83] | 99% [97–100] | 13.4 [8.1–18.2] | 20.0 [12.0–28.1] |
| **full system** (audit + repair) | 20% [12–28] | **0%** | 70% [63–82] | 100% | 13.4 [8.1–18.2] | 20.0 [12.0–28.1] |
| full + strict judge filter (offline) | 13% [8–18] | 0% | 100% by construction | 100% | 13.4 | 13.4 |

Pooled over 235 lines: editor alone 51 edits (4 unsafe; 33 accepted by both judges, 50 by at least one; 31 / 47 useful); full 47 edits (**0 unsafe**; 31 / 47; 31 / 47 useful). Paired over stories, full vs editor alone:
unsafe −5.5 points (CI −11.9 … 0.0), useful per 100 lines 0.0 (CI 0.0 … 0.0), judge-accepted (both) +0.1 (CI −2.4 … +2.6), (either) +1.0 (CI 0.0 … +3.1). Reading: the repair removes the remaining
unsafe edits at **no cost in useful edits**; its effect on judge acceptance is nil here (the editor v6 + gates already produces few audit-flagged edits, 4 of 51). The earlier "−12 points" was a property of the
older editor/gates (v5). Note the strict useful rate is 13.4 vs 15.1 for the v5 run: a 1.7 difference is inside the seed noise (about ±2 per 100 lines), not a claimable change.

Pooled over the 234–235 lines (re-scored v5-era edits): editor alone 51 edits (9 unsafe; 39 accepted by both judges, 47 by at least one; 35 / 42 useful); full 47 edits (5 unsafe; 37 / 45; 35 / 42 useful); no Gate 2 52 edits
(12 unsafe; 38 / 48; 35 / 40 useful); no rule gate 51 edits (12 unsafe; 38 / 47; 32 / 39 useful).

## What the data supports (paired over the same 7 stories, final run)

(Re-scored numbers, see the note under the table; the previously published values are in brackets.)

* **The validator's repair removes unsafe edits, but by less than first reported.** Unsafe-edit rate −4.6 points vs the editor alone (95% CI −10.9 … 0.0; was −12.4, CI −25.6 … −2.1): 5 of 47 edits
  unsafe vs 9 of 51 (was 0 of 48 vs 7 of 54). The repair still helps, but the interval now touches zero, and the remaining unsafe edits are ones the stricter audit flags that the run-time
  validator of that run did not.
* **It does not cost useful edits, and it slightly raises judge acceptance.** Useful edits per 100 lines +0.1 (CI 0.0 … +0.2; was +0.5); share of edits accepted by both judges +1.8 points
  (CI 0.0 … +4.8; was +6.7), by at least one judge +2.8 points (CI 0.0 … +6.4; was +7.6). The judge-acceptance gain shrank a lot and is not distinguishable from zero.
* **Most real edits are acceptable by every yardstick we have.** At least one judge accepts 95% of the full system's edits; a human rater kept 94% of a stratified sample of 32 real edits (mean
  meaning/grammar 4.3 / 5, naturalness 4.2 / 5; `docs/EVALUATION.md` §4a). The strict "both judges" rate (81%) is a lower bound: the Claude judge is stricter than the person.
* **Gate effects are small and NOT stable across runs.** Removing Gate 2: unsafe +4.5 points (CI −0.8 … +10.8), judge acceptance (both) −4.6 (CI −11.0 … −0.3), useful per 100 lines +0.3 (CI −0.7 … +1.6).
  Removing the rule gate: unsafe +3.1 points (CI 0.0 … +9.4), useful −0.7 (CI −2.0 … 0.0). The earlier run (below) suggested the gates cost a little coverage; this run does not. With 7 stories
  these effects are within run-to-run noise, so we claim only that the gates do not hurt and may reduce unsafe edits.

## Two independent runs (replication)

An earlier run on different Pass-1 text (249 lines, code before the nukta / comma fixes; `outputs/ablation_multi/`) gave: editor alone 13% unsafe → full 0%, difference **−12.7 points
(CI −22.7 … −3.6)**; useful edits unchanged. The final run first gave **−12.4 points (CI −25.6 … −2.1)**; both were scored with the audit of that time. Re-scored with today's stricter audit the
final run gives **−4.6 points (CI −10.9 … 0.0)**; the earlier run (`ablation_multi`, no empty-turn scene) has not been re-scored, so the two are no longer like-for-like. What replicated at the time was the
direction of the repair's effect on unsafe edits; the gate-removal effects and the size of the judge-acceptance gain do **not** replicate reliably.

## Limits (state them)

7 stories, one run per arm (the editor is seeded and deterministic; Pass-1 text varies between pipeline runs and is frozen here). "Unsafe" is rule-defined (a safety POLICY, not a naturalness
judgement). A 0% for the full system is by construction only when the audit used for scoring is the one the validator ran with; the 2026-11 numbers were first scored that way (0%) and re-scored with the
stricter current audit (13%), so compare arms only within one scoring. The judges are validated on synthetic perturbations and on one human rater (50 items,
the project author, stratified sample: indicative, not definitive). The earlier Groq pilot in `outputs/ablation/groq_pilot` used the cloud fallback and one old trace and must not be quoted.

## Reproduce

```bash
python -m evaluation.ablation outputs/<story>_trace.json --out-dir outputs/ablation_final2 --backend qwen --modes full,no_validator_repair,no_gate2,no_acceptance_rule_gate,no_editor
python -m evaluation.aggregate_ablation --dir outputs/ablation_final2 --stories "hostel=fin_hostel_trace,cricket=fin_cricket_trace,science_fair=fin_sci_trace,library=fin_t1_trace,himalaya=fin_t2_trace,school_fair=fin_t3_trace,birthday=fin_bday_trace"
```
