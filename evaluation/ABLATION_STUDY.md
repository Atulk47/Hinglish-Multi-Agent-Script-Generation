# Agent 4 ablation study

This experiment compares Agent 4 and its safeguards while holding upstream Pass-1 dialogue fixed. It does not regenerate the story analysis, character profiles, or neutral screenplay.

## Experimental arms

- `full`: Agent 4, the ThreeGateController eligibility gates, the configured edit-acceptance rule, and MuRIL validation with targeted retries.
- `no_eligibility_gates`: bypasses the three ThreeGateController gates as a group. Agent 4's separate speaker/register, respectful-line, lexical meaning, tone, and edit-acceptance checks remain active.
- `no_gate1`: bypasses only Gate 1's speaker eligibility decision. Other controller gates and Agent 4's separate register/respectful-line checks remain active, so this is specifically an ablation of the controller's Gate 1 logic—not every speaker safeguard.
- `no_gate2`: bypasses only Gate 2's pragmatic-intent eligibility decision. Intent is still classified and recorded; Gates 1 and 3 remain active.
- `no_gate3`: bypasses only Gate 3's density blocking decision. It still records utterance token totals, while Gates 1 and 2 remain active.
- `no_acceptance_rule_gate`: bypasses only the deterministic `edit_gate` check in `_judge_ok`. The editor still must produce a new shortlist slang term, and lexical meaning/tone checks plus repetition limits remain active. This mode requires `QWEN_GATE=rules`.
- `no_validation_redo`: one Agent 4 pass, with no MuRIL validation or retry.
- `no_editor`: Pass-1 dialogue unchanged. Validity is undefined when no slang is inserted and is shown as `N/A`.

`no_eligibility_gates` is a group ablation. The three `no_gate*` arms isolate each controller gate separately. This still does not disable Agent 4's additional register/respectful-line checks or editor-quality safeguards.

## Run the experiment

From the repository root, use a saved trace with `neutral_script`, `character_profiles`, and per-scene `events`:

```powershell
python -m evaluation.ablation outputs/ui_20261031_140117_trace.json --out-dir outputs/ablation/ui --backend qwen
```

For your current Groq pilot, select the full arm plus the three individual gate arms with:

```powershell
python -m evaluation.ablation outputs/orch_full_trace.json --out-dir outputs/ablation/gate_study --backend groq --modes full,no_gate1,no_gate2,no_gate3
```

The command writes one trace-compatible JSON per arm plus a manifest containing the input hash, backend, adapter, runtime slang source, seed, per-arm edit counts, validation outcomes, and detected editor API errors. Selecting `--backend qwen` makes the command stop rather than silently compare against a fallback if Qwen cannot load. Keep the backend and environment configuration unchanged across repeated studies.

The run loads the editor and MuRIL and can take time. It may also make API calls through the configured intent/tone/reranking paths. Do not use a run with recorded editor API errors as paper evidence; inspect the manifest and rerun if necessary.

The evaluator reports both whole-scene MuRIL similarity and similarity restricted to changed dialogue turns (`changed_line_semantic_sim_mean`, minimum, and count). The changed-turn score avoids unchanged turns dominating the scene average. Both are embedding-based proxies, not factual entailment tests; validate important factual changes through blinded human review or a separately validated judge.

## Score each arm

Score the generated traces individually, using the same runtime slang bank for every arm:

```powershell
python -m evaluation.evaluate outputs/ablation/ui/ui_20261031_140117_full.json --lexicon data/felix_dataset/slang_bank_v3.csv --out outputs/ablation/ui/metrics_full.json
```

Repeat with each variant filename and a distinct report filename. Add `--judge` only if you want the additional model-judge scores; it costs API usage and is not a substitute for human ratings. The `--lexicon` option ensures slang scoring uses the bank Agent 4 actually used, rather than the older default dataset.

## Paper reporting cautions

- Treat the saved Pass-1 trace as the frozen input; retain it and the manifest with the experiment results.
- Report the exact variant definitions above. In particular, each `no_gate*` mode bypasses only the named controller decision, while additional Agent 4 safeguards remain active; `no_eligibility_gates` disables the three controller gates as a group.
- Compare overall edit rate as well as newly unblocked lines in the eligibility-gate arm; its eligible-line denominator differs by design.
- The built-in report gives descriptive metrics, not statistical significance or confidence intervals. Use multiple held-out stories, compare systems on the same stories, and calculate uncertainty by resampling stories (not individual lines).
- Have fluent Hinglish raters blindly assess meaning preservation, naturalness, slang appropriateness, and character/tone fit. Keep rater instructions and agreement statistics for the paper.
- Slang validity is reported as `N/A` for a no-edit output; do not describe that baseline as having 100% validity.
