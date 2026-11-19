# What is pending (2026-11-18, end of day)

Everything below that needs an LLM call waits for OpenRouter credit (account balance, AND the key's own spend limit: check both at openrouter.ai). Costs are measured, not guessed.

## Needs credit (in this order)
**Done 2026-11-18 late: #1, #3, #4 (results in docs/BASELINE.md, last section). Still open: #2 (frontier judging with the same cheap judge was effectively done by the v5/pad runs against opus_rules, so only the older v4 set is unjudged), #5, #6.**
| # | What | Command | Cost |
|---|---|---|---|
| 1 | **Re-judge the small-model experiment fairly** (judge prompt now states "Roman script required", so Devanagari / looping outputs are penalised; reuses the SAVED Llama outputs, no regeneration) | `python -m evaluation.baseline_vs_pipeline --pipeline-dirs outputs/stress_full_llama8b --baseline-model meta-llama/llama-3.1-8b-instruct --label llama8b --reuse outputs/small_model_experiment.json --judge-model google/gemini-3.6-flash --out outputs/small_model_experiment_v2.json` | ~$0.13 |
| 2 | **Finish the frontier judging** (3 of 7 verdicts were lost to the credit limit). Same cheap judge as #1 so both experiments use one judge | `python -m evaluation.baseline_vs_pipeline --pipeline-dirs outputs/stress_full_v4 --reuse outputs/baseline_vs_pipeline_v2.json --judge-arms opus_rules --judge-model google/gemini-3.6-flash --out outputs/baseline_vs_pipeline_v3.json` | ~$0.10 |
| 3 | **Fresh `--full` run after today's slang fixes** (pool tagging, "sis yaar" fix) to confirm no coverage loss and read the edits (UI closed, ~3 GB RAM) | `python -m evaluation.stress_run --full --only s01 s02 s10 s13 s19 s20 s22 --no-real --out outputs/stress_full_v5` | ~$0.35 |
| 4 | **Coverage vs fidelity trade-off, judged**: the same run with the old padded writer (`AGENT3_STYLE=slotrich_pad`), then judge both | PowerShell: `$env:AGENT3_STYLE="slotrich_pad"; python -m evaluation.stress_run --full ... --out outputs/stress_full_pad` | ~$0.35 + ~$0.13 judge |
| 5 | A second small model (e.g. `google/gemma-3-12b-it`): same experiment as #1 with `OR_MODEL_*` env vars set to it | see BASELINE.md section 2 | ~$0.3 |
| 6 | Several samples per arm (3 seeds) so the comparisons have error bars | repeat the runs | ~$2-3 |

## Needs people, no credit
* **Native-speaker rating**: `outputs/human_eval/sheet.csv` is built (14 blind items, pipeline vs Opus, order randomised). Give it to 2-3 native speakers (README.md inside has the instructions), keep `key.csv` away from them, then `from evaluation.human_eval_sheet import score; score("outputs/human_eval/key.csv", ["rater1.csv", ...])`. This is the single most valuable missing piece for a publication.
* ~~Decide the coverage vs fidelity default~~ DONE 2026-11-19: `slotrich_pad` (more natural, more slang) is the default; `AGENT3_STYLE=slotrich` restores the faithful writer. Demo slang target 2-3 per casual scene: see docs/BASELINE.md, last section.
* Old items that are documentation / decisions, not code: slide-deck wording fixes, whether to train the intent classifier or document that it is an LLM prompt, the Maya voice batch on Kaggle (`notebooks/maya_batch_demo.py`), a human-audited judge-agreement sample.
* git commit and push (many files changed since Nov 12).

## Known limits that are NOT fixable by more runs
* A single frontier call with the rules in the prompt beats this pipeline on fidelity and naturalness (docs/BASELINE.md). The defensible claim is robustness for weak models (7/7 valid outputs vs 1-3/7 for one 8B call), plus the evaluation method.
* A hallucinated character name equal to a common Hindi word ("Sara" vs "सारा") cannot be caught by text matching.
* The slang pool still has idiom-level awkwardness a native speaker should review (choose-among-candidates instead of veto is the real fix and needs live testing).

## Done today without credit
* Slang defects found by reading all 52 saved edits: criticism words that judge a thing ("flop", "bakwaas", "ghatiya"...) can no longer fill "mujhe ... lag raha hai" (29 pool entries tagged `experiencer_ok:false`, backup in data/slang/); "sis"/"didi"/"behen" count as an address word (no "Sis yaar"). Replaying all 52 edits through the gate: exactly the 2 defective ones are rejected, the other 50 untouched.
* `AGENT3_STYLE` switch, fair judge prompt, re-judge-without-regenerating, blind human-eval sheet generator + scorer.
