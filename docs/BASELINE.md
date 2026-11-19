# Does the pipeline beat ONE call to a frontier model? (2026-11-18)

The question every reviewer asks. Run: `python -m evaluation.baseline_vs_pipeline` (7 stories: s01, s02, s10, s13, s19, s20, s22; one sample each).

Arms: **pipeline** (the whole system, saved `--full` outputs), **opus_simple** (one Claude Opus 5.5 call: "turn this Devanagari story into Hinglish dialogue with natural slang"), **opus_rules** (one Opus call with the pipeline's rules written into the prompt: no narrator, no invented facts, slang only between peers, none for elders / traditional / serious).

## Result: the pipeline did NOT win on anything measured

| Measure | pipeline | opus_simple | opus_rules |
|---|---|---|---|
| Real structural defects (invariants, after fixing a checker false-positive bug) | 0 | 1 story: 9 "Narrator" lines | 0 |
| Urban slang in traditional stories (s13, s19) | 0 | 3 lines | 0 |
| Slang density (lines with slang) | 0-44% | 21-46% | 0-47% |
| Judge: invented items / story (lower better) | 5.1 | 6.2 | **0.0** |
| Judge: dropped events / story (lower better) | 6.9 | 2.3 | 5.0 |
| Judge: naturalness (1-5) | 1.9 | 4.3 | 4.0 |
| Judge: slang fit (1-5) | 2.2 | 3.8 | 4.2 |
| Judge preference, overall | 0 wins | 6 of 6 | 5 of 5 |
| Seconds per story | 37-155 | 13-28 | 10-21 |
| API cost per story | about $0.05 | $0.033 | $0.033 |

(judge = gpt-5.4, blind, random side, different family from both systems; 6 and 5 verdicts parsed. It is an LLM proxy, not a native speaker, and n is small. The direction is consistent across stories and matches reading the outputs.)

## What it means
* **A frontier model given the rules in the prompt matches every guarantee the pipeline offers** (no narrator, no invented material, no urban slang where it does not belong) and is faster, cheaper, more natural, and drops fewer events. The plain prompt does show the failure modes the pipeline was built for (Narrator lines, urban slang in a village story, ~6 invented items per story), so those are real; they are just fixable with a paragraph of instructions.
* Why the pipeline loses: bookish Pass-1 Hinglish, "mujhe bahut accha laga" padding appended to lines that carry no feeling in the source (which the slang stage then amplifies), and events dropped by the per-scene line budget (AGENT3_MAX_LINES=12).
* So the paper cannot claim a quality advantage over frontier models. What survives: (1) the failure-mode analysis of naive prompting and the finding that explicit rules close the gap; (2) the evaluation method (invariants + stress corpus + paired tests) and its measured effect on a weaker pipeline; (3) negative results (IndicBART script mismatch, MuRIL drift); (4) the register-tagged slang pool; (5) the open question below.

## The experiment that could still give a positive claim
Scaffolding may help **weak / cheap models**, not Opus. Run the same comparison with a small open model (e.g. gpt-oss-20b, a 7-8B model) in every role: one plain call vs one rules call vs the full pipeline, judged blind. If the pipeline lifts a small model toward the frontier rules-prompt baseline, that is a result (cost / privacy / offline use). Not run yet (about $1 with the judge).

## Honest checker note
The first version of this comparison counted 21 baseline "speaker-not-grounded" errors; all but the Narrator lines were checker false positives (names like Geeta / Rocky / Meenu that ARE in the story but spelled differently from the romanisation). Fixed in `evaluation/invariants._in_source` (now `grounding.name_grounded`).

## Follow-up: fixing the writer (2026-11-18, same evening)
Causes found by reading the outputs, and fixes (agents/agent3_synthesizer.py):
1. **Padding was designed in**: `PEER_STYLE_RULES` demanded "at least two lines in every scene" that state a feeling ("bahut accha laga") to give the slang editor something to replace. Now a feeling is stated only where the original shows one; adding "mujhe bahut accha laga" to a factual line is forbidden.
2. **Scenery spoken as dialogue** (the moneylender "says" the village has a long queue): new rule "description is not speech"; the permission to add "small natural touches (reactions, short connecting sentences)" was replaced by "at most a greeting or a one-word acknowledgement; no added thanks, promises, opinions, jokes, questions, agreements".
3. **Bookish Pass-1 Hinglish**: all characters now get "talk the way people really talk: short spoken sentences, English words where Indians say them".
4. **Dropped events**: the line budget was one line per quote + 1 + sentences/3 (capped at 12); now sentences/2 and never fewer than key events + 1.
5. A wording slip in (2) made the writer too terse on a quote-heavy scene (8 quoted clauses -> 4 lines); fixed ("every quoted line must still be said") and re-checked on that scene (8 lines).
Also: "DamadSahab" / "Damad Sahab" spacing variants are one name.

**Measured (paired: old vs new writer, same 7 stories, same cheap judge gpt-oss-120b against the same Opus-rules reference, n=7, one sample each, so noisy):**
| | old writer | new writer |
|---|---|---|
| padded lines (regex, no judge) | 15 / 121 = 12.4% | 2 / 121 = 1.7% |
| judge: invented per story | 3.14 | 2.14 |
| judge: dropped per story | 2.14 | 2.86 (before fix 5; s02 alone 1 -> 7) |
| judge: naturalness | 3.14 | 3.29 |
Only the padding result is solid. Invention improved, drops and naturalness are within noise. **The headline comparison against Opus (gpt-5.4 judge, full pipeline incl. the slang stage) has NOT been re-run** - the OpenRouter key spend cap was used up; rerun `python -m evaluation.baseline_vs_pipeline` after a fresh `--full` stress run (about $1). Do not claim an improved judge score until that is done.
Note the slang stage is untouched by this: its clumsy picks ("ekdum flop lag raha hai", "bahut pathetic shot") are a separate naturalness cost; the review loop that vetoes them was measured earlier and loses more useful edits than it saves (docs/DIRECTOR.md, result 3).

## Re-run after the writer fixes, and the small-model experiment (2026-11-18, night)
All numbers: 7 stories (s01, s02, s10, s13, s19, s20, s22), one sample per arm, outputs in `outputs/stress_full_v4`, `outputs/stress_full_llama8b`, `outputs/baseline_vs_pipeline_v2.json`, `outputs/small_model_experiment.json`. The OpenRouter credit ran out during both judged runs, so several judge verdicts are missing (stated per result).

### 1. Fresh full pipeline (frontier models) vs Opus-with-rules: the gap did NOT close
* Hard invariants: pipeline 0 errors on 7/7 stories (one transliteration artefact "Ramu" -> "Ramū" was found and fixed: `strip_diacritics`).
* Judge (gpt-5.4, blind, n=4 verdicts; 3 calls failed on credit): Opus-rules preferred 4/4 overall. Mean invented per story: pipeline 5.75 vs Opus 0.75; dropped 7.0 vs 6.5; naturalness 2.5 vs 3.75; slang fit 2.75 vs 3.5.
* What the judge calls invented in the pipeline output: extra dialogue turns and speaker swaps ("Haan Papa" where the story says "Haan Dadaji"), added explanations, "scripted / bookish" phrasing. The writer fixes (padding, description-is-not-speech) did not change the headline.
* **Cost of the padding fix: slang coverage nearly halved** - mean share of lines carrying slang 22.5% -> 12.6% (s22 23% -> 0%, s02 27% -> 6%). The padded "bahut accha laga" lines were what the slang editor was swapping. Coverage vs fidelity is a real trade-off: the old behaviour (`PEER_STYLE_RULES` asking for plain feeling words) gives more slang and more invented warmth.

### 2. Small-model experiment: the same comparison with Llama 3.1 8B in EVERY role
Pipeline = the whole system with every LLM role (extract, writer, director, reviewer ...) set to llama-3.1-8b-instruct (the local Qwen editor unchanged), $0.02 for 7 stories. Baselines = ONE call to the same model, plain prompt and rules prompt.

Judge-free result (deterministic): outputs that are Roman-script AND pass every structural invariant
| | valid outputs |
|---|---|
| pipeline (8B in every role) | **7 / 7** |
| single call, rules prompt | 3 / 7 |
| single call, plain prompt | 1 / 7 |
Failure modes of the single 8B call: writes Devanagari instead of Roman script (6/7 stories with the plain prompt), repeats one exchange until it hits the token limit (279 lines in one story), invents speakers, drops the "Name: line" format. A first, independent run gave the same picture (plain 0/7 clean, rules 2/7 clean). **The scaffolding (grounding, sanitiser, critic, segmentation) makes a small model's output reliably valid.** This is the one positive, judge-independent claim the project can make.

Quality when the baseline output IS valid: judge (gemini-3.6-flash, 9 of 14 verdicts, judge prompt did not state the Roman-script requirement, so Devanagari outputs were not penalised): the single 8B call was preferred as often as or more often than the pipeline (overall 4-1 and 3-1 for the baselines; mean naturalness 3.0-3.4 vs 2.4). With n this small it is "no evidence the scaffolding makes small-model text better", not "it is worse".

### What the paper can honestly say now
1. A single frontier call with the rules in the prompt beats this pipeline on fidelity and naturalness (section 1 + the first comparison).
2. The pipeline's contribution is **robustness for weak models** (section 2: 7/7 valid vs 1-3/7), measured without a judge, plus the evaluation method and failure-mode analysis. Frame it as a reliability layer, not a quality improvement.
3. Not done: native-speaker evaluation; judged comparison with a fixed judge prompt that states the script requirement; more than one sample per arm; a second small model.

## 2026-11-18 (late): the pending runs, done - one cheap judge (gemini-3.6-flash) for everything below, 7 stories, one sample per arm
Total spend for this block about $0.76 (two full runs $0.52, four judged sets $0.25). Files: `outputs/stress_full_pad`, `outputs/stress_full_v5`, `outputs/judge_pad.json`, `outputs/judge_v5.json`, `outputs/small_model_experiment_v2.json`.

### A. Writer style trade-off (frontier models; pipeline vs Opus-with-rules; all 7 verdicts parsed)
| writer | invented / story | dropped / story | naturalness | slang fit | slang coverage | padded neutral lines |
|---|---|---|---|---|---|---|
| `slotrich_pad` (old: forces stated feelings) | 3.14 | 1.43 | 3.29 | 3.43 | 16.7% | 6.1% |
| `slotrich` (current default: faithful) | **1.43** | 1.50 | 2.57 | 3.00 | 13.4% | 2.1% |
| Opus with rules (reference) | 0.0-0.14 | 0.6-0.7 | 4.9-5.0 | 4.9-5.0 | - | - |
Faithful writer halves invented content (3.1 -> 1.4) at the price of lower judged naturalness / slang fit (-0.7 / -0.4) and ~3 points of slang coverage. The judge preferred Opus on overall 7/7 for both. Honest reading: the padded lines sound warmer, which this judge rewards, but they are invented. Keep `slotrich` if fidelity is the claim. n=7, one sample, scene segmentation also varied between runs (s01 1 scene, s22 2 scenes), so differences of ~1 point are not reliable.
Today's slang fixes confirmed on the fresh run (#3): 0 errors, 0 crashes, 0 slang-invariant violations, "Sis yaar" 3 -> 0.

### B. Small model (Llama 3.1 8B in every role), re-judged with the fair prompt (Roman script required), all 14 verdicts parsed
| | pipeline wins | baseline wins | tie |
|---|---|---|---|
| overall vs ONE call, plain prompt | **6** | 1 | 0 |
| overall vs ONE call, rules prompt | **5** | 2 | 0 |
| naturalness vs plain / rules | 7 / 5 | 0 / 2 | 0 |
| fidelity vs plain / rules | 1 / 1 | 5 / 4 | 1 / 2 |
Single 8B call produced a valid output (Roman script + invariants clean) in 1/7 (plain) and 3/7 (rules) stories; the pipeline in 7/7. Fidelity goes to the baselines mostly because a Devanagari copy of the story is trivially "faithful"; restricted to the 4 stories where the single call WAS valid, the baseline was preferred overall in 3 of 4 (s13 plain, s13 rules, s19 rules; the pipeline won s02 rules). So: the scaffolding clearly lifts a weak model from mostly-invalid to valid and is judged better overall (11 of 14), but when the weak model manages valid output on its own, its text is as good or better.

## 2026-11-19: demo defaults - padded writer + a slang target of 2-3 per casual scene
Decision (user): `AGENT3_STYLE` default is now `slotrich_pad` (judged more natural, more slang; the cost is ~1.7 more invented items per story, see section A above), and the Slang Director aims for 2 slang lines per casual scene of 4+ lines with a hard ceiling of 3 (`SLANG_DIRECTOR_TARGET=2`, `SLANG_DIRECTOR_MAX_PER_SCENE=3`, Director prompt v5). Traditional / elder / serious scenes are never targeted.
What the runs showed (7 stories, full pipeline, one sample each):
| setting | coverage | slang per casual scene (4+ lines), mean | scenes reaching >= 2 |
|---|---|---|---|
| padded writer, no target | 16.7% | 1.89 | 6 / 9 |
| padded + target 2 (run 1) | 16.8% | 1.78 | 7 / 9 |
| padded + target 2, final gate (run 2) | 17.2% | 1.46 | 5 / 13 |
| padded + target 3 (trial) | 12.6% | 1.33 | 3 / 9 |
* The Director is already on target (it nominates ~2.1-2.4 lines per casual scene); the local editor then applies ~70% of them, because for the rest no safe candidate exists. Asking for 3 did not help (20 nominated, 12 applied). Differences between these runs are within run-to-run noise (scene splitting and sampling vary), so the target is kept at 2 as a harmless nudge, not as a guarantee.
* A guaranteed 2-3 per scene is NOT safely possible: the scenes that fall short are a grandfather/grandson scene (slang refused, correctly) and plan-making scenes whose lines carry no feeling to replace. Forcing slang there means slang on respectful or factual lines.
* For the demo pick a lively peer story: that is where 2-3 per scene happens (e.g. the hostel / cafe / siblings stories), and keep short scenes (<4 lines) out of the expectation.
* Judge (gemini-3.6-flash, final defaults): Opus-with-rules preferred 7/7; pipeline invented 3.0, dropped 1.7, naturalness 2.7, slang fit 3.0 per story (same range as the padded run, noise ~ +-0.5).

Slang-quality fixes made while reading every edit of these runs (all general frame rules; replaying 100 distinct saved edits, exactly the 4 meaning-changing ones are rejected and 96 pass): `bura laga` (I felt hurt) can no longer become `bore`/`cringe`/a verdict word (`wahiyat` allowed); a thing-praise slang word cannot stand before `laga ki <clause>` ("mujhe jhakkas laga ki Papa khush hain"); a banned word the story itself contains ("super bro") is no longer reported as a slang leak.
