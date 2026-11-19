# Slang Director — an agent that owns the slang decision (default since the agentic-mode run, see the end)

## What it is

The slang decision layer used to look at one line at a time: Gate 2 classified a single utterance into one of 7 intents and Gate 3 was a token counter. It never knew who a
line was spoken **to**, how the two people relate, how the speaker feels at that point of the story, or what slang the scene already used.

`agents/slang_director.py` replaces Gates 2 and 3 with **one LLM call per scene** (Claude Sonnet 5.5, reasoning on, temperature 0, about $0.007 per scene). It receives the cast
(gender, age group, role, traits, emotion and intensity in this scene), the relationships with their registers, the scene's events, the dialogue, and the slang used earlier, and returns a plan per line:
`slang yes/no`, `register` (hype / banter / frustration / stress / urgency), `strength 1-3`, and a one-line `why`.

**Design rule — the agent proposes, the rules dispose.** The Director only nominates lines. The local Qwen editor still writes the edit, Gate 1 (elders / authority), the register guards, the rule gate
and the validator (audit + repair) still apply, so a wrong plan can cost coverage but not safety (`tests/test_director.py` checks that a nominated line to a professor is still vetoed). If the model is down or
its answer is unusable, that scene uses the old line-by-line gates; a scene being retried because it got no slang also uses them. Plans are cached by prompt hash, so ablation arms are paired and repeatable.

Switches: `SLANG_DIRECTOR` (default 1) and `SLANG_AGENTIC` (default 1, see below), or the "Slang Director" checkbox in the UI. Each skipped line shows the Director's reason in the comparison view.

## Result 1 — Director ASSISTING the gates (Gate 1 and the guards still active): it did NOT beat them

Same frozen Pass-1 text, same local editor, same validator repair, paired over the 7 stories of `docs/RESULTS.md` (`outputs/ablation_director`, `outputs/ablation_director_v2`):

| | edit rate | unsafe | judge-accepted (both / either) | useful edits per 100 lines (both) |
|---|---|---|---|---|
| old gates (`full`) | 20% | 13% | 81% / 95% | 15.1 |
| Director v1 prompt | 8% | 0% | 75% / 100% | 6.6  (paired diff −8.5, CI −12.4 … −4.5) |
| Director v2 prompt | 14% | 12% | 75% / 100% | 10.0  (paired diff −5.1, CI −8.2 … −2.0) |

*Re-scored 2026-11-16 (pairing bug fixed in the library story; scoring with the current, stricter audit; the first-published values were gates 21% / 0% / 75% / 95% / 15.5, v1 8% / 0% / 62% / 100% / 6.6 (−8.9),
v2 14% / 0% / 68% / 100% / 9.9 (−5.6); old summaries: `outputs/ablation_director*/multistory_summary.before_pairing_fix.json`). Direction and conclusions are unchanged.*

* **Safety is not clearly different.** The validator repair runs on both arms; first reported as equal (0 unsafe edits). Under the current audit the gates show 5 of 47 edits unsafe, Director v1 0 of 19, v2 4 of 32
  (v1 vs gates: −10.9 points, CI −21.4 … −1.8; v2 vs gates: −1.1, CI −9.7 … +7.3). Those unsafe edits are ones the stricter audit flags that the validator of that run did not, so this is a re-scoring effect, not a
  re-run of the arms.
* **Precision up, coverage down.** At least one judge accepts 100% of the Director's edits vs 95% for the gates (paired +5.2 points, CI 0.0 … +10.7), but it makes 32 edits vs 47 (v2), so net useful edits fall.
* **v1 → v2.** Reading the 33 lines v1 dropped showed over-cautious rules (refusing a feeling word inside a plan line, "avoid consecutive" as a hard rule, "warm praise would cheapen it", a narrator treated as a formal
  addressee). v2 rewrites those rules and keeps the elder / serious-moment / pure-fact rules. Coverage rose 8% → 14% but stayed below the gates.
* **Held-out check (2 stories never used to tune the prompt: `held_cafe`, `held_train`, 50 lines):** the gates made 11 edits, 10 of them useful; the Director made 10 edits, 9 useful (useful per 100 lines 20.0 vs 18.0).
  Essentially a tie, but with 2 stories there is no usable interval.

## Honest limits

* The v2 prompt was written after reading the same 7 stories it was then scored on (in-sample); only the two held-out stories are clean, and they are too few for intervals. A third prompt round on these
  stories would be over-fitting, so we stopped at v2.
* One Director model (Claude Sonnet 5.5); one of the two judges is also Claude Haiku. The Director chooses lines and writes no text, so self-preference should be small, and the other judge (Gemini) is a different family.
* "Useful edits per 100 lines" rewards coverage; if a reader values precision over coverage the Director's 100% either-judge acceptance is the more relevant column.

## Result 2 — Director OWNING the decision (`SLANG_AGENTIC=1`, the default): level with the gates, with fewer policy errors

Why a second run: of the 51 lines the v2 Director nominated only 32 became edits; most of the 19 failures were the editor's own rule-based limits (script-wide filler cap, "no-natural-filler", "already-casual",
plus Gate 1, the respectful-line regex and the serious-scene word list) overruling or fighting the agent. In agentic mode the Director is the only decider for the lines it planned: those rule-based decision
checks are bypassed and its per-scene cap replaces the filler cap. Still in force and independent of the decider: the Qwen editor (wording), the post-edit rule gate, and the validator (audit + repair).
If the model is unreachable the scene uses the old gates. Same 7 stories, same Director plans (cached), paired over stories (`outputs/ablation_agentic`):

| | edit rate | unsafe | judge-accepted (either) | useful / 100 lines (both) |
|---|---|---|---|---|
| gates alone (`no_validator_repair`) | 21% | 18% | 92% | 15.0 |
| **agent alone** (`agentic_no_repair`) | 20% | 14% | 100% | 14.1 |
| gates + validator repair (`full`) | 20% | 13% | 95% | 15.1 |
| Director assisting gates (v2) | 14% | 12% | 100% | 10.0 |
| **agent + validator repair (`agentic`, default)** | 19% | 6% | 100% | 14.0 |

*Re-scored 2026-11-16 (pairing fix + current audit; first-published values: gates alone 23% / 12% / 87% / 15.5, agent alone 21% / 11% / 96% / 14.6, full 21% / 0% / 95% / 15.5, Director v2 14% / 0% / 100% / 9.9,
agentic 19% / 0% / 100% / 14.3; old summary: `outputs/ablation_agentic/multistory_summary.before_pairing_fix.json`). Note that with the current audit the "+ validator repair" rows are no longer 0% unsafe.*

Paired differences (95% bootstrap over stories):
* agentic vs `director` (rules still active): useful edits **+4.0 per 100 lines (CI +1.5 … +6.5)**: taking the rule-based vetoes away is what recovered the coverage.
* agentic vs `full`: useful edits −1.1 (CI −6.5 … +4.2): **not distinguishable** from the gates; edit rate −1.5 points (CI −8.5 … +5.3); judge-accepted (at least one judge) +5.2 points (CI 0.0 … +10.7);
  unsafe −6.0 points (CI −16.7 … +5.5).
* agent alone vs gates alone: unsafe edits −1.0 points (CI −13.0 … +10.6): **equally safe, not clearly safer**; useful −0.9 (CI −6.4 … +4.3); judge-accepted (either) +8.0 points (CI +1.0 … +16.3).
* What kind of unsafe edit remains: the gates alone made 3 policy errors (2 disrespect-to-elder contexts, 1 serious-moment) plus 4 wording errors; the agent alone made none of those two policy kinds among the
  5 flagged edits we classified (4 were editor wording errors: edits touching more than one word, a changed name; the 5th we did not classify). The remaining errors are the editor's wording, which neither
  decider controls and only the validator's audit catches, so the validator stays mandatory.
* Held-out (2 stories, 50 lines, no intervals): agentic 18.0 useful per 100 lines vs 20.0 for `full`.

**Conclusion:** the slang decision can be made agentic without losing measured quality (it is statistically level with the rule-based gates on net useful edits, higher on judge acceptance, and its errors are not the
policy kind), at about a cent per scene. It is the default. The honest claim is "equal, not better, on edit count"; the gain is that decisions come with reasons and generalise by reading context instead of
hand-written sets. Agent-alone safety is no better than rules, so the validator is the safety guarantee, not the agent.

## Result 3 — runtime review loop (`SLANG_REVIEW=1`, built, OFF by default): stricter, but it loses useful edits

After the editor, a second agent (`openai/gpt-5.4`, a third model family so it does not share a family with the Director or the scoring judges) re-reads the edited scene and may revert edits it judges wrong,
garbled, out of place or forced (`SlangDirector.review_scene`, `SlangRewriter._apply_review`; fails open). Paired over the same 7 stories (`outputs/ablation_review`), `agentic_review` vs `agentic`:

| | edit rate | unsafe | judge-accepted (both / either) | useful / 100 lines (both) |
|---|---|---|---|---|
| agentic (default) | 19% | 0% | 73% / 100% | 14.3 |
| agentic + review | 14% | 0% | 83% / 100% | 11.7 |
| paired difference | −4.9 pts (CI −7.3 … −2.5) | 0 | both +9.9 (CI +1.1 … +18.9) / either 0 | **−2.7 (CI −3.9 … −1.4)** |

* It reverted 15 of 49 edits (agent alone, no validator repair; `outputs/review_reverted_edits.txt` lists each with the judges' verdicts). **Every one of the 15 was accepted by at least one scoring judge, 10 by both**,
  so it removes edits the judges consider fine: strict precision rises (+9.9 points), net useful edits fall.
* **It did not make the verifier unnecessary.** Agent alone: 5 unsafe edits; with the review loop 4 (paired −0.3 points, CI −3.3 … +2.3). The audit flagged only 1 of the 15 reverted edits (a changed-meaning edit), so
  the review loop mostly overlaps taste, not the editor's wording errors that the deterministic audit catches.
* Reading the reasons and the author's own reading (a native speaker read all 15 and agreed with most of the reverts): the reviewer is mostly RIGHT. Sound catches include a filler that clutters a reassurance,
  "yaar" in a formal group moment, and three idiom frame errors ("Mujhe dil baith raha hai" x2 and "jaan nikal gayi" with the wrong subject: the idioms need "mera dil" / "meri jaan"), which the judges and the audit both
  missed. The loop is therefore not "wrong"; it exposes weak EDITS: a 59-phrase slang pool (47 edits used only 15 distinct phrases, "yaar" 32%, five phrases 70%), idioms inserted without their grammatical frame, and
  generic "ekdum kamaal / on point" for mild praise. A few reverts are taste, and two came with no reason.
* **Consequence for the metrics:** all 15 reverted edits were accepted by at least one scoring judge (10 by both). The lenient "either judge" rate (95 to 100%) therefore overstates quality; the strict "both judges" rate
  (73%, a lower bound) is closer to what a careful native reader accepts. Report both and say so.
* **Pairing caveat:** the arms are not edit-for-edit identical after the first revert, because reverted slang is no longer in the Director's "slang used so far" context, so later plans differ (10 of 15 displayed pairs
  are the exact edit the reviewer judged). The comparison measures the system with and without the loop, not the loop on fixed edits.
* Held-out (2 stories, 50 lines, no intervals): agentic 18.0 vs agentic + review 14.0 useful per 100 lines.

**Conclusion:** the loop is implemented and tested (about $0.0085 per scene with edits) and its reverts are mostly right, but as a veto it costs net useful edits (-2.7 per 100 lines) and does not touch the
writing errors the audit catches, so it stays off by default. The finding that matters is upstream: fix the slang pool and idiom frames, and use the reviewer to CHOOSE among candidates instead of only vetoing.

## What to say in the paper

A scene-aware agentic decision layer replaces the three hand-written gates at equal measured quality (net useful edits statistically indistinguishable, judge acceptance higher) and about a cent per scene,
with readable per-line reasons. Two honest negative results go with it: as an assistant to the gates it lost coverage (v1 / v2 prompts), and agent-alone safety is not clearly better than rules (14% vs 18% unsafe after re-scoring; first reported 11% vs 12%), so an independent deterministic verifier (the validator) remains part of the design: the agent decides where, the verifier checks how.

## Reproduce

```bash
python -m evaluation.ablation outputs/<story>_trace.json --out-dir outputs/ablation_director_v2 --backend qwen --modes no_validator_repair,full,director,no_editor
python -m evaluation.aggregate_ablation --dir outputs/ablation_director_v2 --arms no_editor,no_validator_repair,full,director --stories "hostel=fin_hostel_trace,..."
```

## v4 (2026-11-18): coverage without loosening any rule
Measured on 2,222 cached Director decisions: 24% nominated; refused for respect/formality 35%, serious moments 8%, no feeling word 23%, **quota-style reasons only 2%** ("only one nomination allowed", "Archana already nominated", "variety" - sub-quotas the model invented from the prompt's density hints). The editor side was already exhausted (mood retry, sampled n-best added 0 edits, filler fallback).
Changes (prompt v4; all gates, register rules, Gate-1 style rules, the rule gate and the validator are untouched):
1. The prompt no longer says "spread nominations over different speakers"; it states that the cap is the only limit, the code enforces it by keeping the strongest lines, and the model must not hold back below it.
2. Optional **filler-only** nomination (`"filler": true`): a casual peer line with no feeling word may get ONE light filler ("yaar", "arre") through the editor's existing filler route only (`_filler_kind` shape rules, register rules, comma-glitch / stacked-filler gates all still apply). Capped at 1 in 4 lines (`SLANG_DIRECTOR_FILLER_SHARE`) and inside the same per-scene cap. Never when the Director does not own the line.
Live check (2 stories, full pipeline, 0 errors): s01 11% -> 17%, s20 37% -> 40% of lines (different neutral text each run, so NOT a controlled comparison; the Director used filler-only once in 43 lines, and the editor found no natural filler there). Expect a few points, not a jump: the remaining refusals are the rules.
The controlled test is `python -m evaluation.director_ab --src outputs/stress4` (old v3 vs v4 prompt on identical neutral scripts, Director calls only, about $0.35, also counts unsafe nominations); it was NOT completed: OpenRouter credit ran out (HTTP 402).

## v6 (2026-11-19): the Director sees what the editor can swap
Measured on 70 runs: 28% of the Director's nominations had nothing the editor could swap (they became fillers or stayed neutral). Every dialogue line now ends with `[swappable: accha=praise]` or `[nothing the editor can swap]` (the same `slot_candidates` the editor uses); the prompt says to nominate slang only on a swappable line (filler-only otherwise), and the plan parser demotes any violating nomination BEFORE the per-scene cap. On the birthday script: 3 old nominations (1 with nothing swappable) -> 2 new, both swappable. Writer side (agents/agent3_synthesizer): `feelings_block` voices each PEER's analysed emotion in plain words, feelings go between peers, elders say tum/tu to the young. Supply of swappable lines: birthday story 20-22% -> 45-55%, short-film story 27% (48 lines).
