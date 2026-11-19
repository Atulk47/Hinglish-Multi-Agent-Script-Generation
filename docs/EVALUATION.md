# Evaluation protocol — how the outputs are validated, and how the validation itself is validated

The question this answers: *"How do you know the slang edits keep the meaning, tense and grammar, make sense, and are appropriate — and why should anyone trust
your metrics?"*  Answer: **three independent layers of evidence, each with confidence intervals, none of which is "I read it and it looked fine".**

## 1. What is measured (criteria → metric)

| Criterion | Metric | Type |
|---|---|---|
| Safety: no slang on health lines, elders / teachers, "aap" lines; no elder speakers | rule audit `evaluation/edit_audit.py` | deterministic |
| Structure: one small local change; names, numbers, negation, question form preserved | rule audit | deterministic |
| Grammar: gender / tense agreement of the slang (`ho rahi thi`, `hoga / hogi`) | shared rule gate `agents/edit_gate.py` (+ judge) | deterministic + judge |
| Mood fit: slang polarity matches the replaced word (praise ↔ criticism) | rule audit | deterministic |
| Meaning preserved, grammar, naturalness, situational fit | two-family LLM rubric judge `evaluation/judge.py` | LLM, validated |
| Story fidelity: events present, nobody absent speaks, secrets kept | Fidelity Critic + LLM event check | LLM, validated |
| Coverage / diversity | edit rate, slang diversity | counts |

Embedding similarity (raw or centred MuRIL) is **not** used as quality evidence: it was shown to be at chance for that (below) and is kept only as a coarse drift diagnostic.

## 2. Layer 1 — known-answer tests ("is the metric sensitive to the failure it claims to detect?")

Following the perturbation-checklist idea (Sai et al. 2021) each metric is tested on **good edits** (222 gold edits of the held-out `kaggle_v5/eval_slots` sentences)
and on the same edits **corrupted five ways** (hallucinated clause, content-word swap, negation flip, unrelated line, wrong-mood slang). `evaluation/eval_validator.py`,
`evaluation/eval_judge.py`; 95% Wilson intervals; nothing is tuned on the corruptions.

| Metric | Corrupted edits caught | Good edits accepted |
|---|---|---|
| raw MuRIL cosine (the old validator) | 34.9% | 95% (threshold set so) |
| centred MuRIL cosine | 41.4% | 95% (threshold set so) |
| bidirectional NLI (mDeBERTa-XNLI) on romanized text | 47.4% (n=340) | 95% (threshold set so) |
| **rule audit** (re-run 2026-11-16) | **97.8%** (n=1,077) | **86.9%** (193/222; the 29 rejections are grammar errors in the v5 gold labels, see VALIDATOR.md; was 99.1% before the stricter grammar rules) |
| judge: Claude Haiku 5.5 | 98.8% [97–99] (n=488) | 79% [70–86] (n=100) |
| judge: Gemini 2.5 Flash | 94.3% [92–96] | 97% [92–99] |
| **both judges must accept** | **99.8%** [99–100] | 77% [68–84] |

Agreement between the two judge families: Cohen's κ = 0.69 (raw 91%). Unparseable judge answers: 6 of ~1,200 (always counted as *not accepted*, never as accepted).
Event coverage (160 scene/event pairs, an event vs its own scene and vs another scene): **LLM check 86.9% [81–91]**, centred MuRIL 73.1%, keyword matching 64.4%.
Real judged edits (1,132, older looser contract): cosine AUC 0.47–0.49 (chance), audit ≈ rule gate (precision 0.75, bad rejected 0.90). LM-fluency score: AUC 0.57 (weak, off).

What this establishes: the rules and the judges are sensitive to each corruption type; embeddings are not. What it does **not** establish: that they agree with people on *real*
edits, whose failures are subtler than synthetic corruptions → layer 3.

## 3. Layer 2 — convergent validity on real outputs

On the real edits of the ablation arms, the audit and the judge agree at κ ≈ 0 on a *single* overall accept — because they measure different things (safety / structure vs
naturalness / fit). The honest reading, used throughout the project: **report them separately** (`unsafe edits`, `judge-accepted`) and never as one number. The multi-story table
(`docs/RESULTS.md`) does exactly that. A second consistency check: with the strict (both judges) rule real edits are accepted at 68–75% and gold edits at 77%, so the judge's strictness, not only the editor, explains the
remainder; with the lenient (either judge) rule 87–95% of real edits are accepted, matching the human rater's 94% (§4a).

## 4. Layer 3 — a small human calibration set (validates the metric once)

`python -m evaluation.calibration create` builds **125 blinded items** (75 real edits from 8 runs of 7 stories, 25 gold edits, 25 deliberately corrupted controls), opaque ids, shuffled
per rater, 4 rater sheets in `outputs/calibration/` with `RATER_INSTRUCTIONS.md` (≈45 min per rater, three ratings per row). After at least two (ideally three or four) sheets are returned:

```bash
python -m evaluation.calibration analyze --dir outputs/calibration
```

reports rater reliability (Krippendorff's α for the two 1–5 ratings; each rater's accuracy on the gold and control items, which doubles as a quality check), the human consensus,
and for every automatic metric (each judge, the ensemble, the audit, audit-and-judge) its agreement with that consensus: Cohen's κ with bootstrap CI, true-positive / true-negative
rates, accept rate per item kind, and the Spearman correlation between the judge's naturalness and the humans' mean rating. **Result: which metric to trust and how much.** After this
step humans are not needed again; every larger or repeated evaluation uses the validated automatic metric. *(Status: one rater done, see below; a larger multi-rater set is ready in `outputs/calibration`.)*

### 4a. Result of the single-rater calibration (50 items, `outputs/calibration_small`)

`python -m evaluation.calibration_small create` built **50 blinded items**: 32 real edits (8 from each combination of rule-audit accepts / blocks x both-judges accept / reject, so the
cases where the validators disagree are rated too), 9 known-good gold edits and 9 deliberately broken controls. The rater is the project author (a possible leniency bias), and there is
**one** rater, so there is no inter-rater agreement; every number is agreement with that one person. Confidence intervals are bootstrap / Wilson and wide (n = 50).

* **Rater reliability:** accepted 9/9 gold edits and rejected 9/9 corrupted controls.
* **Real edits:** kept 30 of 32 (94%); mean meaning/grammar 4.28 and naturalness/fit 4.16 (1-5). Even the edits the validators reject were mostly kept (7/8 of the audit-blocked +
  judge-rejected quadrant).
* **Agreement with the person** (Cohen's kappa [95% CI]; TPR = of the edits the person keeps, how many the metric also accepts; TNR = of the edits the person rejects, how many it also rejects):

| Metric | κ vs "keep" | TPR | TNR | κ vs "both ratings >= 4" |
|---|---|---|---|---|
| Gemini 2.5 Flash judge | **0.83** [0.61-1.00] | 95% | 91% | 0.59 |
| either judge accepts ("any") | 0.75 [0.48-0.94] | 97% | 73% | 0.53 |
| Claude Haiku 5.5 judge | 0.25 [0.01-0.48] | 62% | 73% | 0.27 |
| both judges must accept | 0.33 [0.13-0.55] | 59% | 91% | 0.33 |
| rule audit, QUALITY checks only (grammar, polarity, negation, locality ...) | 0.48 [0.21-0.72] | 77% | 82% | 0.53 |
| rule audit, safety POLICY only (health / elder / "aap") | 0.06 [-0.19-0.37] | 87% | 18% | -0.06 |
| rule audit, all checks | 0.39 [0.17-0.61] | 64% | 91% | 0.39 |
| Spearman, judge naturalness vs the person's mean rating | 0.38 | | | |

What this says: (1) the **Claude judge is stricter than the person**, so the "both must accept" rule rejects ~40% of edits the person would keep; Gemini alone, or "at least one judge accepts",
tracks the person well. (2) The audit's **quality** checks agree moderately with the person (they catch real grammar errors the person also noticed, e.g. "full tension ho rahi tha"); its **policy**
checks are unrelated to naturalness *by design* (they forbid slang on elders and health lines even when the edit reads fine) and must be justified as a design policy, not as a quality metric.
(3) The strict bar matters: under "both ratings >= 4" every metric agrees less, so these numbers describe agreement with one person's bar.

Decisions taken from it: the optional run-time judge now reverts an edit only when **every judge rejects** it (`VALIDATOR_JUDGE_RULE=any`, default; `all` keeps the strict rule); the ablation reports
the strict ("both") and lenient ("either") judge rates side by side; the audit is described as safety policy + structural quality, not as a general quality score. The thresholds were chosen after seeing
this set (n = 50, one rater), so they are indicative: confirm them with the multi-rater set before treating them as final.

## 5. Statistics

System-level claims resample **stories** (the independent unit), 95% bootstrap CIs, paired differences over the same stories (`evaluation/stats.py`). With 7 stories the intervals are wide
and no significance is claimed beyond what the paired intervals show (Card et al. 2020 on small-sample power). Rates use Wilson intervals. All judge verdicts are cached on disk
by (rubric version, model, text), so every number is reproducible for free; rubric `edit-judge-v1`, temperature 0, two model families neither of which is the editor.

## 6. Threats to validity (say them before the reviewer does)

* Synthetic corruptions are authored by us and the audit's rules overlap them; hence the independent judge, and hence the human layer.
* LLM judges can be strict or biased (here: Claude strict, Gemini lenient; the ensemble inherits the strict one). κ between judge families is 0.69, not 1.
* Judge validity on real edits is only as good as its agreement with the calibration set (pending).
* 7 stories; Pass-1 text varies between pipeline runs (frozen in the ablation); one editor seed.
* Slang naturalness is partly subjective: rules encode only objective failures (we deliberately did not turn a few judge opinions into rules — they conflicted with the gold data).

## 7. One-command reproduction

```bash
python -m evaluation.eval_validator            # E1-E3: cosine / audit / events (outputs/validator_eval.json)
python -m evaluation.eval_judge                # J1-J2: judge known-answer + event check (outputs/judge_eval.json)
python -m evaluation.aggregate_ablation --dir outputs/ablation_multi --stories "..."      # the multi-story table
python -m evaluation.calibration create|analyze                                           # human calibration set
```
