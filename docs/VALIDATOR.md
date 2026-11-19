# The Validator (Layer 5) — design, evaluation, limits

Code: `evaluation/muril_validator.py` (entry point), `evaluation/edit_audit.py` (the audit + repair), `evaluation/fluency.py` (optional LM check),
`tools/calibrate_validator.py` (fits the centring), `evaluation/eval_validator.py` (validates the validator). Calibration files: `data/validator/`.

## What was wrong with version 1

Version 1 compared the neutral and slang text with the **raw cosine of MuRIL embeddings** and a keyword match for events.

* Raw MuRIL embeddings of Hinglish lines are almost parallel (*anisotropy*): the mean cosine of two **random** project lines is **0.995**.
  Every scene scored ~0.999, no threshold could ever fire, and the redo loop was effectively dead code (measured: across the 16 scene validations of
  the 2-story ablation, the only non-passing scenes were `NO_SLANG`, never `MEANING_DRIFT`).
* Edit quality is not semantic similarity: "yeh bahut bura hua → yeh ekdum wahiyat hua" is good, "mujhe ekdum jhand lag raha hai" on a fear line or slang on
  "Papa, aap thik ho?" is bad, and both change the embedding by the same tiny amount.
* Event recall counted an English event as covered if **any single word** appeared in the Hinglish dialogue.

## Version 2

| Part | What it does | Training? |
|---|---|---|
| **Centred MuRIL** | subtract the mean embedding of the project's own lines (3,000) and remove the top-5 principal directions (Mu & Viswanath "all-but-the-top"). Mean random-pair cosine **0.995 → 0.001**. | none (statistics only, 2 min) |
| **Edit audit** | re-verifies every changed line from the text alone: *locality* (one small change, no appended clause), *preservation* (names, numbers, negation, question vs statement), *polarity* (the slang's mood vs the word it replaced), *context* (no slang on health lines, on lines addressed to an elder / teacher / "aap", or from elder speakers), the deterministic *rule gate* re-run independently, *style* (crude slang, stacked fillers). | none (rules, ~ms per line) |
| **Repair** | a line that fails a **blocking** check is reverted to its neutral Pass-1 text. Safe by construction, no model call, no retry loop. The pipeline does this in `_validate` and stores the repaired script. | none |
| **Semantic event coverage** | event description ↔ best dialogue line by centred MuRIL similarity. `EVENT_LOSS` is raised only when the **editor** lost events (recall below the neutral text's recall); events Pass 1 never contained are the Fidelity Critic's job and cannot be fixed by re-running the editor. | none |
| **LM fluency** (optional, off) | NLL change of the line under the local GGUF model. | none |

Failure modes: `OK | NO_SLANG | BAD_EDIT | LOW_COVERAGE | MEANING_DRIFT | EVENT_LOSS | MONOTONOUS`. `NO_SLANG` and `BAD_EDIT` scenes are accepted
(neutral by design / after repair), the others go through the targeted redo loop exactly as before.

**No retraining of any model is needed**: everything is statistics, rules, or calibrated thresholds.

## Evidence (`python -m evaluation.eval_validator`, results in `outputs/validator_eval.json`)

**E2 — meaning-drift detection.** Good edits = 222 gold edits of the held-out `kaggle_v5/eval_slots` sentences. Each is corrupted five ways. Cosine
thresholds are set so that 95% of the good edits pass; nothing is tuned on the corruptions.

| Corruption (n≈222 each) | raw cosine (v1) | centred cosine | **edit audit** |
|---|---|---|---|
| hallucinated clause appended | 35.6% | 54.5% | **100%** |
| content word swapped for another | 23.4% | 27.9% | **89.6%** |
| negation flipped | 11.7% | 11.3% | **100%** |
| unrelated line | 100% | 100% | **100%** |
| wrong-mood slang (praise ↔ criticism) | 7.4% | 7.9% | **99.5%** |
| **overall detection** | **36.5%** | **41.3%** | **97.8%** |
| good edits wrongly rejected | 5% (by design) | 5% (by design) | **13.1%** (29 of 222) |

*Re-run 2026-11-16 after the v6 gate/audit changes (previous run, 2026-11-12: 97.7% detection, 0.9% good rejected). Detection is unchanged. The
rejection rate rose because the audit is now stricter about grammar (idiom frames, gender agreement, usage rules), and the 222 "good" edits are the
v5 gold labels, which contain such errors. All 29 rejections are grammar errors in that gold set, not in the validator: 23 are `"mujhe dimaag fry ho raha
hai"` / `"mujhe jaan nikal gayi"` / `"mujhe pasine chhoot gaye"` / `"mujhe fatti padi hai"` / `"mujhe dil baith raha hai"` (these idioms need `mera/meri`),
2 are `"bada panga ho gayi"` (panga is masculine), and 4 are `"ekdum on point laga"` (a usage rule added in the v6 round that is pending native review,
so those 4 are debatable). The edit audit therefore found bugs in the gold set, again. Excluding the 4 debatable rows, 25 of 29 are unambiguous gold errors.*

**E1 — real judged edits** (1,132 teacher edits labelled keep / not-keep by a strict judge, older, looser contract): cosine (raw 0.47, centred 0.49 AUC) is
**at chance**, as expected: quality is not similarity. The audit passes 74.8% good among the edits it accepts and rejects 89.6% of the bad ones; its
rule-gate component accounts for almost all of that (the same labels give 74.1% / 89.2% for the rule gate alone). The remaining misses are
*fluency / word-choice* judgements ("jo ek sahi hai sanket hai") that rules cannot see.

**Fluency probe** (`outputs/validator_fluency_probe.json`, 500 judged edits): the LM check is a **weak** signal — AUC 0.57 overall, 0.70 among the edits the
audit already passes; at a threshold that keeps 90% of good edits it rejects only 18.5% of the bad ones, at 2.6 s/edit. It is therefore **off by default**
(`VALIDATOR_FLUENCY=1` to enable) and reported as a negative result. (Loading llama.cpp inside the same process as torch also triggers the machine's known
intermittent native crash, another reason to keep it optional.)

**E3 — event coverage** (278 scenes, 2,208 event/scene pairs: an event vs its own scene and vs a random other scene): AUC **0.69 (keyword) → 0.77
(semantic)**. Still a weak detector, which is why `EVENT_LOSS` is relative (editor-caused loss only) and the Fidelity Critic owns Pass-1 completeness.

## Found and fixed by the final end-to-end check

Running the finished pipeline on a fresh story exposed two grammar defects that both the editor's rule gate and the audit let through: `"Mujhe full tension ho rahi tha"`
(feminine idiom, masculine auxiliary) and `"yeh jhamela hogi"` (masculine noun, feminine future). The gender-agreement rule in `agents/edit_gate.py` only looked at the
form right after `ho`; it now checks the whole verb chain (`ho rahi thi`, `hoga / hogi`, `jayega / jayegi`), which fixes the editor **and** the audit (shared rule).
`MONOTONOUS` (one slang reused in a 3-edit scene) is now a reported warning instead of a re-do: the editor is seeded and deterministic, and three retries produced the
identical scene (measured).

Two unseen stories (`data/raw/heldout_cafe_story.txt`, `heldout_train_story.txt`, written after the rules were fixed, so they test generalisation) were then run end to end and every
edit read by hand. 5 of 25 lines were edited in each story; all edits passed the audit and both judges. Reading the output found two more defects the metrics had missed:

* **Nukta transliteration bug** (`preprocessing/transliterate.py`): ज़ फ़ ख़ ग़ क़ ड़ ढ़ were dropped or given a stray vowel in either Unicode form ("ज़ोया" -> "Jaoya", "फ़िर" -> "phair",
  "प्लेटफ़ॉर्म" -> "Pletapharma"), which also garbled the text given to Agent 1. They now romanize as z / f / kh / g / q / d / dh in both forms (`tests/test_nukta_transliteration.py`).
* **Comma glitch** (`agents/edit_gate.py`, shared by the editor and the audit): "Accha hai, lekin" -> "Arre accha, hai, lekin". A new comma is now allowed only directly after an inserted
  filler ("Arre,"); 2 of 222 gold edits are still rejected, none of the 41 real full-system edits was affected.
* **Event recall is a diagnostic, not a re-do trigger**: with the LLM check on, an `EVENT_LOSS` re-do fired although the judge answers were identical for the neutral and the edited scene (noise);
  audited edits are small and local, so a real editor-caused loss is implausible.

The Gradio app shows the validator's work (a "Validation" box under the screenplay with every reverted edit and its reason, an "Edits reverted" counter, and a checkbox for the AI-judge
second opinion); verified by driving the app in a browser on the hostel story (5 edits kept, 4 reverted with reasons).

## Honest limits

* Slang *quality* (is it funny, natural, in the right register for *this* character) is only partly rule-checkable. The audit blocks the clearly unsafe
  edits; subtler ones need human judgement (see `evaluation/human_eval.py`, ids are now opaque and identical line pairs are rated once).
* The labelled judged edits come from an older, looser contract and a single (LLM) judge; they are an *indicative* yardstick, not ground truth.
* Thresholds (`data/validator/thresholds.json`) are calibrated on this project's stories and gold edits; re-run `tools/calibrate_validator.py` and
  `evaluation/eval_validator.py --write-thresholds` after changing the corpus.
* The audit is intentionally conservative (health / elder / "aap" lines never get slang); this lowers coverage slightly, never raises risk.

## Update 2026-11-16: the centred cosine no longer triggers a re-do on audit-clean edits
Found on a real voice-input run: an idiom swap ("mujhe bahut khushi ho rahi hai" -> "mere mann mein laddu phoot rahe hain") passed the edit audit but scored a centred cosine of 0.53 (< the calibrated 0.6655),
so the scene was marked `MEANING_DRIFT` and re-injected three times (about +140 s), each time re-picking the same idiom, then accepted as "best effort". An idiom changes the words and therefore the embedding while keeping the meaning;
on judged real edits the cosine is at chance (AUC 0.47-0.49) and adds 0.1 point of detection over the audit (97.9% vs 97.8%, table above). Now `MEANING_DRIFT` from the cosine fires only below **half** the calibrated
threshold (a safety net for gross drift); between half and the threshold the cosine is reported as a diagnostic note. The audit is unchanged and still reverts unsafe edits. Test: `tests/test_validator_drift.py`.

**About `event_recall`.** It is a weak, lenient diagnostic, not an overfitted one (nothing is trained on it): an event counts as covered when its centred similarity to the best single dialogue line is >= `event_sim_min`
(0.113, the 25th percentile of true event/scene pairs, so 75% of true events pass), but a *random* wrong event reaches 0.176 at the 90th percentile (`event_sim_chance`), so most events pass whether or not they are present
(AUC 0.77). A scene with no key events reports 1.0 by definition. Read `recall = 1.00` as "no evidence of loss", not "every event verified"; event completeness rests on the Fidelity Critic (and the optional LLM check, 86.9%).

**Drift is repaired by reverting the line, not by re-running the scene (2026-11-16).** `validate_and_repair` now handles `MEANING_DRIFT` like any other unsafe edit: the changed line with the lowest centred similarity (below half the calibrated threshold) is
marked with the blocking flag `meaning-drift` and reverted to its neutral Pass-1 text through `edit_audit.repair_scene`, the scene is re-diagnosed, and this repeats until no line is that far off. Other edits in the scene are kept; a scene left without edits is accepted as neutral
(`NO_SLANG`), never re-run. Reason: with a cached Director plan and a deterministic editor a re-run reproduced the same edit three times. Tests: `tests/test_validator_drift.py`.
