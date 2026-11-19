# Slang pool as data, grammatical frames, and the native-speaker review

## Why
The review loop (docs/DIRECTOR.md, Result 3) reverted 15 edits that every scoring judge accepted, and the author (a native speaker) agreed with most reverts. The causes were in the pool, not the model:
the slot lists were 42 hard-coded phrases (59 distinct in the whole bank; 47 edits used only 15 of them, "yaar" 32%), and subject-bound idioms were inserted without their grammar
("Mujhe dil baith raha hai", "Mujhe jaan nikal gayi": the idioms need "mera dil", "meri jaan").

## What changed
* `data/slang/slang_pool.json` is the source of truth for the slot lists used by `agents/edit_gate.py` (praise, criticism, worry, trouble, ease, predicate), with per-phrase metadata: sign, intensity, gender,
  grammatical frame, situational context, control labels, status (`core` = used, `candidate` / `rejected` / `disabled` = not used). The legacy lists remain as the fallback; `tests/test_slang_pool.py` checks every legacy phrase and its order are preserved.
* **Idiom frames.** Five subject-bound idioms carry `frame.possessive_before`. `repair_frame()` converts the experiencer pronoun in the editor's output ("Mujhe dil baith raha hai" -> "Mera dil baith raha hai"); `frame_ok()` makes the rule gate
  reject anything it cannot repair ("Mujhe ekdum dil baith raha hai" -> `idiom-frame`). The swap check normalises the possessive back to the dative so a repaired edit is still judged as the span swap it is.
* **Situational phrases.** "paisa vasool" is offered only when the sentence is about money / value (film, food, show, purchase).
* **Hyphenated phrases** ("haath-pair phool gaye") were split by the phrase tokeniser but not by the text tokeniser, so they could never pass; fixed.
* **Candidate drafting + review.** `tools/draft_slang_candidates.py` asks a strong model for new phrases per slot (with frames, intensity, gender, risk, a correct example), classifies the unused phrases already in the CSV bank, drops malformed drafts,
  and re-runs the rule gate on every example with all candidates temporarily promoted. `outputs/slang_pool_review/review.html` is the native-speaker form (Keep / Fix / Drop); `tools/apply_slang_review.py` promotes Keeps (with control labels so the polarity
  guards and control-based pools know them), marks Drops, and lists Fixes. Nothing drafted is used until reviewed.

## Measured effect of the frame fix (agentic arm, same 7 stories, same Director plans)
* Ungrammatical idiom edits ("mujhe" + idiom): 2 before, **0 after**; correct ("mera/meri" + idiom): 0 before, 2 after.
* Useful edits per 100 lines fell 2.3 (CI -3.4 ... -1.0) and judge-both acceptance 34 -> 29 of 44/42 edits. The line-by-line diff shows this is mostly NOT the idioms: removing "paisa vasool" from non-money sentences changed the
  seeded shortlist shuffle, so many lines got a different (equally plausible) slang word, some of which the stricter judge liked less; three lines that previously received a "yaar"/"kamaal" edit stayed neutral.
  **Sensitivity of the metric to the editor's seeded choices is therefore about 2 useful edits per 100 lines; effects of that size elsewhere in these docs (review loop -2.7, gate removals) must be read as within noise**
  (seed runs: see the end of this file once complete).
* A gap the diff exposed: "Ab sab ki dimaag fry kam ho gayi hai" (neutral: "...sab ki chinta kam ho gayi hai") was accepted by the rule gate and the audit but rejected by both judges. "dimaag fry" is an experiencer predicate
  ("mera dimaag fry ho raha hai"), not a noun that can replace "chinta" in a genitive. Frames need a second type (predicate vs noun-modifier); the native speaker's answer on "dimaag fry" in the review form decides the rule.

## Round 1 review applied, seed noise, and round 2 (2026-11-15)
* **Review 1 (native speaker):** 68 cards: 62 keep, 3 drop (bawaal, "default mein", "setting kharab"), 2 fix without a note (load, overthinking: not promoted), "full tension" confirmed. 54 new phrases were promoted, so the pool went from 45 to 99 live entries
  (praise 39, criticism 20, worry 14, trouble 14, ease 9). Two rules from the reviewer's notes are encoded: "dimaag fry" needs "mera" (frame on the noun "dimaag"; "sab ki dimaag fry" is rejected), and "on point" is fine as "on point hai" but never
  before "laga" (`forbid_after`, gate reason `slang-usage`).
* **New emotion slots.** Only 26% of the dialogue lines we generate contain a word the editor can replace. Joy ("khushi", "maza": 20 lines the editor could never touch), pride, anger and liking ("pasand") were added as slots defined in the pool file
  (`slot_defs`); the gate, the shortlist and the audit pick them up generically. Round 2 (`python -m tools.draft_slang_candidates --round 2`) asked the model for Hindi-first, more varied phrases (the round-1 list was dominated by English loanwords),
  including these slots, with the reviewer's approvals and drops as guidance; 111 drafts, 81 pass the rule gate and are shown in `outputs/slang_pool_review/review_round2.html` (candidates that could never pass the gate are not shown).
* **Seed noise (agentic arm, same Director plans, 3 editor seeds, 6 stories, scored by the current audit):** useful edits per 100 lines (both judges) 13.1 / 13.1 / 11.2 (max-min 1.9, sd 0.9); (at least one judge) 17.5 / 15.5 / 12.6 (max-min 4.9, sd 2.0); edit rate
  19.4 / 21.4 / 21.4%. **Differences of about 2 useful edits per 100 lines or less (the frame-fix -2.3, the review loop -2.7, the gate-removal effects) are within seed noise and should not be claimed as effects.** Larger, replicated effects stand:
  the validator repair's unsafe-edit reduction (-12 points, replicated in two independent runs) and the Director-assisted vs owning difference (+4.5, direction only). The stories' confidence intervals do not include this seed variance.
* The audit now includes the reviewer's rules, so older edits such as "on point laga" are flagged (unsafe 4 to 6 in the seed table, where earlier tables showed 0). Tables before this date used the older rules.

## Round 2 applied and measured (2026-11-15)
* **Review 2 (native speaker):** 81 cards: 78 keep, 2 drop ("baanchhein khil gayi", "jam raha hai"), 1 fix ("mast ekdum" -> "ekdum mast", already a kept card). Pool: 99 -> 177 live entries (praise 60, criticism 34, worry 23, trouble 21, ease 19; extra slots joy 7, pride 6, anger 4).
* **Measurement** (agentic arm, 7 stories, seed 7, same Director, `outputs/ablation_pool2` vs `outputs/ablation_pool1`): edits 42 -> 41, unsafe 4 -> 1, judges both 29 -> 25, either 41 -> 37, useful per 100 lines 11.9 -> 10.6 (both) and 16.2 -> 15.3 (either);
  paired difference -1.05 [-2.63, +0.87]. This is inside seed noise (sd ~0.9-2.0): **a larger pool did not measurably change quality or quantity.**
* **New phrases are used, but little:** 13 of 41 edits (8 distinct new phrases: jabra, balle balle, kamaal ka, dhaansu, pataka, ekdum mast, ek mazaak, bina sar-pair ka). Several read awkwardly (e.g. "yeh bina sar-pair ka baat hai", "hum balle balle karenge",
  "mujhe pataka laga"); the pool is validated per phrase but not per sentence frame, so a phrase approved as a word can still be wrong in a sentence slot.
* **The new emotion slots (joy / pride / anger) were never used.** 28 generated lines contain a trigger word (khushi, garv, naaz, gussa); 8 were edited, all on a different word ("accha", "kharab"). Two causes: the Director only chooses a register
  (hype/banter/frustration/...) and describes swappable words in praise terms, and when it did nominate a joy line (e.g. Mira, "khushi se ro rahi hoon") the Qwen v5 editor abstained ("kept neutral"), because it was never trained on these slots.
* **Implication:** more phrases alone do not help; the bottleneck is the editor's use of them. Next options: (a) retrain editor v6 with joy/pride/anger/like examples and the new phrases, (b) n-best candidate selection (a stronger model picks among editor
  proposals) so the editor's first guess is not final, (c) tell the Director about the extra slots.

## Retraining prepared: editor v6 (2026-11-16)
* **Gate fixes found while building the data (both would also have blocked these edits at runtime):** (1) `phrase_swap_ok` did not allow the verbs of the replaced span ("bahut gussa **aaya**", "bahut maza **aa raha hai**"): added `REMOVABLE_VERBS`; (2) the drafted idiom frames were wrong for oblique and feminine nouns
  ("mere mann mein laddu...", "mere pair zameen par nahi hain", "meri neend udd gayi", "meri rooh kaanp gayi", ...): 10 frames corrected in `slang_pool.json`.
* **Director prompt v3** now lists joy / pride / anger / liking words as swappable feeling words (cache key changes with the prompt version, so the next measurement re-queries the Director).
* **v6 data** (`python -m data_prep.gen_v6_rows sentences|rows|assemble`, ~$3.8 of API cost): 1,601 re-gated v5 gold edits + 3,514 new ones (Haiku writes the edit, the runtime rule gate incl. frame repair and a strict Sonnet judge >= 4/5 must both accept it; ~12% of writer items survive).
  Final: train 6,211 rows (172 distinct gold slang, none above 4%), val 390, held-out slot eval 879 (184 rows on the new slots; two domains never trained on), real-pipeline eval 2,013. Gold rows per slot: praise 1330, criticism 722, joy 608, ease 605, trouble 534, worry 403,
  anger 396, pride 384. New-slot abstain rows (negated, elder / teacher, third-person) added. 41 of the 132 new phrases have < 8 examples (idioms the writer / gate / judge rarely accept: e.g. "bina sar-pair ka", "sone pe suhaga", "rooh kaanp gayi" have none), so v6 will rarely pick those.
* Kaggle package: `kaggle_v6/` + `notebooks/train_slang_editor_lora_qwen_v6.py` (loads the pool into the gate; checklist adds per-slot checks for joy / pride / anger). Dry-run audit passes locally. After training: unzip into `models/slang-editor-lora-v6/`, convert with `tools/export_gguf.py`, re-run the 7-story measurement
  (agentic arm, same seed) and compare to `outputs/ablation_pool2`.

## Editor v6 trained, adopted, fixed and measured (2026-11-16)
* **v6 is the default** (`config.py`: `models/slang-editor-lora-v6`, `models/gguf/slang-editor-v6-Q5_K_M.gguf`). Kaggle checklist: all 13 items pass; held-out joy 84% / pride 100% / anger 100% gate-valid (v5: about 0-3%); abstain kept on real v5 sentences 0.89.
* **Why joy/pride/anger stayed at 0 edits first:** the editor *did* write the idioms, but `audit_edit` false positives made validator repair revert them (a sentence-initial capital counted as a changed entity; a negation anywhere in the line, or the "nahi" inside an idiom,
  counted as a negation change; a hyphenated idiom could not be inferred; the non-local limit was too tight for 6-7 word idioms). Fixes: negation checked only in the span's own sentence, sentence starters exempt, idiom-internal "nahi" ignored, slang inferred from the text,
  removed-word limit 8 for emotion idioms. Two gate changes: emotion idioms may replace a span of up to **7** words (others 5; `span_limit`, `SPAN_MAX` / `EMOTION_SPAN_MAX`), and a line with a praise word AND a joy / pride / anger / liking word is tried on the emotion slot first (`mood_x` route, +2.0 sentence-order bonus).
* **New usage rules** (`forbid_after` / new `forbid_before`, gate reason `slang-usage`; marked "pending native review" in the pool notes): "aag-babula" never after mujhe / mujhko / humein / ...; "ghazab ka" / "kamaal ka" never before laga-forms;
  "bina sar-pair ka" / "do kaudi ka" never before a feminine noun; the pride nouns naaz / fakhr / garv / gurur / maan never before laga-forms ("mujhe bahut naaz laga" slipped through once and is now blocked).
* **Measured (agentic arm, 7 stories, seed 7, same audit for both; `outputs/ablation_pool2` v5 vs `outputs/ablation_v6e` v6):** edits 41 / 45, unsafe 2 / 1, useful per 100 lines (both judges) 11.1 / 10.6, (either judge) 14.9 / 18.3, edits on the new slots 0 / 5 (joy 3, anger 1, pride 1),
  round-2 phrases 13 / 7. **No claimable effect on "both judges" (seed noise is about +-2); the one clear change is that joy / pride / anger slots are now used at all.**
* **Retrain v7: not done, deliberately.** 24 of 6,231 v6 training rows teach usages the gate now forbids (aag-babula + "mujhe" x12, "ghazab ka" x7, "kamaal ka" x5). They cost yield, not safety (the gate rejects the output). If v7 is ever trained: drop those rows, and update the drift guard / gate in `data_prep/v3_common.py` to the 7-word emotion span.
* **Read-through of the 44 edits of the final run (2026-11-16, by Claude, not a native speaker; needs native confirmation):** about three quarters read naturally. Weaker ones: "mujhe ekdum kadak laga", "mujhe ekdum bindaas laga", "aaj ka din bahut dhansu gaya", "mujhe bahut dope laga", "mujhe bahut pakau lag raha hai" (for "bura lag raha hai": hurt becomes annoyed),
  and a heavy reuse of "ekdum" / "yaar". These pass the audit and the lenient "either judge" rule, so that rule overstates quality; report the strict "both judges" rate as the lower bound. Candidates for rules or removal after native review.
* **Awaiting native-speaker review:** "mera paara chadh gaya hai", "mera dil garden-garden ho gaya hai", "mere pair zameen par nahi hain", "mera dil baag-baag ho gaya" (the two judge families split on these), the usage rules above, and the weak phrases in the previous bullet.
* **Repeatability:** re-running the default arm on the 7 frozen stories reproduced 6 of 7 stories exactly (the seventh changed only because of the pride-noun rule added after the first run). `tools/check_determinism.py`: 0 of 60 prompts changed output across 4 call orders, with llama.cpp prefix reuse on or off; the pipeline gave identical scripts under different `PYTHONHASHSEED`s.
  Claim "repeatable under fixed settings" only; `GGUFEditor` resets the context per call as a precaution (`QWEN_GGUF_REUSE_PREFIX=1` turns that off).

## Findings from a real voice-input run (2026-11-16)
A birthday story told by voice got only 3 edits in 30 lines. Causes, all fixed: (1) a sentence containing "Bhaiya" (in `PLAIN_FILLER_WORDS`) was skipped entirely as "already casual" - now it only blocks a second filler; (2) the trained editor abstains on any line that
already carries an address word, so the address word ("Bhaiya, ..." / "..., Bhaiya!") is set aside, the rest is edited as a normal sentence and the address word is re-attached (`_split_vocative`, switch `SLANG_VOCATIVE_STRIP`);
(3) "darr lag raha hai" was not a fear slot (only "dar" was) although the writer model almost always writes "darr"; (4) merely *mentioning* an elder ("Nani ka janamdin hai, ...") blocked mood slang - the gate now uses the same addressed-vs-mentioned rule as the audit (`addresses_elder`);
(5) a filler could land mid-phrase ("Accha yaar, hai, ...") because a sentence-initial capital counted as a name; a filler may now bring a new comma only as an opener or before a name.
Same 7 stories before/after: 44 -> 45 edits, unsafe 3% -> 0%, useful per 100 lines (both judges) 9.5 -> 9.5, (either) 17.3 -> 18.3 (no claimable change); the user's story went from 3 to 5 edits.

## 2026-11-18: thing-criticism words are not feelings
Reading every saved slang edit showed `mujhe ekdum flop lag raha hai` / `mujhe ekdum bakwaas lag raha hai ki tum aise le gaye`: a word that judges a THING replaced the feeling in "I feel bad / hurt", which changes the meaning. 29 criticism phrases that describe things or events (bakwaas, flop, ghatiya, kachra, bekaar, faltu, pathetic, ek mazaak, ...) now carry `experiencer_ok: false`, like the praise adjectives before them; real feeling words (bore, wahiyat, cringe, mid, dead) stay allowed. `yeh plan flop lag raha hai` is still fine. Backup: `slang_pool.backup_before_experiencer_criticism.json`. Test: tests/test_slang_naturalness.py (replays the saved defects).

## 2026-11-19 (demo day): what the pool contains, and what was added
File: `data/slang/slang_pool.json` (183 phrases: 125 words, 58 idioms; register: 105 general + 78 urban_youth). Fillers (yaar, arre, bro, bhai, abey, boss, chill, sis ...) are NOT in the pool; they are a short list in `agents/agent4_slang_rewriter.py` / `edit_gate.FILLERS`.
Per entry: phrase, slots (which feeling it may replace), sign, intensity 1-3, kind, gender, frame (idioms: "mera dil ..."), context, note (141 have a usage note), register, experiencer_ok (80 flagged), forbid_after / forbid_before, needs_copula.
Phrases per feeling slot: praise 60, criticism 34, worry 23, trouble 21, ease 19, predicate 9, joy 7, pride 6, sad/hurt 6 (new), anger 4, like 0 (trigger "pasand" exists, no phrases). NOT covered at all: excitement, surprise / disbelief, relief, tiredness, embarrassment, teasing.
Added today: slot `sad` for "bura lag raha hai" (mood off ho gaya, dil toot gaya, dil bhaari ho gaya, mann kharab ho gaya, low, down), fires only in the "bura lag-" frame.
Fillers are now capped: 1 per scene, the same filler at most 2x per story (`SLANG_MAX_FILLERS_PER_SCENE`, `SLANG_FILLER_REPEAT_MAX`), Director filler-only share 25% -> 15%.
Also fixed today: characters whose romanised spelling differs from the correct one ("adity", "simaran") were dropped from the profiles, so the Director treated two of four friends as formal (agents/agent2_character.name_in_story).
Backups of the files changed today: outputs/_bak_agent4_slang_rewriter.py, outputs/_bak_config.py, outputs/_bak_slang_pool.json.
