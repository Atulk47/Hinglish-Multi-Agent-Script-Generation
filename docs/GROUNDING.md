# Grounding: no claim enters the shared scratchpad without evidence from the story

## Why
The agents share a scratchpad so each can build on the others. That also lets one agent's mistake become every later agent's "fact".
A real run on a classical Hindi novel excerpt (auto-romanised, then read by the analysts) showed it:

| What the source says | What entered the scratchpad / the script |
|---|---|
| झुर्रियों (wrinkles) | "forehead filled with **hair**" |
| बँधी (tied) | "something is **locked** at the door", location "where the object is locked" |
| रस-पानी करना (have a light refreshment) | "the rasa-pani has **spilled**" -> "rasa-pani gir gaya" in the dialogue |
| "...बोला—रायसाहब के यहाँ" | "going to **Bolarayasahaba's** place" |
| nothing | a **secret** ("the cow was bought with a shady loan") applied to 18 scenes; 17 **"Narrator:"** lines; a model note "Wait, that violates the rule: narrator cannot speak" shipped as a speaker |
| one-sentence paragraphs | 35 scenes, each padded to up to 6 invented lines |
| a 1930s village | "teri gaay **dope** lag rahi hai", "**cringe**", "**panic**", "Dhaniya **yaar**" between husband and wife |

The rule that stops this, **for any story**: *a claim is accepted into the scratchpad only if it carries a quotation from the story and that quotation really is in the
story.* Everything below is deterministic code around the model calls (no story-specific word lists).

## Mechanisms (P0: hallucination)
| Where | What |
|---|---|
| `agents/grounding.py` | `span_supported` (a quotation must be in the source, Devanagari or romanised; bag-of-words does not count), `clean_location` (unhedged AND quoted, else `Unknown`), `ground_events`, `spoken_lines` (the author's own words: dash after a speech verb / dialogue paragraph / quotes), `shows_concealment` (a "secret" must speak of hiding / surprise / not telling), `role_speaks_in` (an unnamed speaker must precede a speech verb in the text) |
| Agent 1 (`agent1_narrative.py`) | reads the **Devanagari original** (romanisation is only an aid), must give one quotation per event and for the location; events without a real quotation are dropped, a location without one becomes `Unknown`; answers are forced to Roman script (leftover Devanagari is transliterated); what was dropped is recorded in `grounding` |
| Persona (`agent2b_persona.py`) | secrets need a quotation **and** a concealment cue; unnamed speakers need a quotation with speech or a role word before a speech verb; records the story **setting** (era / milieu / formality) |
| Agent 3 (`agent3_synthesizer.py`) | no "narrator" listener fallback; the line budget follows the source (speech lines + 1 + sentences/3, not a flat 6); the author's spoken words are handed over as text that must survive translation; `sanitize_pass1` keeps only lines whose speaker is a real character (one canonical name per person), drops Narrator / self-notes / empty labels / English prose |
| Critic (`agent3b_critic.py`) | a narrator is always an issue (it used to be whitelisted); runs against the whole cast when the scene has none; new places / destinations / money / deal terms / family relations count as invented facts; **fail closed**: lines still flagged after the retries are removed instead of shipped (never emptying a scene) |
| Output | a location that is `Unknown` is left out of the screenplay instead of printed as a guess |

## Mechanisms (P2: register follows the setting)
* `data/slang/slang_pool.json`: every phrase carries `register`: `urban_youth` (English-origin / modern campus slang: dope, cringe, full tension, red flag ...) or `general` (Hindi idioms:
  jaan nikal gayi, dil baith raha hai, paara chadh gaya ...). Tagged by `tools/tag_slang_register.py` (two documented rules, rerun after adding phrases).
* The Persona Agent records `setting` = era / milieu / formality (model answer, lexical second opinion as fallback, modern casual default).
* In a traditional / historical / rural story `edit_gate` offers no `urban_youth` phrase or filler (`bro`, `chill`, `boss`, `abey`), `rule_gate` rejects one (`register-mismatch`),
  the Director is told the setting, and "yaar" is not placed between people the Persona Agent relates as spouses, parent and child, in-laws or master and servant.
* Per-story name sets (`FEMALE_NAMES`, `RESPECT_ADDR`) are reset at the start of every story, so one story's people no longer leak into the next in a long session.

## Verified (2026-11-17, three very different stories through segmenter -> Agent 1 -> Persona -> Agent 3 + critic)
| Story | Narrator / meta lines | Devanagari in names / locations | Setting found | Secret |
|---|---|---|---|---|
| classical village excerpt (34 scenes) | 0 (was 17) | 0 | historical / rural / traditional | invented one dropped |
| birthday surprise (5 scenes) | 0 | 0 | contemporary / urban / casual | real surprise kept |
| cafe (4 scenes) | 0 | 0 | contemporary / urban / casual | none |

Limits (state them): the grounding check proves that a quotation EXISTS in the story, not that it supports the claim (hence the extra concealment / speech cues); the critic is an LLM and can
miss an invented line; location is `Unknown` for most paragraph-sized scenes of literary prose (that is the honest answer); the scene segmenter still treats each author paragraph as a scene
(P1, not done).

Tests: `tests/test_grounding_p0.py`, `tests/test_register_p2.py`, updated `tests/test_fidelity.py`.

## Round 2 (2026-11-17): scenes, hidden feelings, sentence frames
Found on a second real run (Premchand's "Idgah", 15 one-paragraph scenes; a raw English line in scene 1; Amina saying aloud the worry the author says she hid; "mujhe bahut lajawaab lag raha hai").

* **Scenes follow the story (`preprocessing/segmenter.py`).** Author paragraphs are *atoms*, not scenes. Gap signals between paragraphs: a time / place cue opening the next paragraph (now also arrival / return / on-the-road cues such as
  "पहुँचकर", "वापसी के रास्ते में", "घर पहुँचते ही"), lexical cohesion of the surrounding paragraphs, and speech continuing across the break (an exchange stays one scene). A boundary is kept where the score is high; a scene shorter than 3 sentences is merged into a
  neighbour only across a *weak* boundary (a clear "next day" keeps even a two-sentence scene); a scene longer than max(14, 2 x sentences / cap) is split at its best internal gap. **There is no target number**: a 4-paragraph story gives 3 scenes, Idgah 10, the village excerpt 8-14.
  **Hard upper limit `SEGMENT_MAX_SCENES` = 15** (weakest boundaries are merged first). For 6+ paragraphs an LLM referee may regroup them ("one continuous beat: place, time, people, activity"), but its answer is rejected (and the deterministic result used) if it exceeds the cap or the size ceiling.
* **Narration-only scenes.** A scene with no named person and no spoken words (pure scene-setting) is kept *without dialogue* ("*(scene-setting narration, no spoken dialogue)*"); the writer is not called, so it cannot invent a speaker. If the writer's answer has no valid line, it is retried with the cast named, then left empty. Raw model text is never passed on.
* **What was NOT said aloud (`grounding.unspoken_cues`).** Sentences of the source that say something was hidden, silent, only thought or unanswered ("उसने अपनी चिंता छिपा ली", "वह चुप रहा", "शब्द नहीं निकल रहे थे") are given to the writer ("never turn these into spoken lines") and to the critic, whose `contradiction` type now includes a character saying aloud what the author says they hid, or speaking where the source says they stayed silent.
* **Sentence frames for praise slang (`edit_gate.experiencer_frame_ok`).** A praise *adjective that describes things* (lajawaab, zabardast, kamaal ...; pool field `experiencer_ok: false`, set by `tools/tag_slang_register.py`) cannot stand in "mujhe (bahut) ___ lag raha/rahi/rahe hai" (I FEEL ...), where only feel-good words (mast, bindaas, badiya, cool ...) fit. "Mujhe yeh bahut lajawaab laga" and "mujhe ekdum zabardast laga" (past, "I found it ...") stay valid. The shortlist for such a sentence leaves the thing-adjectives out, so the editor does not waste an attempt.

Verified on Idgah: 10 scenes at real place / time changes (village morning, Hamid and Amina, the walk, the fair, the shop, the friends' mockery, the way back, home); scene 1 narration-only; Amina no longer states her worry and Hamid (not Amina) says "meri chinta mat karna"; 4 unsupported lines removed by the fail-closed critic; no Narrator; locations grounded or `Unknown`.
Known limit: when the critic flags *every* line of a scene (scene 10: Hamid and Amina speak where the source has them silent) the scene keeps its lines (fail-closed never empties a scene), and the issue is reported.

## Round 3 (2026-11-17): a failed model call must not lose a scene (Mohan and Raju story)
A 5-paragraph story came out as 2 scenes; scene 2 had no events and no characters, Raju (the second main character) never entered the cast, his umbrella exchange was rewritten as Mohan retelling it to his mother, and the log blamed a "Groq rate limit" although every call had gone through OpenRouter. Cause: one Agent 1 call came back empty and nothing retried it.
* **Agent 1** now tries up to 3 times (original call, then a 4000-token budget, then a second non-reasoning model from the same gateway) and marks a scene it still could not analyse (`analysis_failed`); the warning no longer blames a rate limit.
* **A speaker named in the scene's own text** (difflib match against the source tokens) is kept even if Agent 1 missed him; an invented name is still rejected. Such a speaker joins the scene's cast for the critic.
* **Scene size.** The ceiling is `max(12, 1.5 x sentences / cap)` and a strong time / place cue opening a sentence cuts a scene even inside a paragraph when both sides keep 3+ sentences (`cue_cuts`). Mohan: 4 scenes (the evening; Raju and the umbrella; the rain and arriving home; indoors).

## Round 4 (2026-11-17): reported speech, one-scene stories, predicate idioms (Kabir and Arav on a bench)
* **Reported (indirect) speech** ("कबीर ने कहा कि ...", "याद दिलाया कि ...") is extracted (`grounding.reported_speech`), handed to the writer as "(reported)" lines to be written as direct words, and counted as speech by the segmenter. A story told only this way had no spoken-words list, so a line of the exchange went missing.
* **Segmenter:** connectives ("फिर", "लेकिन", "उसके बाद") are weak evidence of a scene change (0.15, was 0.45); the ceiling floor is 16 sentences and an oversized group is cut with ONE balanced cut where possible. One continuous conversation in one place stays one scene.
* **Chunks = scenes.** `ProseChunker.from_scenes` replaces `chunk(...)` in the pipeline: `chunk` re-split a one-scene story into ~4-sentence pieces and broke the scene <-> original-text alignment.
* **Critic:** setting facts, positions, gestures and silent thoughts are never "missing events".
* **Predicate idioms** ("bacchon ka khel", "left hand ka khel", "sone pe suhaga", "paisa vasool"; pool field `needs_copula`) must be followed by a linking verb or the end of the clause.
* **Agent 1 retry order:** primary model, then the OTHER model, then primary with a bigger budget; the log shows what a failed answer looked like.

## Round 5 (2026-11-17): evening scene, person-subject agreement (Riya and Siya)
* **Time-of-day openers.** A paragraph that opens with a time of day (सुबह, शाम, रात, दोपहर, सवेरे ...) is a strong time jump, and a strong cue waives the "speech continues across the break" penalty. "शाम होने तक ..." now starts its own scene (the 13-sentence scene that swallowed the evening and the plan to meet again is gone).
* **Person-subject idioms** (pool set `PERSON_SUBJECT`: aag-babula, bore): the verb after "ho" agrees with the speaker ("main ... thi, isliye aag-babula ho gayi"). Repaired from the Persona gender (else from the line's own "thi" / "tha"), rejected (`person-agreement`) if it cannot be.
* **"bro" and "bhai"** are removed from the shortlist for a line addressing a girl (`drop_male_address_for_female_names`).
* Residual: an invented object ("solar system wala") can pass the LLM critic.

## Rounds 6-8 (stress corpus, 2026-11-17)
* Round 6-7: fuzz/fault-injection found a quadratic regex in `speech_subjects` (121 s), `None`/wrong-shape JSON crashes, writer/critic exceptions killing scenes, a non-idempotent sanitiser; empty-last-attempt no longer beats real lines; referee scene count bounded.
* Round 8 (story with no personal names): Persona now always runs (roles go to `extra_speakers` with evidence); with no cast the sanitiser requires the speaker to be grounded in the source (`_name_in_source`, fuzzy + consonant skeleton, so "Team Lead" ~ "tim lid") and drops prompt-field echoes ("Listener"); referee window `hi = max(ceil(1.4n)+1, n+3)`.
* Result: stress corpus 27 -> 28 -> 31 -> 32 of 32 clean. See `docs/TESTING.md`.

## Round 9 (2026-11-18, found by the full-pipeline runs): one person, two names
* Persona now returns `aliases` (`alias`, `same_as`, `evidence`). An alias is accepted only if the quotation is in the story AND holds both names (spelling-tolerant; Devanagari loanwords are romanised differently) AND a nickname cue ("nam se", "pukarte", "urf", "known as") sits within 2 words of one of the names (`grounding.names_one_person`). Accepted aliases are merged everywhere (events, pronoun map, cast, relationships, emotions, secrets) and the writer gets a "SAME PERSON" note, so Rahul no longer talks about "Rocky" as someone else. Two people who merely appear together, or a wrong pair, are never merged.
* Agent 1 `_canonicalize`: an English kin word next to exactly one Hindi kin name ("Grandma" + "Dadiji") is one person (runs before and after the spelling merge); one-letter spelling variants of names of 6+ letters ("Dadiiji"/"Dadiji") merge.
* Tests: `tests/test_aliases.py`.
* Self-reference: a speaker who names himself or his alias as the subject ("Rahul: Rocky ne Gita ki ball rok di") now says "maine" (`agent3_synthesizer.fix_self_reference`; only the "<own name> ne" form, only the speaker's own names).

## Round 10 (2026-11-18)
* Character names are grounded like events: Agent 1 drops a character whose name is not in the scene's own text (spelling-tolerant `grounding.name_grounded`: Geeta/Gita, Rocky/roki, spacing "DadaJi"/"dada ji", titles ignored; a text under 5 words vouches for everything). Found because in a story that names nobody the model invented "Sara" and the whole script spoke as her.
* The consonant skeleton treats g and j as one sound ("Engineer" ~ "injiniyar"), so English role labels written as Devanagari loanwords are accepted as speakers.
* Known limit: an invented name that equals a common word of the story ("Sara" vs "सारा" = whole) cannot be told apart by text matching.
