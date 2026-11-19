# Making the trained slang editor close to a frontier model (plan, 2026-11-19; NOT started)

Goal (author): a small local slang editor that matches, or is close to, frontier models. Today's v6 (Qwen2.5-1.5B + LoRA, GGUF) picks ONE phrase from a code-chosen shortlist for ONE clause, or abstains. It copies the shortlist; a bigger pool alone does not teach it new language and does not raise how many lines are eligible.

## Measured today (70 saved runs, 1,528 lines; docs/PENDING.md for the funnel)
28% of lines had a swappable feeling word; the Director nominated 32%; of 495 nominated lines 150 got real slang, 121 a filler, 141 had nothing to swap, 32 the editor abstained on (6.5%), 8 the gates rejected. The editor is not what limits quantity; the one-swap-per-line design and the supply of feeling words are.

## Plan
0. (done today, before the demo) D1 slot-aware Director, W1 emotion-arc feelings in the writer, W3 peers-only feelings and elder 'tum' rule. Supply of swappable lines on the birthday story 20-22% -> 45-55%.
1. **Benchmark first** (days, ~$5-10): 300-500 held-out neutral lines with context (speaker, listener register, emotion, setting). Arms: frontier teacher (Sonnet/Opus) as slang editor under the same constraints, v6, prompted small models. Metrics: judged acceptability (native speaker + LLM judge), meaning preserved (edit audit / MuRIL), register respected, slang per eligible line, distinct slang per 100 lines, cost, latency. This says how far v6 is, and is the paper's central table.
2. **v7 = pool-conditioned generative line editor**: input = neutral line + speaker/listener register + emotion/intensity + setting + optional shortlist; output = the line with 0-3 edits (swap, interjection, light restructure) or unchanged. Teacher data from the frontier model with rejection sampling by the gates (rule gate adapted, edit audit, MuRIL, judge): 10-20k pairs, about $20-100 depending on teacher. The pool stays as optional hint and as safety lexicon (grow it 177 -> ~400, new entries reviewed by a native speaker).
3. Train 1.5B vs 3B vs 7B LoRA on Kaggle; adopt the smallest model that reaches >= ~90% of the teacher's judged score on the benchmark. Check CPU latency (1.5B ~3 s per generation today; 3B about 2x, 7B several times: may need GPU or a hosted small model for the demo).
4. Claims: "to our knowledge" only, after a literature check; report the head-to-head, the cost/latency, and the ablation (rules only vs frontier-as-editor vs trained editor).

## What switching costs (honest)
- Hard guarantees: v6 edits are pool-bound and frame-checked; free generation can paraphrase or hallucinate (v2 did on long lines). Safety must be rebuilt as post-hoc gates; some good edits will be rejected.
- Time: roughly 1-2 weeks (benchmark, data, training, evaluation). Nothing of this fits before the demo or the Nov 20 capstone deadline; the capstone can contain the plan and the benchmark design.
- Code and tests: Agent 4 and ~100 tests assume shortlist / pool semantics; a refactor with regression risk.
- Control: pool = controllable vocabulary and register; generation is steered only by conditioning.
- Possibly latency / footprint if the student is larger than 1.5B.
## What it gains
Coverage beyond hand-written trigger lists (the supply bottleneck), several edits per line, learned filler placement and variety, and a defensible novelty claim.

## v7 build status (2026-11-19)
Built and tested (no spend yet): `data_prep/v7_format.py` (the contract: context + MAX EDITS + optional hints -> edited line or UNCHANGED; deterministic `context_allows_edit`), `data_prep/v7_corpus.py`
(4,035 unique real lines with real context from the saved runs: 2,087 allow slang, 1,948 must stay unchanged, 1,211 have a swappable slot; run `python -m data_prep.v7_corpus build`),
`data_prep/v7_filter.py` (deterministic safety checks reusing evaluation/edit_audit), `data_prep/v7_teacher.py` (`run` = frontier teacher, `judge` = cross-family judge, `stats`; resumable, budget-stopped),
tests/test_v7_pipeline.py (23 tests). Estimated cost: pilot 300 lines ~$0.40; the whole first dataset (2,087 lines) ~$2.70 (teacher $2.4 + judge $0.3).
Next: (1) pilot + read the accepted / rejected examples, (2) scale, (3) assembler + Kaggle training notebook (derived from the v6 notebook: data audit first, LoRA SFT on 1.5B and 3B, held-out stories),
(4) benchmark harness (v6, v7 1.5B / 3B, frontier teacher, prompted small model; judge + audit + human sheet), (5) integrate v7 into Agent 4 behind a flag.

## v7 pilot result (2026-11-19, 300 real lines, Sonnet 5.5 teacher, Gemini 3.6 flash judge, ~$0.55 total for two rounds)
Round 1 (first-4-hints prompt): the teacher wrote "mast" and "kadak" for 72% of its edits and appended ", ekdum X!" tails; the judge call lost most replies to truncation (reasoning tokens): fixed (tolerant JSON parsing, larger limits).
Round 2 (hints sampled at random, "never repeat a slang word within a request", "do not append a tail"): 300 lines -> 108 left unchanged by the teacher (valid), 192 edited; 88 pass the deterministic filter (46%); the judge accepts 61 of them (69%, mean natural 4.11) = 20% of the lines become training edits.
Diversity is now real (jhakaas, zabardast, wahiyat, dhaansu, tagda, solid, tension, "sone pe suhaga", "hawa mein ho", "jugaad se", "maza aa gaya") and several accepted edits are NOT pool-slot swaps (idiom or phrase edits), which v6 cannot make. Judge rejections are sensible taste calls ("bhaukaal lag raha hai" for a slide). Rule-gate rejections that remain: 52 of 88 are appended comma-tails, the rest non-feeling swaps or real frame errors: principled.
Implication: ~20% yield means the 2,087 existing allowed lines give ~420 accepted edits, too few for a generative 3B; more lines are needed (synthetic scenes and / or the existing sentence banks) and a best-of-3 teacher would raise the yield. Files: data/v7/ (contexts.jsonl, teacher_raw.jsonl, judge.jsonl, pilot1/).
