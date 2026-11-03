# Human evaluation: next experiment step

Human ratings provide evidence that the automatic metrics alone cannot establish. This helper creates shuffled, blinded CSV sheets from the Groq pilot variants and summarizes completed ratings.

## Prepare rater sheets

Run from the repository root:

```powershell
python -m evaluation.human_eval create outputs/ablation/groq_pilot --raters 3 --seed 41
```

This creates `outputs/ablation/groq_pilot/human_eval/` with one randomized CSV per rater, a manifest, and `condition_key.json`. **Do not share `condition_key.json` with raters**; it reveals the variant. Give each rater only their own CSV.

## Rater instructions and scale

Have fluent Hinglish speakers rate every neutral/edited line pair independently. Keep the order randomized and do not tell raters which system generated an edited line. Use 1–5, where 1 is very poor and 5 is excellent:

- **meaning_preservation**: Does the edited line keep the original facts, intent, and polarity?
- **naturalness**: Would this sound natural in spoken Hinglish?
- **slang_appropriateness**: Does the slang fit the words and situation, rather than sounding forced?
- **character_tone_fit**: Does it suit this speaker and the emotional/social context?

Use the comment column for brief notes, especially if meaning changed or slang feels forced. If a pair is identical, rate naturalness and meaning normally; slang appropriateness may be left blank when no slang is present. Use the same rule for all raters.

## Collect and summarize

Raters fill in their respective CSVs and return them without renaming the files. Put the completed sheets back in the same folder, preserving the `item_id` and `rater_id` columns. Then run:

```powershell
python -m evaluation.human_eval summarize outputs/ablation/groq_pilot/human_eval
```

The script writes `human_eval_summary.json` with per-variant descriptive mean scores and ordinal Krippendorff alpha for inter-rater agreement. Review that file and the comments; the small pilot is descriptive and should not be treated as conclusive statistical evidence.

## Important study limits

- This particular ablation uses the Groq fallback, not the missing Qwen LoRA adapter. Describe it as a Groq pilot; do not generalize its findings to Qwen.
- It currently contains only five scenes. For paper claims, repeat the same experiment across more held-out stories with varied topics, character roles, and emotional situations.
- The comparison is paired: raters see the same neutral dialogue alongside each variant's output. This helps control for differences in source content.
- Report the number of stories, scenes, turns, raters, missing ratings, and inter-rater agreement. Do not claim significance from the current pilot.
