# Slang editor v5 — Kaggle run (~45–70 min on a T4)

## What is in this folder
| file | what |
|---|---|
| `lora_train.jsonl` (≈3,076 rows after the audit) | v4 rows that the CURRENT runtime contract accepts (fillers, formal abstains, older edit patterns) + ~2,050 new rows whose gold edit was built by code per slot kind and approved by the runtime rule gate **and** an LLM judge (≥4/5) + 742 abstain rows (incl. hard ones: negation, "aap", addressed to an elder, hospital, facts, questions) |
| `lora_val.jsonl` (≈180) | random held-out slice of the same distribution |
| `eval_slots.jsonl` (≈385 + abstains) | NEW held-out sentences from two domains that never appear in training (festival & shaadi, tech / robotics) |
| `eval_realv5.jsonl` (≈614) | real Agent-3 sentences from saved pipeline runs |
| `eval_real.jsonl`, `eval_synth.jsonl`, `eval_filler.jsonl` | the v4 evaluation sets, for before/after comparison |
| `edit_gate.py` | the runtime rule gate, so Kaggle measures what the pipeline will accept |
| `old_adapter/` | the v4 adapter (so the notebook reports v5 vs v4 vs untouched base in one run) |

Why v4 rows were dropped: 38% of the old edit rows taught things the runtime gate rejects (content-word swaps, tail appends,
"Boss/kasam se" with rewrites). Training on them wasted the model's accepted-edit capacity.

## Steps
1. kaggle.com → Datasets → **New Dataset** → drag in EVERYTHING in this folder (including `old_adapter/`) → name it `slang-v5`.
2. **New Notebook** → Accelerator **GPU T4 x1** → **Add Data** → attach only `slang-v5`.
3. Paste ALL of `notebooks/train_slang_editor_lora_qwen_v5.py` into one cell. (torchao error? `!pip uninstall -y torchao` in a cell above, restart.)
4. **Run All.** The notebook first AUDITS the data (it refuses to train on contract-violating rows) — you will see `AUDIT PASSED`
   within a minute. Then it trains, prints SANITY outputs, VAL CHECK, the per-file evaluation table, and a final **CHECKLIST**
   with PASS / FAIL lines and a RECOMMENDATION.
5. Download from the Output tab: `slang-editor-lora.zip` (adapter + `training_report.json`), `eval_outputs_new.jsonl`,
   `eval_outputs_old.jsonl`, `eval_outputs_base.jsonl`.
6. Unzip into `models/slang-editor-lora-v5/`, then tell Claude "v5 ready" and paste the CHECKLIST block.

Adopt only if the checklist passes and the held-out `eval_slots` gate-valid rate beats v4. Nothing is replaced automatically.
