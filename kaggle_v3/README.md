# Slang editor v3 — Kaggle run (≈30 min)

## What is in this folder
| file | what it is |
|---|---|
| `lora_train.jsonl` (1,167 rows) / `lora_val.jsonl` (58) | on-distribution training data: Agent-3-style stiff sentences + verified slang edits, abstain examples, a small share of old short gold rows |
| `eval_real.jsonl` (142) | **real pipeline sentences** (never trained on) — 61 where a slang fits, 81 where nothing fits |
| `eval_synth.jsonl` (82) | held-out synthetic scenes (never trained on) |
| `old_adapter/` | your CURRENT adapter, so the before/after comparison runs in the same job |

## Steps (all in the browser)
1. kaggle.com → **Datasets → New Dataset** → drag in **everything in this folder** (the 4 `.jsonl` files and the `old_adapter` folder) → name it `slang-v3` → Create (private is fine).
2. **New Notebook** → right panel **Session options → Accelerator → GPU T4 x1**.
3. **Add Data** → choose your `slang-v3` dataset.
4. Paste the whole of `notebooks/train_slang_editor_lora_qwen_v3.py` into one code cell. (If you hit a *torchao* error, put `!pip uninstall -y torchao` in a cell above and restart the session.)
5. **Run All.** Paths are auto-detected — nothing to edit.
6. Read the end of the output: the `VAL CHECK (strict)` block and three lines `[new] / [base] / [old] valid-edit rate ...`.
7. **Output tab → download:** `slang-editor-lora.zip`, `eval_outputs_new.jsonl`, `eval_outputs_base.jsonl`, `eval_outputs_old.jsonl`.
8. Locally:
   * unzip the adapter into `models/slang-editor-lora-v3/`
   * put the three `eval_outputs_*.jsonl` into `data/felix_dataset/v3/`
9. Tell Claude “outputs ready” — the held-out comparison (old vs new vs untuned base) is then scored and the adapter is adopted **only if it clearly wins**.
