# Slang editor v4 — Kaggle run (~35 min)  — same steps as v3

1. kaggle.com → Datasets → **New Dataset** → drag in everything in this folder (5 `.jsonl` files + the `old_adapter` folder) → name it `slang-v4` → Create.
2. **New Notebook** (fresh one) → Accelerator **GPU T4 x1** → **Add Data** → attach **only `slang-v4`** (not the old slang-v3 dataset).
3. Paste all of `notebooks/train_slang_editor_lora_qwen_v3.py` into one cell (it is the same notebook; it now also reads `eval_filler.jsonl`). If a *torchao* error appears: `!pip uninstall -y torchao` in a cell above, restart.
4. Run All. Download from the Output tab: `slang-editor-lora.zip`, `eval_outputs_new.jsonl`, `eval_outputs_old.jsonl`, `eval_outputs_base.jsonl`.
5. Unzip the adapter to `models/slang-editor-lora-v4/`; put the three `eval_outputs_*.jsonl` into `data/felix_dataset/v3/` **and** copy `eval_filler.jsonl` from `data/felix_dataset/v4/` into `data/felix_dataset/v3/` (the scorer reads that folder).
6. Tell Claude "v4 ready".

In this run `old` = your current v3 adapter, `new` = v4, so the comparison is v4 vs v3.

What changed in the data: +520 filler-placement rows (vocative after a name, opener after "Haan / Theek hai", "yaar" closing a "Chinta mat karo" clause) and +17 formal-line abstains, all placed by the same rules the runtime gate enforces. Train 1,678 / val 84. New real-sentence test: `eval_filler.jsonl` (59 rows, never trained on).
