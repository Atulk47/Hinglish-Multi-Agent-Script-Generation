"""
notebooks/train_slang_editor_kaggle_v4.py
FULL fine-tune of mt5-small as the slang editor (Agent 4) — RUN ON KAGGLE (GPU T4).

==============================================================================
WHY v4 (the version-mismatch bug, and the fixes for every error we hit)
==============================================================================
v3 trained on Kaggle's newest transformers, but the saved model generated
degenerate output when loaded by the project's local transformers -> Agent 4 kept
falling back to Groq. Root cause = train/runtime transformers version gap.

THE FIX: pin BOTH training (here) and the local runtime to transformers 4.46.3.
4.46.3 is recent enough to install via wheels on Kaggle (no Rust/source build like
the old 4.40 pin), works with the local torch 2.2 runtime (no torch upgrade), and
its saved-model format loads+generates correctly locally. The local env is already
set to 4.46.3; this notebook pins Kaggle to it too.

Pins chosen to avoid the exact errors seen while setting this up:
  * transformers==4.46.3  -> matches local; wheels exist (4.40 had no wheel -> build fail)
  * accelerate==0.33.0    -> <0.34, so the optimizer has no .train() method
                             (Kaggle's newer accelerate caused: AttributeError:
                              'AdamW' object has no attribute 'train')
    NOTE: accelerate 0.33.0 requires numpy<2, so it pulls numpy 1.26.4 - that's
    fine (torch imports and trains with it). Do NOT also pin numpy>=2.0: it makes
    the install unsatisfiable (ResolutionImpossible).

4.46 Trainer API used below:  torch_dtype=  (not dtype=) ,  eval_strategy=  (not
evaluation_strategy=) ,  processing_class=  (not tokenizer=).

==============================================================================
HOW TO RUN ON KAGGLE  (follow in order)
==============================================================================
 0. New Notebook -> Settings: Accelerator = GPU T4.
 1. Add Data -> add your dataset that contains editor_train.jsonl + editor_val.jsonl
    (built locally by  python -m data_prep.build_editor_dataset --src <final csv>).
    Note the mount path it gives you, e.g. /kaggle/input/hinglish-editor-data .
 2. Paste CELL 1 below into the FIRST cell, run it, then  RESTART THE SESSION
    (Run menu -> "Restart & Clear Cell Outputs", or the restart button).
    *** You MUST restart, or the old transformers stays loaded in memory. ***
 3. After restart: paste CELL 2 (everything from the import line to the end) into a
    new cell, set DATA_DIR to your mount path from step 1, and Run.
 4. Training writes a FULL model (~1.2 GB) to /kaggle/working/slang-editor-full .
 5. Run CELL 3 (bottom) to zip it; download the zip from the right-hand Output panel.
 6. Locally: extract it to  ./models/slang-editor-full/  (overwrite the old one).
    Agent 4 already prefers that folder, so no code change. Then test with:
        python run_pipeline.py --story data/raw/demo_science_fair.txt --session mt5test
    If you no longer see "falling back to Groq", the trained editor is live.

READ THE METRICS, NOT THE LOSS:  slang_hit HIGH, copies_input ~0, has_sentinel 0.
"""

# ============================================================================
# CELL 1  — run this FIRST, then RESTART THE SESSION before running CELL 2
# ============================================================================
#   !pip install -q "transformers==4.46.3" "accelerate==0.33.0"
#
#   # (fallback, only if the line above ever fails building tokenizers:)
#   # !pip install -q "transformers==4.46.3" "tokenizers==0.20.3" "accelerate==0.33.0"
#
#   >>> NOW RESTART THE SESSION (Run -> Restart & Clear Cell Outputs) <<<
# ============================================================================


# ============================================================================
# CELL 2  — the training run (paste from here to the end, after restarting)
# ============================================================================
import json, re, os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter
from datasets import Dataset
import transformers
from transformers import (AutoTokenizer, MT5ForConditionalGeneration,
                          DataCollatorForSeq2Seq, Seq2SeqTrainingArguments,
                          Seq2SeqTrainer, TrainerCallback)

# Guard: make sure the pin + restart worked. If this fails, you skipped the
# restart in step 2 — run CELL 1, RESTART, then run this cell again.
assert transformers.__version__.startswith("4.46"), (
    f"Expected transformers 4.46.x (run CELL 1 and RESTART the session) — found "
    f"{transformers.__version__}.")
print("transformers", transformers.__version__, "| torch", torch.__version__)

# ---- CONFIG: set DATA_DIR to your Kaggle dataset mount path -----------------
MODEL_NAME = "google/mt5-small"
DATA_DIR   = "/kaggle/input/<your-dataset-slug>"   # <-- folder with editor_train.jsonl / editor_val.jsonl
OUT_DIR    = "/kaggle/working/slang-editor-full"
MAX_INPUT, MAX_TARGET = 96, 96
EPOCHS, BATCH, ACCUM, LR = 6, 16, 2, 2e-4
EDIT_WEIGHT = 2.0
# ----------------------------------------------------------------------------


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def neutral_of(input_str):
    return input_str.split(" | ")[-1].strip()

def words(s):
    return set(re.findall(r"[a-z]+", s.lower()))


class EditWeightedTrainer(Seq2SeqTrainer):
    """Target tokens absent from the source count EDIT_WEIGHT x more, so copying is
    discouraged; a full fine-tune has the capacity to satisfy it with real slang."""
    def __init__(self, *args, edit_weight=EDIT_WEIGHT, **kwargs):
        super().__init__(*args, **kwargs)
        self.edit_weight = edit_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]; input_ids = inputs["input_ids"]
        outputs = model(**inputs); logits = outputs.logits
        B, T, V = logits.shape
        ce = F.cross_entropy(logits.view(-1, V), labels.view(-1),
                             reduction="none", ignore_index=-100).view(B, T)
        valid = (labels != -100).float()
        present = (labels[:, :, None] == input_ids[:, None, :]).any(-1)
        edit = (~present).float() * valid
        w = valid + (self.edit_weight - 1.0) * edit
        loss = (ce * w).sum() / w.sum().clamp(min=1)
        return (loss, outputs) if return_outputs else loss


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    # 4.46 API: torch_dtype (NOT dtype)
    model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)

    # hard-suppress ALL mT5 sentinels at generation (vocab scan, since
    # convert_tokens_to_ids returns unk for these)
    sentinel_ids = sorted(i for s, i in tok.get_vocab().items() if "extra_id" in s)
    print(f"suppressing {len(sentinel_ids)} sentinel ids at generation")
    model.generation_config.suppress_tokens = sentinel_ids
    model.generation_config.max_length = MAX_TARGET
    model.generation_config.num_beams = 4
    model.generation_config.no_repeat_ngram_size = 3

    train_raw = load_jsonl(os.path.join(DATA_DIR, "editor_train.jsonl"))
    val_raw = load_jsonl(os.path.join(DATA_DIR, "editor_val.jsonl"))
    val_neutrals = [neutral_of(ex["input"]) for ex in val_raw]
    print(f"train={len(train_raw)}  val={len(val_raw)}")

    STOP = {"hai","hain","ho","hota","hoti","gaya","gayi","gaye","ka","ki","ke","ko",
            "se","mein","me","par","aur","ya","ye","yeh","wo","woh","na","nahi","nahin",
            "kar","karo","kya","hi","to","tha","thi","the","raha","rahi","rahe","ab",
            "mera","meri","tera","teri","apna","is","us","kal","aaj","ek","bahut","bhi"}
    ins = Counter()
    for ex in train_raw:
        ins.update(words(ex["target"]) - words(neutral_of(ex["input"])))
    SLANG_LEX = {w for w, c in ins.items() if c >= 2 and len(w) >= 2} - STOP
    print(f"corpus slang lexicon size: {len(SLANG_LEX)}  e.g. {sorted(SLANG_LEX)[:12]}")

    train = Dataset.from_list(train_raw); val = Dataset.from_list(val_raw)

    def tokenize(batch):
        mi = tok(batch["input"], max_length=MAX_INPUT, truncation=True, padding="max_length")
        labels = tok(text_target=batch["target"], max_length=MAX_TARGET,
                     truncation=True, padding="max_length")
        mi["labels"] = [[(t if t != tok.pad_token_id else -100) for t in seq]
                        for seq in labels["input_ids"]]
        return mi

    train = train.map(tokenize, batched=True, remove_columns=train.column_names)
    val = val.map(tokenize, batched=True, remove_columns=val.column_names)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple): preds = preds[0]
        preds = np.where(preds < 0, tok.pad_token_id, preds)
        labels = np.where(labels < 0, tok.pad_token_id, labels)
        dec = [d.strip() for d in tok.batch_decode(preds, skip_special_tokens=True)]
        refs = [r.strip() for r in tok.batch_decode(labels, skip_special_tokens=True)]
        n = len(dec); neu = val_neutrals[:n]
        copies = sum(dec[i] == neu[i] for i in range(n)) / max(n, 1)
        sent = sum("<extra_id_" in dec[i] for i in range(n)) / max(n, 1)
        slang = sum(bool((words(dec[i]) - words(neu[i])) & SLANG_LEX) for i in range(n)) / max(n, 1)
        mref = sum(dec[i] == refs[i] for i in range(n)) / max(n, 1)
        return {"slang_hit": round(slang,3), "copies_input": round(copies,3),
                "has_sentinel": round(sent,3), "matches_ref": round(mref,3)}

    # 4.46 API: eval_strategy (NOT evaluation_strategy)
    args = Seq2SeqTrainingArguments(
        output_dir=OUT_DIR, num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH, per_device_eval_batch_size=BATCH,
        gradient_accumulation_steps=ACCUM, learning_rate=LR, warmup_steps=200,
        weight_decay=0.01, gradient_checkpointing=True, eval_accumulation_steps=1,
        logging_steps=50, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="slang_hit",
        greater_is_better=True, save_total_limit=1,
        predict_with_generate=True, generation_max_length=MAX_TARGET,
        generation_num_beams=4, fp16=False, report_to="none",
    )

    class PrintLoss(TrainerCallback):
        def on_log(self, a, s, c, logs=None, **k):
            if logs:
                print({kk: round(v,4) for kk, v in logs.items() if isinstance(v, float)})

    # 4.46 API: processing_class (NOT tokenizer)
    trainer = EditWeightedTrainer(
        model=model, args=args, train_dataset=train, eval_dataset=val,
        processing_class=tok,
        data_collator=DataCollatorForSeq2Seq(tok, model=model, padding=True),
        compute_metrics=compute_metrics, callbacks=[PrintLoss()],
        edit_weight=EDIT_WEIGHT,
    )
    trainer.train()
    model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    print("saved FULL model ->", OUT_DIR)

    # sanity generation (sentinels already suppressed via generation_config)
    model.eval(); hits = 0
    for ex in val_raw[:10]:
        ids = tok(ex["input"], return_tensors="pt", truncation=True, max_length=MAX_INPUT)
        out = model.generate(**{k: v.to(model.device) for k, v in ids.items()}, max_new_tokens=MAX_TARGET)
        got = tok.decode(out[0], skip_special_tokens=True).strip()
        neu = neutral_of(ex["input"]); inserted = words(got) - words(neu)
        hit = bool(inserted & SLANG_LEX); hits += hit
        print("IN :", ex["input"]); print("GOT:", got, "  [slang_hit]" if hit else "  [no slang]")
        print("REF:", ex["target"], "\n")
    print(f"slang_hit on 10 samples: {hits}/10")


if __name__ == "__main__":
    main()


# ============================================================================
# CELL 3  — run AFTER training to zip the model, then download from Output panel
# ============================================================================
#   import shutil
#   shutil.make_archive("/kaggle/working/slang-editor-full", "zip",
#                       "/kaggle/working/slang-editor-full")
#   print("zipped -> /kaggle/working/slang-editor-full.zip  (download it)")
# ============================================================================
