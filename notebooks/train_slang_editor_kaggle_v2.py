"""
notebooks/train_slang_editor_kaggle_v2.py
Train the mT5-small "slang editor" (Agent 4) with LoRA + EDIT-WEIGHTED LOSS.
RUN THIS ON KAGGLE (GPU T4). This is the v2 fix for the copy-collapse failure.

------------------------------------------------------------------------------
WHY v2 EXISTS (read this — it's the whole point)
------------------------------------------------------------------------------
v1 (train_slang_editor_kaggle.py) reached a beautiful eval_loss of 2.59 but the
model learned to COPY the input verbatim (or prepend a broken "Bhi,") and
inserted no real slang. Diagnosis:

  * The input neutral and the target differ by only ~2-3 slang tokens out of ~8.
  * v1's loss was plain token cross-entropy with predict_with_generate=False, so
    eval_loss only measured teacher-forced next-token loss, never generation.
  * Copying the input scores very low loss (80% of tokens are right for free),
    so the model minimizes loss by converging to near-identity. The slang tokens
    are too sparse a signal to force real learning. Classic edit-task trap.

v2 fixes this with THREE changes:
  1. EDIT-WEIGHTED LOSS: target tokens that are NOT present in the source input
     (i.e. the inserted slang) are weighted `EDIT_WEIGHT`x higher. Copying no
     longer minimizes the loss — the model is forced to get the slang right.
  2. GENERATION-BASED EVAL (predict_with_generate=True) so eval_loss + metrics
     reflect what the model actually generates, not teacher-forced loss.
  3. A `copies_input` METRIC printed every epoch = fraction of val examples where
     the generated text equals the input neutral. This is the direct measure of
     copy-collapse. WATCH THIS GO DOWN. (v1 would have scored ~0.6-1.0 here.)

Everything else (mT5 rationale, fp32, LoRA on q/v) is unchanged from v1 — see
that file's header for why mT5 over IndicBART.

------------------------------------------------------------------------------
STEPS ON KAGGLE
------------------------------------------------------------------------------
  1. Locally, AFTER the 3k augmentation finishes:
       python -m data_prep.build_editor_dataset --extra data/felix_dataset/augmented_v3.csv
     -> produces data/felix_dataset/editor_train.jsonl + editor_val.jsonl
  2. Upload those two JSONL files as a Kaggle Dataset.
  3. New Kaggle notebook, Accelerator = GPU T4 x2, add that dataset.
  4. Set DATA_DIR below to the dataset mount path, run this file.
  5. Zip + download /kaggle/working/slang-editor-lora, unzip to ./models/slang-editor-lora.
     Agent 4 (backend="auto") loads it automatically.

TUNING: if copies_input is still high after a few epochs, raise EDIT_WEIGHT
(4 -> 6 -> 8). If generations become garbled / non-grammatical, lower it (4 -> 3).
"""

import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import (AutoTokenizer, MT5ForConditionalGeneration,
                          DataCollatorForSeq2Seq, Seq2SeqTrainingArguments,
                          Seq2SeqTrainer, TrainerCallback)
from peft import LoraConfig, get_peft_model, TaskType

MODEL_NAME = "google/mt5-small"
DATA_DIR = "/kaggle/input/datasets/atulk4721/editor"   # <-- set to your Kaggle dataset mount
OUT_DIR = "/kaggle/working/slang-editor-lora"
MAX_INPUT, MAX_TARGET = 96, 96
EPOCHS, BATCH, LR = 8, 32, 3e-4
EDIT_WEIGHT = 4.0        # how much harder inserted-slang tokens count vs copied tokens


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def neutral_of(input_str):
    """Recover the neutral sentence from 'insert slang | control: X | NEUTRAL'."""
    return input_str.split(" | ")[-1].strip()


# ----------------------------------------------------------------------------
# Edit-weighted trainer: up-weight target tokens absent from the source input.
# ----------------------------------------------------------------------------
class EditWeightedTrainer(Seq2SeqTrainer):
    def __init__(self, *args, edit_weight=EDIT_WEIGHT, **kwargs):
        super().__init__(*args, **kwargs)
        self.edit_weight = edit_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]              # (B, T)
        input_ids = inputs["input_ids"]        # (B, S)
        outputs = model(**inputs)              # labels passed -> logits aligned to labels
        logits = outputs.logits                # (B, T, V)
        B, T, V = logits.shape

        ce = F.cross_entropy(
            logits.view(-1, V), labels.view(-1),
            reduction="none", ignore_index=-100,
        ).view(B, T)

        valid = (labels != -100).float()                       # real target tokens
        # A label token is an "edit" if it does NOT appear anywhere in that row's
        # source input_ids (the prompt + neutral). Inserted slang -> edit.
        present = (labels[:, :, None] == input_ids[:, None, :]).any(-1)   # (B, T)
        edit = (~present).float() * valid

        # base weight 1.0 on every valid token, +extra on edit tokens
        w = valid + (self.edit_weight - 1.0) * edit
        loss = (ce * w).sum() / w.sum().clamp(min=1)
        return (loss, outputs) if return_outputs else loss


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME, dtype=torch.float32)

    lora = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=16, lora_alpha=32,
                      target_modules=["q", "v"], lora_dropout=0.05, bias="none")
    model = get_peft_model(model, lora)
    model.enable_input_require_grads()   # needed for grad checkpointing + PEFT
    model.print_trainable_parameters()

    train_raw = load_jsonl(os.path.join(DATA_DIR, "editor_train.jsonl"))
    val_raw = load_jsonl(os.path.join(DATA_DIR, "editor_val.jsonl"))
    # neutrals for the copy-collapse metric (val eval order is sequential, not shuffled)
    val_neutrals = [neutral_of(ex["input"]) for ex in val_raw]

    train = Dataset.from_list(train_raw)
    val = Dataset.from_list(val_raw)

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
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds < 0, tok.pad_token_id, preds)
        labels = np.where(labels < 0, tok.pad_token_id, labels)
        dec = [d.strip() for d in tok.batch_decode(preds, skip_special_tokens=True)]
        refs = [r.strip() for r in tok.batch_decode(labels, skip_special_tokens=True)]
        n = len(dec)
        # align neutrals; guard against any length mismatch from padding batches
        neu = val_neutrals[:n] if len(val_neutrals) >= n else [""] * n
        copies_input = sum(d == neu[i] for i, d in enumerate(dec)) / max(n, 1)
        edits_made = sum(d != neu[i] for i, d in enumerate(dec)) / max(n, 1)
        matches_ref = sum(d == refs[i] for i, d in enumerate(dec)) / max(n, 1)
        return {"copies_input": round(copies_input, 3),   # WANT LOW
                "edits_made": round(edits_made, 3),        # WANT HIGH
                "matches_ref": round(matches_ref, 3)}      # exact target match (bonus)

    args = Seq2SeqTrainingArguments(
        output_dir=OUT_DIR, num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH, per_device_eval_batch_size=BATCH,
        gradient_accumulation_steps=1, learning_rate=LR, warmup_steps=200,
        gradient_checkpointing=True, eval_accumulation_steps=1,
        logging_steps=50, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, save_total_limit=2,
        predict_with_generate=True, generation_max_length=MAX_TARGET,
        generation_num_beams=4,
        fp16=False, report_to="none",
    )

    class PrintLoss(TrainerCallback):
        def on_log(self, a, s, c, logs=None, **k):
            if logs:
                print({kk: round(v, 4) for kk, v in logs.items() if isinstance(v, float)})

    trainer = EditWeightedTrainer(
        model=model, args=args, train_dataset=train, eval_dataset=val,
        processing_class=tok,
        data_collator=DataCollatorForSeq2Seq(tok, model=model, padding=True),
        compute_metrics=compute_metrics,
        callbacks=[PrintLoss()],
        edit_weight=EDIT_WEIGHT,
    )
    trainer.train()
    model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    print("saved LoRA adapter ->", OUT_DIR)

    # quick sanity generation with an explicit copy check
    model.eval()
    copied = 0
    samples = load_jsonl(os.path.join(DATA_DIR, "editor_val.jsonl"))[:8]
    for ex in samples:
        ids = tok(ex["input"], return_tensors="pt", truncation=True, max_length=MAX_INPUT)
        out = model.generate(**{k: v.to(model.device) for k, v in ids.items()},
                             max_new_tokens=MAX_TARGET, num_beams=4)
        got = tok.decode(out[0], skip_special_tokens=True).strip()
        neu = neutral_of(ex["input"])
        is_copy = (got == neu)
        copied += is_copy
        print("IN :", ex["input"])
        print("GOT:", got, "  <-- COPIED INPUT" if is_copy else "")
        print("REF:", ex["target"], "\n")
    print(f"copy rate on 8 samples: {copied}/8  (lower is better)")


if __name__ == "__main__":
    main()
