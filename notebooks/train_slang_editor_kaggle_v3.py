"""
notebooks/train_slang_editor_kaggle_v3.py
FULL fine-tune (NO LoRA) of mt5-small as the slang editor (Agent 4).
RUN ON KAGGLE (GPU T4). This is the capacity-wall test after v1/v2 both collapsed.

------------------------------------------------------------------------------
THE STORY SO FAR (why v3 exists)
------------------------------------------------------------------------------
  v1 (plain LoRA)          -> COPY-COLLAPSE: model copies input verbatim.
  v2 (LoRA + edit-weight)  -> SENTINEL-COLLAPSE: model inserts mT5's pretraining
                              sentinel <extra_id_0> (id 250099) as a fake "edit".
  v2 + ban sentinels       -> GARBAGE-COLLAPSE: model inserts random script
                              ("શ્વર") then copies. matches_ref stayed 0.0 throughout.

Diagnosis: a rank-16 LoRA on q/v had enough capacity to learn the STRUCTURE
("insert a non-source token here, then copy") but NOT the SEMANTICS (which slang
fits which context), and it could not override mT5's decoder prior (mT5 is
pretrained to start its output with <extra_id_0>). v3 attacks the root cause:

  1. FULL FINE-TUNE (no LoRA): every parameter is trainable, so the model has the
     capacity to override the sentinel prior and actually learn slang. Every
     training target starts with a real word -> strong signal against sentinels.
  2. MILD edit-weighting (EDIT_WEIGHT=2.0, was 4.0): still discourages pure copy,
     but with full capacity the cheapest way to earn the edit reward becomes
     "emit the real slang from training", not "emit a sentinel".
  3. HARD sentinel suppression at generation (generation_config.suppress_tokens),
     so eval/inference can never hide in <extra_id_*>. Ban ids come from a VOCAB
     SCAN (convert_tokens_to_ids returns unk for these — that bit us before).
  4. REAL metric: `slang_hit` = did the generation insert a word from the corpus
     slang lexicon? `matches_ref` (exact) is too harsh — there are many valid
     slangs per line, so exact match is the wrong bar.

READ THE METRICS, NOT THE LOSS:
  * slang_hit  -> WANT HIGH  (inserted a genuine slang word)   <-- the real signal
  * copies_input -> WANT LOW (0 = never a bare copy)
  * has_sentinel -> WANT 0   (suppression working)
If slang_hit climbs above ~0.5-0.7 with copies_input ~0 and has_sentinel 0, the
editor WORKS. If slang_hit stays near 0, the task genuinely needs an LLM (keep
Groq Agent 4) and you have an ironclad negative result (LoRA AND full-FT fail).

------------------------------------------------------------------------------
STEPS ON KAGGLE (same dataset as v1/v2 — no re-upload needed)
------------------------------------------------------------------------------
  1. Copy & Edit your v2 notebook (keeps the dataset mounted) OR new T4 notebook
     with the editor_train.jsonl / editor_val.jsonl dataset added.
  2. Paste this file, set DATA_DIR, Run All.
  3. If it works: the saved model is a FULL model (~1.2 GB), not an adapter.
     Zip cell at the bottom makes one archive; or push to HF Hub (see note).
     It goes to ./models/slang-editor-full — the Agent 4 loader will need a small
     tweak to load a full model instead of base+adapter (ping Claude to wire it).
"""

import json, re
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter
from datasets import Dataset
from transformers import (AutoTokenizer, MT5ForConditionalGeneration,
                          DataCollatorForSeq2Seq, Seq2SeqTrainingArguments,
                          Seq2SeqTrainer, TrainerCallback)

MODEL_NAME = "google/mt5-small"
DATA_DIR = "/kaggle/input/<your-dataset-slug>"     # <-- same path you trained v1/v2 from
OUT_DIR = "/kaggle/working/slang-editor-full"
MAX_INPUT, MAX_TARGET = 96, 96
EPOCHS, BATCH, ACCUM, LR = 6, 16, 2, 2e-4          # full-FT: lower LR than LoRA, eff batch 32
EDIT_WEIGHT = 2.0                                   # sweet spot: 1.5 let copy-collapse back (slang_hit 0.77->0.44);
                                                    # the "yaar..yaar" double-wrap is cleaned in Agent 4 post-processing instead


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def neutral_of(input_str):
    return input_str.split(" | ")[-1].strip()


def words(s):
    return set(re.findall(r"[a-z]+", s.lower()))


class EditWeightedTrainer(Seq2SeqTrainer):
    """Mild edit-weighting: target tokens absent from the source input count
    EDIT_WEIGHT x more, so copying is discouraged. With a FULL fine-tune the
    model has the capacity to satisfy this by learning real slang."""
    def __init__(self, *args, edit_weight=EDIT_WEIGHT, **kwargs):
        super().__init__(*args, **kwargs)
        self.edit_weight = edit_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        input_ids = inputs["input_ids"]
        outputs = model(**inputs)
        logits = outputs.logits
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
    model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME, dtype=torch.float32)

    # --- hard-suppress ALL mT5 sentinels at generation (vocab scan, not
    #     convert_tokens_to_ids which returns unk for these) ---
    sentinel_ids = sorted(i for s, i in tok.get_vocab().items() if "extra_id" in s)
    print(f"suppressing {len(sentinel_ids)} sentinel ids at generation")
    model.generation_config.suppress_tokens = sentinel_ids
    model.generation_config.max_length = MAX_TARGET
    model.generation_config.num_beams = 4
    model.generation_config.no_repeat_ngram_size = 3

    train_raw = load_jsonl(os.path.join(DATA_DIR, "editor_train.jsonl"))
    val_raw = load_jsonl(os.path.join(DATA_DIR, "editor_val.jsonl"))
    val_neutrals = [neutral_of(ex["input"]) for ex in val_raw]

    # empirical slang lexicon: words that appear in a target but not its neutral,
    # seen in >=2 examples (filters incidental rewordings). Subtract a small
    # stoplist of pure grammatical words that leak in via rephrasing, so a
    # slang_hit means genuine slang and not just "the model emitted 'hai'".
    STOP = {"hai", "hain", "ho", "hota", "hoti", "gaya", "gayi", "gaye", "ka",
            "ki", "ke", "ko", "se", "mein", "me", "par", "aur", "ya", "ye", "yeh",
            "wo", "woh", "na", "nahi", "nahin", "kar", "karo", "kya", "hi", "to",
            "tha", "thi", "the", "raha", "rahi", "rahe", "ab", "mera", "meri",
            "tera", "teri", "apna", "is", "us", "kal", "aaj", "ek", "bahut", "bhi"}
    ins = Counter()
    for ex in train_raw:
        ins.update(words(ex["target"]) - words(neutral_of(ex["input"])))
    SLANG_LEX = {w for w, c in ins.items() if c >= 2 and len(w) >= 2} - STOP
    print(f"corpus slang lexicon size: {len(SLANG_LEX)}  e.g. {sorted(SLANG_LEX)[:12]}")

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
        neu = val_neutrals[:n]
        copies = sum(dec[i] == neu[i] for i in range(n)) / max(n, 1)
        sent = sum("<extra_id_" in dec[i] for i in range(n)) / max(n, 1)
        slang = sum(bool((words(dec[i]) - words(neu[i])) & SLANG_LEX)
                    for i in range(n)) / max(n, 1)
        mref = sum(dec[i] == refs[i] for i in range(n)) / max(n, 1)
        return {"slang_hit": round(slang, 3),       # WANT HIGH  <- the real signal
                "copies_input": round(copies, 3),   # WANT LOW
                "has_sentinel": round(sent, 3),     # WANT 0
                "matches_ref": round(mref, 3)}      # bonus (harsh)

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
                print({kk: round(v, 4) for kk, v in logs.items() if isinstance(v, float)})

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
    model.eval()
    hits = 0
    samples = val_raw[:10]
    for ex in samples:
        ids = tok(ex["input"], return_tensors="pt", truncation=True, max_length=MAX_INPUT)
        out = model.generate(**{k: v.to(model.device) for k, v in ids.items()},
                             max_new_tokens=MAX_TARGET)
        got = tok.decode(out[0], skip_special_tokens=True).strip()
        neu = neutral_of(ex["input"])
        inserted = words(got) - words(neu)
        hit = bool(inserted & SLANG_LEX)
        hits += hit
        print("IN :", ex["input"])
        print("GOT:", got, "  [slang_hit]" if hit else "  [no slang]")
        print("REF:", ex["target"], "\n")
    print(f"slang_hit on 10 samples: {hits}/10  (higher is better)")


if __name__ == "__main__":
    main()
