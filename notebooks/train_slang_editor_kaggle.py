"""
notebooks/train_slang_editor_kaggle.py
Train the mT5-small "slang editor" (Agent 4) with LoRA. RUN THIS ON KAGGLE (GPU T4).

WHY mT5 and not IndicBART: IndicBART's vocabulary is Devanagari-only, so romanized
Hinglish is out-of-distribution (see indicbart_pivot_report.md). mT5 was pretrained on
mC4 which contains romanized Hinglish, so its tokenizer already knows these subwords.

WHY these settings differ from the IndicBART notebook:
  * model class      : MT5ForConditionalGeneration (not MBart)
  * tokenizer        : AutoTokenizer (mT5 SentencePiece) — no keep_accents / <2hi> token
  * NO decoder_start_token_id hack (mT5 uses pad as decoder start automatically)
  * LoRA target_modules = ["q", "v"]  (T5 attention linears) NOT ["q_proj","v_proj"]
  * fp32 training (mT5 is numerically unstable in fp16 -> NaN loss on T4)

STEPS ON KAGGLE
  1. Locally:  python -m data_prep.build_editor_dataset --extra data/felix_dataset/augmented_v2.csv
     -> produces data/felix_dataset/editor_train.jsonl + editor_val.jsonl
  2. Upload those two JSONL files as a Kaggle Dataset (e.g. "hinglish-slang-editor").
  3. New Kaggle notebook, Accelerator = GPU T4 x2, add that dataset.
  4. Set DATA_DIR below to the dataset mount path, run this file.
  5. Download /kaggle/working/slang-editor-lora and place it at ./models/slang-editor-lora
     in the repo. Agent 4 (backend="auto") will then load it automatically.
"""
import json
import os
import torch
from datasets import Dataset
from transformers import (AutoTokenizer, MT5ForConditionalGeneration,
                          DataCollatorForSeq2Seq, Seq2SeqTrainingArguments,
                          Seq2SeqTrainer, TrainerCallback)
from peft import LoraConfig, get_peft_model, TaskType

MODEL_NAME = "google/mt5-small"
DATA_DIR = "/kaggle/input/hinglish-slang-editor"   # <-- set to your Kaggle dataset mount
OUT_DIR = "/kaggle/working/slang-editor-lora"
MAX_INPUT, MAX_TARGET = 96, 96
EPOCHS, BATCH, LR = 8, 32, 3e-4


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME, dtype=torch.float32)

    lora = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=16, lora_alpha=32,
                      target_modules=["q", "v"], lora_dropout=0.05, bias="none")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train = Dataset.from_list(load_jsonl(os.path.join(DATA_DIR, "editor_train.jsonl")))
    val = Dataset.from_list(load_jsonl(os.path.join(DATA_DIR, "editor_val.jsonl")))

    def tokenize(batch):
        mi = tok(batch["input"], max_length=MAX_INPUT, truncation=True, padding="max_length")
        labels = tok(text_target=batch["target"], max_length=MAX_TARGET,
                     truncation=True, padding="max_length")
        mi["labels"] = [[(t if t != tok.pad_token_id else -100) for t in seq]
                        for seq in labels["input_ids"]]
        return mi

    train = train.map(tokenize, batched=True, remove_columns=train.column_names)
    val = val.map(tokenize, batched=True, remove_columns=val.column_names)

    args = Seq2SeqTrainingArguments(
        output_dir=OUT_DIR, num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH, per_device_eval_batch_size=BATCH,
        gradient_accumulation_steps=1, learning_rate=LR, warmup_steps=200,
        logging_steps=50, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        fp16=False, report_to="none", predict_with_generate=False,
    )

    class PrintLoss(TrainerCallback):
        def on_log(self, a, s, c, logs=None, **k):
            if logs:
                print({kk: round(v, 4) for kk, v in logs.items() if isinstance(v, float)})

    trainer = Seq2SeqTrainer(
        model=model, args=args, train_dataset=train, eval_dataset=val,
        processing_class=tok,
        data_collator=DataCollatorForSeq2Seq(tok, model=model, padding=True),
        callbacks=[PrintLoss()],
    )
    trainer.train()
    model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    print("saved LoRA adapter ->", OUT_DIR)

    # quick sanity generation
    model.eval()
    for ex in load_jsonl(os.path.join(DATA_DIR, "editor_val.jsonl"))[:5]:
        ids = tok(ex["input"], return_tensors="pt", truncation=True, max_length=MAX_INPUT)
        out = model.generate(**{k: v.to(model.device) for k, v in ids.items()},
                             max_new_tokens=MAX_TARGET, num_beams=4)
        print("IN :", ex["input"])
        print("GOT:", tok.decode(out[0], skip_special_tokens=True))
        print("REF:", ex["target"], "\n")


if __name__ == "__main__":
    main()
