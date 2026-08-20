"""
data_prep/train_editor_local.py
QUICK LOCAL (CPU) de-risk train of the mT5 slang editor.

Purpose: prove the loop works end to end — train runs, adapter saves, Agent 4
(backend="mt5") loads it and generates. It trains on a SUBSET for a few steps,
so the adapter will be under-trained; the real quality run is on Kaggle
(notebooks/train_slang_editor_kaggle.py). Same data + input format.

Usage:
  python -m data_prep.train_editor_local --subset 800 --epochs 1
"""
import argparse, json, os
import torch
from datasets import Dataset
from transformers import (AutoTokenizer, MT5ForConditionalGeneration,
                          DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer)
from peft import LoraConfig, get_peft_model, TaskType
import config


def load_jsonl(p, limit=None):
    out = []
    with open(p, encoding="utf-8") as f:
        for l in f:
            if l.strip():
                out.append(json.loads(l))
            if limit and len(out) >= limit:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/felix_dataset")
    ap.add_argument("--out", default=config.EDITOR_ADAPTER_PATH)  # ./models/slang-editor-lora
    ap.add_argument("--subset", type=int, default=800)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(max(1, os.cpu_count() // 2))
    tok = AutoTokenizer.from_pretrained(config.EDITOR_BASE_MODEL)
    model = MT5ForConditionalGeneration.from_pretrained(config.EDITOR_BASE_MODEL, dtype=torch.float32)
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, r=16, lora_alpha=32,
        target_modules=["q", "v"], lora_dropout=0.05, bias="none"))
    model.print_trainable_parameters()

    train = Dataset.from_list(load_jsonl(os.path.join(args.data, "editor_train.jsonl"), args.subset))
    val = Dataset.from_list(load_jsonl(os.path.join(args.data, "editor_val.jsonl"), max(40, args.subset // 8)))

    def tok_fn(b):
        mi = tok(b["input"], max_length=config.EDITOR_MAX_INPUT, truncation=True, padding="max_length")
        lab = tok(text_target=b["target"], max_length=config.EDITOR_MAX_TARGET, truncation=True, padding="max_length")
        mi["labels"] = [[(t if t != tok.pad_token_id else -100) for t in s] for s in lab["input_ids"]]
        return mi
    train = train.map(tok_fn, batched=True, remove_columns=train.column_names)
    val = val.map(tok_fn, batched=True, remove_columns=val.column_names)

    trainer = Seq2SeqTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(
            output_dir="./models/_editor_ckpt", num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
            learning_rate=3e-4, warmup_ratio=0.05, logging_steps=20,
            eval_strategy="no", save_strategy="no", fp16=False, report_to="none"),
        train_dataset=train, eval_dataset=val, processing_class=tok,
        data_collator=DataCollatorForSeq2Seq(tok, model=model, padding=True))
    trainer.train()

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print("saved adapter ->", args.out)

    model.eval()
    for ex in load_jsonl(os.path.join(args.data, "editor_val.jsonl"), 4):
        ids = tok(ex["input"], return_tensors="pt", truncation=True, max_length=config.EDITOR_MAX_INPUT)
        gen = model.generate(**ids, max_new_tokens=config.EDITOR_MAX_TARGET, num_beams=4)
        print("\nIN :", ex["input"])
        print("GOT:", tok.decode(gen[0], skip_special_tokens=True))
        print("REF:", ex["target"])


if __name__ == "__main__":
    main()
