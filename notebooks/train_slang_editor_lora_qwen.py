"""
notebooks/train_slang_editor_lora_qwen.py   (prompt v2: shortlist-pick + ABSTAIN)
LoRA fine-tune Qwen2.5-1.5B-Instruct as the slang editor.  RUN ON KAGGLE (GPU T4).

WHAT THE MODEL LEARNS
  Given a Hinglish line + a shortlist of 5-8 candidate slangs, reply with the line
  rewritten using the ONE slang that fits best - or, if NONE fits naturally, with the
  line UNCHANGED. (v1 had no "none fits" option, so when the code-chosen shortlist had
  no good candidate it forced a bad one: "Respect, Riy, ...", "scene set hai" on a line
  about a machine burning.) The training data now includes ~15% abstain examples.

==============================================================================
KAGGLE SETUP  (NO version pins - use Kaggle's native stack; pinning broke everything)
==============================================================================
 0. New Notebook -> Accelerator = GPU T4.
 1. Add Data -> a dataset containing the NEW lora_train.jsonl + lora_val.jsonl
    (rebuilt locally with  python -m data_prep.build_lora_dataset ).
    !! Replace/refresh the old files - the format changed (v2). Old files = wrong prompt.
 2. transformers / peft / datasets are already installed on Kaggle - DO NOT pip-pin
    anything. Set DATA_DIR below and Run All. (If peft is missing: `!pip install -q peft`.)

    *** If you hit "incompatible version of torchao" during get_peft_model: ***
        FIRST CELL:  !pip uninstall -y torchao      (then RESTART the session)
 3. Read the printed "VAL CHECK" block at the end:
        abstain_ok   (on lines where nothing fits)    want >= ~0.70
        pick_ok      (on lines where a slang fits)    want >= ~0.85
        false_abstain(refuses although one fits)      want <= ~0.10
    If false_abstain is high the model learned to copy -> tell Claude (we lower the
    abstain share and retrain). If abstain_ok is ~0, abstention didn't take.
 4. Output: the adapter + prompt_version.txt at /kaggle/working/slang-editor-lora .
    Run the LAST cell to zip it, download, and extract to ./models/slang-editor-lora/
    (delete the old folder contents first). Agent 4 reads prompt_version.txt
    automatically, so no config change is needed.
==============================================================================
"""

import json, os, re
import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments, Trainer, DataCollatorForSeq2Seq)
from peft import LoraConfig, get_peft_model

# ---- CONFIG ----------------------------------------------------------------
MODEL    = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_DIR = "/kaggle/input/<your-dataset-slug>"   # folder with lora_train.jsonl / lora_val.jsonl
OUT_DIR  = "/kaggle/working/slang-editor-lora"   # final adapter (this gets zipped)
CKPT_DIR = "/kaggle/working/ckpt"                # trainer checkpoints, kept OUT of the zip
MAXLEN   = 320                                   # prompts are a bit longer (8-slang lists)
EPOCHS, BATCH, ACCUM, LR = 3, 4, 4, 2e-4         # eff. batch = 16
PROMPT_VERSION = 2
# ----------------------------------------------------------------------------

# Must be IDENTICAL to data_prep/build_lora_dataset.py and agents/agent4_slang_rewriter.py
# (checked locally by data_prep/check_prompt_sync.py).
SYSTEM_V2 = ("You are a Hinglish (romanized Hindi-English) dialogue editor. Given a sentence and "
             "a few candidate slang expressions, pick the ONE that fits best and insert it "
             "naturally, keeping the meaning EXACTLY the same. If none of the candidates fits "
             "naturally, reply with the sentence UNCHANGED. Reply with ONLY the sentence.")
USER_V2 = ("Pick the slang from this list that best fits the line, and insert it naturally "
           "without changing the meaning. If none of them fits naturally, return the line "
           "UNCHANGED:\nSlangs: {slangs}\nLine: {line}")

print("transformers", __import__("transformers").__version__, "| torch", torch.__version__)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16)
model.config.use_cache = False

lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()

ds = load_dataset("json", data_files={
    "train": os.path.join(DATA_DIR, "lora_train.jsonl"),
    "val":   os.path.join(DATA_DIR, "lora_val.jsonl")})

# sanity: the data on disk must be the v2 format, or we'd train the wrong prompt
_sys = ds["train"][0]["messages"][0]["content"]
assert _sys == SYSTEM_V2, ("lora_train.jsonl has an OLD system prompt - rebuild it locally with "
                           "data_prep.build_lora_dataset and re-upload it to Kaggle.")
val_messages = [ex["messages"] for ex in ds["val"]]       # keep a copy for the VAL CHECK


def tokenize(ex):
    msgs = ex["messages"]
    full = tok.apply_chat_template(msgs, tokenize=False)
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    full_ids = tok(full, truncation=True, max_length=MAXLEN, add_special_tokens=False)["input_ids"]
    prompt_ids = tok(prompt, truncation=True, max_length=MAXLEN, add_special_tokens=False)["input_ids"]
    n_prompt = min(len(prompt_ids), len(full_ids))        # completion-only loss
    labels = [-100] * n_prompt + full_ids[n_prompt:]
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


ds = ds.map(tokenize, remove_columns=ds["train"].column_names)
print("tokenized. sample len:", len(ds["train"][0]["input_ids"]),
      "| train", len(ds["train"]), "| val", len(ds["val"]))

args = TrainingArguments(
    output_dir=CKPT_DIR, num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH, per_device_eval_batch_size=BATCH,
    gradient_accumulation_steps=ACCUM, learning_rate=LR, warmup_steps=100,
    weight_decay=0.0, lr_scheduler_type="cosine", logging_steps=25,
    eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
    load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
    fp16=True, gradient_checkpointing=True, report_to="none",
)

trainer = Trainer(
    model=model, args=args,
    train_dataset=ds["train"], eval_dataset=ds["val"],
    processing_class=tok,
    data_collator=DataCollatorForSeq2Seq(tok, model=model, padding=True, label_pad_token_id=-100),
)
trainer.train()

model.save_pretrained(OUT_DIR)     # saves ONLY the LoRA adapter (tiny)
tok.save_pretrained(OUT_DIR)
with open(os.path.join(OUT_DIR, "prompt_version.txt"), "w") as f:
    f.write(str(PROMPT_VERSION))   # Agent 4 reads this to pick the matching prompt + shortlist size
print("saved LoRA adapter + prompt_version.txt ->", OUT_DIR)

# ---- generation helper ------------------------------------------------------
model.config.use_cache = True
model.eval()


def generate(shortlist, line):
    user = USER_V2.replace("{slangs}", ", ".join(shortlist)).replace("{line}", line)
    enc = tok.apply_chat_template([{"role": "system", "content": SYSTEM_V2},
                                   {"role": "user", "content": user}],
                                  add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True).to(model.device)
    out = model.generate(**enc, max_new_tokens=60, do_sample=False)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def norm(s):
    return re.sub(r"\s+", " ", s.strip().lower())


# ---- sanity samples: fitting shortlists, and shortlists where NOTHING fits --------
print("\n=== SANITY: a slang that fits (expect a rewrite) ===")
for shortlist, line in [
    (["jhakaas", "mast", "solid", "kadak", "bindaas"], "Aaj ka match bahut accha tha."),
    (["dimaag kharab", "waat lag gayi", "pak gaya", "bakwaas", "jhand"], "Yeh kaam baar baar bigad raha hai."),
    (["bawaal", "dhamaal", "gazab", "bomb", "next level"], "Humari team final mein pahunch gayi."),
]:
    print(f"\n[{', '.join(shortlist)}]\n  in : {line}\n  out: {generate(shortlist, line)}")

print("\n=== SANITY: nothing fits (expect the line back UNCHANGED) ===")
for shortlist, line in [
    (["bakwaas", "ghatiya", "flop", "bekaar", "waat lag gayi"], "Humari team final mein pahunch gayi."),
    (["jhakaas", "bawaal", "mast", "dhamaal", "kadak"], "Yeh movie bilkul acchi nahi thi."),
    (["note chhapna", "peti-pack", "respect", "scene set", "jackpot"],
     "Haan, lekin pehle safety check kar lete hain; aaj ki machine baar baar jalti hai."),
]:
    got = generate(shortlist, line)
    flag = "ABSTAINED (good)" if norm(got) == norm(line) else "forced an edit"
    print(f"\n[{', '.join(shortlist)}]\n  in : {line}\n  out: {got}   <- {flag}")


# ---- quantitative VAL CHECK ------------------------------------------------------
def parse_user(u):
    slangs = re.search(r"Slangs: (.*)\nLine:", u).group(1).split(", ")
    line = u.split("\nLine: ", 1)[1]
    return slangs, line


abst, pick = [], []
for msgs in val_messages:
    slangs, line = parse_user(msgs[1]["content"])
    (abst if norm(msgs[2]["content"]) == norm(line) else pick).append((slangs, line))
abst, pick = abst[:60], pick[:80]

abstain_ok = sum(norm(generate(s, l)) == norm(l) for s, l in abst)
pick_out = [(generate(s, l), s, l) for s, l in pick]
pick_ok = sum(norm(o) != norm(l) and any(c.lower() in o.lower() for c in s) for o, s, l in pick_out)
false_abstain = sum(norm(o) == norm(l) for o, s, l in pick_out)

print("\n================ VAL CHECK ================")
print(f"abstain_ok    {abstain_ok}/{len(abst)} = {abstain_ok/max(len(abst),1):.2f}   (nothing fits -> returned unchanged; want >= ~0.70)")
print(f"pick_ok       {pick_ok}/{len(pick)} = {pick_ok/max(len(pick),1):.2f}   (a slang fits -> used one from the list; want >= ~0.85)")
print(f"false_abstain {false_abstain}/{len(pick)} = {false_abstain/max(len(pick),1):.2f}   (refused although one fits; want <= ~0.10)")
print("===========================================")


# ============================================================================
# LAST CELL - zip the adapter for download
# ============================================================================
#   import shutil
#   shutil.make_archive("/kaggle/working/slang-editor-lora", "zip",
#                       "/kaggle/working/slang-editor-lora")
#   print("zipped -> /kaggle/working/slang-editor-lora.zip  (download it)")
# ============================================================================
