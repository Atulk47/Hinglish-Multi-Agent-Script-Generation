"""
notebooks/train_slang_editor_lora_qwen_v7.py      (v7 = context-conditioned LINE editor; see docs/EDITOR_ROADMAP.md)
LoRA fine-tune Qwen2.5 (1.5B or 3B Instruct) to rewrite a Hinglish line with 0-3 slang edits given the whole situation (setting, speaker, listener, emotion, scene, optional hints).
Audits the data FIRST, trains, then writes the model's answers for the held-out TEST stories to eval_outputs_v7.jsonl (scored locally by the benchmark).   RUN ON KAGGLE (GPU T4).

KAGGLE SETUP
 1. Run locally:  python -m data_prep.v7_assemble   -> data/v7/train.jsonl dev.jsonl test.jsonl
 2. kaggle.com -> Datasets -> New Dataset -> upload train.jsonl, dev.jsonl, test.jsonl  (name: slang-v7).
 3. New Notebook -> Accelerator GPU T4 x1 -> Add Data -> slang-v7 -> paste this file into one cell. Set MODEL below (1.5B first; 3B only after 1.5B looks sane).
 4. Download from the Output tab: slang-editor-v7.zip (adapter + report) and eval_outputs_v7.jsonl. Tell Claude "v7 ready".
Estimate on a T4: audit 1 min, training 30-60 min (1.5B), generation 5 min.
"""
import collections, glob, json, os, random, re, shutil, sys, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
DRY = os.environ.get("DRY_RUN") == "1"
T0 = time.time()

MODEL = os.environ.get("V7_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")        # or "Qwen/Qwen2.5-3B-Instruct"
MAXLEN = 512
EPOCHS, BATCH, ACCUM, LR = 3, 4, 4, 1.5e-4
R, ALPHA, DROPOUT = 32, 64, 0.05
SEED = 42
TAG = "3b" if "3B" in MODEL else "1_5b"
OUT_DIR = "outputs/_dry/v7" if DRY else f"/kaggle/working/slang-editor-v7-{TAG}"


def _find(name):
    hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
    return os.path.dirname(hits[0]) if hits else None


DATA_DIR = os.environ.get("DATA_DIR") or _find("train.jsonl")
assert DATA_DIR, "train.jsonl not found - attach the slang-v7 dataset"
read = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
train_rows, dev_rows, test_rows = (read(os.path.join(DATA_DIR, f"{n}.jsonl")) for n in ("train", "dev", "test"))
print(f"train {len(train_rows)} | dev {len(dev_rows)} | test {len(test_rows)} | model {MODEL}")

# ---- 1. data audit (hard failures stop the run before GPU time is spent) -------------------------------
import torch, transformers
transformers.set_seed(SEED)
random.seed(SEED)
bad = collections.Counter()
for r in train_rows + dev_rows:
    m = r["messages"]
    if [x["role"] for x in m] != ["system", "user", "assistant"]:
        bad["roles"] += 1
    t = m[2]["content"]
    if not t.strip() or "\n" in t:
        bad["empty / multi-line target"] += 1
    if re.search(r"[ऀ-ॿ]", t):
        bad["devanagari target"] += 1
    if r["kind"] == "edit" and t.strip().lower() == r["line"].strip().lower():
        bad["edit row equals the line"] += 1
    if r["kind"].startswith("keep") and t.strip() != r["line"].strip():
        bad["keep row differs from the line"] += 1
kinds = collections.Counter(r["kind"] for r in train_rows)
print("train kinds:", dict(kinds), "| violations:", dict(bad))
assert sum(bad.values()) == 0, "data violates the contract - rebuild with data_prep.v7_assemble"
test_lines = {r["line"].lower() for r in test_rows}
leak = sum(r["line"].lower() in test_lines for r in train_rows)
print("train rows whose line also occurs in test:", leak)
assert leak <= 0.02 * len(train_rows), "train/test leakage"
assert not {r["story"] for r in train_rows} & {r["story"] for r in test_rows}, "a story is in both train and test"
assert len(train_rows) >= 150, "dataset is suspiciously small"

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


def tokenize_msgs(msgs):
    full = tok.apply_chat_template(msgs, tokenize=False)
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    return tok(full, add_special_tokens=False)["input_ids"], tok(prompt, add_special_tokens=False)["input_ids"]


lens, trunc = [], 0
for r in train_rows + dev_rows:
    f, p = tokenize_msgs(r["messages"])
    lens.append(len(f))
    trunc += len(f) > MAXLEN
lens.sort()
print(f"tokens/row: median {lens[len(lens) // 2]} | p95 {lens[int(len(lens) * .95)]} | max {lens[-1]} | truncated {trunc}")
assert trunc == 0, "raise MAXLEN"
if DRY:
    print("DRY RUN OK")
    sys.exit(0)

# ---- 2. model + LoRA ------------------------------------------------------------------------------------
assert torch.cuda.is_available(), "No GPU. Notebook settings -> Accelerator -> GPU T4 x1"
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, DataCollatorForSeq2Seq, Trainer, TrainingArguments

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16)
model.config.use_cache = False
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=R, lora_alpha=ALPHA, lora_dropout=DROPOUT, bias="none", task_type="CAUSAL_LM",
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
for _n, _p in model.named_parameters():
    if _p.requires_grad:
        _p.data = _p.data.float()
model.print_trainable_parameters()


def to_features(rows):
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    for r in rows:
        f, p = tokenize_msgs(r["messages"])
        out["input_ids"].append(f)
        out["attention_mask"].append([1] * len(f))
        out["labels"].append([-100] * len(p) + f[len(p):])
    return Dataset.from_dict(out)


args = TrainingArguments(output_dir="/kaggle/working/ckpt", num_train_epochs=EPOCHS, per_device_train_batch_size=BATCH, per_device_eval_batch_size=BATCH,
                         gradient_accumulation_steps=ACCUM, learning_rate=LR, warmup_steps=20, weight_decay=0.01, lr_scheduler_type="cosine", logging_steps=10,
                         eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, load_best_model_at_end=True, metric_for_best_model="eval_loss",
                         greater_is_better=False, fp16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
                         report_to="none", seed=SEED, data_seed=SEED)
trainer = Trainer(model=model, args=args, train_dataset=to_features(train_rows), eval_dataset=to_features(dev_rows), processing_class=tok,
                  data_collator=DataCollatorForSeq2Seq(tok, model=model, padding=True, label_pad_token_id=-100))
trainer.train()
ev = [h["eval_loss"] for h in trainer.state.log_history if "eval_loss" in h]
print("validation loss per epoch:", [round(x, 4) for x in ev])
os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR)
tok.save_pretrained(OUT_DIR)

# ---- 3. answer the held-out TEST stories (greedy) ---------------------------------------------------------
model.config.use_cache = True
model.eval()
tok.padding_side = "left"


@torch.no_grad()
def generate(rows, bs=16):
    outs = []
    for s in range(0, len(rows), bs):
        chunk = rows[s:s + bs]
        prompts = [tok.apply_chat_template(r["messages"][:2], tokenize=False, add_generation_prompt=True) for r in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        n = max(len(tok(r["line"], add_special_tokens=False)["input_ids"]) for r in chunk)
        g = model.generate(**enc, max_new_tokens=min(120, int(n * 1.6) + 12), do_sample=False, pad_token_id=tok.eos_token_id)
        for row in g:
            outs.append(tok.decode(row[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip().split("\n")[0].strip().strip('"'))
    return outs


outs = generate(test_rows)
with open(f"/kaggle/working/eval_outputs_v7_{TAG}.jsonl", "w", encoding="utf-8") as f:
    for r, o in zip(test_rows, outs):
        f.write(json.dumps({"id": r["id"], "output": o}, ensure_ascii=False) + "\n")
changed = sum(o.strip().lower() != r["line"].strip().lower() for r, o in zip(test_rows, outs))
keep_ok = sum(o.strip().lower() == r["line"].strip().lower() for r, o in zip(test_rows, outs) if r["kind"] == "keep_forced")
n_forced = sum(r["kind"] == "keep_forced" for r in test_rows)
print(f"test rows {len(test_rows)} | edited {changed} | forced-keep rows left unchanged {keep_ok}/{n_forced}")
for r, o in list(zip(test_rows, outs))[:25]:
    print(f"[{r['kind']}] {r['line']}\n      -> {o}")
json.dump({"model": MODEL, "val_loss": ev, "train_rows": len(train_rows), "kinds": dict(kinds)}, open(os.path.join(OUT_DIR, "training_report.json"), "w"), indent=1)
shutil.make_archive(f"/kaggle/working/slang-editor-v7-{TAG}", "zip", OUT_DIR)
print(f"DONE in {(time.time() - T0) / 60:.0f} min")
