"""
notebooks/train_slang_editor_lora_qwen_v3.py   (prompt v2 format, ON-DISTRIBUTION data)
LoRA fine-tune Qwen2.5-1.5B-Instruct as the slang editor, then generate held-out evaluation
outputs for  (a) the untouched base model  (b) the OLD adapter (optional)  (c) the NEW adapter.
RUN ON KAGGLE (GPU T4).

WHAT CHANGED vs the previous notebook
  * data = Agent-3-style whole SENTENCES (mean ~18 words, stiff register) labelled offline by a
    teacher LLM and double-checked by an independent judge; the old short casual rows are a minor
    share. Same v2 prompt format, so Agent 4 needs no change (prompt_version.txt = 2).
  * the slang bank is a curated one (113 expressions) - the same file used at inference.
  * evaluation is STRICT and on held-out data: synthetic held-out scenes AND real pipeline
    sentences. pick_ok only counts an output when the slang is NEWLY inserted from the shortlist
    AND the runtime drift guard passes (the old pick_ok=0.94 only checked "a listed word appears").

==============================================================================
KAGGLE SETUP  (no pip pins)
==============================================================================
 0. New Notebook -> Accelerator = GPU T4.
 1. Add Data -> a dataset (e.g. "slang-v3") containing ALL of:
        lora_train.jsonl  lora_val.jsonl  eval_synth.jsonl  eval_real.jsonl
    (from data/felix_dataset/v3/ after  python -m data_prep.build_v3_dataset ).
 2. OPTIONAL but recommended: add your OLD adapter as a second dataset (the folder that holds
    adapter_model.safetensors from models/slang-editor-lora) and set OLD_ADAPTER below,
    so the before/after comparison is produced in the same run.
 3. If peft errors with torchao:  FIRST CELL  !pip uninstall -y torchao   then restart.
 4. Run All. Download from /kaggle/working:
        slang-editor-lora.zip        (the new adapter  -> models/slang-editor-lora-v3/)
        eval_outputs_new.jsonl  eval_outputs_old.jsonl  eval_outputs_base.jsonl
    and put the three eval_outputs_*.jsonl into data/felix_dataset/v3/ locally.
==============================================================================
"""

import json, os, re, random
import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments, Trainer, DataCollatorForSeq2Seq)
from peft import LoraConfig, get_peft_model

# ---- CONFIG ----------------------------------------------------------------
MODEL       = "Qwen/Qwen2.5-1.5B-Instruct"
# Paths are auto-detected under /kaggle/input (no edits needed): the folder that holds
# lora_train.jsonl, and (optionally) the folder that holds the OLD adapter_config.json.
import glob
def _find_dir(pattern):
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    return os.path.dirname(hits[0]) if hits else None
DATA_DIR    = _find_dir("eval_filler.jsonl") or _find_dir("lora_train.jsonl")   # v4 data has eval_filler.jsonl
OLD_ADAPTER = _find_dir("adapter_config.json")           # None if you did not add the old adapter
assert DATA_DIR, "lora_train.jsonl not found - add the slang-v3 dataset to the notebook (Add Data)"
print("DATA_DIR   :", DATA_DIR)
print("OLD_ADAPTER:", OLD_ADAPTER)
OUT_DIR     = "/kaggle/working/slang-editor-lora"
CKPT_DIR    = "/kaggle/working/ckpt"
MAXLEN      = 384
EPOCHS, BATCH, ACCUM, LR = 4, 8, 2, 1.5e-4               # eff. batch = 16; best epoch kept by val loss
R, ALPHA, DROPOUT = 32, 64, 0.05
PROMPT_VERSION = 2
SEED = 42
# ----------------------------------------------------------------------------

# IDENTICAL to data_prep/build_lora_dataset.py and agents/agent4_slang_rewriter.py
SYSTEM_V2 = ("You are a Hinglish (romanized Hindi-English) dialogue editor. Given a sentence and "
             "a few candidate slang expressions, pick the ONE that fits best and insert it "
             "naturally, keeping the meaning EXACTLY the same. If none of the candidates fits "
             "naturally, reply with the sentence UNCHANGED. Reply with ONLY the sentence.")
USER_V2 = ("Pick the slang from this list that best fits the line, and insert it naturally "
           "without changing the meaning. If none of them fits naturally, return the line "
           "UNCHANGED:\nSlangs: {slangs}\nLine: {line}")

random.seed(SEED); torch.manual_seed(SEED)
print("transformers", __import__("transformers").__version__, "| torch", torch.__version__)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16)
model.config.use_cache = False
lora = LoraConfig(r=R, lora_alpha=ALPHA, lora_dropout=DROPOUT, bias="none", task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, lora)
model.print_trainable_parameters()

ds = load_dataset("json", data_files={"train": os.path.join(DATA_DIR, "lora_train.jsonl"),
                                      "val": os.path.join(DATA_DIR, "lora_val.jsonl")})
assert ds["train"][0]["messages"][0]["content"] == SYSTEM_V2, "data has an OLD system prompt - rebuild it"
val_messages = [ex["messages"] for ex in ds["val"]]


def tokenize(ex):
    msgs = ex["messages"]
    full = tok.apply_chat_template(msgs, tokenize=False)
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    full_ids = tok(full, truncation=True, max_length=MAXLEN, add_special_tokens=False)["input_ids"]
    prompt_ids = tok(prompt, truncation=True, max_length=MAXLEN, add_special_tokens=False)["input_ids"]
    n_prompt = min(len(prompt_ids), len(full_ids))          # completion-only loss
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids),
            "labels": [-100] * n_prompt + full_ids[n_prompt:]}


ds = ds.map(tokenize, remove_columns=ds["train"].column_names)
print("tokenized | train", len(ds["train"]), "| val", len(ds["val"]),
      "| max len", max(len(x) for x in ds["train"]["input_ids"]))

args = TrainingArguments(
    output_dir=CKPT_DIR, num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH, per_device_eval_batch_size=BATCH,
    gradient_accumulation_steps=ACCUM, learning_rate=LR, warmup_steps=60,
    weight_decay=0.01, lr_scheduler_type="cosine", logging_steps=25,
    eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
    load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
    fp16=True, gradient_checkpointing=True, report_to="none", seed=SEED,
)
trainer = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["val"],
                  processing_class=tok,
                  data_collator=DataCollatorForSeq2Seq(tok, model=model, padding=True, label_pad_token_id=-100))
trainer.train()

model.save_pretrained(OUT_DIR)
tok.save_pretrained(OUT_DIR)
with open(os.path.join(OUT_DIR, "prompt_version.txt"), "w") as f:
    f.write(str(PROMPT_VERSION))
print("saved LoRA adapter + prompt_version.txt ->", OUT_DIR)

# ---- batched generation helper --------------------------------------------------
model.config.use_cache = True
model.eval()
tok.padding_side = "left"


def build_prompt(shortlist, line):
    user = USER_V2.replace("{slangs}", ", ".join(shortlist)).replace("{line}", line)
    return tok.apply_chat_template([{"role": "system", "content": SYSTEM_V2},
                                    {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate_many(items, bs=24):
    """items: list of (shortlist, line) -> list of output strings (greedy)."""
    outs = []
    for s in range(0, len(items), bs):
        chunk = items[s:s + bs]
        enc = tok([build_prompt(sl, ln) for sl, ln in chunk], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        n_line = max(len(tok(ln, add_special_tokens=False)["input_ids"]) for _, ln in chunk)
        g = model.generate(**enc, max_new_tokens=min(120, int(n_line * 1.4) + 14), do_sample=False,
                           pad_token_id=tok.eos_token_id)
        for row in g:
            t = tok.decode(row[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            outs.append(t.split("\n")[0].strip().strip('"'))
    return outs


def norm(s):
    return re.sub(r"\s+", " ", s.strip().lower())


# ---- the runtime drift guard (identical to SlangRewriter._keeps_meaning) -----------
def keeps_meaning(neutral, output):
    nlist = re.findall(r"[a-z]+", neutral.lower())
    nwords = {w for w in nlist if len(w) >= 3}
    if not nwords:
        return True
    owords = set(re.findall(r"[a-z]+", output.lower()))
    dropped, added = len(nwords - owords), len(owords - set(nlist))
    return dropped <= (2 if len(nwords) >= 10 else (len(nwords) + 1) // 2) and added <= 7


def strict_ok(line, shortlist, out):
    """An output counts as a valid edit only if it differs, a shortlist slang is NEWLY inserted,
    and the runtime drift guard passes."""
    if norm(out) == norm(line):
        return False
    used = [s for s in shortlist if s.lower() in out.lower() and s.lower() not in line.lower()]
    return bool(used) and keeps_meaning(line, out)


def parse_user(u):
    return re.search(r"Slangs: (.*)\nLine:", u).group(1).split(", "), u.split("\nLine: ", 1)[1]


# ---- VAL CHECK (random-held-out val rows, STRICT) ----------------------------------
abst, pick = [], []
for msgs in val_messages:
    sl, ln = parse_user(msgs[1]["content"])
    (abst if norm(msgs[2]["content"]) == norm(ln) else pick).append((sl, ln))
abst, pick = abst[:120], pick[:200]
a_out = generate_many(abst)
p_out = generate_many(pick)
abstain_ok = sum(norm(o) == norm(ln) for o, (sl, ln) in zip(a_out, abst))
pick_ok = sum(strict_ok(ln, sl, o) for o, (sl, ln) in zip(p_out, pick))
false_abs = sum(norm(o) == norm(ln) for o, (sl, ln) in zip(p_out, pick))
print("\n================ VAL CHECK (strict) ================")
print(f"abstain_ok    {abstain_ok}/{len(abst)} = {abstain_ok/max(len(abst),1):.2f}   (want >= 0.70)")
print(f"pick_ok       {pick_ok}/{len(pick)} = {pick_ok/max(len(pick),1):.2f}   (newly-inserted slang + drift guard; want >= 0.85)")
print(f"false_abstain {false_abs}/{len(pick)} = {false_abs/max(len(pick),1):.2f}   (want <= 0.10)")
print("====================================================")


# ---- held-out evaluation outputs for base / old / new ----------------------------------
def load_eval():
    rows = []
    for name in ("eval_real.jsonl", "eval_synth.jsonl", "eval_filler.jsonl"):
        p = os.path.join(DATA_DIR, name)
        if os.path.exists(p):
            rows += [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    return rows


EVAL = load_eval()
print(f"\nevaluating on {len(EVAL)} held-out rows "
      f"({sum(r['src']=='real' for r in EVAL)} real pipeline, {sum(r['src']=='synth' for r in EVAL)} synthetic)")


def dump(tag):
    outs = generate_many([(r["shortlist"], r["line"]) for r in EVAL])
    with open(f"/kaggle/working/eval_outputs_{tag}.jsonl", "w", encoding="utf-8") as f:
        for r, o in zip(EVAL, outs):
            f.write(json.dumps({"id": r["id"], "output": o}, ensure_ascii=False) + "\n")
    ok_edit = [strict_ok(r["line"], r["shortlist"], o) for r, o in zip(EVAL, outs) if r["expect"] == "edit"]
    ab = [norm(o) == norm(r["line"]) for r, o in zip(EVAL, outs) if r["expect"] == "abstain"]
    print(f"[{tag}] valid-edit rate on edit rows {sum(ok_edit)}/{len(ok_edit)} = {sum(ok_edit)/max(len(ok_edit),1):.2f} | "
          f"abstain rate on no-fit rows {sum(ab)}/{len(ab)} = {sum(ab)/max(len(ab),1):.2f}")


if EVAL:
    model.set_adapter("default")
    dump("new")
    with model.disable_adapter():
        dump("base")                                    # zero-shot Qwen, no fine-tuning
    if OLD_ADAPTER and os.path.exists(os.path.join(OLD_ADAPTER, "adapter_config.json")):
        model.load_adapter(OLD_ADAPTER, adapter_name="old")
        model.set_adapter("old")
        dump("old")
        model.set_adapter("default")
    else:
        print("OLD_ADAPTER not found - skipped (set it to compare against the previous adapter)")

import shutil
shutil.make_archive("/kaggle/working/slang-editor-lora", "zip", OUT_DIR)
print("DONE. Download from the Output tab: slang-editor-lora.zip, eval_outputs_new.jsonl, "
      "eval_outputs_base.jsonl, eval_outputs_old.jsonl")
