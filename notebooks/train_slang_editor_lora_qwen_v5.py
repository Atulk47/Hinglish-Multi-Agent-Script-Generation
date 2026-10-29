"""
notebooks/train_slang_editor_lora_qwen_v5.py      (prompt v2 format, slot-typed mood data + phrase swaps)
LoRA fine-tune Qwen2.5-1.5B-Instruct as the slang editor, audit the data FIRST, train, then run a full strict evaluation
against the previous adapter and the untouched base model, and finish with a PASS/FAIL checklist.   RUN ON KAGGLE (GPU T4).

WHAT IS NEW vs v4
  * data = v4 rows (fillers, formal abstains, older edit patterns) + NEW rows whose gold edit was BUILT BY CODE for each slot
    kind (praise / criticism / worry / trouble / ease) incl. idiom-for-phrase swaps ("bahut dar lag raha hai" -> "fatti padi hai"),
    verified by the same rule gate the runtime uses, then rated >= 4/5 for naturalness by an LLM judge. Plus "hard abstain" rows
    (negation, respectful 'aap', addressed to an elder, hospital scenes, plain facts, questions).
  * two whole domains are held out of training (festival and shaadi, tech / robotics project) -> honest generalisation test.
  * the notebook REFUSES to train on bad data: schema, prompt, shortlist, target consistency, duplicates, truncation, train/eval
    leakage are all checked first (section 2). Nothing is silently dropped except exact duplicates of eval lines.
  * evaluation also runs the REAL runtime rule gate (edit_gate.py shipped in the dataset folder), per slot type, per eval file.

==============================================================================
KAGGLE SETUP (no pip pins)
==============================================================================
 1. kaggle.com -> Datasets -> New Dataset -> upload EVERYTHING in kaggle_v5/ :
      lora_train.jsonl  lora_val.jsonl  eval_slots.jsonl  eval_realv5.jsonl  eval_real.jsonl  eval_synth.jsonl  eval_filler.jsonl
      edit_gate.py  (the runtime rule gate)   and the folder  old_adapter/  (your v4 adapter files).  Name it  slang-v5.
 2. New Notebook -> Accelerator GPU T4 x1 -> Add Data -> attach ONLY slang-v5 -> paste this whole file into one cell -> Run All.
    (If peft complains about torchao:  !pip uninstall -y torchao  in a cell above, then restart.)
 3. Read the CHECKLIST at the end. Download from the Output tab:  slang-editor-lora.zip (adapter + training_report.json)
    and eval_outputs_new.jsonl / eval_outputs_old.jsonl / eval_outputs_base.jsonl.
 4. Unzip into models/slang-editor-lora-v5/ and tell Claude "v5 ready".
Runtime estimate on a T4: audit ~1 min, training ~40-60 min, evaluation ~10 min.
==============================================================================
"""
import collections, glob, hashlib, importlib.util, json, os, random, re, shutil, sys, time

DRY = os.environ.get("DRY_RUN") == "1"          # local dry run: audit + tokenisation checks only (no GPU, no training)
T0 = time.time()

# ---- CONFIG ----------------------------------------------------------------------------------------
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAXLEN = 384
EPOCHS, BATCH, ACCUM, LR = 3, 8, 2, 1.5e-4        # effective batch 16; best epoch kept by validation loss
R, ALPHA, DROPOUT = 32, 64, 0.05
PROMPT_VERSION = 2
SEED = 42
OUT_DIR, CKPT_DIR = "/kaggle/working/slang-editor-lora", "/kaggle/working/ckpt"
if DRY:
    OUT_DIR, CKPT_DIR = "outputs/_dry/adapter", "outputs/_dry/ckpt"


def _find(pattern, root="/kaggle/input"):
    hits = sorted(glob.glob(f"{root}/**/{pattern}", recursive=True))
    return os.path.dirname(hits[0]) if hits else None


DATA_DIR = os.environ.get("DATA_DIR") or _find("lora_train.jsonl")
assert DATA_DIR, "lora_train.jsonl not found - attach the slang-v5 dataset (Add Data)"
OLD_ADAPTER = os.environ.get("OLD_ADAPTER") or (None if DRY else _find("adapter_config.json"))
print("DATA_DIR   :", DATA_DIR, "\nOLD_ADAPTER:", OLD_ADAPTER)

# ---- 0. environment checks --------------------------------------------------------------------------
import torch, transformers
print("python", sys.version.split()[0], "| torch", torch.__version__, "| transformers", transformers.__version__)
if not DRY:
    assert torch.cuda.is_available(), "No GPU. Notebook settings -> Accelerator -> GPU T4 x1"
    print("GPU:", torch.cuda.get_device_name(0), "| VRAM %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1e9))
transformers.set_seed(SEED)
random.seed(SEED)

# IDENTICAL to data_prep/build_lora_dataset.py and agents/agent4_slang_rewriter.py
SYSTEM_V2 = ("You are a Hinglish (romanized Hindi-English) dialogue editor. Given a sentence and "
             "a few candidate slang expressions, pick the ONE that fits best and insert it "
             "naturally, keeping the meaning EXACTLY the same. If none of the candidates fits "
             "naturally, reply with the sentence UNCHANGED. Reply with ONLY the sentence.")
USER_V2 = ("Pick the slang from this list that best fits the line, and insert it naturally "
           "without changing the meaning. If none of them fits naturally, return the line "
           "UNCHANGED:\nSlangs: {slangs}\nLine: {line}")


# ---- helpers (the runtime drift guard is replicated EXACTLY: agents/agent4_slang_rewriter.py::_keeps_meaning) ------------
def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def keyline(s):
    return re.sub(r"[^a-z ]", "", (s or "").lower()).strip()


def keeps_meaning(neutral, output):
    nlist = re.findall(r"[a-z]+", neutral.lower())
    nwords = {w for w in nlist if len(w) >= 3}
    if not nwords:
        return True
    blist = re.findall(r"[a-z]+", output.lower())
    owords = set(blist)
    dropped, added = len(nwords - owords), len(owords - set(nlist))
    allowed_drop = 2 if len(nwords) >= 10 else (len(nwords) + 1) // 2
    p = 0
    while p < min(len(nlist), len(blist)) and nlist[p] == blist[p]:
        p += 1
    s = 0
    while s < min(len(nlist), len(blist)) - p and nlist[-1 - s] == blist[-1 - s]:
        s += 1
    span = nlist[p:len(nlist) - s]
    if 0 < len(span) <= 5:                                   # one contiguous phrase swap
        allowed_drop = max(allowed_drop, len({w for w in span if len(w) >= 3}))
    return dropped <= allowed_drop and added <= 7


def used_slang(line, shortlist, out):
    return next((s for s in shortlist if s.lower() in out.lower() and s.lower() not in line.lower()), None)


def strict_ok(line, shortlist, out):
    """A valid edit: differs, a shortlist slang is NEWLY inserted, and the runtime drift guard passes."""
    return norm(out) != norm(line) and used_slang(line, shortlist, out) is not None and keeps_meaning(line, out)


def parse_user(u):
    m = re.search(r"Slangs: (.*)\nLine: (.*)$", u, re.DOTALL)
    assert m, "user message does not match the v2 template"
    return m.group(1).split(", "), m.group(2)


# the REAL runtime rule gate (shipped next to the data); optional but strongly recommended
GATE = None
_gp = os.path.join(DATA_DIR, "edit_gate.py")
if os.path.exists(_gp):
    spec = importlib.util.spec_from_file_location("edit_gate", _gp)
    GATE = importlib.util.module_from_spec(spec)
    sys.modules["edit_gate"] = GATE
    spec.loader.exec_module(GATE)
    print("runtime rule gate loaded:", _gp)
else:
    print("WARNING: edit_gate.py not found in the dataset folder - gate metrics will be skipped")


def gate_ok(line, shortlist, out):
    if GATE is None:
        return None
    u = used_slang(line, shortlist, out)
    return bool(u) and GATE.rule_gate(line, out, u)[0] and keeps_meaning(line, out)


def read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


# ---- 1. load data ------------------------------------------------------------------------------------
train_rows = read_jsonl(os.path.join(DATA_DIR, "lora_train.jsonl"))
val_rows = read_jsonl(os.path.join(DATA_DIR, "lora_val.jsonl"))
EVAL_FILES = sorted(glob.glob(os.path.join(DATA_DIR, "eval_*.jsonl")))
EVAL = []
for f in EVAL_FILES:
    for r in read_jsonl(f):
        r["file"] = os.path.basename(f).replace(".jsonl", "")
        EVAL.append(r)
print(f"loaded train {len(train_rows)} | val {len(val_rows)} | eval {len(EVAL)} rows from {[os.path.basename(f) for f in EVAL_FILES]}")

# ---- 2. DATA AUDIT (hard fails stop the run BEFORE any GPU time is spent) ----------------------------------
AUDIT = {}


def audit(rows, name, hard=True):
    bad = collections.defaultdict(list)
    abst = edit = 0
    words, pos_gold = [], collections.Counter()
    gold_slang = collections.Counter()
    for i, ex in enumerate(rows):
        try:
            m = ex["messages"]
            assert [x["role"] for x in m] == ["system", "user", "assistant"], "roles"
            assert m[0]["content"] == SYSTEM_V2, "system prompt differs from v2"
            sl, line = parse_user(m[1]["content"])
            assert 2 <= len(sl) <= 8 and len(set(map(str.lower, sl))) == len(sl), "shortlist size/dups"
            tgt = m[2]["content"].strip()
            assert tgt and "\n" not in tgt, "empty / multi-line target"
            words.append(len(line.split()))
            if norm(tgt) == norm(line):
                abst += 1
            else:
                edit += 1
                assert strict_ok(line, sl, tgt), "target is not a valid edit (slang not newly inserted from the list, or drift)"
                u = used_slang(line, sl, tgt)
                gold_slang[u.lower()] += 1
                pos_gold[sl.index(u) if u in sl else -1] += 1
                if GATE is not None:
                    assert GATE.rule_gate(line, tgt, u)[0], "target is rejected by the runtime rule gate"
        except AssertionError as e:
            bad[str(e)].append(i)
        except Exception as e:                                   # malformed row
            bad["malformed: " + type(e).__name__].append(i)
    n_bad = sum(len(v) for v in bad.values())
    users = collections.defaultdict(set)
    for ex in rows:
        users[ex["messages"][1]["content"]].add(ex["messages"][2]["content"])
    conflicts = sum(1 for v in users.values() if len(v) > 1)
    dups = len(rows) - len(users)
    print(f"\n[{name}] rows {len(rows)} | edit {edit} | abstain {abst} ({abst / max(len(rows), 1):.0%}) | "
          f"avg line {sum(words) / max(len(words), 1):.1f} words | exact-duplicate prompts {dups} | conflicting duplicates {conflicts}")
    if gold_slang:
        top = gold_slang.most_common(5)
        print(f"   gold slang: {len(gold_slang)} distinct | top {top} | largest share {top[0][1] / max(edit, 1):.0%}")
        print(f"   gold position in shortlist: {dict(sorted(pos_gold.items()))}")
    for why, idx in bad.items():
        print(f"   !! {len(idx):4d} rows: {why}   e.g. row {idx[:3]}")
    AUDIT[name] = {"rows": len(rows), "bad": n_bad, "bad_reasons": {k: len(v) for k, v in bad.items()}, "dups": dups,
                   "conflicts": conflicts, "abstain_share": round(abst / max(len(rows), 1), 3),
                   "distinct_slang": len(gold_slang)}
    return bad, n_bad


bad_tr, nb_tr = audit(train_rows, "train")
bad_va, nb_va = audit(val_rows, "val")
assert nb_tr <= 0.02 * len(train_rows), f"{nb_tr} / {len(train_rows)} train rows violate the contract (>2%) - rebuild the dataset"
assert nb_va <= 0.03 * len(val_rows), f"{nb_va} val rows violate the contract"
# drop the (few) contract violations from training so they cannot teach a wrong behaviour
bad_idx = {i for v in bad_tr.values() for i in v}
train_rows = [r for i, r in enumerate(train_rows) if i not in bad_idx]
val_bad = {i for v in bad_va.values() for i in v}
val_rows = [r for i, r in enumerate(val_rows) if i not in val_bad]
# conflicting duplicates (same prompt, different answers) are removed entirely: they are label noise
seen = collections.defaultdict(list)
for i, r in enumerate(train_rows):
    seen[r["messages"][1]["content"]].append(i)
drop = set()
for k, idxs in seen.items():
    outs = {train_rows[i]["messages"][2]["content"] for i in idxs}
    if len(outs) > 1:
        drop |= set(idxs)
    else:
        drop |= set(idxs[1:])                                    # exact duplicates: keep one
train_rows = [r for i, r in enumerate(train_rows) if i not in drop]
print(f"removed {len(drop)} duplicate / conflicting train rows -> train {len(train_rows)}")

# train/eval leakage: no evaluation line may be in train or val
eval_keys = {keyline(r["line"]) for r in EVAL}
leak_tr = [i for i, r in enumerate(train_rows) if keyline(parse_user(r["messages"][1]["content"])[1]) in eval_keys]
leak_va = [i for i, r in enumerate(val_rows) if keyline(parse_user(r["messages"][1]["content"])[1]) in eval_keys]
print(f"leakage: {len(leak_tr)} train rows and {len(leak_va)} val rows duplicate an evaluation line -> removed")
assert len(leak_tr) <= 0.02 * len(train_rows), "too much train/eval leakage - rebuild the dataset"
train_rows = [r for i, r in enumerate(train_rows) if i not in set(leak_tr)]
val_rows = [r for i, r in enumerate(val_rows) if i not in set(leak_va)]
AUDIT["leak_removed"] = len(leak_tr) + len(leak_va)
EVAL_BAD = [r["id"] for r in EVAL if not (2 <= len(r["shortlist"]) <= 8 and r["expect"] in ("edit", "abstain"))]
assert not EVAL_BAD, f"malformed eval rows: {EVAL_BAD[:5]}"
assert len(train_rows) >= 1500 and len(val_rows) >= 60, "dataset is suspiciously small"
data_hash = hashlib.sha256("".join(r["messages"][1]["content"] + r["messages"][2]["content"] for r in train_rows).encode()).hexdigest()[:12]
print(f"AUDIT PASSED | final train {len(train_rows)} | val {len(val_rows)} | data hash {data_hash} | {time.time() - T0:.0f}s")

# ---- 3. tokenisation checks -----------------------------------------------------------------------
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


def tokenize_msgs(msgs):
    full = tok.apply_chat_template(msgs, tokenize=False)
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    full_ids = tok(full, add_special_tokens=False)["input_ids"]
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    return full_ids, prompt_ids


lens, n_trunc, n_nolabel = [], 0, 0
for r in train_rows + val_rows:
    f, p = tokenize_msgs(r["messages"])
    lens.append(len(f))
    n_trunc += len(f) > MAXLEN
    n_nolabel += (len(f) - len(p)) < 3
lens.sort()
print(f"tokens/row: median {lens[len(lens) // 2]} | p95 {lens[int(len(lens) * .95)]} | max {lens[-1]} | truncated {n_trunc} | rows with <3 label tokens {n_nolabel}")
assert n_trunc == 0 and n_nolabel == 0, "some rows would be truncated / have no completion tokens - raise MAXLEN or fix the data"

if DRY:
    print("\nDRY RUN OK - data audit and tokenisation passed. (No model was loaded.)")
    sys.exit(0)

# ---- 4. model + LoRA --------------------------------------------------------------------------------
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, DataCollatorForSeq2Seq, Trainer, TrainingArguments

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16)
model.config.use_cache = False
model.enable_input_require_grads()                                # needed for gradient checkpointing with a frozen base
lora = LoraConfig(r=R, lora_alpha=ALPHA, lora_dropout=DROPOUT, bias="none", task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, lora)
for _n, _p in model.named_parameters():
    if _p.requires_grad:
        _p.data = _p.data.float()                                 # fp32 adapter weights: stable fp16 mixed-precision training
model.print_trainable_parameters()
assert any(p.requires_grad for p in model.parameters()), "no trainable parameters"


def to_features(rows):
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    for r in rows:
        f, p = tokenize_msgs(r["messages"])
        out["input_ids"].append(f)
        out["attention_mask"].append([1] * len(f))
        out["labels"].append([-100] * len(p) + f[len(p):])          # completion-only loss
    return Dataset.from_dict(out)


ds_train, ds_val = to_features(train_rows), to_features(val_rows)
print("features:", len(ds_train), len(ds_val))
args = TrainingArguments(
    output_dir=CKPT_DIR, num_train_epochs=EPOCHS, per_device_train_batch_size=BATCH, per_device_eval_batch_size=BATCH,
    gradient_accumulation_steps=ACCUM, learning_rate=LR, warmup_steps=60, weight_decay=0.01, lr_scheduler_type="cosine",
    logging_steps=25, eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, load_best_model_at_end=True,
    metric_for_best_model="eval_loss", greater_is_better=False, fp16=True, gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}, report_to="none", seed=SEED, data_seed=SEED)
trainer = Trainer(model=model, args=args, train_dataset=ds_train, eval_dataset=ds_val, processing_class=tok,
                  data_collator=DataCollatorForSeq2Seq(tok, model=model, padding=True, label_pad_token_id=-100))
trainer.train()
hist = trainer.state.log_history
ev = [h for h in hist if "eval_loss" in h]
tr = [h for h in hist if "loss" in h]
print("\nvalidation loss per epoch:", [round(h["eval_loss"], 4) for h in ev])
assert ev and all(h["eval_loss"] == h["eval_loss"] for h in ev), "NaN validation loss"
assert tr and tr[-1]["loss"] < tr[0]["loss"], "training loss did not go down - something is wrong"
best_val_loss = min(h["eval_loss"] for h in ev)

os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR)
tok.save_pretrained(OUT_DIR)
open(os.path.join(OUT_DIR, "prompt_version.txt"), "w").write(str(PROMPT_VERSION))
print("saved adapter ->", OUT_DIR)

# ---- 5. generation helpers ------------------------------------------------------------------------------
model.config.use_cache = True
model.eval()
tok.padding_side = "left"


def build_prompt(shortlist, line):
    user = USER_V2.replace("{slangs}", ", ".join(shortlist)).replace("{line}", line)
    return tok.apply_chat_template([{"role": "system", "content": SYSTEM_V2}, {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate_many(items, bs=24):
    outs = []
    for s in range(0, len(items), bs):
        chunk = items[s:s + bs]
        enc = tok([build_prompt(sl, ln) for sl, ln in chunk], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        n_line = max(len(tok(ln, add_special_tokens=False)["input_ids"]) for _, ln in chunk)
        g = model.generate(**enc, max_new_tokens=min(120, int(n_line * 1.4) + 14), do_sample=False, pad_token_id=tok.eos_token_id)
        for row in g:
            t = tok.decode(row[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            outs.append(t.split("\n")[0].strip().strip('"'))
    return outs


# ---- 6. sanity prompts (read these by eye) ----------------------------------------------------------------
SANITY = [
    (["bindaas", "zabardast", "top class", "badiya"], "Kal ki presentation mein tumne bahut accha bola, Meera."),
    (["fatti padi hai", "full tension", "dimaag fry", "jaan nikal gayi"], "Mujhe kal ke result ki bahut chinta ho rahi hai."),
    (["panga", "locha", "jhamela", "gadbad"], "Kabir, yeh toh bahut mushkil situation hai."),
    (["bakwaas", "faltu", "wahiyat", "ghatiya"], "Mujhe woh movie bahut kharab lagi, bilkul time waste."),
    (["chutki mein", "bacchon ka khel", "left hand ka khel"], "Hum yeh sawaal aasaani se solve kar lenge."),
    (["bakwaas", "zabardast", "faltu"], "Sir, aapka lecture bahut accha tha, dhanyavaad."),                 # respectful -> unchanged
    (["bakwaas", "zabardast", "faltu"], "Yeh plan accha nahi hai, hum kuch aur sochte hain."),            # negation  -> unchanged
    (["full tension", "dimaag fry", "fatti padi hai"], "Dadaji ki tabiyat bahut kharab hai aur mujhe chinta ho rahi hai."),   # serious
]
print("\n================ SANITY ================")
for (sl, ln), o in zip(SANITY, generate_many(SANITY)):
    print(f"{'OK  ' if strict_ok(ln, sl, o) else ('SAME' if norm(o) == norm(ln) else 'ODD ')} | {ln}\n       -> {o}")

# ---- 7. VAL CHECK -----------------------------------------------------------------------------------------
abst, pick = [], []
for r in val_rows:
    sl, ln = parse_user(r["messages"][1]["content"])
    (abst if norm(r["messages"][2]["content"]) == norm(ln) else pick).append((sl, ln))
abst, pick = abst[:150], pick[:250]
a_out, p_out = generate_many(abst), generate_many(pick)
val_abstain = sum(norm(o) == norm(ln) for o, (sl, ln) in zip(a_out, abst)) / max(len(abst), 1)
val_pick = sum(strict_ok(ln, sl, o) for o, (sl, ln) in zip(p_out, pick)) / max(len(pick), 1)
val_gate = (sum(bool(gate_ok(ln, sl, o)) for o, (sl, ln) in zip(p_out, pick)) / max(len(pick), 1)) if GATE else None
val_false_abs = sum(norm(o) == norm(ln) for o, (sl, ln) in zip(p_out, pick)) / max(len(pick), 1)
print(f"\n================ VAL CHECK (strict) ================\nabstain_ok {val_abstain:.2f} | pick_ok {val_pick:.2f} | "
      f"gate_ok {val_gate if val_gate is None else round(val_gate, 2)} | false_abstain {val_false_abs:.2f}")

# ---- 8. held-out evaluation: new vs old adapter vs untouched base ------------------------------------------
REPORT = {"data_hash": data_hash, "audit": AUDIT, "epochs": EPOCHS, "best_val_loss": best_val_loss,
          "val": {"abstain_ok": val_abstain, "pick_ok": val_pick, "gate_ok": val_gate, "false_abstain": val_false_abs}, "eval": {}}


def evaluate(tag):
    outs = generate_many([(r["shortlist"], r["line"]) for r in EVAL])
    with open(f"/kaggle/working/eval_outputs_{tag}.jsonl", "w", encoding="utf-8") as f:
        for r, o in zip(EVAL, outs):
            f.write(json.dumps({"id": r["id"], "output": o}, ensure_ascii=False) + "\n")
    res = {}
    groups = collections.defaultdict(list)
    for r, o in zip(EVAL, outs):
        groups["ALL"].append((r, o))
        groups[r["file"]].append((r, o))
        if r["file"] == "eval_slots" and r.get("expect") == "edit":
            groups["slots:" + r.get("slot", "?")].append((r, o))
    for g, items in groups.items():
        e = [(r, o) for r, o in items if r["expect"] == "edit"]
        a = [(r, o) for r, o in items if r["expect"] == "abstain"]
        res[g] = {"n_edit": len(e), "n_abstain": len(a),
                  "valid_edit": sum(strict_ok(r["line"], r["shortlist"], o) for r, o in e) / max(len(e), 1),
                  "gate_edit": (sum(bool(gate_ok(r["line"], r["shortlist"], o)) for r, o in e) / max(len(e), 1)) if GATE else None,
                  "abstain": sum(norm(o) == norm(r["line"]) for r, o in a) / max(len(a), 1),
                  "bad_edit_on_abstain": sum(norm(o) != norm(r["line"]) for r, o in a) / max(len(a), 1)}
    REPORT["eval"][tag] = res
    print(f"\n--- [{tag}] ---")
    for g in sorted(res):
        x = res[g]
        print(f"{g:22s} edit rows {x['n_edit']:4d} valid {x['valid_edit']:.2f} gate {('%.2f' % x['gate_edit']) if x['gate_edit'] is not None else ' n/a'}"
              f" | abstain rows {x['n_abstain']:4d} kept-unchanged {x['abstain']:.2f}")
    return outs


model.set_adapter("default")
new_outs = evaluate("new")
with model.disable_adapter():
    evaluate("base")
if OLD_ADAPTER and os.path.exists(os.path.join(OLD_ADAPTER, "adapter_config.json")):
    model.load_adapter(OLD_ADAPTER, adapter_name="old")
    model.set_adapter("old")
    evaluate("old")
    model.set_adapter("default")
else:
    print("OLD_ADAPTER not found - the before/after comparison is skipped")

# ---- 9. CHECKLIST ------------------------------------------------------------------------------------------
new_e = REPORT["eval"]["new"]
old_e = REPORT["eval"].get("old")
checks = {
    "audit: <=2% contract violations in train": nb_tr <= 0.02 * (len(train_rows) + nb_tr),
    "no truncated rows": n_trunc == 0,
    "val abstain_ok >= 0.70": val_abstain >= 0.70,
    "val pick_ok >= 0.85": val_pick >= 0.85,
    "val false_abstain <= 0.10": val_false_abs <= 0.10,
    "held-out slots: gate-valid edits >= 0.75": (new_e.get("eval_slots", {}).get("gate_edit") or 0) >= 0.75,
    "held-out slots: abstain kept >= 0.80": new_e.get("eval_slots", {}).get("abstain", 0) >= 0.80,
    "real v5 sentences: abstain kept >= 0.80": new_e.get("eval_realv5", {}).get("abstain", 1) >= 0.80,
}
if old_e:
    checks["no regression on filler rows (vs old, -0.05)"] = (new_e.get("eval_filler", {}).get("valid_edit", 1)
                                                               >= old_e.get("eval_filler", {}).get("valid_edit", 0) - 0.05)
    checks["slots: better than old"] = ((new_e.get("eval_slots", {}).get("gate_edit") or 0)
                                        >= (old_e.get("eval_slots", {}).get("gate_edit") or 0))
REPORT["checklist"] = {k: bool(v) for k, v in checks.items()}
json.dump(REPORT, open(os.path.join(OUT_DIR, "training_report.json"), "w"), indent=1)
print("\n================ CHECKLIST ================")
for k, v in checks.items():
    print(("PASS  " if v else "FAIL  ") + k)
print("RECOMMENDATION:", "ADOPT v5 (all checks pass)" if all(checks.values()) else
      "REVIEW before adopting - paste this checklist and the per-file table to Claude")
shutil.make_archive("/kaggle/working/slang-editor-lora", "zip", OUT_DIR)
print(f"\nDONE in {(time.time() - T0) / 60:.0f} min. Download slang-editor-lora.zip, eval_outputs_new/old/base.jsonl from the Output tab.")
