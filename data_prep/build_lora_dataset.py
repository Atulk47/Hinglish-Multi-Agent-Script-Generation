"""
data_prep/build_lora_dataset.py   (prompt v2: shortlist-pick + ABSTAIN)
Build instruction data for LoRA fine-tuning a small instruction LLM as the slang editor.

Per example the model sees a neutral Hinglish line + a shortlist of 5-8 candidate slangs
and must reply with the line rewritten using the ONE that fits best - OR, if none fits
naturally, with the line UNCHANGED.

Why abstention (the fix for "forced bad slang"): v1 always had the right slang in the
shortlist. At inference the shortlist is chosen by code from sentiment only, so often
nothing fits - and a model that can't abstain forces the least-bad candidate
("Respect, Riy, ...", "kuch koi problem hi nahi, scene set hai"). v2 teaches abstention
with two kinds of examples (target = the line, unchanged):
  (a) OPPOSITE-sentiment shortlists - e.g. only negative slangs offered for a happy line.
      Verifiably wrong, no labelling needed.
  (b) SAME-sentiment shortlists WITHOUT the true slang, kept only if a Groq judge says
      none of the candidates fits the line. (Judge-vetted so we don't teach the model to
      refuse when a generic slang like "mast" would have been fine.) Skipped, with a
      warning, if Groq is unavailable.
Abstain examples are capped at ~15% of the data so the model doesn't collapse to copying.

Output = chat "messages" jsonl for SFT; the system prompt is stored in every example.
KEEP the two V2 prompt strings below identical to agents/agent4_slang_rewriter.py and
notebooks/train_slang_editor_lora_qwen.py - run data_prep/check_prompt_sync.py to verify.

Usage:
  python -m data_prep.build_lora_dataset --src data/felix_dataset/final_dataset.csv
  python -m data_prep.build_lora_dataset --no-vet     # skip the Groq-vetted (b) examples
"""
from __future__ import annotations
import argparse, csv, json, os, random, re, time
from collections import defaultdict

SENT = {
    "ASSERT_POSITIVE":"positive","ASSERT_HAPPY":"happy","ASSERT_EXCITED":"excited",
    "ASSERT_FRUSTRATION":"frustrated","ASSERT_ANGER":"angry","ASSERT_NEGATIVE":"negative",
    "ASSERT_SADNESS":"sad","ASSERT_STRESS":"stressed","ASSERT_ANXIETY":"anxious",
    "ASSERT_WARNING":"warning","ASSERT_FEAR":"fearful","REQUEST_URGENCY":"urgent",
    "CASUAL_ASSERTION":"casual","ASSERT_NEUTRAL":"casual","INFO_STATEMENT":"casual",
}
POS = {"positive", "happy", "excited"}
NEG = {"frustrated", "angry", "negative", "sad", "stressed", "anxious", "fearful", "warning"}
PLAIN_FILLERS = {"yaar","bhai","bro","boss","abe","abey","re","yr","bhaiya","arre"}

SYSTEM_V2 = ("You are a Hinglish (romanized Hindi-English) dialogue editor. Given a sentence and "
             "a few candidate slang expressions, pick the ONE that fits best and insert it "
             "naturally, keeping the meaning EXACTLY the same. If none of the candidates fits "
             "naturally, reply with the sentence UNCHANGED. Reply with ONLY the sentence.")
USER_V2 = ("Pick the slang from this list that best fits the line, and insert it naturally "
           "without changing the meaning. If none of them fits naturally, return the line "
           "UNCHANGED:\nSlangs: {slangs}\nLine: {line}")


def make_example(neutral, shortlist, target):
    user = USER_V2.replace("{slangs}", ", ".join(shortlist)).replace("{line}", neutral)
    return {"messages": [{"role": "system", "content": SYSTEM_V2},
                         {"role": "user", "content": user},
                         {"role": "assistant", "content": target}]}


# ── Groq judge for type-(b) abstain candidates ─────────────────────────────
JUDGE_PROMPT = """You are a strict native Hinglish speaker. For each numbered item you get a casual line and a list of candidate slang expressions.
Decide whether ANY candidate could be inserted into that line so that a real person would naturally say it, with the line's meaning and tone unchanged.
Be strict: a slang that is merely the right mood but unrelated to what the line is about does NOT fit.
Return ONLY a JSON object mapping each item number to true (at least one fits) or false (none fit), e.g. {{"1": false, "2": true}}.

{block}"""


def judge_none_fit(cands, batch=20, model="openai/gpt-oss-20b"):
    """cands: list of (neutral, shortlist). Returns list[bool|None]: True = judge says
    NONE fit (valid abstain example); False = something fits; None = judge unavailable."""
    from dotenv import load_dotenv
    load_dotenv()
    from groq import Groq
    import config
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    out = [None] * len(cands)
    fails = 0
    for s in range(0, len(cands), batch):
        chunk = cands[s:s + batch]
        block = "\n".join(f'{i+1}. Line: "{n}" | Candidates: {", ".join(sl)}'
                          for i, (n, sl) in enumerate(chunk))
        res, last_err = None, ""
        for _ in range(3):
            try:
                r = client.chat.completions.create(
                    model=model, temperature=0.0, max_tokens=1500,
                    reasoning_effort=getattr(config, "GROQ_REASONING_EFFORT", "low"),
                    messages=[{"role": "user", "content": JUDGE_PROMPT.format(block=block)}])
                txt = r.choices[0].message.content or ""
                m = re.search(r"\{.*\}", txt, re.DOTALL)
                if m:
                    res = json.loads(m.group()); break
                last_err = f"no JSON in reply (finish={r.choices[0].finish_reason}, {txt[:60]!r})"
            except Exception as e:
                last_err = str(e)[:160]
                time.sleep(6 if "429" in last_err or "rate" in last_err.lower() else 3)
        if res is None:
            print(f"  judge batch @{s} failed: {last_err}")
            fails += 1
            if fails >= 4:
                print("  judge: 4 consecutive failed batches (rate limit?) - stopping vetting early")
                break
            continue
        fails = 0
        for i in range(len(chunk)):
            v = res.get(str(i + 1))
            if isinstance(v, bool):
                out[s + i] = (v is False)
        if (s // batch) % 10 == 0:
            print(f"  judged {min(s+batch, len(cands))}/{len(cands)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/felix_dataset/final_dataset.csv")
    ap.add_argument("--outdir", default="data/felix_dataset")
    ap.add_argument("--val-frac", type=float, default=0.06)
    ap.add_argument("--kmin", type=int, default=5)
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--abstain-frac", type=float, default=0.15)
    ap.add_argument("--no-vet", action="store_true", help="skip Groq-vetted type-(b) abstain examples")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    raw, by_sent, all_col = [], defaultdict(set), set()
    with open(args.src, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            neutral = (r.get("neutral") or "").strip()
            output = (r.get("output") or "").strip()
            slang = (r.get("slangs") or r.get("slang") or "").strip()
            ctrl = (r.get("control") or "CASUAL_ASSERTION").strip().upper()
            if not (neutral and output and slang) or neutral.lower() == output.lower():
                continue
            if slang.lower() not in output.lower():
                continue
            sent = SENT.get(ctrl, "casual")
            raw.append((neutral, output, slang, sent))
            if slang.lower() not in PLAIN_FILLERS:
                by_sent[sent].add(slang); all_col.add(slang)
    fillers = sorted({s for _, _, s, _ in raw if s.lower() in PLAIN_FILLERS})
    pos_pool = sorted(set().union(*[by_sent[s] for s in POS]))
    neg_pool = sorted(set().union(*[by_sent[s] for s in NEG]))

    # ── positives: variable-size shortlist containing the true slang ──
    examples, n_pos = [], 0
    for neutral, output, slang, sent in raw:
        k = random.randint(args.kmin, args.kmax)
        pool = sorted(by_sent.get(sent, set()) - {slang}) or sorted(all_col - {slang})
        random.shuffle(pool)
        d = pool[:k - 1]
        if fillers and random.random() < 0.25:          # inference shortlists can contain fillers
            d[-1:] = [random.choice(fillers)]
        sl = d + [slang]
        random.shuffle(sl)
        examples.append(make_example(neutral, sl, output)); n_pos += 1

    # ── abstain budget: ~abstain_frac of the FINAL set ──
    budget = int(args.abstain_frac / (1 - args.abstain_frac) * n_pos)

    # (a) opposite-polarity shortlists (verifiably no fit)
    a_rows = [r for r in raw if r[3] in POS or r[3] in NEG]
    random.shuffle(a_rows)
    n_a_target = int(budget * 0.45)
    n_a = 0
    for neutral, output, slang, sent in a_rows:
        if n_a >= n_a_target:
            break
        opp = neg_pool if sent in POS else pos_pool
        k = random.randint(args.kmin, args.kmax)
        sl = random.sample(opp, min(k, len(opp)))
        if any(s.lower() in neutral.lower() for s in sl):
            continue
        examples.append(make_example(neutral, sl, neutral)); n_a += 1

    # (b) same-sentiment shortlists WITHOUT the true slang, kept only if the judge says none fit
    n_b = 0
    if not args.no_vet:
        b_cands, b_meta = [], []
        pool_rows = raw[:]
        random.shuffle(pool_rows)
        for neutral, output, slang, sent in pool_rows[: int(budget * 2.2)]:
            pool = sorted(by_sent.get(sent, set()) - {slang})
            if len(pool) < args.kmin:
                continue
            k = random.randint(args.kmin, args.kmax)
            sl = random.sample(pool, min(k, len(pool)))
            if any(s.lower() in neutral.lower() for s in sl):
                continue
            b_cands.append((neutral, sl)); b_meta.append(neutral)
        print(f"judging {len(b_cands)} same-sentiment abstain candidates with Groq ...")
        try:
            verdict = judge_none_fit(b_cands)
        except Exception as e:
            print(f"  judge unavailable ({e}); skipping type-(b) examples")
            verdict = [None] * len(b_cands)
        for (neutral, sl), v in zip(b_cands, verdict):
            if v is True and n_b < budget - n_a:
                examples.append(make_example(neutral, sl, neutral)); n_b += 1
        judged = sum(v is not None for v in verdict)
        print(f"  judge answered {judged}/{len(b_cands)}; kept {n_b} 'none fit' examples")
        if judged == 0:
            print("  WARNING: no type-(b) examples (Groq unavailable). Dataset still has type-(a) "
                  "abstain examples; re-run later for the full set.")

    random.shuffle(examples)
    n_val = int(len(examples) * args.val_frac)
    val, train = examples[:n_val], examples[n_val:]
    os.makedirs(args.outdir, exist_ok=True)
    for name, split in [("lora_train.jsonl", train), ("lora_val.jsonl", val)]:
        with open(os.path.join(args.outdir, name), "w", encoding="utf-8") as f:
            for ex in split:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    total = len(examples); n_abs = n_a + n_b
    print(f"\ntotal {total} | train {len(train)} | val {len(val)}")
    print(f"  pick examples   : {n_pos} (shortlist {args.kmin}-{args.kmax}, true slang inside)")
    print(f"  abstain (a)     : {n_a}  opposite-sentiment shortlist")
    print(f"  abstain (b)     : {n_b}  same-sentiment, judge-vetted none-fit")
    print(f"  abstain share   : {100*n_abs/total:.1f}%  (target ~{100*args.abstain_frac:.0f}%)")
    print(f"wrote {args.outdir}/lora_train.jsonl and lora_val.jsonl")


if __name__ == "__main__":
    main()
