"""
data_prep/gen_claude.py
High-quality slang-pair generator using Claude (Anthropic API).

Batch generation: one API call produces ~25 natural neutral->slang pairs, each
targeting a specific slang (with its correct sentiment) from the hand-authored
gold vocabulary, steered by gold few-shot examples. Claude writes far more natural
Hinglish than gemma/gpt-oss, so quality is high and cost is tiny (Haiku 5.5).

Needs ANTHROPIC_API_KEY in .env.

Usage:
  python -m data_prep.gen_claude --n 100 --out data/felix_dataset/gen_claude.csv   # test
  python -m data_prep.gen_claude --n 2700 --out data/felix_dataset/gen_claude.csv
"""
from __future__ import annotations
import argparse, csv, json, os, random, re, time
from collections import defaultdict
from dotenv import load_dotenv
import anthropic

load_dotenv()

try:
    from rapidfuzz import fuzz
    def sim(a, b): return fuzz.token_sort_ratio(a, b)
except Exception:
    import difflib
    def sim(a, b): return 100 * difflib.SequenceMatcher(None, a, b).ratio()

MODEL = os.getenv("CLAUDE_GEN_MODEL", "claude-sonnet-5-5")
# ($/1M input, $/1M output)
PRICING = {"claude-haiku-5-5":(0.10,0.50), "claude-sonnet-5-5":(2.0,10.0),
           "claude-opus-5-5":(4.0,20.0)}
PIN, POUT = PRICING.get(MODEL, (2.0,10.0))
GOLD  = "data/felix_dataset/claude_generated.csv"
DEVANAGARI = re.compile(r"[ऀ-ॿ]")

CONTROL_SENTIMENT = {
    "ASSERT_POSITIVE":"positive","ASSERT_HAPPY":"happy","ASSERT_EXCITED":"excited",
    "ASSERT_FRUSTRATION":"frustrated","ASSERT_ANGER":"angry",
    "ASSERT_NEGATIVE":"negative/disappointed","ASSERT_SADNESS":"sad",
    "ASSERT_STRESS":"stressed","ASSERT_ANXIETY":"anxious",
    "REQUEST_URGENCY":"urgent","CASUAL_ASSERTION":"casual",
}
CONTEXTS = ["college","cricket","hostel","office","family dinner","traffic","exam results",
    "street food","gaming","festival shopping","gym","job interview","road trip","wedding",
    "power cut","phone repair","coffee shop","group project","missed train","food delivery",
    "new phone","movie night","internship","placements","chai break","bike ride","wifi down",
    "birthday","old friends","cooking","late-night study","salary day","monsoon","first job"]

SYSTEM = ("You write natural, casual Hinglish (romanized Hindi-English code-mix) exactly the "
          "way young urban Indians actually speak and text. You are creating high-quality "
          "training data for a slang-insertion model. Every line must sound like a real person "
          "- never forced, never awkward, never literal-translated.")

def key(s): return re.sub(r"[^a-z0-9 ]","",(s or "").lower()).strip()

def load_gold():
    rows=list(csv.DictReader(open(GOLD,encoding="utf-8-sig")))
    by_ctrl=defaultdict(list); vocab={}
    for r in rows:
        c=(r.get("control") or "CASUAL_ASSERTION").strip().upper()
        s=(r.get("slangs") or "").strip()
        by_ctrl[c].append(r)
        if s: vocab[s]=c
    return rows, by_ctrl, vocab

def few_shot_block(by_ctrl, targets, k=8):
    # prefer examples whose control is among the batch's targets
    want={c for _,c in targets}
    pool=[r for c in want for r in by_ctrl.get(c,[])]
    random.shuffle(pool); picks=pool[:k]
    if len(picks)<k:
        extra=[r for rs in by_ctrl.values() for r in rs]; random.shuffle(extra)
        picks+=extra[:k-len(picks)]
    random.shuffle(picks)
    return "\n".join(f'- neutral: "{r["neutral"]}"  ->  output: "{r["output"]}"' for r in picks)

def build_prompt(targets, by_ctrl):
    ex=few_shot_block(by_ctrl, targets)
    lines=[f'{i+1}. "{s}"  ({CONTROL_SENTIMENT.get(c,"casual")})'
           for i,(s,c) in enumerate(targets)]
    ctxs=random.sample(CONTEXTS, min(len(targets), len(CONTEXTS)))
    return f"""Examples of the EXACT style and quality required:
{ex}

Now create {len(targets)} NEW, diverse pairs - ONE per target slang below. For each:
- "neutral": a plain, natural Hinglish sentence (NO slang) on a fresh everyday topic
- "output": the SAME sentence rewritten to weave in that slang naturally (meaning unchanged)

Target slangs (with the sentiment they must match):
{chr(10).join(lines)}

Rules:
- romanized Roman script only (NO Devanagari); each sentence 5-16 words
- the slang must fit the given sentiment and sound genuinely natural (not tacked on at the end)
- VARY the topics widely (e.g. {", ".join(ctxs[:8])}, etc.) - no two pairs on the same topic
- meaning identical between neutral and output except the added slang
- use the EXACT slang text and control token given

Return ONLY a JSON array of {len(targets)} objects, in order:
[{{"neutral":"...","output":"...","slang":"<exact slang>","control":"<exact control>"}}]"""

def parse_array(txt):
    m=re.search(r"\[.*\]", txt or "", re.DOTALL)
    if not m: return []
    try: return json.loads(m.group())
    except Exception:
        try: return json.loads(re.sub(r",\s*([\]}])", r"\1", m.group()))
        except Exception: return []

def structural_ok(neutral, output, slang, seen):
    if not (neutral and output and slang): return False
    if DEVANAGARI.search(neutral) or DEVANAGARI.search(output): return False
    if slang.lower() not in output.lower(): return False
    if neutral.strip().lower()==output.strip().lower(): return False
    if not (4 <= len(output.split()) <= 22): return False
    if key(output) in seen: return False
    stripped=re.sub(r"\s+"," ",re.sub(re.escape(slang),"",output,flags=re.I)).strip(" ,.!")
    if sim(neutral.lower(), stripped.lower()) < 55: return False
    return True

def flush(out, kept):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cols=["neutral","output","slangs","control","source"]; tmp=out+".tmp"
    with open(tmp,"w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); w.writerows(kept)
    os.replace(tmp,out)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--n",type=int,default=100)
    ap.add_argument("--out",default="data/felix_dataset/gen_claude.csv")
    ap.add_argument("--batch",type=int,default=25)
    ap.add_argument("--cap",type=int,default=0)
    args=ap.parse_args()
    random.seed(7)

    gold, by_ctrl, vocab = load_gold()
    slangs=list(vocab.items())
    cap=args.cap or max(10, args.n//max(len(slangs),1)+6)
    client=anthropic.Anthropic()   # reads ANTHROPIC_API_KEY
    print(f"MODEL={MODEL} | gold={len(gold)} seed, {len(slangs)} slangs | target={args.n}, batch={args.batch}, cap={cap}")

    kept=[]; seen=set(); used=defaultdict(int)
    for r in gold: seen.add(key(r["output"]))
    # dedup against salvaged good rows too
    # dedup against salvaged + all sibling generated files so a 2nd model makes net-new rows
    for other in ["data/felix_dataset/salvaged_good_rows.csv",
                  "data/felix_dataset/gen_claude.csv",
                  "data/felix_dataset/gen_quality.csv"]:
        if other != args.out and os.path.exists(other):
            for r in csv.DictReader(open(other, encoding="utf-8-sig")):
                seen.add(key(r.get("output","")))
    if os.path.exists(args.out):
        for r in csv.DictReader(open(args.out,encoding="utf-8-sig")):
            kept.append(r); seen.add(key(r.get("output",""))); used[(r.get("slangs") or "").strip()]+=1
        if kept: print(f"RESUME: {len(kept)} existing rows")

    t0=time.time(); calls=0; in_tok=out_tok=0
    while len(kept)<args.n:
        # pick a batch of under-used slangs
        avail=[(s,c) for s,c in slangs if used[s]<cap]
        if not avail: avail=slangs
        random.shuffle(avail)
        targets=avail[:min(args.batch, args.n-len(kept))]
        # Disable/minimize thinking - this is a direct generation task, thinking
        # just inflates output tokens (cost). Param differs per model.
        think = ({"type":"between_tools"} if "sonnet" in MODEL
                 else {"type":"disabled"} if "haiku" in MODEL else None)
        kw=dict(model=MODEL, max_tokens=4000, output_config={"effort":"low"},
                system=SYSTEM, messages=[{"role":"user","content":build_prompt(targets,by_ctrl)}])
        if think: kw["thinking"]=think
        try:
            resp=client.messages.create(**kw)
            calls+=1; in_tok+=resp.usage.input_tokens; out_tok+=resp.usage.output_tokens
            txt="".join(b.text for b in resp.content if b.type=="text")
        except anthropic.APIStatusError as e:
            print(f"  API error {getattr(e,'status_code','?')}: {str(e)[:100]}"); time.sleep(3); continue
        except Exception as e:
            print(f"  error: {str(e)[:100]}"); time.sleep(3); continue
        got=0
        for o in parse_array(txt):
            n=str(o.get("neutral","")).strip(); out=str(o.get("output","")).strip()
            sl=str(o.get("slang","")).strip() or str(o.get("slangs","")).strip()
            ct=str(o.get("control","")).strip().upper() or vocab.get(sl,"CASUAL_ASSERTION")
            if not sl: continue
            if structural_ok(n,out,sl,seen):
                seen.add(key(out)); used[sl]+=1; got+=1
                kept.append({"neutral":n,"output":out,"slangs":sl,"control":ct,"source":"claude_gen2"})
        est=in_tok/1e6*PIN + out_tok/1e6*POUT
        print(f"  kept {len(kept)}/{args.n} (+{got}/{len(targets)} this batch) | calls {calls} "
              f"| ~${est:.3f} | {time.time()-t0:.0f}s", flush=True)
        flush(args.out, kept)
    est=in_tok/1e6*PIN + out_tok/1e6*POUT
    print(f"\nDONE kept {len(kept)} in {calls} calls, {time.time()-t0:.0f}s. "
          f"tokens in={in_tok} out={out_tok} | est cost ~${est:.3f}")
    print("saved:", args.out)

if __name__=="__main__":
    main()
