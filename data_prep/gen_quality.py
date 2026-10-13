"""
data_prep/gen_quality.py
Quality-first slang-pair generator.

Approach (path B): use a hand-authored GOLD seed (data/felix_dataset/claude_generated.csv)
as few-shot examples to steer a strong LLM, and gate EVERY candidate through a strict
naturalness judge (keep only score >= 4). The slang vocabulary + its sentiment come from
the gold seed (correctly labelled), so no forced rare/mislabelled slang. NOVEL-style
generation (model writes a fresh neutral too) also grows neutral-sentence diversity.

Usage:
  python -m data_prep.gen_quality --n 20                      # small test
  python -m data_prep.gen_quality --n 1800 --out data/felix_dataset/gen_quality.csv
"""
from __future__ import annotations
import argparse, csv, itertools, json, os, random, re, time
from collections import defaultdict
from dotenv import load_dotenv
import config

load_dotenv()
from groq import Groq

try:
    from rapidfuzz import fuzz
    def sim(a, b): return fuzz.token_sort_ratio(a, b)
except Exception:
    import difflib
    def sim(a, b): return 100 * difflib.SequenceMatcher(None, a, b).ratio()

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
GEN_MODEL   = os.getenv("GEN_MODEL",   "openai/gpt-oss-120b")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-oss-20b")
GOLD = "data/felix_dataset/claude_generated.csv"

CONTEXTS = ["college","cricket match","hostel","office work","family dinner","traffic",
    "exam results","street food","online gaming","festival shopping","gym","job interview",
    "road trip","wedding","power cut","phone repair","coffee shop","group project",
    "missed train","food delivery","new phone","movie night","internship","placements",
    "chai break","bike ride","wifi down","birthday party","old friends","cooking",
    "late night study","salary day","monsoon","relationship","startup pitch","first job"]

CONTROL_SENTIMENT = {
    "ASSERT_POSITIVE":"positive", "ASSERT_HAPPY":"happy", "ASSERT_EXCITED":"excited",
    "ASSERT_FRUSTRATION":"frustrated", "ASSERT_ANGER":"angry",
    "ASSERT_NEGATIVE":"negative / disappointed", "ASSERT_SADNESS":"sad",
    "ASSERT_STRESS":"stressed", "ASSERT_ANXIETY":"anxious",
    "REQUEST_URGENCY":"urgent", "CASUAL_ASSERTION":"casual",
}

GEN_PROMPT = """You write natural, casual Hinglish (romanized Hindi-English mix) the way young Indians actually speak.
Write ONE short casual sentence, then the SAME sentence rewritten to naturally include the slang "{slang}" (a {sentiment} expression), WITHOUT changing the meaning.
The slang must fit naturally and sound like a real person said it - never forced or tacked on at the end.

Good examples (match this style):
{examples}

Now write a NEW, different pair using the slang "{slang}" (you can loosely set it around "{context}" if it fits naturally - don't force it):
Return ONLY JSON: {{"neutral": "<plain sentence, NO slang>", "output": "<same sentence WITH '{slang}' woven in naturally>"}}
Rules: romanized Roman script only (no Devanagari); 5-16 words; meaning identical except the added slang; must sound genuinely natural."""

JUDGE_PROMPT = """You are a STRICT native Hinglish speaker judging a slang insertion for a training dataset.
neutral: {neutral}
slang output: {output}
inserted slang: {slang}
Rate how NATURAL the slang sounds in this exact sentence, 1-5:
 5 = a real person would say exactly this; 4 = natural; 3 = a bit off; 2 = forced/awkward; 1 = wrong or nonsensical.
Return ONLY JSON:
{{"score": <1-5 int>, "is_hinglish": <true/false, a real Hindi-English mix not pure English>, "meaning_preserved": <true/false>, "fits": <true/false, slang fits the tone AND grammar>}}"""


def parse_json(txt):
    raw = re.sub(r"```(?:json)?", "", txt or "").strip()
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m: return None
    try: return json.loads(m.group())
    except Exception:
        try: return json.loads(re.sub(r",\s*}", "}", m.group()))
        except Exception: return None


def call(client, prompt, model, max_tokens=220):
    for _ in range(4):
        try:
            r = client.chat.completions.create(
                model=model, messages=[{"role":"user","content":prompt}],
                temperature=0.8 if model==GEN_MODEL else 0.0, max_tokens=max_tokens,
                reasoning_effort=getattr(config,"GROQ_REASONING_EFFORT","low"))
            return parse_json(r.choices[0].message.content)
        except Exception as e:
            msg=str(e).lower()
            if "429" in msg or "rate" in msg or "quota" in msg:
                time.sleep(6); continue
            time.sleep(2)
    return None


def load_gold():
    rows=list(csv.DictReader(open(GOLD,encoding="utf-8-sig")))
    by_ctrl=defaultdict(list); vocab={}
    for r in rows:
        c=(r.get("control") or "CASUAL_ASSERTION").strip().upper()
        s=(r.get("slangs") or "").strip()
        by_ctrl[c].append(r)
        if s: vocab[s]=c
    return rows, by_ctrl, vocab


def few_shot(by_ctrl, ctrl, k=6):
    pool=list(by_ctrl.get(ctrl, []))
    random.shuffle(pool)
    picks=pool[:k-2]
    others=[r for c,rs in by_ctrl.items() if c!=ctrl for r in rs]
    random.shuffle(others); picks+=others[:k-len(picks)]
    random.shuffle(picks)
    return "\n---\n".join(f"neutral: {r['neutral']}\noutput: {r['output']}" for r in picks)


def structural_ok(neutral, output, slang, seen, min_fuzz=62):
    if not (neutral and output and slang): return "empty"
    if DEVANAGARI.search(neutral) or DEVANAGARI.search(output): return "devanagari"
    if slang.lower() not in output.lower(): return "slang_absent"
    if neutral.strip().lower()==output.strip().lower(): return "no_change"
    if not (4 <= len(output.split()) <= 22): return "length"
    key=re.sub(r"[^a-z0-9 ]","",output.lower()).strip()
    if key in seen: return "duplicate"
    stripped=re.sub(r"\s+"," ",re.sub(re.escape(slang),"",output,flags=re.I)).strip(" ,.!")
    if sim(neutral.lower(), stripped.lower()) < min_fuzz: return "drift"
    return "ok"


def flush(out, kept):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cols=["neutral","output","slangs","control","source"]
    tmp=out+".tmp"
    with open(tmp,"w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); w.writerows(kept)
    os.replace(tmp,out)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--n",type=int,default=20)
    ap.add_argument("--out",default="data/felix_dataset/gen_quality.csv")
    ap.add_argument("--min-score",type=int,default=4)
    ap.add_argument("--cap",type=int,default=0)       # per-slang cap (0=auto)
    ap.add_argument("--seed",type=int,default=11)
    args=ap.parse_args()
    random.seed(args.seed)

    gold, by_ctrl, vocab = load_gold()
    slangs=list(vocab.items())                        # [(slang, control)]
    cap=args.cap or max(6, args.n//max(len(slangs),1)+4)
    client=Groq(api_key=os.environ["GROQ_API_KEY"])
    print(f"GEN={GEN_MODEL} JUDGE={JUDGE_MODEL} | gold={len(gold)} seed rows, {len(slangs)} slangs | target={args.n}, per-slang cap={cap}")

    kept=[]; seen=set(); used=defaultdict(int); reasons=defaultdict(int)
    # resume
    if os.path.exists(args.out):
        for r in csv.DictReader(open(args.out,encoding="utf-8-sig")):
            kept.append(r); seen.add(re.sub(r"[^a-z0-9 ]","",(r.get("output") or "").lower()).strip())
            used[(r.get("slangs") or "").strip()]+=1
        if kept: print(f"RESUME: {len(kept)} existing rows")
    # also dedup against gold outputs
    for r in gold: seen.add(re.sub(r"[^a-z0-9 ]","",r["output"].lower()).strip())

    cyc=itertools.cycle(sorted(slangs, key=lambda x:x[0].lower()))
    attempts=0; t0=time.time(); max_attempts=args.n*8; consec_fail=0
    while len(kept)<args.n and attempts<max_attempts:
        if consec_fail>=45:   # sustained failures = likely daily token cap; stop cleanly, resume later
            print(f"  [stop] {consec_fail} consecutive failures — likely hit the daily cap. "
                  f"Saved {len(kept)} rows; rerun the same command later to resume.", flush=True)
            break
        attempts+=1
        slang,ctrl=next(cyc)
        if used[slang]>=cap:   # skip over-used slang
            # find a non-capped one
            for _ in range(len(slangs)):
                slang,ctrl=next(cyc)
                if used[slang]<cap: break
            else: break
        sent=CONTROL_SENTIMENT.get(ctrl,"casual")
        obj=call(client, GEN_PROMPT.format(slang=slang,sentiment=sent,context=random.choice(CONTEXTS),
                 examples=few_shot(by_ctrl,ctrl)), GEN_MODEL)
        if not obj: reasons["unparseable"]+=1; consec_fail+=1; continue
        neutral=str(obj.get("neutral","")).strip(); output=str(obj.get("output","")).strip()
        st=structural_ok(neutral,output,slang,seen)
        if st!="ok": reasons[st]+=1; consec_fail+=1; continue
        j=call(client, JUDGE_PROMPT.format(neutral=neutral,output=output,slang=slang), JUDGE_MODEL, 120)
        if not j or not all(bool(j.get(k)) for k in ("is_hinglish","meaning_preserved","fits")) \
           or float(j.get("score",0))<args.min_score:
            reasons["judge"]+=1; consec_fail+=1; continue
        consec_fail=0
        seen.add(re.sub(r"[^a-z0-9 ]","",output.lower()).strip()); used[slang]+=1; reasons["ok"]+=1
        kept.append({"neutral":neutral,"output":output,"slangs":slang,"control":ctrl,"source":"gen_q"})
        if len(kept)%10==0:
            print(f"  kept {len(kept)}/{args.n} | attempts {attempts} | accept {100*len(kept)/attempts:.0f}% "
                  f"| distinct slang {len({k['slangs'] for k in kept})}", flush=True)
            flush(args.out,kept)
    flush(args.out,kept)
    print(f"\nDONE kept {len(kept)} in {time.time()-t0:.0f}s ({attempts} attempts, "
          f"{100*len(kept)/max(attempts,1):.0f}% accept). reasons={dict(reasons)}")
    print("saved:",args.out)


if __name__=="__main__":
    main()
