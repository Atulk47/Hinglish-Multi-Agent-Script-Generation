"""
data_prep/vet_rows.py
Quality gate: rate each row's slang naturalness 1-5 with a (free) Groq judge and
keep only rows scoring >= --min-score. Used on the newly-generated rows before
they enter the training set. Batches 25 rows per call; checkpoints + resumes.

Usage:
  python -m data_prep.vet_rows --in data/felix_dataset/gen_claude.csv --out data/felix_dataset/gen_claude_vetted.csv --min-score 4
"""
from __future__ import annotations
import argparse, csv, json, os, re, time
from dotenv import load_dotenv
import config
load_dotenv()
from groq import Groq

MODEL = os.getenv("VET_MODEL", "openai/gpt-oss-20b")

PROMPT = """You are a strict native Hinglish speaker judging slang insertions for a training set.
For each numbered line rate 1-5 how NATURAL the slang sounds in that exact sentence:
 5 = a real person would say exactly this; 4 = natural; 3 = a bit off; 2 = forced/awkward; 1 = wrong/nonsensical.
Also judge tone-match and that it's genuine Hinglish (not pure English).
Return ONLY a JSON object mapping each number to its score, e.g. {{"1":5,"2":3,...}}.

Lines:
{block}"""


def parse_scores(txt):
    m = re.search(r"\{.*\}", txt or "", re.DOTALL)
    if not m: return {}
    try: d = json.loads(m.group())
    except Exception: return {}
    out = {}
    for k, v in d.items():
        try: out[int(k)] = float(v)
        except Exception: pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-score", type=int, default=4)
    ap.add_argument("--batch", type=int, default=25)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.inp, encoding="utf-8-sig")))
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    print(f"VET {len(rows)} rows from {args.inp} (keep score>={args.min_score}), judge={MODEL}")

    cols = ["neutral", "output", "slangs", "control", "source"]
    kept = []; done = 0
    # resume
    if os.path.exists(args.out):
        kept = list(csv.DictReader(open(args.out, encoding="utf-8-sig")))
        done = 0  # we re-scan from start but skip already-kept by output
    kept_out = {(r.get("output") or "").strip().lower() for r in kept}

    t0 = time.time(); rejected = 0
    for start in range(0, len(rows), args.batch):
        chunk = rows[start:start + args.batch]
        # skip if all already processed (rough: if their outputs are in kept)
        block = "\n".join(f'{i+1}. "{r.get("output","")}"  [slang: {r.get("slangs","")}]'
                          for i, r in enumerate(chunk))
        try:
            resp = client.chat.completions.create(model=MODEL, temperature=0.0, max_tokens=400,
                reasoning_effort=getattr(config, "GROQ_REASONING_EFFORT", "low"),
                messages=[{"role": "user", "content": PROMPT.format(block=block)}])
            sc = parse_scores(resp.choices[0].message.content)
        except Exception as e:
            print(f"  batch @{start} failed: {str(e)[:70]} - keeping batch unrated"); sc = {}
        for i, r in enumerate(chunk):
            out_l = (r.get("output") or "").strip().lower()
            if out_l in kept_out:
                continue
            score = sc.get(i + 1, 4 if not sc else 0)  # if the whole batch failed, keep; else need a real >=min
            if score >= args.min_score:
                kept.append({c: r.get(c, "") for c in cols}); kept_out.add(out_l)
            else:
                rejected += 1
        if start % (args.batch * 10) == 0:
            print(f"  rated {min(start+args.batch,len(rows))}/{len(rows)} | kept {len(kept)} | {time.time()-t0:.0f}s", flush=True)
            _flush(args.out, kept, cols)
    _flush(args.out, kept, cols)
    print(f"\nVET done: kept {len(kept)}/{len(rows)} (rejected {rejected}). saved {args.out}")


def _flush(out, kept, cols):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(kept)
    os.replace(tmp, out)


if __name__ == "__main__":
    main()
