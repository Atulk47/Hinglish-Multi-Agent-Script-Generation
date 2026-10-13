"""
data_prep/rank_augmented.py
Rate every augmented slang row for how NATURAL the inserted slang sounds in the
sentence (1-5 via an LLM judge), then write a showcase CSV sorted best-first so a
screenshot of the top rows shows only the strongest examples.

Usage:
  python -m data_prep.rank_augmented
  python -m data_prep.rank_augmented --top 150 --batch 20
"""
from __future__ import annotations
import argparse, csv, json, os, re, time

from dotenv import load_dotenv
import config

load_dotenv()
from groq import Groq

SRC = "data/felix_dataset/augmented_cerebras.csv"
OUT = "data/felix_dataset/augmented_showcase_ranked.csv"
COLS = ["neutral", "output", "slangs", "control", "source"]

RATE_PROMPT = """You are a native Hinglish speaker judging a slang dataset.
For each numbered line, rate 1-5 how NATURAL and FITTING the slang sounds in that
exact sentence:
  5 = sounds perfect, a real person would say it this way
  4 = good, natural
  3 = acceptable but a little off
  2 = awkward / forced placement
  1 = wrong tone or nonsensical
Judge naturalness + tone match (e.g. harsh slang on a happy line = low).
Return ONLY a JSON object mapping each number to its score, e.g. {{"1":5,"2":2,...}}.

Lines:
{block}"""


def parse_scores(txt, n):
    raw = re.sub(r"```(?:json)?", "", txt or "").strip()
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return {}
    try:
        d = json.loads(m.group())
    except Exception:
        return {}
    out = {}
    for k, v in d.items():
        try:
            out[int(k)] = float(v)
        except Exception:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=150)
    ap.add_argument("--batch", type=int, default=20)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    print(f"Rating {len(rows)} rows in batches of {args.batch}...")

    scores = [3.0] * len(rows)   # default neutral if a batch fails
    t0 = time.time()
    for start in range(0, len(rows), args.batch):
        chunk = rows[start:start + args.batch]
        block = "\n".join(
            f'{i+1}. "{r.get("output","")}"  [slang: {r.get("slangs","")}]'
            for i, r in enumerate(chunk))
        try:
            resp = client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=[{"role": "user", "content": RATE_PROMPT.format(block=block)}],
                temperature=0.0, max_tokens=400,
                reasoning_effort=getattr(config, "GROQ_REASONING_EFFORT", "low"))
            sc = parse_scores(resp.choices[0].message.content, len(chunk))
        except Exception as e:
            print(f"  batch @{start} failed: {str(e)[:80]}")
            sc = {}
        for i in range(len(chunk)):
            if (i + 1) in sc:
                scores[start + i] = sc[i + 1]
        if start % (args.batch * 10) == 0:
            print(f"  rated {min(start+args.batch,len(rows))}/{len(rows)} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    order = sorted(range(len(rows)), key=lambda i: scores[i], reverse=True)
    ranked = [rows[i] for i in order]

    # Save raw scores first (cheap insurance: if the CSV target is locked in
    # Excel, we still keep the expensive LLM ratings and can rebuild instantly).
    json.dump({"scores": scores}, open(SRC + ".scores.json", "w"))

    def _write(path):
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS); w.writeheader()
            for r in ranked:
                w.writerow({c: r.get(c, "") for c in COLS})

    global OUT
    try:
        _write(OUT)
    except PermissionError:                      # target open in Excel -> fallback name
        OUT = OUT.replace(".csv", f"_{int(time.time())}.csv")
        _write(OUT)
        print(f"  (original was locked; wrote to {OUT} instead)")

    top = [scores[i] for i in order[:args.top]]
    print(f"\nWrote {OUT} (all {len(rows)} rows, best-first).")
    print(f"Top {args.top}: avg score {sum(top)/len(top):.2f}, "
          f"min {min(top):.0f}, count>=4.5 = {sum(1 for s in scores if s>=4.5)}")


if __name__ == "__main__":
    main()
