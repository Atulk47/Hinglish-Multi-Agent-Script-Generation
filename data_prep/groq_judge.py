"""
data_prep/groq_judge.py
FREE fallback judge (Groq) with the SAME strict rubric as the Sonnet judge in label_v3.JUDGE_PROMPT.
Because the Anthropic credit can run out, this module is first CALIBRATED against the Sonnet verdicts
already collected (agreement report), and only then used to grade new candidates.

  python -m data_prep.groq_judge --calibrate 240                 # agreement vs Sonnet on stage-1 verdicts
  python -m data_prep.groq_judge --grade labeled2.jsonl --out judged2.jsonl
"""
from __future__ import annotations
import argparse, json, os, random, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from data_prep.v3_common import parse_json, read_jsonl, write_jsonl
from data_prep.label_v3 import SYSTEM, JUDGE_PROMPT

load_dotenv()
D = "data/felix_dataset/v3/"
MODEL = os.getenv("GROQ_JUDGE_MODEL", "openai/gpt-oss-120b")


def _client():
    from groq import Groq
    return Groq(api_key=os.environ["GROQ_API_KEY"])


def judge_batch(client, chunk):
    """chunk: list of (clause, edit, slang) -> list of {score, meaning} or None per item."""
    items = "\n".join(f'{i+1}. ORIGINAL: "{c}"\n   EDITED: "{e}"\n   SLANG: {s}'
                      for i, (c, e, s) in enumerate(chunk))
    prompt = SYSTEM + "\n\n" + JUDGE_PROMPT.format(items=items)
    for attempt in range(4):
        try:
            r = client.chat.completions.create(model=MODEL, temperature=0.0, max_tokens=2500,
                                               reasoning_effort="low",
                                               messages=[{"role": "user", "content": prompt}])
            arr = parse_json(r.choices[0].message.content or "")
            out = [None] * len(chunk)
            for a in arr or []:
                if isinstance(a, dict) and "i" in a and 1 <= int(a["i"]) <= len(chunk):
                    out[int(a["i"]) - 1] = {"score": float(a.get("score", 0)), "meaning": bool(a.get("meaning", False))}
            return out
        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate" in msg.lower():
                time.sleep(8 + 6 * attempt)
            else:
                time.sleep(2)
    return [None] * len(chunk)


def judge_many(items, batch=20, workers=3):
    client = _client()
    res = [None] * len(items)
    chunks = [(s, items[s:s + batch]) for s in range(0, len(items), batch)]
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(judge_batch, client, ch): s for s, ch in chunks}
        for fu in as_completed(futs):
            s = futs[fu]
            for k, v in enumerate(fu.result()):
                res[s + k] = v
    return res


def calibrate(n):
    rows = read_jsonl(D + "judged.jsonl")
    cands = []
    for r in rows:
        if r["label"] != "EDIT":
            continue
        if r.get("judge"):
            cands.append((r["clause"], r["edit"], r["slang"], r["judge"]))
        for ak in ("alt", "alt2"):
            if r.get(ak) and r[ak].get("judge"):
                cands.append((r["clause"], r[ak]["edit"], r[ak]["slang"], r[ak]["judge"]))
    random.Random(2).shuffle(cands)
    # balance: half Sonnet-good, half Sonnet-not-good, so precision/recall are both measurable
    good = [c for c in cands if c[3]["score"] >= 4 and c[3]["meaning"]]
    bad = [c for c in cands if not (c[3]["score"] >= 4 and c[3]["meaning"])]
    sample = good[: n // 2] + bad[: n // 2]
    random.Random(3).shuffle(sample)
    out = judge_many([(c[0], c[1], c[2]) for c in sample])
    tp = fp = fn = tn = miss = 0
    for c, o in zip(sample, out):
        if o is None:
            miss += 1; continue
        s_good = c[3]["score"] >= 4 and c[3]["meaning"]
        g_good = o["score"] >= 4 and o["meaning"]
        tp += s_good and g_good; fp += (not s_good) and g_good
        fn += s_good and (not g_good); tn += (not s_good) and (not g_good)
    tot = tp + fp + fn + tn
    print(f"Groq judge ({MODEL}) vs Sonnet on {tot} items ({miss} unanswered)")
    print(f"  agreement {(tp+tn)/max(tot,1):.2f} | precision of 'good' {tp/max(tp+fp,1):.2f} "
          f"| recall of 'good' {tp/max(tp+fn,1):.2f}")
    print(f"  Groq-good that Sonnet rejects (false accepts): {fp}   Sonnet-good that Groq rejects: {fn}")
    return tp / max(tp + fp, 1), tp / max(tp + fn, 1)


def grade(inp, out, min_score=4.0, strict_margin=0.0):
    rows = read_jsonl(D + inp)
    items, keys = [], []
    for i, r in enumerate(rows):
        if r["label"] != "EDIT":
            continue
        items.append((r["clause"], r["edit"], r["slang"])); keys.append((i, "main"))
        for ak in ("alt", "alt2"):
            if r.get(ak):
                items.append((r["clause"], r[ak]["edit"], r[ak]["slang"])); keys.append((i, ak))
    print(f"grading {len(items)} candidates with Groq")
    res = judge_many(items)
    for (i, k), v in zip(keys, res):
        tgt = rows[i] if k == "main" else rows[i][k]
        tgt["judge"] = v
        tgt["keep"] = bool(v and v["score"] >= min_score + strict_margin and v["meaning"])
    write_jsonl(D + out, rows)
    kept = sum(1 for (i, k), v in zip(keys, res) if (rows[i] if k == "main" else rows[i][k]).get("keep"))
    print(f"kept {kept}/{len(items)} ({100*kept/max(len(items),1):.0f}%) -> {D}{out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", type=int, default=0)
    ap.add_argument("--grade", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--margin", type=float, default=0.0)
    a = ap.parse_args()
    if a.calibrate:
        calibrate(a.calibrate)
    if a.grade:
        grade(a.grade, a.out or a.grade.replace("labeled", "judged"), strict_margin=a.margin)
