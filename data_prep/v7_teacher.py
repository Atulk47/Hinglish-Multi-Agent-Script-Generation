"""
data_prep/v7_teacher.py  -  a frontier model writes the slang edit for each context line; v7_filter keeps only the safe ones.

    python -m data_prep.v7_teacher run   --n 300  --budget 1.0 [--model anthropic/claude-sonnet-5.5]     # pilot: 300 lines
    python -m data_prep.v7_teacher judge --budget 1.0 [--model google/gemini-3.6-flash]                   # cross-family judge on the filter survivors
    python -m data_prep.v7_teacher stats

Resumable (results are appended to data/v7/teacher_raw.jsonl), batches 6 lines per call, STOPS at --budget (USD, from the provider's usage report). Only lines whose context
allows slang are sent; lines that must stay unchanged become "keep as is" examples in the assembler without any model call.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import re
import sys
from collections import Counter

sys.path.insert(0, os.getcwd())
from dotenv import load_dotenv
load_dotenv()
import config
import llm_client as L
from data_prep.v7_format import render_user
from data_prep import v7_filter

D = "data/v7"
NL = chr(10)
TEACHER_SYSTEM = """You are an expert in how Indian college students and young people really talk in Hinglish (Hindi written in Roman letters, mixed with English).
You get screenplay lines, each with its situation. For EACH line write the version a real young speaker would say, with at most MAX EDITS small edits.
An edit is: swapping a plain feeling or opinion word for fitting slang, using an idiom for a feeling, or adding ONE light interjection.

Rules:
1. Keep the meaning, names, numbers, negation, the question / statement form and the speaker's gender verb forms EXACTLY.
2. Prefer VARIED, natural slang over a stock filler: for example kadak, solid, bindaas, mast, jhakaas, next level, epic, ekdum fire, mood off, dil toot gaya, full tension, fatti padi,
   chhaa gaya, kya baat hai, ek number, dhaansu, paisa vasool, bakwaas, wahiyat, scene ho gaya. Use a filler (yaar, arre, bro, bhai) only when no feeling or opinion word can be swapped,
   and never more than one per line. Do not use "yaar" or "arre" as your default.
3. If nothing fits naturally, return the line UNCHANGED. Not every line needs an edit, and a forced edit is worse than none.
4. Never edit lines that thank, apologise, grieve or talk about illness. Match the slang to the speaker's emotion and to the register with the listener.
5. Roman letters only. One line in, one line out.
6. VARIETY: use a DIFFERENT slang word or idiom for every line of one request; never repeat a slang word within a request, and do not fall back on "mast" or "kadak".
   HINTS are only suggestions: use one when it fits, or any other natural slang. Do NOT append a tag such as ", ekdum mast!" at the end of a line: swap a word inside the line.
Return ONLY JSON: [{"id": "<id>", "edited": "<the line>", "edits": ["old -> new", ...]}] in the same order, "edits" empty when unchanged."""


def _read(path):
    return [json.loads(x) for x in open(path, encoding="utf-8")] if os.path.exists(path) else []


def _json_array(text):
    """The list of objects in a model reply. Tolerant: a reply cut off mid-array (a reasoning model can run out of tokens) still yields every COMPLETE object before the cut."""
    a, b = text.find("["), text.rfind("]")
    if a >= 0 and b > a:
        try:
            r = json.loads(text[a:b + 1])
            if isinstance(r, list):
                return r
        except Exception:
            pass
    out = []
    for m in re.finditer(r"\{[^{}]*\}", text):
        try:
            out.append(json.loads(m.group(0)))
        except Exception:
            continue
    return out


def build_batch_prompt(batch):
    """One request: every line with its situation. The optional slang HINTS are 4 phrases sampled at random from everything the pool offers for that line (not the first four of the
    pool, which made the teacher write 'mast' and 'kadak' for 72% of its edits)."""
    parts = []
    for c in batch:
        pool = c.get("hints_all") or c.get("hints") or []
        c = dict(c, hints=random.Random(c["id"]).sample(pool, min(4, len(pool))) if pool else [])
        parts.append(f"### id: {c['id']}" + NL + render_user(c))
    return (NL * 2).join(parts)


def run(n: int, budget: float, model: str, batch_size: int = 6, seed: int = 7, caller=None):
    ctxs = [c for c in _read(f"{D}/contexts.jsonl") if c.get("allows_edit")]
    done = {r["id"] for r in _read(f"{D}/teacher_raw.jsonl")}
    rng = random.Random(seed)
    rng.shuffle(ctxs)
    todo = [c for c in ctxs if c["id"] not in done][:n]
    for c in todo:
        c["max_edits"] = rng.choice([1, 2, 2, 3])
    config.OPENROUTER_MODELS["teacher"] = model
    c0 = float(L.STATS["cost"])
    out = open(f"{D}/teacher_raw.jsonl", "a", encoding="utf-8")
    sent = 0
    for i in range(0, len(todo), batch_size):
        if float(L.STATS["cost"]) - c0 >= budget:
            print(f"budget ${budget} reached after {sent} lines")
            break
        batch = todo[i:i + batch_size]
        msgs = [{"role": "system", "content": TEACHER_SYSTEM}, {"role": "user", "content": build_batch_prompt(batch)}]
        try:
            text = caller(msgs) if caller else L.complete("teacher", msgs, temperature=0.7, max_tokens=4000, only="openrouter")[0]
        except Exception as e:
            print("call failed:", str(e)[:100])
            continue
        by = {str(r.get("id")): r for r in _json_array(text) if isinstance(r, dict)}
        for c in batch:
            r = by.get(c["id"])
            if r is None:
                continue
            out.write(json.dumps({"id": c["id"], "max_edits": c["max_edits"], "edited": r.get("edited", ""), "claimed": r.get("edits", []), "model": model}, ensure_ascii=False) + "\n")
            sent += 1
        out.flush()
    out.close()
    print(f"{sent} teacher answers written | cost this run ${float(L.STATS['cost']) - c0:.3f}")


JUDGE_SYSTEM = """You are a native speaker of Hindi and Hinglish judging slang edits of screenplay lines. For each item you get the situation, the ORIGINAL line and the EDITED line.
Rate: natural = 1-5 (would a young person really say the edited line? 5 = completely natural, 3 = odd but understandable, 1 = broken), meaning_kept = true / false (same meaning,
same feeling direction, same names / numbers / gender forms), register_ok = true / false (slang fits the speaker, listener, setting and emotion; false for slang toward an elder,
in grief, in an apology, or in a traditional setting).
Return ONLY JSON: [{"id": "<id>", "natural": 4, "meaning_kept": true, "register_ok": true}] in the same order."""


def judge(budget: float, model: str, batch_size: int = 8, caller=None):
    ctx = {c["id"]: c for c in _read(f"{D}/contexts.jsonl")}
    done = {r["id"] for r in _read(f"{D}/judge.jsonl")}
    todo = []
    for r in _read(f"{D}/teacher_raw.jsonl"):
        c = dict(ctx[r["id"]], max_edits=r["max_edits"])
        v = v7_filter.check(c, r["edited"])
        if v["ok"] and v["changed"] and r["id"] not in done:
            todo.append((c, r))
    config.OPENROUTER_MODELS["judge_ext"] = model
    c0 = float(L.STATS["cost"])
    out = open(f"{D}/judge.jsonl", "a", encoding="utf-8")
    n = 0
    for i in range(0, len(todo), batch_size):
        if float(L.STATS["cost"]) - c0 >= budget:
            print(f"budget ${budget} reached after {n} judgements")
            break
        batch = todo[i:i + batch_size]
        body = (NL * 2).join(f"### id: {c['id']}" + NL + render_user(c) + NL + f"EDITED: {r['edited']}" for c, r in batch)
        msgs = [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": body}]
        try:
            text = caller(msgs) if caller else L.complete("judge_ext", msgs, temperature=0.0, max_tokens=6000, only="openrouter")[0]
        except Exception as e:
            print("call failed:", str(e)[:100])
            continue
        by = {str(x.get("id")): x for x in _json_array(text) if isinstance(x, dict)}
        for c, r in batch:
            x = by.get(c["id"])
            if x:
                out.write(json.dumps({"id": c["id"], "natural": x.get("natural"), "meaning_kept": x.get("meaning_kept"), "register_ok": x.get("register_ok"), "judge": model}) + NL)
                n += 1
        out.flush()
    out.close()
    print(f"{n} judgements | cost this run ${float(L.STATS['cost']) - c0:.3f}")


def accepted(min_natural: int = 4):
    """ids of the teacher edits that passed the deterministic filter AND the judge (natural >= min_natural, meaning kept, register ok)."""
    j = {r["id"]: r for r in _read(f"{D}/judge.jsonl")}
    soft = {r["id"] for r in _read(f"{D}/teacher_raw.jsonl") if "appended-tail" in v7_filter.check(_ctx_of(r), r["edited"]).get("soft", [])}
    return {i for i, r in j.items() if (r.get("natural") or 0) >= (5 if i in soft else min_natural) and r.get("meaning_kept") is True and r.get("register_ok") is True}


def _ctx_of(r):
    c = _CTX.get(r["id"])
    if c is None:
        _CTX.update({x["id"]: x for x in _read(f"{D}/contexts.jsonl")})
        c = _CTX[r["id"]]
    return dict(c, max_edits=r["max_edits"])


_CTX = {}


def stats():
    ctx = {c["id"]: c for c in _read(f"{D}/contexts.jsonl")}
    raw = _read(f"{D}/teacher_raw.jsonl")
    ok = Counter(); why = Counter(); slang = Counter(); filler_only = 0; edited = 0
    for r in raw:
        c = dict(ctx[r["id"]], max_edits=r["max_edits"])
        v = v7_filter.check(c, r["edited"])
        if not v["changed"]:
            ok["unchanged (valid)"] += 1
            continue
        edited += 1
        if v["ok"]:
            ok["edited & accepted"] += 1
            filler_only += v7_filter.is_filler_only(c, r["edited"])
            slang.update(w for w in set(r["edited"].lower().split()) - set(c["line"].lower().split()))
        else:
            ok["edited & REJECTED"] += 1
            why.update(v["reasons"])
    print(dict(ok), "| of the accepted edits, filler-only:", filler_only)
    print("rejection reasons:", dict(why.most_common(10)))
    print("most added words:", dict(slang.most_common(25)))
    j = _read(f"{D}/judge.jsonl")
    if j:
        acc = accepted()
        print(f"judge: {len(j)} judged | accepted (natural>=4, meaning kept, register ok): {len(acc)} ({len(acc) / len(j):.0%}) | mean natural {sum((x.get('natural') or 0) for x in j) / len(j):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "judge", "stats"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--model", default="anthropic/claude-sonnet-5.5")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.n, a.budget, a.model)
    elif a.cmd == "judge":
        judge(a.budget, a.model if "--model" in sys.argv else "google/gemini-3.6-flash")
    else:
        stats()


if __name__ == "__main__":
    main()
