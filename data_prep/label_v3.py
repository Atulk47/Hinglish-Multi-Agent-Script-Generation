"""
data_prep/label_v3.py
Step 2 (+3) of the v3 on-distribution dataset: OFFLINE teacher labelling and judging.

  --stage label : for each clause cut from an Agent-3-style line, the teacher (Sonnet) either
                  rewrites it the way a real young person would say it (<=2 stiff-word swaps +
                  exactly ONE slang from a cleaned bank) or answers NONE (nothing fits naturally).
                  Every EDIT is checked locally against the SAME guards the pipeline applies at
                  runtime (slang newly inserted, drift guard, no Devanagari, punctuation).
  --stage judge : an independent strict pass grades each surviving edit (naturalness 1-5, meaning
                  preserved) and double-checks every NONE. Only score>=4 + meaning-preserved edits and
                  doubly-confirmed NONEs survive = the "platinum" set.

The teacher is used ONLY to build training data. At inference the small Qwen LoRA works alone.

Usage:
  python -m data_prep.label_v3 --stage label --lines data/felix_dataset/v3/lines.jsonl \
        --out data/felix_dataset/v3/labeled.jsonl --max-clauses 3000 --max-usd 5
  python -m data_prep.label_v3 --stage judge --inp data/felix_dataset/v3/labeled.jsonl \
        --out data/felix_dataset/v3/judged.jsonl --max-usd 2
"""
from __future__ import annotations
import argparse, collections, json, os, random, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from data_prep.v3_common import (Teacher, parse_json, load_lexicon, candidate_units, keeps_meaning,
                                 newly_inserted, DEVANAGARI, PLAIN_FILLERS, read_jsonl, write_jsonl, norm)

from data_prep.slang_bank_v3 import BANK, TONE_NAME


def clean_bank():
    """-> {slang_lower: (display, tone_name)} from the curated bank."""
    return {k: (k, TONE_NAME[v[0]]) for k, v in BANK.items()}


SYSTEM = ("You are a native Hinglish speaker (young urban Indian, early 20s) and an expert editor of "
          "screenplay dialogue. You make stiff, textbook dialogue sound like what real friends say, "
          "without ever changing its meaning.")

LABEL_PROMPT = """Each numbered SENTENCE below was cut from a longer line spoken by a young person (a peer, not an elder or official). The sentences are stiff, textbook Hinglish written by a weaker model.

For each sentence choose ONE:
(A) EDIT - rewrite it the way a real friend would say it, using exactly ONE slang from the SLANG BANK that genuinely fits THIS sentence's meaning, emotion and grammar. You may also swap up to 2 stiff / bookish words for everyday equivalents (maksad->matlab, prashansa->tareef, avashyak->zaroori, uttam->badhiya, sahayata->help, nirasha->tension, prayaas->koshish, vishwas->bharosa ...). Everything else stays word-for-word. Keep all facts, names, tense, negation and attitude. Add NO new information, feeling or clause. At most 2 words of the original may disappear (counting words the slang replaces). Keep the final punctuation.
(B) NONE - if no bank slang fits naturally. Strictness is correct: a forced slang is worse than none.

WHAT WORKS (a native speaker accepts these):
 1. SUBSTITUTION - the slang replaces an evaluative / intensifier word: "khana bahut kharab tha" -> "khana ekdum ghatiya tha"; "kaafi uttam sunai dega" -> "kaafi zabardast sunai dega".
 2. VOCATIVE / OPENER - a short filler next to a name or at the start: "Neha, humein jaldi nikalna chahiye" -> "Neha yaar, humein jaldi nikalna chahiye"; "Haan Pooja, mujhe bhi aisa lag raha hai" -> "Arre haan Pooja, mujhe bhi aisa lag raha hai".
 3. FITTING IDIOM - an idiom that restates the SAME emotion the sentence already expresses: "mujhe bahut chinta ho rahi hai" -> "mujhe full tension ho rahi hai".
WHAT FAILS (do NOT do these - answer NONE instead):
 x tacking an idiom onto the end of a calm / informational sentence ("...report jama kar dunga, sorted.");
 x adding a feeling or fact the sentence does not state ("...naukri par garv hoga, jhakaas hoon." / "...shaq na rahe" -> "zero tension mein rahe");
 x a slang that changes what the sentence says, or that is in the wrong mood (negative slang on a happy sentence, or the reverse);
 x a slang that merely matches the mood but is unrelated to the content.
Typically 25-35% of sentences should be NONE.

Neutral fillers (yaar, bhai, bro, chill, arre, boss, abey, kasam se) work well as vocatives/openers but use them in at most 1 of every 4 edits; prefer a wide variety across the batch.

Also return up to TWO alternative edits ("alt", "alt2"), each using a DIFFERENT slang from the bank, each following the same rules and each genuinely natural (null if you cannot find a natural one). "also_ok" lists up to 4 other bank slangs that would also be natural in this sentence.

SLANG BANK
positive: {pos}
negative: {neg}
neutral / filler: {neu}

SENTENCES
{clauses}

Return ONLY a JSON array, one object per sentence, in order:
[{{"i":1,"slang":"<exact bank slang or NONE>","edit":"<rewritten sentence, or the sentence unchanged for NONE>","also_ok":[...],"alt":{{"slang":"..","edit":".."}} or null,"alt2":{{"slang":"..","edit":".."}} or null}}]"""


JUDGE_PROMPT = """You are a very strict native speaker of casual Indian Hinglish (age 22, Delhi/Mumbai). Grade each edited clause.
For each item you see the ORIGINAL (stiff) clause, the EDITED clause and the slang that was inserted.
Score NATURALNESS 1-5: 5 = a real friend would say exactly this; 4 = natural, the slang fits; 3 = a bit off or tacked on; 2 = awkward/forced; 1 = wrong or nonsense. Be harsh: a slang that merely matches the mood but is unrelated to what the clause is about, or sits oddly in the grammar, is 3 or lower.
"meaning" = true only if the EDITED clause says exactly the same thing (no fact, tense, negation, attitude or polarity change; no new information).

{items}

Return ONLY JSON: [{{"i":1,"score":4,"meaning":true}}, ...]"""

NONE_PROMPT = """You are a very strict native speaker of casual Indian Hinglish. For each item below, decide whether ANY of its candidate slangs could be inserted into the clause so that a real friend would naturally say it, with the clause's meaning and tone unchanged. If the clause is serious/informational, or every candidate is the wrong mood or unrelated to the content, answer false.

{items}

Return ONLY JSON: [{{"i":1,"fits":false}}, ...]"""


def pick_bank(bank, usage, rng, n_pos=16, n_neg=16, n_neu=5):
    def draw(tone, n):
        items = [(k, v[0]) for k, v in bank.items() if v[1] == tone]
        w = [1.0 / (1 + usage[k]) ** 1.5 for k, _ in items]
        out, seen = [], set()
        while len(out) < min(n, len(items)):
            k, d = rng.choices(items, weights=w)[0]
            if k not in seen:
                seen.add(k); out.append(d)
        return out
    return draw("positive", n_pos), draw("negative", n_neg), draw("neutral", n_neu)


_SENT_POS = {"accha", "acchi", "achha", "achhi", "badiya", "mast", "sahi", "shaandaar", "zabardast", "kamaal",
             "behtareen", "khush", "maza", "pasand", "best", "great", "uttam", "sundar", "prashansa", "khushi",
             "safal", "saflata", "garv", "himmat", "pyaar", "mazedaar", "shandar", "tareef", "jeet", "jeetna"}
_SENT_NEG = {"bekaar", "bura", "buri", "kharab", "bakwaas", "ghatiya", "boring", "pareshan", "dukhi", "gussa",
             "thak", "fail", "problem", "dikkat", "mushkil", "galat", "tension", "chinta", "nirasha", "haar",
             "kathin", "galti", "darr", "ghabra", "ghabrahat", "pareshani", "dukh", "musibat", "afsos",
             "niraasha", "uljha", "gadbad", "der", "toot", "rukawat"}


def has_sentiment(c):
    t = set(re.findall(r"[a-z]+", c.lower()))
    return bool(t & _SENT_POS or t & _SENT_NEG)


def choose_clauses(lines, rng, max_clauses, exclude=None, calm_keep=0.4):
    cl, seen = [], set()
    by_line = []
    for r in lines:
        cs = candidate_units(r["line"])
        cs = [c for c in cs if not re.search(r"\b(yaar|bhai|bro)\b", c.lower())]
        if not cs:
            continue
        # questions are blocked upstream (Gate 2), so keep only a few; everything else is fair game
        pick = [c for c in cs if not c.endswith("?") or rng.random() < 0.15]
        by_line.append((r, pick))
    for r, pick in by_line:
        for c in pick:
            k = norm(c)
            if k in seen or (exclude and k in exclude):
                continue
            # sentences with no sentiment/evaluative word are mostly NONE; keep a calm sample only
            if not has_sentiment(c) and rng.random() > calm_keep:
                continue
            seen.add(k)
            cl.append({"scene_id": r["scene_id"], "speaker": r.get("speaker", ""), "line": r["line"], "clause": c})
    rng.shuffle(cl)
    return cl[:max_clauses]


def end_punct(s):
    m = re.search(r"([.!?,;]+)\s*$", s)
    return m.group(1) if m else ""


def fix_punct(clause, edit):
    cp, ep = end_punct(clause), end_punct(edit)
    if cp == ep:
        return edit
    return re.sub(r"[.!?,;]+\s*$", "", edit).rstrip() + cp


def validate_edit(clause, edit, slang, bank_names):
    """Local gate: the same checks the runtime applies. -> (ok, why)"""
    if not edit or not slang or slang not in bank_names:
        return False, "off-bank"
    if DEVANAGARI.search(edit) or "\n" in edit or len(edit) > len(clause) + 80:
        return False, "format"
    edit = fix_punct(clause, edit)
    if norm(edit) == norm(clause):
        return False, "unchanged"
    if not newly_inserted(clause, edit, slang):
        return False, "slang-not-inserted"
    if not keeps_meaning(clause, edit):
        return False, "drift"
    if clause.rstrip().endswith(",") and re.search(re.escape(slang) + r"\W*$", edit, re.I):
        return False, "slang-at-fragment-end"
    return True, ""


def label_batch(teacher, batch, bank, usage, rng):
    pos, neg, neu = pick_bank(bank, usage, rng)
    clauses = "\n".join(f'{i+1}. "{b["clause"]}"' for i, b in enumerate(batch))
    txt = teacher.ask(SYSTEM, LABEL_PROMPT.format(pos=", ".join(pos), neg=", ".join(neg),
                                                  neu=", ".join(neu), clauses=clauses), 5000)
    arr = parse_json(txt)
    names = {s.lower(): s for s in pos + neg + neu}
    out = []
    if not isinstance(arr, list):
        return out
    for o in arr:
        try:
            b = batch[int(o["i"]) - 1]
        except Exception:
            continue
        clause = b["clause"]
        slang = str(o.get("slang", "")).strip()
        rec = dict(b, bank=pos + neg + neu, label="NONE", slang=None, edit=None, also_ok=[], alt=None, alt2=None, why="")
        if slang.upper() == "NONE" or not slang:
            out.append(rec); continue
        sl = names.get(slang.lower())
        ok, why = validate_edit(clause, str(o.get("edit", "")).strip(), sl, set(names.values()))
        if not ok:
            rec["label"] = "REJECT"; rec["why"] = why; out.append(rec); continue
        rec.update(label="EDIT", slang=sl, edit=fix_punct(clause, str(o["edit"]).strip()))
        rec["also_ok"] = [names[x.lower()] for x in (o.get("also_ok") or [])
                          if isinstance(x, str) and x.lower() in names and x.lower() != sl.lower()][:4]
        for ak in ("alt", "alt2"):
            alt = o.get(ak)
            if isinstance(alt, dict):
                a_sl = names.get(str(alt.get("slang", "")).strip().lower())
                a_ed = str(alt.get("edit", "")).strip()
                used_sl = {sl.lower()} | {(rec.get(k) or {}).get("slang", "").lower() for k in ("alt", "alt2")}
                if (a_sl and a_sl.lower() not in used_sl
                        and validate_edit(clause, a_ed, a_sl, set(names.values()))[0]):
                    rec[ak] = {"slang": a_sl, "edit": fix_punct(clause, a_ed)}
        out.append(rec)
    return out


def stage_label(args):
    rng = random.Random(args.seed)
    lines = read_jsonl(args.lines)
    sid = lambda r: int(r["scene_id"].replace("v3s", "")) if r["scene_id"].startswith("v3s") else -1
    lines = [r for r in lines if args.scene_min <= sid(r) < args.scene_max or sid(r) < 0]
    bank = clean_bank()
    print(f"bank: {len(bank)} slangs ({collections.Counter(v[1] for v in bank.values())})")
    done = read_jsonl(args.out)
    done_k = {norm(d["clause"]) for d in done}
    excl = set()
    for pth in (args.exclude or "").split(","):
        if pth.strip():
            excl |= {norm(d["clause"]) for d in read_jsonl(pth.strip())}
    clauses = [c for c in choose_clauses(lines, rng, args.max_clauses, excl, args.calm_keep)
               if norm(c["clause"]) not in done_k]
    print(f"{len(lines)} lines -> {len(clauses)} new clauses to label (have {len(done)})")
    usage = collections.Counter()
    for d in done:
        if d.get("slang"):
            usage[d["slang"].lower()] += 1
    teacher = Teacher(args.model, args.max_usd)
    batches = [clauses[i:i + args.batch] for i in range(0, len(clauses), args.batch)]
    rows = list(done)
    ulock = __import__("threading").Lock()

    def work(bi, b):
        r = random.Random(args.seed * 7919 + bi + len(done))
        with ulock:
            u = collections.Counter(usage)
        return label_batch(teacher, b, bank, u, r)

    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(work, bi, b) for bi, b in enumerate(batches)]
        for k, fu in enumerate(as_completed(futs), 1):
            try:
                res = fu.result()
            except Exception as e:
                print("  batch failed:", str(e)[:120])
                if "spend cap" in str(e) or "credit" in str(e).lower():
                    break
                continue
            rows += res
            with ulock:
                for d in res:
                    if d.get("slang"):
                        usage[d["slang"].lower()] += 1
            if k % 10 == 0 or k == len(batches):
                write_jsonl(args.out, rows)
                c = collections.Counter(r["label"] for r in rows)
                print(f"  {k}/{len(batches)} batches | {dict(c)} | slangs used {len(usage)} | ~${teacher.cost:.2f}",
                      flush=True)
    write_jsonl(args.out, rows)
    c = collections.Counter(r["label"] for r in rows)
    rej = collections.Counter(r["why"] for r in rows if r["label"] == "REJECT")
    print(f"\nDONE labels {dict(c)} | rejects by reason {dict(rej)} | ~${teacher.cost:.2f}")


def stage_judge(args):
    rows = read_jsonl(args.inp)
    done = {r["cid"]: r for r in read_jsonl(args.out)} if os.path.exists(args.out) else {}
    # assign stable ids
    for i, r in enumerate(rows):
        r["cid"] = i
    items = []   # (cid, kind, clause, edit, slang)
    for r in rows:
        if r["cid"] in done:
            continue
        if r["label"] == "EDIT":
            items.append((r["cid"], "main", r["clause"], r["edit"], r["slang"]))
            for ak in ("alt", "alt2"):
                if r.get(ak):
                    items.append((r["cid"], ak, r["clause"], r[ak]["edit"], r[ak]["slang"]))
        elif r["label"] == "NONE":
            items.append((r["cid"], "none", r["clause"], None, None))
    print(f"judging {len(items)} items")
    teacher = Teacher(args.model, args.max_usd)
    verdict = {}   # (cid, kind) -> dict

    def run_edit_batch(chunk):
        txt = teacher.ask(SYSTEM, JUDGE_PROMPT.format(items="\n".join(
            f'{i+1}. ORIGINAL: "{c}"\n   EDITED: "{e}"\n   SLANG: {s}' for i, (_, _, c, e, s) in enumerate(chunk))), 2500)
        arr = parse_json(txt) or []
        return [(chunk[int(o["i"]) - 1][:2], {"score": float(o.get("score", 0)), "meaning": bool(o.get("meaning", False))})
                for o in arr if isinstance(o, dict) and "i" in o and 1 <= int(o["i"]) <= len(chunk)]

    bankcache = {r["cid"]: r for r in rows}

    def run_none_batch(chunk):
        txt = teacher.ask(SYSTEM, NONE_PROMPT.format(items="\n".join(
            f'{i+1}. CLAUSE: "{c}"\n   CANDIDATES: {", ".join(bankcache[cid]["bank"][:36])}'
            for i, (cid, _, c, _, _) in enumerate(chunk))), 1500)
        arr = parse_json(txt) or []
        return [(chunk[int(o["i"]) - 1][:2], {"fits": bool(o.get("fits", True))})
                for o in arr if isinstance(o, dict) and "i" in o and 1 <= int(o["i"]) <= len(chunk)]

    edit_items = [t for t in items if t[1] in ("main", "alt", "alt2")]
    none_items = [t for t in items if t[1] == "none"]
    jobs = [("e", edit_items[i:i + 25]) for i in range(0, len(edit_items), 25)]
    jobs += [("n", none_items[i:i + 8]) for i in range(0, len(none_items), 8)]
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(run_edit_batch if k == "e" else run_none_batch, ch) for k, ch in jobs]
        for n, fu in enumerate(as_completed(futs), 1):
            try:
                for key, v in fu.result():
                    verdict[key] = v
            except Exception as e:
                print("  judge batch failed:", str(e)[:120])
            if n % 20 == 0:
                print(f"  {n}/{len(jobs)} judge batches | ~${teacher.cost:.2f}", flush=True)

    out = []
    for r in rows:
        if r["cid"] in done:
            out.append(done[r["cid"]]); continue
        r = dict(r)
        if r["label"] == "EDIT":
            m = verdict.get((r["cid"], "main"))
            r["judge"] = m
            r["keep"] = bool(m and m["score"] >= args.min_score and m["meaning"])
            for ak in ("alt", "alt2"):
                if r.get(ak):
                    a = verdict.get((r["cid"], ak))
                    r[ak]["judge"] = a
                    r[ak]["keep"] = bool(a and a["score"] >= args.min_score and a["meaning"])
        elif r["label"] == "NONE":
            n = verdict.get((r["cid"], "none"))
            r["keep"] = bool(n and n["fits"] is False)
        out.append(r)
    write_jsonl(args.out, out)
    ed = [r for r in out if r["label"] == "EDIT"]
    kept_m = sum(r.get("keep", False) for r in ed)
    kept_a = sum(1 for r in ed for ak in ("alt", "alt2") if r.get(ak) and r[ak].get("keep"))
    nn = [r for r in out if r["label"] == "NONE"]
    print(f"\nEDIT clauses {len(ed)}: main kept {kept_m} ({100*kept_m/max(len(ed),1):.0f}%), alt kept {kept_a}")
    print(f"NONE clauses {len(nn)}: confirmed {sum(r.get('keep', False) for r in nn)}")
    sc = [r["judge"]["score"] for r in ed if r.get("judge")]
    print(f"mean judge score (main) {sum(sc)/max(len(sc),1):.2f} | ~${teacher.cost:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["label", "judge"], required=True)
    ap.add_argument("--lines", default="data/felix_dataset/v3/lines.jsonl")
    ap.add_argument("--inp", default="data/felix_dataset/v3/labeled.jsonl")
    ap.add_argument("--out", default="data/felix_dataset/v3/labeled.jsonl")
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--max-clauses", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-usd", type=float, default=5.0)
    ap.add_argument("--min-score", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--exclude", default="", help="comma-separated labeled jsonl whose sentences to skip")
    ap.add_argument("--calm-keep", type=float, default=0.4, help="keep-prob for sentences without sentiment words")
    ap.add_argument("--scene-min", type=int, default=0)
    ap.add_argument("--scene-max", type=int, default=10**9)
    args = ap.parse_args()
    (stage_label if args.stage == "label" else stage_judge)(args)


if __name__ == "__main__":
    main()
