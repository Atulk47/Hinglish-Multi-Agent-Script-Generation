"""
data_prep/gen_v6_rows.py   - training data for the v6 slang editor ("the grown pool + joy / pride / anger slots")

WHY: the native-speaker reviews grew the pool from 45 to 177 live phrases and added three emotion slots (joy, pride, anger) that the v5 editor was never trained on.
Measured on 7 stories (docs/SLANG_POOL.md, "Round 2 applied and measured"): the new slots were never used (the editor abstained on every joy / pride / anger line) and
the new phrases appeared in only 13 of 41 edits, some awkwardly. More phrases alone do not help; the editor has to be taught them.

HOW (same contract as v5, only the gold-edit WRITER changed):
  stage 1  `sentences`  a cheap LLM writes neutral sentences for the three new slots (16 domains) + extra hard-abstain sentences for them
                        (negated, addressed to an elder / teacher, ...). The v5 sentence bank (data/felix_dataset/v5/sentences.jsonl) is reused for the old slots.
  stage 2  `rows`       for every slot sentence the script picks pool phrases the runtime would offer for that slot (slot_candidates, so context-filtered),
                        prefers the NEW phrases (least-used first), and asks an LLM writer to make the minimal edit. The edit is kept only if the SAME runtime rule
                        gate (frame repair included) + drift guard accept it AND a strong judge (default: Sonnet) rates it >= 4/5 on naturalness. v5 gold edits are
                        re-gated with the current gate.
  stage 3  `assemble`   balance, split (two whole domains held out), carry over the v5 abstain / filler / formal rows that the current contract still accepts,
                        add real-pipeline eval rows, write kaggle_v6/.

  python -m data_prep.gen_v6_rows sentences --budget 1.0
  python -m data_prep.gen_v6_rows rows --budget 6.0
  python -m data_prep.gen_v6_rows assemble
Cost is tracked from the provider's own usage report and the script STOPS at --budget (USD).
"""
from __future__ import annotations
import argparse, collections, concurrent.futures as cf, json, os, random, re, sys, threading, time

sys.path.insert(0, os.getcwd())
from dotenv import load_dotenv
load_dotenv()
import config
import llm_client as L
from agents import edit_gate as eg
from agents.edit_gate import rule_gate, slot_candidates, repair_frame
from data_prep.build_lora_dataset import make_example
from data_prep.v3_common import read_jsonl, write_jsonl, keeps_meaning, newly_inserted, norm
from data_prep import gen_v5_rows as V5

OUT = "data/felix_dataset/v6/"
os.makedirs(OUT, exist_ok=True)
RUN = os.environ.get("V6_RUN", "1")
rng = random.Random(20261116 + int(RUN))
HELD_OUT_DOMAINS = V5.HELD_OUT_DOMAINS
DOMAINS = V5.DOMAINS
NEW_SLOTS = ("joy", "pride", "anger")
WRITER_TASK = "dialogue"          # Haiku: cheap, writes the candidate edits
JUDGE_TASK = os.environ.get("V6_JUDGE", "director")      # Sonnet: stricter naturalness judge (the reviewers found awkward edits that Haiku-level judging let through)

POOLS = {"praise": list(eg.POS_SLANG), "criticism": list(eg.NEG_SLANG), "worry": sorted(eg.FEEL_SLANG),
         "trouble": sorted(eg.PROB_SLANG), "ease": sorted(eg.EASE_SLANG)}
for _n, _x in eg.EXTRA_SLOTS.items():
    POOLS[_n] = list(_x["slangs"])
OLD_POOL = {e["phrase"].lower() for e in json.load(open("data/slang/slang_pool.backup_before_review1.json", encoding="utf-8"))["entries"] if e.get("status") == "core"}
INFO = {c["phrase"].lower(): c for c in json.load(open("data/slang/slang_candidates.json", encoding="utf-8"))["entries"]}
_lock = threading.Lock()

SLOT_SPECS = {
    "joy": "Every line says the speaker is HAPPY / delighted, using one of these plain constructions: 'mujhe bahut khushi ho rahi hai', 'khushi hui', 'bahut maza aa raha hai', "
           "'mujhe bahut maza aaya', 'khushi ki baat hai'. Examples: 'Tumhe dekhkar mujhe bahut khushi hui.', 'Kal ki party mein mujhe bahut maza aaya.'",
    "pride": "Every line says the speaker feels PROUD of someone or of themselves, using 'garv' (or 'gurur'): 'mujhe tum par bahut garv hai', 'mujhe apne aap par garv hai', "
             "'humein apni team par garv hai'. Examples: 'Mujhe apni behen par bahut garv hai.', 'Humein tumhari mehnat par garv hai.'",
    "anger": "Every line says the speaker is ANGRY or furious, using 'gussa': 'mujhe bahut gussa aa raha hai', 'mujhe uspar bahut gussa aaya', 'usne mujhe bahut gussa dilaya'. "
             "Examples: 'Mujhe uski is harkat par bahut gussa aa raha hai.', 'Match haarne par mujhe bahut gussa aaya.'",
}
ABSTAIN_NEW = {
    "neg_new": "Every line contains 'khushi', 'garv', 'gussa' or 'maza' but NEGATED or hedged, e.g. 'Mujhe koi khushi nahi hui.', 'Mujhe gussa nahi hai, bas thoda dukh hai.', "
               "'Ismein garv karne jaisa kuch nahi hai.', 'Mujhe maza nahi aaya.'",
    "respect_new": "Every line is spoken by a young person to a teacher / elder / coach and uses 'aap' or addresses them as 'Sir', 'Madam', 'Papa', 'Dadaji', 'Coach', "
                   "and contains 'khushi', 'garv' or 'gussa', e.g. 'Sir, mujhe aapke saath kaam karke bahut khushi hui.', 'Dadaji, mujhe aap par bahut garv hai.'",
    "third_new": "Every line is a THIRD-person statement or report about someone else's joy / pride / anger (not the speaker's own feeling), e.g. "
                 "'Unhe bahut gussa aaya aur woh chale gaye.', 'Teacher ko class par garv tha.'",
}


def glob_files(pat):
    import glob
    return glob.glob(pat)


def spent() -> float:
    return float(L.STATS["cost"])


def ask(prompt, task, max_tokens=1800, temperature=0.9):
    text, _ = L.complete(task, [{"role": "user", "content": prompt}], temperature=temperature, max_tokens=max_tokens)
    return text


def pmap(fn, items, workers=8, budget=None):
    out = [None] * len(items)
    def run(i):
        if budget is not None and spent() > budget:
            return i, None
        try:
            return i, fn(items[i])
        except Exception as e:
            print("   call failed:", str(e)[:100], flush=True)
            return i, None
    with cf.ThreadPoolExecutor(workers) as ex:
        for i, r in ex.map(run, range(len(items))):
            out[i] = r
    return out


# ------------------------------------------------------------------------------------------------------ stage 1
def stage_sentences(budget, per_call=14):
    path = OUT + "sentences_new.jsonl"
    have = read_jsonl(path) if os.path.exists(path) else []
    seen = {norm(r["text"]) for r in have}
    jobs = []
    for dom in DOMAINS:
        for slot, n in (("joy", 3), ("pride", 2), ("anger", 3)):
            jobs += [("slot", slot, dom)] * n
        for kind in ABSTAIN_NEW:
            jobs.append(("abstain", kind, dom))
    rng.shuffle(jobs)

    def one(job):
        kind, sub, dom = job
        spec = (SLOT_SPECS if kind == "slot" else ABSTAIN_NEW)[sub]
        prompt = (f"{V5.STYLE}\n\nSituation / domain: {dom}.\nTask: write {per_call} DIFFERENT lines. {spec}\nReturn ONLY a JSON array of {per_call} strings.")
        return job, V5.parse_lines(ask(prompt, WRITER_TASK))

    res = pmap(one, jobs, budget=budget)
    for r in res:
        if not r:
            continue
        (kind, sub, dom), lines = r
        for ln in lines:
            ln = V5.clean_sentence(ln)
            if ln and norm(ln) not in seen and not (set(re.findall(r"[a-z]+", ln.lower())) & {"yaar", "bhai", "bro", "arre", "mast", "solid", "kamaal", "zabardast"}):
                seen.add(norm(ln))
                have.append({"text": ln, "kind": kind, "sub": sub, "domain": dom})
    write_jsonl(path, have)
    print("new sentences:", dict(collections.Counter((r["kind"], r["sub"]) for r in have)), f"| cost ${spent():.3f}")


# ------------------------------------------------------------------------------------------------------ stage 2
WRITER = """You edit Hinglish (romanized Hindi-English) dialogue. For each item, rewrite the SENTENCE so that the plain {slot} expression is replaced by the given SLANG, changing NOTHING else
(same speaker, tense, names, other words). The result must be grammatical, natural spoken Hinglish that a young Indian would really say.
HARD RULES: (1) copy every word and punctuation mark of the SENTENCE exactly, except ONE contiguous span of at most 5 words that expresses the feeling / opinion (e.g. 'bahut khushi hui', 'bahut gussa aa raha hai', 'bahut accha', 'chinta ho rahi hai', 'aasaani se', 'badi problem'); replace that span by the slang (plus the verb it needs, e.g. 'mera dil garden-garden ho gaya'); (2) never add or drop any other word, never add intensifiers like 'ekdum' unless they are part of the slang you were given or already in the span; (3) the slang must express the SAME feeling as the span it replaces.
Other rules: use the slang phrase exactly as written; fix only the minimum grammar around it (idioms about the body or feelings need a possessive subject: "mujhe dil baith raha hai" is wrong,
"mera dil baith raha hai" is right; gender and verb agreement must be right). If the slang cannot sit naturally in this sentence, answer null for that item - do NOT force it.
Return ONLY a JSON array with one entry per item, in order: the rewritten sentence or null.

{block}"""


def _hint(slang):
    c = INFO.get(slang.lower())
    if not c:
        return ""
    ex = f" e.g. '{c['example_neutral']}' -> '{c['example_edited']}'" if c.get("example_neutral") and c.get("example_edited") else ""
    nu = f" (avoid when: {c['do_not_use_when']})" if c.get("do_not_use_when") else ""
    return ex + nu


def write_edits(batch):
    """batch: list of dicts with text, slot, slang -> list of edited sentence | None"""
    block = "\n".join(f"{i+1}. SLOT: {b['slot']} | SLANG: {b['slang']}{_hint(b['slang'])}\n   SENTENCE: {b['text']}" for i, b in enumerate(batch))
    txt = ask(WRITER.format(slot="feeling / opinion", block=block), WRITER_TASK, max_tokens=1500, temperature=0.2)
    m = re.search(r"\[.*\]", txt, re.DOTALL)
    arr = json.loads(m.group())
    return [(str(x).strip() if isinstance(x, str) else None) for x in arr][:len(batch)] + [None] * max(0, len(batch) - len(arr))


JUDGE = V5.JUDGE.replace("one opinion/feeling phrase with a slang expression", "one opinion/feeling phrase with a slang expression or idiom") + \
    "\nBe strict about GRAMMAR (possessive subjects for body/feeling idioms, gender and verb agreement) and about whether the slang really means what the plain word meant."


def judge(items, batch=12):
    out = [0] * len(items)
    chunks = [(s, items[s:s + batch]) for s in range(0, len(items), batch)]

    def one(ch):
        s, chunk = ch
        block = "\n".join(f"{i+1}. ORIGINAL: {o}\n   EDITED: {e}" for i, (o, e) in enumerate(chunk))
        txt = ask(JUDGE.format(block=block), JUDGE_TASK, max_tokens=900, temperature=0.0)
        d = json.loads(re.search(r"\{.*\}", txt, re.DOTALL).group())
        return s, [int(d.get(str(i + 1), 0)) for i in range(len(chunk))]

    for attempt in range(3):
        todo = [ch for ch in chunks if not all(out[ch[0]:ch[0] + len(ch[1])])]
        if not todo:
            break
        for r in pmap(one, todo, workers=4 if attempt else 5):
            if r:
                s, vals = r
                out[s:s + len(vals)] = vals
    return out


def stage_rows(budget, pilot=0):
    sents = read_jsonl(OUT + "sentences_new.jsonl") + [r for r in read_jsonl(V5.OUT + "sentences.jsonl") if r["kind"] == "slot"]
    use = collections.Counter()
    done = set()
    for f in sorted(glob_files(OUT + "gold_new*.jsonl")):
        if f.endswith("pilot.jsonl"):
            continue
        for c in read_jsonl(f):
            use[c["gold"]] += 1
            done.add((norm(c["text"]), c["gold"]))
    items = []
    for r in sents:
        text, slot = r["text"], r["sub"]
        slots = [s for s in slot_candidates(text) if s["type"] == slot]
        if not slots:
            continue
        allowed = [p for p in slots[0]["slangs"] if p.lower() not in text.lower()]
        new = [p for p in allowed if p.lower() not in OLD_POOL]
        want = 6 if slot in NEW_SLOTS else 4
        pick = []
        for p in sorted(new, key=lambda p: (use[p], rng.random()))[:want]:
            pick.append(p)
            use[p] += 1
        for p in pick:
            if (norm(text), p) in done:
                continue
            items.append({"text": text, "slot": slot, "slang": p, "domain": r["domain"]})
    rng.shuffle(items)
    if pilot:
        items = items[:pilot]
    print(f"writer items: {len(items)} ({len({i['slang'] for i in items})} distinct phrases)", flush=True)
    batches = [items[i:i + 8] for i in range(0, len(items), 8)]
    res = pmap(write_edits, batches, budget=budget * 0.35)
    stats = collections.Counter()
    cands = []
    for b, r in zip(batches, res):
        if not r:
            stats["writer-failed"] += len(b)
            continue
        for it, ed in zip(b, r):
            if not ed:
                stats["writer-null"] += 1
                continue
            ed = repair_frame(ed, [it["slang"]])
            ok, why = rule_gate(it["text"], ed, it["slang"])
            if not ok:
                stats["gate:" + why] += 1
                continue
            if not keeps_meaning(it["text"], ed) or not newly_inserted(it["text"], ed, it["slang"]):
                stats["drift/new"] += 1
                continue
            cands.append({"text": it["text"], "slot": it["slot"], "domain": it["domain"], "gold": it["slang"], "edited": ed})
    print("gate/writer outcomes:", dict(stats), "| candidates:", len(cands), f"| cost ${spent():.3f}", flush=True)
    ratings = []
    for s in range(0, len(cands), 240):
        if spent() > budget:
            print(f"BUDGET reached (${spent():.3f}) during judging - the rest is dropped")
            break
        ratings += judge([(c["text"], c["edited"]) for c in cands[s:s + 240]])
        print(f"  judged {len(ratings)}/{len(cands)}  ${spent():.3f}", flush=True)
    kept, dist = [], collections.Counter()
    for c, rt in zip(cands, ratings):
        dist[rt] += 1
        if rt >= 4:
            c["rating"] = rt
            kept.append(c)
    print("judge ratings:", dict(sorted(dist.items())), "| kept", len(kept))
    print("kept per slot:", dict(collections.Counter(c["slot"] for c in kept)))
    print("kept per slang:", dict(collections.Counter(c["gold"] for c in kept).most_common()))
    write_jsonl(OUT + ("gold_new_pilot.jsonl" if pilot else f"gold_new_run{RUN}.jsonl"), kept)
    print(L.cost_summary())


# ------------------------------------------------------------------------------------------------------ stage 3
FIRST_PERSON = re.compile(r"\b(mujhe|main|mera|meri|mere|hum|humein|humara|hamara|hamari|humari|maine|mujhko)\b", re.I)


def trigger_type(text):
    low = set(re.findall(r"[a-z']+", text.lower()))
    for name, x in eg.EXTRA_SLOTS.items():
        if low & x["triggers"]:
            return name
    for typ, words in (("praise", ("accha", "achha", "achhi", "badhiya")), ("criticism", ("kharab", "bura", "bekaar")),
                       ("worry", ("chinta", "dar", "ghabrahat")), ("trouble", ("problem", "dikkat", "mushkil"))):
        if low & set(words):
            return typ
    return None


def stage_assemble():
    D4, D5 = "kaggle_v4/", "kaggle_v5/"
    # v5 gold edits, re-gated with the current gate (the pool / frame / usage rules changed since)
    v5g = read_jsonl(V5.OUT + "gold_edits.jsonl")
    regated = []
    for c in v5g:
        if c["gold"].lower() in {e["phrase"].lower() for e in eg.POOL_ENTRIES} and rule_gate(c["text"], c["edited"], c["gold"])[0]:
            regated.append(c)
    print(f"v5 gold edits still valid under the current gate/pool: {len(regated)}/{len(v5g)}")
    new, seen_nn = [], set()
    for f in sorted(glob_files(OUT + "gold_new_run*.jsonl")):
        for c in read_jsonl(f):
            k = (norm(c["text"]), c["gold"])
            if k not in seen_nn:
                seen_nn.add(k)
                new.append(c)
    print("new gold edits (all runs, deduped):", len(new))
    kept = regated + new
    kept = [c for c in kept if c["slot"] not in ("worry", "joy", "pride", "anger") or FIRST_PERSON.search(c["text"])]
    # balance: no phrase above ~4% of the edit rows (the pool is now 177 phrases), and the NEW phrases are not capped below the old ones
    cap = max(30, int(0.04 * len(kept)))
    by = collections.defaultdict(list)
    for c in kept:
        by[c["gold"]].append(c)
    balanced = []
    for s, lst in by.items():
        rng.shuffle(lst)
        balanced += lst[:cap]
    print(f"gold edits: {len(kept)} -> {len(balanced)} after the per-phrase cap {cap} | phrases {len(by)} | per slot {dict(collections.Counter(c['slot'] for c in balanced))}")
    train_e, eval_e = [], []
    for c in balanced:
        (eval_e if c["domain"] in HELD_OUT_DOMAINS else train_e).append(c)
    rng.shuffle(train_e)
    nval = max(40, int(0.06 * len(train_e)))
    val_e, train_e = train_e[:nval], train_e[nval:]

    def shortlist(gold, slot, line):
        pool = [x for x in POOLS.get(slot, POOLS["praise"]) if x != gold and x.lower() not in line.lower()]
        rng.shuffle(pool)
        return rng.sample(pool, min(len(pool), rng.choice([3, 4, 4, 5]))) + [gold]

    def edit_row(c):
        sl = shortlist(c["gold"], c["slot"], c["text"])
        rng.shuffle(sl)
        return make_example(c["text"], sl, c["edited"]), sl

    rows_train = [edit_row(c)[0] for c in train_e]
    rows_val = [edit_row(c)[0] for c in val_e]
    eval_rows = []
    for i, c in enumerate(eval_e):
        ex, sl = edit_row(c)
        eval_rows.append({"id": f"v6s{i:04d}", "src": "synth", "slot": c["slot"], "shortlist": sl, "line": c["text"], "expect": "edit", "gold": c["edited"]})

    # abstain sentences: v5 bank + the new ones
    abst = read_jsonl(V5.OUT + "abstain_sentences.jsonl") + [r for r in read_jsonl(OUT + "sentences_new.jsonl") if r["kind"] == "abstain"]
    a_train, a_eval = [], []
    for r in abst:
        (a_eval if r["domain"] in HELD_OUT_DOMAINS else a_train).append(r)
    CAP = {"negation": 170, "respect": 170, "elder": 100, "serious": 100, "facts": 60, "question": 60, "neg_new": 90, "respect_new": 90, "third_new": 60}
    rng.shuffle(a_train)
    per_kind, capped = collections.Counter(), []
    for r in a_train:
        if per_kind[r["sub"]] < CAP.get(r["sub"], 60):
            per_kind[r["sub"]] += 1
            capped.append(r)
    a_train = capped
    print("abstain rows per kind:", dict(per_kind))

    def abstain_row(r):
        t = trigger_type(r["text"]) or rng.choice(list(POOLS))
        t = t if t in POOLS else "praise"
        sl = rng.sample([x for x in POOLS[t] if x.lower() not in r["text"].lower()], k=min(len(POOLS[t]), rng.choice([3, 4, 5])))
        return make_example(r["text"], sl, r["text"]), sl, t

    na_val = max(10, int(0.06 * len(a_train)))
    rows_train += [abstain_row(r)[0] for r in a_train[na_val:]]
    rows_val += [abstain_row(r)[0] for r in a_train[:na_val]]
    for i, r in enumerate(a_eval):
        ex, sl, t = abstain_row(r)
        eval_rows.append({"id": f"v6a{i:04d}", "src": "synth", "slot": r["sub"], "shortlist": sl, "line": r["text"], "expect": "abstain"})

    # v4 rows (fillers, formal abstains) that the current contract still accepts: reuse v5's filter by calling its helper through the v5 train file instead of v4
    v5_train, v5_val = read_jsonl(D5 + "lora_train.jsonl"), read_jsonl(D5 + "lora_val.jsonl")

    def row_ok(ex):
        try:
            m = ex["messages"]
            sl, line = re.search(r"Slangs: (.*)\nLine: (.*)$", m[1]["content"], re.S).groups()
            sl = sl.split(", ")
            tgt = m[2]["content"].strip()
            if norm(tgt) == norm(line):
                return True
            used = next((x for x in sl if x.lower() in tgt.lower() and x.lower() not in line.lower()), None)
            return bool(used) and keeps_meaning(line, tgt) and rule_gate(line, tgt, used)[0]
        except Exception:
            return False

    # carry over only v5 rows that are NOT mood-slot rows we regenerated: fillers + formal/serious abstains; mood edit rows are rebuilt above
    def is_filler_or_abstain_row(ex):
        sl, line = re.search(r"Slangs: (.*)\nLine: (.*)$", ex["messages"][1]["content"], re.S).groups()
        tgt = ex["messages"][2]["content"].strip()
        return norm(tgt) == norm(line) or any(f in {x.strip().lower() for x in sl.split(",")} for f in ("yaar", "bhai", "bro", "arre", "boss"))
    carry_t = [r for r in v5_train if is_filler_or_abstain_row(r) and row_ok(r)]
    carry_v = [r for r in v5_val if is_filler_or_abstain_row(r) and row_ok(r)]
    print(f"carried over from v5: {len(carry_t)} train / {len(carry_v)} val (fillers + abstains)")
    all_train = carry_t + rows_train
    all_val = carry_v + rows_val
    rng.shuffle(all_train)

    real = real_pipeline_eval()

    def key(s): return re.sub(r"[^a-z ]", "", s.lower()).strip()
    def line_of(ex): return ex["messages"][1]["content"].split("\nLine: ", 1)[1]
    trainset = [set(key(line_of(e)).split()) for e in all_train + all_val]
    trainexact = {key(line_of(e)) for e in all_train + all_val}
    leaks, keep_eval = 0, []
    for r in eval_rows + real:
        k = key(r["line"]); toks = set(k.split())
        if k in trainexact or any(len(toks & t) / max(1, len(toks | t)) >= 0.8 for t in trainset if t):
            leaks += 1
            continue
        keep_eval.append(r)
    print("eval rows removed for leakage:", leaks)
    os.makedirs("kaggle_v6", exist_ok=True)
    write_jsonl("kaggle_v6/lora_train.jsonl", all_train)
    write_jsonl("kaggle_v6/lora_val.jsonl", all_val)
    ev_synth = [r for r in keep_eval if r["src"] == "synth"]
    ev_real = [r for r in keep_eval if r["src"] == "real"]
    write_jsonl("kaggle_v6/eval_slots.jsonl", ev_synth)
    write_jsonl("kaggle_v6/eval_realv5.jsonl", ev_real)              # file name kept so the v5 notebook logic reads it unchanged
    for name in ("eval_real.jsonl", "eval_synth.jsonl", "eval_filler.jsonl"):
        write_jsonl("kaggle_v6/" + name, read_jsonl(D5 + name))
    print(f"kaggle_v6: train {len(all_train)} | val {len(all_val)} | eval slots {len(ev_synth)} (new-slot rows: {sum(1 for r in ev_synth if r['slot'] in NEW_SLOTS)}) | eval real {len(ev_real)}")
    json.dump({"train": len(all_train), "val": len(all_val), "eval_slots": len(ev_synth), "eval_real": len(ev_real), "leak_removed": leaks,
               "gold_per_slang": dict(collections.Counter(c["gold"] for c in balanced).most_common()),
               "gold_per_slot": dict(collections.Counter(c["slot"] for c in balanced)), "domains_heldout": sorted(HELD_OUT_DOMAINS)},
              open(OUT + "build_report.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)


def real_pipeline_eval():
    import glob
    sents, seen = [], set()
    for f in sorted(glob.glob("outputs/*trace*.json")) + sorted(glob.glob("outputs/ablation_pool2/*_agentic.json")):
        try:
            t = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        for sc in t.get("neutral_script") or []:
            for ln in str(sc.get("dialogue", "")).split("\n"):
                if ":" not in ln:
                    continue
                spk, txt = ln.split(":", 1)
                if spk.strip().lower() in ("dadaji", "dadi", "mummy", "papa", "coach varma", "sharm", "narrator"):
                    continue
                for s in re.split(r"(?<=[.!?])\s+", txt.strip()):
                    k = norm(s)
                    if 5 <= len(s.split()) <= 30 and k not in seen:
                        seen.add(k)
                        sents.append(s.strip())
    rows = []
    for i, s in enumerate(sents):
        slots = slot_candidates(s)
        if slots:
            typ = slots[0]["type"] if slots[0]["type"] in POOLS else "praise"
            pool = [x for x in POOLS[typ] if x.lower() not in s.lower()]
            rows.append({"id": f"v6r{i:04d}", "src": "real", "slot": typ, "shortlist": rng.sample(pool, k=min(len(pool), 5)), "line": s, "expect": "edit"})
        else:
            typ = rng.choice(list(POOLS))
            rows.append({"id": f"v6r{i:04d}", "src": "real", "slot": "none", "shortlist": rng.sample(POOLS[typ], k=min(4, len(POOLS[typ]))), "line": s, "expect": "abstain"})
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sentences", "rows", "assemble"])
    ap.add_argument("--budget", type=float, default=2.0, help="USD cap for THIS run (provider-reported cost)")
    ap.add_argument("--pilot", type=int, default=0, help="rows stage only: process just N writer items (cost / quality probe)")
    a = ap.parse_args()
    config.LLM_COST_CAP_USD = max(a.budget + 0.5, 1.0)
    {"sentences": lambda: stage_sentences(a.budget), "rows": lambda: stage_rows(a.budget, a.pilot), "assemble": stage_assemble}[a.stage]()
