"""
data_prep/gen_v5_rows.py   - training data for the v5 slang editor ("mood slots + phrase swaps")

WHY: the runtime now builds the shortlist from the sentence's own SLOT (praise / criticism / worry / trouble / ease) and the
rule gate also accepts an idiom-for-phrase swap ("bahut dar lag raha hai" -> "fatti padi hai"). v4 never saw those patterns.

HOW (gold labels BY CONSTRUCTION, not by a noisy LLM):
  stage 1  `sentences`  an LLM (cheap) only WRITES neutral Hinglish sentences: slot sentences per slot type x domain, plus
                        "hard abstain" sentences (negation, aap/respect, addressed to elder, health, plain facts, questions).
  stage 2  `rows`       code builds the gold edit for every slot sentence (substitute the slot word/phrase with a bank slang
                        of the right KIND, grammar-agreed), keeps it only if the SAME runtime rule gate + drift guard accept
                        it, then an LLM judge scores naturalness (keep >= 4/5). Shortlists are built like the runtime does
                        (same slot type, 4-6 entries, gold at a random position). Abstain rows keep the line UNCHANGED.
  stage 3  `assemble`   dedupe, split (two whole domains held out for evaluation), add real-pipeline eval rows, write kaggle_v5/.

  python -m data_prep.gen_v5_rows sentences --budget 3.0
  python -m data_prep.gen_v5_rows rows --budget 1.0
  python -m data_prep.gen_v5_rows assemble
Cost is tracked from the provider's own usage report and the script STOPS at --budget (USD).
"""
from __future__ import annotations
import argparse, collections, hashlib, json, os, random, re, sys, time

sys.path.insert(0, os.getcwd())
from dotenv import load_dotenv
load_dotenv()
import config
import llm_client as L
from agents.edit_gate import (rule_gate, slot_candidates, POS_SLANG, NEG_SLANG, FEEL_SLANG, PROB_SLANG, EASE_SLANG,
                              NOUN_GENDER, FEMALE_NAMES, RESPECT_ADDR)
from data_prep.build_lora_dataset import make_example
from data_prep.v3_common import read_jsonl, write_jsonl, keeps_meaning, newly_inserted, norm

OUT = "data/felix_dataset/v5/"
os.makedirs(OUT, exist_ok=True)
SEED = 20261025
rng = random.Random(SEED)

# slot type -> slang pool (identical to the runtime lists in agents/edit_gate.py)
POOLS = {"praise": list(POS_SLANG), "criticism": list(NEG_SLANG), "worry": sorted(FEEL_SLANG),
         "trouble": sorted(PROB_SLANG), "ease": sorted(EASE_SLANG)}
DOMAINS = ["school life", "cricket and sports", "family at home", "college friends", "office / internship", "festival and shaadi",
           "exam preparation", "music and dance practice", "tech / robotics project", "road trip with friends", "neighbourhood and market",
           "cooking and food", "movies and OTT shows", "hospital and recovery", "village and farming", "startup and money"]
HELD_OUT_DOMAINS = {"festival and shaadi", "tech / robotics project"}      # never trained on -> honest generalisation test

STYLE = """Write natural everyday ROMANIZED HINGLISH (Hindi in Roman letters, with some English words), the way a screenwriter would
write short spoken lines for young characters. Each line is ONE sentence of 8-24 words, a complete thought, with a speaker's
feeling or opinion. Use plain common words (not Sanskritised Hindi). NO slang words and NO fillers (never use: yaar, bhai, bro, arre,
mast, solid, zabardast, kamaal, jhakaas, gazab, panga, locha, bakwaas, kachra, faltu, ghatiya, fatti, tension idiom etc.).
Vary the speaker's situation, names (Indian first names), tense and sentence shape. Do not number the lines."""

SLOT_SPECS = {
    "praise": "Every line states that something is GOOD using exactly one of these plain words as the opinion word: "
              "bahut accha / bahut achhi / bahut achha / kaafi accha / badhiya / bahut badhiya / shandar / behtareen. Examples: "
              "'Tumhari performance bahut achhi thi.', 'Mujhe yeh plan bahut accha laga.'",
    "criticism": "Every line states that something is BAD using exactly one of these plain words as the opinion word: "
                 "bahut kharab / bahut bura laga / bekaar / boring. Examples: 'Yeh movie bahut kharab thi.', "
                 "'Mujhe uska behaviour bahut bura laga.'",
    "worry": "Every line expresses the speaker's WORRY or FEAR using one of these plain constructions: 'bahut chinta ho rahi hai', "
             "'chinta ho rahi hai', 'bahut dar lag raha hai', 'dar lag raha hai', 'ghabrahat ho rahi hai'. "
             "Examples: 'Mujhe kal ke result ki bahut chinta ho rahi hai.', 'Andhere mein mujhe bahut dar lag raha hai.'",
    "trouble": "Every line says there is a PROBLEM or difficult situation, using one of: 'badi problem', 'bahut badi dikkat', "
               "'mushkil situation', 'samasya', 'dikkat', 'problem'. Examples: 'Yeh toh bahut badi problem ho gayi.', "
               "'Humare saamne ek mushkil situation hai.'",
    "ease": "Every line says something can be done EASILY, using 'aasaani se' or 'asaani se'. Examples: "
            "'Hum yeh kaam aasaani se kar lenge.', 'Woh sawaal maine aasaani se solve kar diya.'",
}
ABSTAIN_SPECS = {
    "negation": "Every line contains an opinion word but NEGATED or inside a negative sentence, e.g. 'Yeh idea accha nahi hai.', "
                "'Mujhe koi problem nahi hai.', 'Chinta mat karo, sab theek ho jayega.', 'Yeh itna kharab bhi nahi tha.'",
    "respect": "Every line is RESPECTFUL / FORMAL, spoken to a teacher, boss or elder using 'aap', 'aapka', 'aapki', 'sir', 'madam' or 'ji', "
               "and contains an opinion or feeling word like accha, kharab, chinta, problem. e.g. 'Sir, aapka lecture bahut accha tha.'",
    "elder": "Every line is spoken by a young person TO an elder or coach and addresses them by 'Papa', 'Mummy', 'Dadaji', 'Dadi', "
             "'Nani', 'Coach' or 'Sir', and contains an opinion or feeling word like accha, bura, chinta, dar. "
             "e.g. 'Papa, mujhe aaj ka match bahut accha laga.'",
    "serious": "Every line is about illness, injury, hospital, doctor or a death in the family, and contains a feeling or opinion word like "
               "kharab, chinta, dar, accha, problem. e.g. 'Dadaji ki tabiyat bahut kharab hai.', 'Doctor ne kaha ki ab sab accha hai.'",
    "facts": "Every line is a plain FACT, plan or instruction with NO opinion, feeling or problem word at all. e.g. "
             "'Kal subah hum station par milenge.', 'Match shaam chhe baje shuru hoga.'",
    "question": "Every line is a QUESTION or a request between friends with no opinion word. e.g. 'Tum kal kitne baje aaoge?'",
}


# ------------------------------------------------------------------------------------------------------ LLM helpers
def spent() -> float:
    return float(L.STATS["cost"])


def ask(prompt: str, max_tokens: int = 1800, temperature: float = 0.9, task: str = "dialogue") -> str:
    msgs = [{"role": "user", "content": prompt}]
    text, _ = L.complete(task, msgs, temperature=temperature, max_tokens=max_tokens)
    return text


def parse_lines(txt: str):
    m = re.search(r"\[.*\]", txt, re.DOTALL)
    if m:
        try:
            arr = json.loads(m.group())
            return [str(x).strip() for x in arr if isinstance(x, (str,))]
        except Exception:
            pass
    return [re.sub(r"^[\-\*\d\.\)\s]+", "", l).strip().strip('"') for l in txt.splitlines() if len(l.split()) >= 5]


def clean_sentence(s: str) -> str | None:
    s = re.sub(r"\s+", " ", s).strip().strip('"')
    if not (6 <= len(s.split()) <= 30) or re.search(r"[ऀ-ॿ]", s) or ":" in s[:20]:
        return None
    if not s.endswith((".", "!", "?")):
        s += "."
    return s


# ------------------------------------------------------------------------------------------------------ stage 1
def stage_sentences(budget: float, per_call: int = 14):
    path = OUT + "sentences.jsonl"
    have = read_jsonl(path) if os.path.exists(path) else []
    seen = {norm(r["text"]) for r in have}
    jobs = []
    for dom in DOMAINS:
        for slot in SLOT_SPECS:
            jobs += [("slot", slot, dom)] * (2 if slot in ("praise", "criticism", "worry", "trouble") else 1)
        for kind in ABSTAIN_SPECS:
            jobs.append(("abstain", kind, dom))
    done = collections.Counter((r["kind"], r["sub"], r["domain"]) for r in have)
    rng.shuffle(jobs)
    n_new = 0
    t0 = time.time()
    for kind, sub, dom in jobs:
        if spent() > budget:
            print(f"BUDGET reached (${spent():.3f}) - stopping")
            break
        spec = (SLOT_SPECS if kind == "slot" else ABSTAIN_SPECS)[sub]
        key = (kind, sub, dom)
        if done[key] >= (2 if kind == "slot" and sub in ("praise", "criticism", "worry", "trouble") else 1) * per_call * 0.7:
            continue
        prompt = (f"{STYLE}\n\nSituation / domain: {dom}.\nTask: write {per_call} DIFFERENT lines. {spec}\n"
                  f"Return ONLY a JSON array of {per_call} strings.")
        try:
            lines = parse_lines(ask(prompt))
        except Exception as e:
            print("  call failed:", str(e)[:100])
            continue
        add = []
        for ln in lines:
            ln = clean_sentence(ln)
            if ln and norm(ln) not in seen and not (set(re.findall(r"[a-z]+", ln.lower())) &
                                                    {"yaar", "bhai", "bro", "arre", "mast", "solid", "kamaal", "zabardast"}):
                seen.add(norm(ln))
                add.append({"text": ln, "kind": kind, "sub": sub, "domain": dom})
        have += add
        n_new += len(add)
        write_jsonl(path, have)
        print(f"  {kind}/{sub}/{dom}: +{len(add)}  total={len(have)}  ${spent():.3f}  {time.time()-t0:.0f}s", flush=True)
    print(f"sentences: {len(have)} (+{n_new} new) | cost ${spent():.3f}")
    print(L.cost_summary())


# ------------------------------------------------------------------------------------------------------ stage 2: gold edits
INTENS = r"(?:bahut|kaafi|bilkul|ekdum|bohot|itna|itni|ek dum|kitna)"
PRAISE_RE = re.compile(rf"\b(?:({INTENS})\s+)?(accha|achha|acchi|achhi|acha|achi|achcha|achchi|achche|acche|achhe|badhiya|shandar|shaandaar|behtareen)\b", re.I)
CRIT_RE = re.compile(rf"\b(?:({INTENS})\s+)?(kharab|bura|buri|bekaar|bekar|boring)\b", re.I)
TROUBLE_RE = re.compile(rf"\b(?:(?:ek\s+)?(?:bahut\s+|kaafi\s+)?(badi|bada)\s+)?(problem|dikkat|samasya|mushkil situation|mushkil)\b", re.I)
EASE_RE = re.compile(r"\b(aasaani|asaani|aasani)\s+se\b", re.I)
WORRY_RE = re.compile(rf"\b(?:({INTENS})\s+)?(chinta|ghabrahat)\s+(ho\s+rahi\s+hai|ho\s+raha\s+hai|ho\s+gayi\s+hai|ho\s+gaya\s+hai|hai)\b", re.I)
FEAR_RE = re.compile(rf"\b(?:({INTENS})\s+)?dar\s+lag\s+(raha|rahi)\s+hai\b", re.I)
# predicate-like worry slangs replace the whole "(bahut) chinta ho rahi hai"; noun-like ones keep the verb
WORRY_NOUN = {"full tension", "dimaag fry"}


def _masc_verb(v: str) -> str:
    return v.replace("rahi", "raha").replace("gayi", "gaya")


def _sub(sentence: str, m: re.Match, repl: str) -> str:
    out = sentence[:m.start()] + repl + sentence[m.end():]
    if m.start() == 0 and repl:
        out = repl[0].upper() + repl[1:] + sentence[m.end():]
    return out


def build_edits(sentence: str, slot_type: str):
    """-> list of (slang, edited_sentence) - every grammatically-agreed candidate edit for this slot type."""
    outs = []
    low = sentence.lower()
    if slot_type == "praise":
        m = PRAISE_RE.search(sentence)
        if m and not (m.start() == 0 and sentence[m.end():m.end() + 1] == ","):
            for s in POOLS["praise"]:
                if s not in low:
                    outs.append((s, _sub(sentence, m, f"ekdum {s}")))
    elif slot_type == "criticism":
        m = CRIT_RE.search(sentence)
        if m and not re.match(r"\s*(maan|mat)", sentence[m.end():]):
            for s in POOLS["criticism"]:
                if s not in low:
                    outs.append((s, _sub(sentence, m, f"ekdum {s}")))
    elif slot_type == "trouble":
        m = TROUBLE_RE.search(sentence)
        if m:
            for s in POOLS["trouble"]:
                if s in low:
                    continue
                size = m.group(1)
                if size or m.group(2).lower() == "mushkil situation":
                    adj = "badi" if NOUN_GENDER.get(s) == "f" else "bada"
                    outs.append((s, _sub(sentence, m, f"{adj} {s}")))
                else:
                    outs.append((s, _sub(sentence, m, s)))
    elif slot_type == "ease":
        m = EASE_RE.search(sentence)
        if m:
            for s in POOLS["ease"]:
                if s not in low:
                    outs.append((s, _sub(sentence, m, s)))
    elif slot_type == "worry":
        m = WORRY_RE.search(sentence)
        if m:
            verb = m.group(3)
            for s in POOLS["worry"]:
                if s in low:
                    continue
                if s in WORRY_NOUN:
                    v = verb if s == "full tension" else _masc_verb(verb)
                    if v == "hai":
                        v = "ho raha hai" if s == "dimaag fry" else "ho rahi hai"
                    outs.append((s, _sub(sentence, m, f"{s} {v}")))
                else:
                    outs.append((s, _sub(sentence, m, s if s.endswith(("hai", "gayi", "gaye")) else s + " hai")))
        m = FEAR_RE.search(sentence)
        if m:
            for s in POOLS["worry"]:
                if s in low:
                    continue
                if s in WORRY_NOUN:
                    v = "ho rahi hai" if s == "full tension" else "ho raha hai"
                    outs.append((s, _sub(sentence, m, f"{s} {v}")))
                else:
                    outs.append((s, _sub(sentence, m, s if s.endswith(("hai", "gayi", "gaye")) else s + " hai")))
    return outs


def make_shortlist(gold: str, slot_type: str, line: str, k=None):
    pool = [x for x in POOLS[slot_type] if x != gold and x not in line.lower()]
    rng.shuffle(pool)
    k = k or rng.choice([3, 4, 4, 5])
    sl = pool[:k] + [gold]
    rng.shuffle(sl)
    return sl


JUDGE = """You are a strict native Hinglish speaker (romanized Hindi-English). For each item an ORIGINAL sentence was edited by replacing
one opinion/feeling phrase with a slang expression. Rate the EDITED sentence 1-5:
5 = a real young person would say exactly this, grammar and mood are right, meaning unchanged
4 = natural enough, tiny awkwardness
3 = understandable but awkward / slightly wrong register or grammar
2 = unnatural or the slang does not suit this sentence
1 = wrong meaning, wrong mood or nonsense
Return ONLY a JSON object mapping the item number to the integer rating, e.g. {{"1":5,"2":3}}.

{block}"""


def judge(items, batch=12):
    """items: list of (original, edited) -> list of int ratings (0 = judge failed)."""
    out = [0] * len(items)
    for s in range(0, len(items), batch):
        chunk = items[s:s + batch]
        block = "\n".join(f'{i+1}. ORIGINAL: {o}\n   EDITED: {e}' for i, (o, e) in enumerate(chunk))
        try:
            txt = ask(JUDGE.format(block=block), max_tokens=400, temperature=0.0, task="dialogue")
            d = json.loads(re.search(r"\{.*\}", txt, re.DOTALL).group())
            for i in range(len(chunk)):
                out[s + i] = int(d.get(str(i + 1), 0))
        except Exception as e:
            print("   judge batch failed:", str(e)[:80])
    return out


def stage_rows(budget: float):
    sents = read_jsonl(OUT + "sentences.jsonl")
    cands, abst = [], []
    stats = collections.Counter()
    for r in sents:
        text = r["text"]
        if r["kind"] == "slot":
            slots = slot_candidates(text)
            types = {s["type"] for s in slots}
            if r["sub"] not in types:
                stats["slot-not-detected"] += 1
                continue
            edits = build_edits(text, r["sub"])
            good = []
            for slang, ed in edits:
                ok, why = rule_gate(text, ed, slang)
                if not ok:
                    stats["gate:" + why] += 1
                    continue
                if not keeps_meaning(text, ed) or not newly_inserted(text, ed, slang):
                    stats["drift/new"] += 1
                    continue
                good.append((slang, ed))
            if not good:
                stats["no-valid-edit"] += 1
                continue
            rng.shuffle(good)
            # up to 2 DIFFERENT gold slangs per sentence (teaches variety), balanced by usage later
            for slang, ed in good[:2]:
                cands.append({"text": text, "slot": r["sub"], "domain": r["domain"], "gold": slang, "edited": ed})
        else:
            abst.append(r)
    print("candidate gold edits:", len(cands), "| abstain sentences:", len(abst), "|", dict(stats))
    # ---- LLM naturalness judge (cost-capped) ----
    ratings = []
    t0 = time.time()
    for s in range(0, len(cands), 120):
        if spent() > budget:
            print(f"BUDGET reached (${spent():.3f}) during judging - the rest is dropped")
            break
        part = cands[s:s + 120]
        ratings += judge([(c["text"], c["edited"]) for c in part])
        print(f"  judged {len(ratings)}/{len(cands)}  ${spent():.3f}  {time.time()-t0:.0f}s", flush=True)
    kept = []
    dist = collections.Counter()
    for c, rt in zip(cands, ratings):
        dist[rt] += 1
        if rt >= 4:
            c["rating"] = rt
            kept.append(c)
    print("judge ratings:", dict(sorted(dist.items())), "| kept", len(kept))
    per_slang = collections.Counter(c["gold"] for c in kept)
    print("kept per slang:", dict(per_slang.most_common()))
    write_jsonl(OUT + "gold_edits.jsonl", kept)
    write_jsonl(OUT + "abstain_sentences.jsonl", abst)
    print(L.cost_summary())


# ------------------------------------------------------------------------------------------------------ stage 3
def stage_assemble():
    D4 = "kaggle_v4/"
    kept = read_jsonl(OUT + "gold_edits.jsonl")
    abst = read_jsonl(OUT + "abstain_sentences.jsonl")
    # balance: no single slang may exceed 9% of the edit rows (keeps the picker from learning a favourite)
    # worry idioms ("dimaag fry ho raha hai", "fatti padi hai") only read right when the SPEAKER is the one worried
    # ("mujhe / mera / hum ..."); third-person "Priya ko ... chinta" + "dimaag fry ho raha hai" is broken Hindi.
    first_person = re.compile(r"\b(mujhe|main|mera|meri|mere|hum|humein|humara|hamara|hamari|humari|maine|mujhko)\b", re.I)
    n0 = len(kept)
    kept = [c for c in kept if c["slot"] != "worry" or first_person.search(c["text"])]
    print(f"worry edits without a first-person experiencer removed: {n0 - len(kept)}")
    cap = max(25, int(0.09 * len(kept)))
    by = collections.defaultdict(list)
    for c in kept:
        by[c["gold"]].append(c)
    balanced = []
    for s, lst in by.items():
        rng.shuffle(lst)
        balanced += lst[:cap]
    train_e, eval_e = [], []
    for c in balanced:
        (eval_e if c["domain"] in HELD_OUT_DOMAINS else train_e).append(c)
    # random 5% of the training domains -> val
    rng.shuffle(train_e)
    nval = max(40, int(0.06 * len(train_e)))
    val_e, train_e = train_e[:nval], train_e[nval:]
    rows_train, rows_val, eval_rows = [], [], []

    def edit_row(c):
        sl = make_shortlist(c["gold"], c["slot"], c["text"])
        return make_example(c["text"], sl, c["edited"]), sl

    for c in train_e:
        rows_train.append(edit_row(c)[0])
    for c in val_e:
        rows_val.append(edit_row(c)[0])
    for i, c in enumerate(eval_e):
        ex, sl = edit_row(c)
        eval_rows.append({"id": f"v5s{i:04d}", "src": "synth", "slot": c["slot"], "shortlist": sl, "line": c["text"],
                          "expect": "edit", "gold": c["edited"]})
    # abstain rows: shortlist of the slot kind the sentence would have, or a filler list; target UNCHANGED
    a_train, a_eval = [], []
    for r in abst:
        (a_eval if r["domain"] in HELD_OUT_DOMAINS else a_train).append(r)
    # abstain mix: the HARD kinds (an opinion word is present but editing it would be wrong) matter most; plain facts/questions
    # never reach the mood route at runtime, so they stay few. Keeps the overall abstain share near a quarter of the data.
    CAP = {"negation": 170, "respect": 170, "elder": 100, "serious": 100, "facts": 60, "question": 60}
    rng.shuffle(a_train)
    per_kind = collections.Counter()
    capped = []
    for r in a_train:
        if per_kind[r["sub"]] < CAP.get(r["sub"], 60):
            per_kind[r["sub"]] += 1
            capped.append(r)
    a_train = capped
    print("abstain rows kept per kind:", dict(per_kind))
    na_val = max(10, int(0.06 * len(a_train)))

    def abstain_row(r):
        # use a shortlist of the slot type that an opinion word in the sentence would trigger, else a random mood pool
        t = rng.choice(list(POOLS))
        for s in slot_candidates(r["text"]) or []:
            t = s["type"]
            break
        else:
            low = r["text"].lower()
            for typ, words in (("praise", ("accha", "achha", "achhi", "badhiya")), ("criticism", ("kharab", "bura", "bekaar")),
                               ("worry", ("chinta", "dar lag", "ghabrahat")), ("trouble", ("problem", "dikkat", "mushkil"))):
                if any(w in low for w in words):
                    t = typ
                    break
        if t not in POOLS:
            t = "praise"                                  # e.g. the "predicate" slot (sahi / theek) uses the praise pool
        sl = rng.sample([x for x in POOLS[t] if x not in r["text"].lower()], k=min(len(POOLS[t]), rng.choice([3, 4, 5])))
        return make_example(r["text"], sl, r["text"]), sl, t

    for r in a_train[na_val:]:
        rows_train.append(abstain_row(r)[0])
    for r in a_train[:na_val]:
        rows_val.append(abstain_row(r)[0])
    for i, r in enumerate(a_eval):
        ex, sl, t = abstain_row(r)
        eval_rows.append({"id": f"v5a{i:04d}", "src": "synth", "slot": r["sub"], "shortlist": sl, "line": r["text"], "expect": "abstain"})

    # ---- merge with v4 rows (fillers, formal abstains, older edit patterns) ----
    v4_train, v4_val = read_jsonl(D4 + "lora_train.jsonl"), read_jsonl(D4 + "lora_val.jsonl")

    def row_ok(ex):
        """v4 rows are kept only if the CURRENT runtime contract accepts them: an abstain stays unchanged; an edit must insert a
        shortlist slang that the rule gate + drift guard accept. (~38% of v4 edit rows taught things the gate now rejects:
        content-word swaps, tail appends, 'Boss'/'kasam se' with rewrites. Training on them wastes accepted-edit capacity.)"""
        try:
            m = ex["messages"]
            sl, line = re.search(r"Slangs: (.*)\nLine: (.*)$", m[1]["content"], re.S).groups()
            sl = sl.split(", ")
            tgt = m[2]["content"].strip()
            if not (2 <= len(sl) <= 8) or len({x.lower() for x in sl}) != len(sl):
                return False
            if norm(tgt) == norm(line):
                return True
            used = next((x for x in sl if x.lower() in tgt.lower() and x.lower() not in line.lower()), None)
            return bool(used) and keeps_meaning(line, tgt) and rule_gate(line, tgt, used)[0]
        except Exception:
            return False

    n4 = (len(v4_train), len(v4_val))
    v4_train, v4_val = [r for r in v4_train if row_ok(r)], [r for r in v4_val if row_ok(r)]
    print(f"v4 rows kept by the current runtime contract: train {len(v4_train)}/{n4[0]} | val {len(v4_val)}/{n4[1]}")
    all_train = v4_train + rows_train
    all_val = v4_val + rows_val
    rng.shuffle(all_train)
    # ---- real-pipeline held-out rows: sentences from saved Agent-3 traces (never used for training) ----
    real = real_pipeline_eval()
    # ---- leak check: no eval sentence may appear (even near-duplicate) in train / val ----
    def key(s): return re.sub(r"[^a-z ]", "", s.lower()).strip()
    def line_of(ex): return ex["messages"][1]["content"].split("\nLine: ", 1)[1]
    trainset = [set(key(line_of(e)).split()) for e in all_train + all_val]
    trainexact = {key(line_of(e)) for e in all_train + all_val}
    leaks = 0
    keep_eval = []
    for r in eval_rows + real:
        k = key(r["line"])
        toks = set(k.split())
        if k in trainexact or any(len(toks & t) / max(1, len(toks | t)) >= 0.8 for t in trainset[:: 1] if t):
            leaks += 1
            continue
        keep_eval.append(r)
    print("eval rows removed for leakage:", leaks)
    os.makedirs("kaggle_v5", exist_ok=True)
    write_jsonl("kaggle_v5/lora_train.jsonl", all_train)
    write_jsonl("kaggle_v5/lora_val.jsonl", all_val)
    ev_synth = [r for r in keep_eval if r["src"] == "synth"]
    ev_real = [r for r in keep_eval if r["src"] == "real"]
    write_jsonl("kaggle_v5/eval_slots.jsonl", ev_synth)               # new held-out slot sentences (held-out domains)
    write_jsonl("kaggle_v5/eval_realv5.jsonl", ev_real)                # real Agent-3 sentences from saved runs
    for name in ("eval_real.jsonl", "eval_synth.jsonl", "eval_filler.jsonl"):   # the v4 eval sets, for before/after
        write_jsonl("kaggle_v5/" + name, read_jsonl(D4 + name))
    print(f"kaggle_v5: train {len(all_train)} (v4 {len(v4_train)} + new {len(rows_train)}) | val {len(all_val)} | "
          f"eval slots {len(ev_synth)} | eval realv5 {len(ev_real)}")
    rep = {"train": len(all_train), "val": len(all_val), "new_edit_rows": len(train_e), "new_abstain_rows": len(a_train) - na_val,
           "eval_slots": len(ev_synth), "eval_real": len(ev_real), "leak_removed": leaks,
           "gold_per_slang": dict(collections.Counter(c["gold"] for c in balanced).most_common()),
           "domains_heldout": sorted(HELD_OUT_DOMAINS)}
    json.dump(rep, open(OUT + "build_report.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)


def real_pipeline_eval():
    """Real Agent-3 sentences from the saved traces -> eval rows (expectation = what the runtime slot logic says)."""
    import glob
    sents, seen = [], set()
    for f in sorted(glob.glob("outputs/*trace*.json")):
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
            pool = [x for x in POOLS[typ] if x not in s.lower()]
            sl = rng.sample(pool, k=min(len(pool), 5))
            rows.append({"id": f"v5r{i:04d}", "src": "real", "slot": typ, "shortlist": sl, "line": s, "expect": "edit"})
        else:
            typ = rng.choice(list(POOLS))
            sl = rng.sample(POOLS[typ], k=min(4, len(POOLS[typ])))
            rows.append({"id": f"v5r{i:04d}", "src": "real", "slot": "none", "shortlist": sl, "line": s, "expect": "abstain"})
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sentences", "rows", "assemble"])
    ap.add_argument("--budget", type=float, default=2.0, help="USD cap for THIS run (provider-reported cost)")
    a = ap.parse_args()
    config.LLM_COST_CAP_USD = max(a.budget + 0.5, 1.0)          # the llm_client's own hard stop
    if a.stage == "sentences":
        stage_sentences(a.budget)
    elif a.stage == "rows":
        stage_rows(a.budget)
    else:
        stage_assemble()
