"""
data_prep/augment_slang_dataset.py  (hardened v3)
Grow the slang dataset with HIGH-QUALITY validated pairs for the "platinum" set.

Upgrades over the naive version (which over-used 'bindaas', broke grammar, mislabeled):
  1. VARIANT mode (default, ~70%): take a REAL neutral sentence from the clean dataset
     and insert a CHOSEN slang into it. The neutral is guaranteed grammatical, and we
     control which slang goes in -> no garbling, no runaway 'bindaas'.
  2. NOVEL mode (~30%): invent a new neutral+output for a context, but the target slang
     is still chosen by US from the lexicon (not the model) -> coverage without drift.
  3. ANTI-CONCENTRATION: target slang is drawn rarity-weighted from the 576-phrase lexicon
     with a hard per-slang cap, so the dataset spreads across the vocabulary.
  4. LLM-JUDGE: every candidate is scored by a second model call on
     {intent_match, grammatical, meaning_preserved, is_real_slang}. All must pass.
  5. Structural + semantic-preservation checks, with a transparent quarantine file.

Usage:
  python -m data_prep.augment_slang_dataset --n 40                       # hardened pilot
  python -m data_prep.augment_slang_dataset --n 4000 --out data/felix_dataset/augmented_v3.csv
"""
from __future__ import annotations
import argparse, csv, itertools, json, os, random, re, time
from collections import defaultdict

from groq import Groq, RateLimitError
from dotenv import load_dotenv
import config

# Free tier gives 1000 requests/DAY *per model*. Round-robin across both gpt-oss
# models so we draw from two independent daily buckets (~2x throughput) and can
# fail over instantly when one bucket is rate-limited. Both accept
# reasoning_effort="low"; qwen does not, so it is intentionally excluded.
AUG_MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]
_model_cycle = itertools.cycle(AUG_MODELS)

load_dotenv()

try:
    from rapidfuzz import fuzz
    def sim(a, b): return fuzz.token_sort_ratio(a, b)
except Exception:
    import difflib
    def sim(a, b): return 100 * difflib.SequenceMatcher(None, a, b).ratio()

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
CONTEXTS = ["college lecture","hostel room","cricket match","office meeting","family dinner",
    "traffic jam","exam results","breakup","street food","online gaming","festival shopping",
    "startup pitch","gym workout","job interview","road trip","wedding function","power cut",
    "phone repair","coffee shop","group project","missed train","food delivery late","new phone",
    "movie review","internship","placement season","chai break","bike breakdown","wifi down",
    "surprise party","old friends meetup"]

INSERT_PROMPT = """Insert the slang expression "{slang}" into this neutral Hinglish line so it sounds natural and casual, WITHOUT changing the meaning.
Return ONLY JSON: {{"output": "<the line with '{slang}' inserted naturally>"}}
Rules: romanized Roman script only; keep all facts; one line; 'output' must contain "{slang}".
Neutral line: {neutral}"""

NOVEL_PROMPT = """Write ONE new casual sentence in romanized HINGLISH (a natural mix of Hindi and English words in Roman script, the way young Indians actually text — NOT pure English) about "{context}", expressing the intent {control}. Then insert the slang "{slang}" into it naturally.
Return ONLY JSON: {{"neutral": "<sentence WITHOUT the slang>", "output": "<same sentence WITH '{slang}' inserted>"}}
Rules: MUST be Hinglish (include Hindi words like hai/nahi/kar/gaya/bahut/mera etc.), NOT all-English; romanized Roman script only; 6-18 words; meaning identical except the added slang."""

JUDGE_PROMPT = """You are a strict data-quality judge for a Hindi-Hinglish slang dataset.
Neutral: {neutral}
Slang output: {output}
Intended intent: {control}
Inserted slang: {slang}
Answer ONLY JSON with booleans:
{{"is_hinglish": <is the output genuinely Hinglish, i.e. a Hindi-English code-mix, NOT pure English?>,
  "grammatical": <is the slang output fluent, natural Hinglish?>,
  "meaning_preserved": <does output keep the neutral's meaning, only adding the slang?>,
  "intent_match": <does the output plausibly express {control}?>,
  "is_real_slang": <is "{slang}" genuine casual slang/filler here, not a normal word?>}}"""


def load_real(path):
    rows = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    return rows


def build_lexicon(rows):
    by_control, freq = defaultdict(set), defaultdict(int)
    for r in rows:
        c = (r.get("control") or "").strip().upper()
        for s in str(r.get("slangs") or "").split("|"):
            s = s.strip()
            if s:
                if c:
                    by_control[c].add(s)
                freq[s] += 1
    return {k: sorted(v) for k, v in by_control.items()}, freq


def _parse_json(content):
    raw = re.sub(r"```(?:json)?", "", content or "").strip()
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        try:
            return json.loads(re.sub(r",\s*}", "}", m.group()))
        except Exception:
            return None


def call_json(client, prompt, max_tokens=300):
    """Round-robin across AUG_MODELS. On a 429 (per-model daily/minute cap) try
    the next model immediately; only pause briefly if EVERY model is capped, so
    we never sit in the SDK's long retry backoff (client uses max_retries=0)."""
    for _ in range(len(AUG_MODELS)):
        model = next(_model_cycle)
        try:
            resp = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                temperature=0.7, max_tokens=max_tokens,
                reasoning_effort=getattr(config, "GROQ_REASONING_EFFORT", "low"),
                timeout=30)
        except RateLimitError:
            continue                     # this bucket is capped -> fail over to the other
        except Exception:
            time.sleep(2)                # transient network/API error -> brief cooldown
            return None
        return _parse_json(resp.choices[0].message.content)
    time.sleep(5)                        # all models rate-limited this round
    return None


def pick_slang(by_control, control, freq, run_used, cap):
    pool = by_control.get(control) or [s for v in by_control.values() for s in v]
    pool = [s for s in pool if run_used[s] < cap] or pool
    weights = [1.0 / (1 + 3 * run_used[s] + 0.1 * freq.get(s, 0)) for s in pool]
    return random.choices(pool, weights=weights, k=1)[0]


def _key(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def structural_ok(neutral, output, slang, existing_outputs, min_fuzz, existing_neutrals=None):
    if not (neutral and output and slang):
        return "empty"
    if DEVANAGARI.search(neutral) or DEVANAGARI.search(output):
        return "devanagari"
    if slang.lower() not in output.lower():
        return "slang_absent"
    if neutral.lower() == output.lower():
        return "no_change"
    if not (4 <= len(neutral.split()) <= 26):
        return "length"
    # dedup on the OUTPUT (the slang line) so multiple slang-variants of a real
    # neutral are allowed — that's how the base dataset is built.
    if _key(output) in existing_outputs:
        return "duplicate"
    # only NOVEL mode forbids reusing an existing neutral (new neutrals must be new)
    if existing_neutrals is not None and _key(neutral) in existing_neutrals:
        return "duplicate"
    stripped = re.sub(r"\s+", " ", re.sub(re.escape(slang), "", output, flags=re.IGNORECASE)).strip(" ,.!")
    if sim(neutral.lower(), stripped.lower()) < min_fuzz:
        return "semantic_drift"
    return "ok"


def flush_out(args, kept, rejected):
    """Write current kept/rejected to disk. Called periodically so a late
    crash loses at most one checkpoint interval instead of the whole run."""
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fields = ["neutral","output","control","function","positions","multi_allowed",
              "social_context","context","slangs","source"]
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(kept)
    os.replace(tmp, args.out)  # atomic: never leaves a half-written CSV
    if rejected:
        rtmp = args.quarantine + ".tmp"
        with open(rtmp, "w", encoding="utf-8", newline="") as f:
            cols = sorted({k for r in rejected for k in r})
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rejected)
        os.replace(rtmp, args.quarantine)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--src", default=config.FELIX_DATASET_CLEAN)
    ap.add_argument("--out", default="data/felix_dataset/augmented_v3.csv")
    ap.add_argument("--quarantine", default="data/felix_dataset/augment_v3_rejected.csv")
    ap.add_argument("--min-fuzz", type=int, default=70)
    ap.add_argument("--variant-frac", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-attempts", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    real = load_real(args.src)
    by_control, freq = build_lexicon(real)
    real_pairs = [(r["neutral"].strip(), (r.get("control") or "ASSERT_NEUTRAL").strip().upper())
                  for r in real if r.get("neutral")]
    existing_neutrals = {_key(n) for n, _ in real_pairs}
    existing_outputs = {_key(r.get("output") or "") for r in real}
    client = Groq(api_key=os.environ["GROQ_API_KEY"], timeout=30.0, max_retries=0)

    cap = max(3, args.n // 40)          # per-slang usage cap this run
    run_used = defaultdict(int)
    max_attempts = args.max_attempts or args.n * 6
    kept, rejected, attempts = [], [], 0
    reasons = defaultdict(int)

    # RESUME: if the output CSV already exists (from checkpoints of a prior run
    # that crashed/was killed), reload it so we continue instead of redoing work.
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                kept.append(r)
                existing_outputs.add(_key(r.get("output") or ""))
                s = (r.get("slangs") or "").strip()
                if s:
                    run_used[s] += 1
                if str(r.get("source", "")).endswith("novel"):
                    existing_neutrals.add(_key(r.get("neutral") or ""))
        if kept:
            print(f"RESUME: loaded {len(kept)} existing rows from {args.out}", flush=True)
    t0 = time.time()

    while len(kept) < args.n and attempts < max_attempts:
        attempts += 1
        if attempts % 40 == 0:   # liveness heartbeat even during reject/ratelimit streaks
            print(f"  [hb] attempts={attempts} kept={len(kept)} "
                  f"rate={len(kept)/max(time.time()-t0,1)*60:.1f}/min reasons={dict(reasons)}",
                  flush=True)
        mode = "variant" if random.random() < args.variant_frac else "novel"
        if mode == "variant":
            neutral, control = random.choice(real_pairs)
            context = ""
        else:
            control = random.choice(list(by_control.keys()))
            context = random.choice(CONTEXTS)
            neutral = None
        slang = pick_slang(by_control, control, freq, run_used, cap)

        try:
            if mode == "variant":
                obj = call_json(client, INSERT_PROMPT.format(slang=slang, neutral=neutral))
                if obj:
                    obj["neutral"] = neutral
            else:
                obj = call_json(client, NOVEL_PROMPT.format(context=context, control=control, slang=slang))
        except Exception:
            reasons["api_error"] += 1; time.sleep(2); continue
        if not obj:
            reasons["unparseable"] += 1; continue

        neutral_f = str(obj.get("neutral", "")).strip()
        output_f = str(obj.get("output", "")).strip()
        # variant mode reuses a real neutral by design -> don't forbid duplicate neutral
        neu_check = None if mode == "variant" else existing_neutrals
        r = structural_ok(neutral_f, output_f, slang, existing_outputs, args.min_fuzz, neu_check)
        if r != "ok":
            reasons[r] += 1
            rejected.append({"neutral": neutral_f, "output": output_f, "slang": slang,
                             "control": control, "_reason": r}); continue

        # LLM-judge
        try:
            j = call_json(client, JUDGE_PROMPT.format(neutral=neutral_f, output=output_f,
                                                      control=control, slang=slang), max_tokens=200)
        except Exception:
            j = None
        if not j or not all(bool(j.get(k)) for k in
                            ("is_hinglish", "grammatical", "meaning_preserved", "intent_match", "is_real_slang")):
            reasons["judge_reject"] += 1
            rejected.append({"neutral": neutral_f, "output": output_f, "slang": slang,
                             "control": control, "_reason": "judge", "_judge": json.dumps(j)}); continue

        existing_outputs.add(_key(output_f))
        if mode == "novel":
            existing_neutrals.add(_key(neutral_f))
        run_used[slang] += 1; reasons["ok"] += 1
        kept.append({"neutral": neutral_f, "output": output_f, "control": control,
                     "function": "", "positions": "", "multi_allowed": False,
                     "social_context": "", "context": context, "slangs": slang,
                     "source": f"groq_aug_v3_{mode}"})
        if len(kept) % 5 == 0:
            distinct = len({k["slangs"] for k in kept})
            print(f"  kept {len(kept)}/{args.n} (attempts {attempts}, "
                  f"accept {100*len(kept)/attempts:.0f}%, distinct slang {distinct})",
                  flush=True)
            flush_out(args, kept, rejected)  # checkpoint: crash loses <=5 rows

    flush_out(args, kept, rejected)

    print(f"\nKept {len(kept)} in {time.time()-t0:.0f}s ({attempts} attempts, "
          f"{100*len(kept)/max(attempts,1):.0f}% accept, "
          f"{len({k['slangs'] for k in kept})} distinct slang used).")
    print("reasons:", dict(reasons))
    print(f"Saved: {args.out}" + (f"  | rejected -> {args.quarantine}" if rejected else ""))


if __name__ == "__main__":
    main()
