"""
data_prep/build_v3_dataset.py
Assemble the v3 (on-distribution) LoRA training set from the judged teacher labels.

Inputs  : data/felix_dataset/v3/judged.jsonl        (train scenes, judge-filtered)
          data/felix_dataset/v3/judged_eval.jsonl   (held-out synthetic scenes)       [optional]
          data/felix_dataset/v3/real_judged.jsonl   (REAL pipeline sentences, eval only) [optional]
          data/felix_dataset/final_dataset.csv      (old short gold rows, small share)
Outputs : data/felix_dataset/v3/lora_train.jsonl / lora_val.jsonl   (prompt v2 chat format)
          data/felix_dataset/v3/eval_synth.jsonl / eval_real.jsonl  (for before/after comparison)
          data/felix_dataset/v3/build_report.json

Prompt format is UNCHANGED (v2: shortlist-pick + abstain), so Agent 4 and the adapter loader need no
code change; only the data distribution changed.

  python -m data_prep.build_v3_dataset
"""
from __future__ import annotations
import argparse, collections, csv, json, os, random, re
from data_prep.build_lora_dataset import make_example, SYSTEM_V2, USER_V2   # single source of truth
from data_prep.slang_bank_v3 import BANK, TONE_NAME
from data_prep.v3_common import read_jsonl, write_jsonl, norm, keeps_meaning, newly_inserted
from data_prep.label_v3 import has_sentiment

D = "data/felix_dataset/v3/"
FILLERS = {s for s, v in BANK.items() if v[0] == "F"}
BY_TONE = {t: sorted(s for s, v in BANK.items() if v[0] == t) for t in "PNF"}
K_CHOICES = [3, 4, 4, 5, 5, 6, 6, 7]


def tone_of(slang):
    return BANK[slang.lower()][0]


def build_shortlist(sentence, true_slang, also_ok, rng, same_tone_only=True):
    """true slang + plausible companions (teacher's also_ok first) + same-tone fillers-in-moderation."""
    k = rng.choice(K_CHOICES)
    low = sentence.lower()
    sl = [true_slang]
    for a in also_ok:
        if len(sl) >= k:
            break
        if a.lower() in BANK and a not in sl and a.lower() not in low:
            sl.append(a)
    tone = tone_of(true_slang)
    pool = [s for s in BY_TONE[tone if tone != "F" else rng.choice("PF")]
            if s not in sl and s not in low and (s not in FILLERS or rng.random() < 0.3)]
    rng.shuffle(pool)
    sl += pool[: max(0, k - len(sl))]
    rng.shuffle(sl)
    return sl


def opposite_shortlist(sentence, slang_tone, rng):
    opp = "N" if slang_tone == "P" else "P"
    low = sentence.lower()
    pool = [s for s in BY_TONE[opp] if s not in low]
    return rng.sample(pool, min(rng.choice([3, 4, 5, 6]), len(pool)))


GENERIC_POS = {"mast", "solid", "badiya", "kamaal", "zabardast", "sorted", "kadak", "sahi hai", "top class",
               "on point", "number one", "ek number", "lajawaab", "chutki mein", "sab set hai"}
EMOTIVE = sorted(s for s in BANK if s not in FILLERS and s not in GENERIC_POS)


def none_shortlist(sentence, rng):
    """Calm / informational sentence + strongly emotive slang = verifiably no fit (no fillers, no
    generic praise words that could be argued to fit anything)."""
    low = sentence.lower()
    pool = [s for s in EMOTIVE if s not in low]
    return rng.sample(pool, rng.choice([4, 5, 6]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--old-rows", type=int, default=800, help="old short gold rows mixed in")
    ap.add_argument("--per-slang-cap", type=int, default=45)
    ap.add_argument("--filler-share", type=float, default=0.10)
    ap.add_argument("--opp-share", type=float, default=0.07, help="opposite-tone abstain rows (share of picks)")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--judged", default="judged.jsonl,pilot2_j.jsonl,judged2.jsonl")
    ap.add_argument("--holdout-from", type=int, default=240,
                    help="judged rows from synthetic scenes >= this id go to eval_synth, never to training")
    ap.add_argument("--real-file", default="real_labeled.jsonl")
    ap.add_argument("--none-share", type=float, default=0.08, help="teacher-NONE (calm sentence) abstain rows")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    judged = []
    for fn in args.judged.split(","):
        judged += read_jsonl(D + fn.strip())
    def sid_num(r):
        m = re.match(r"v3s(\d+)", r.get("scene_id", ""))
        return int(m.group(1)) if m else -1
    holdout_rows = [r for r in judged if sid_num(r) >= args.holdout_from]
    judged = [r for r in judged if sid_num(r) < args.holdout_from]
    picks, nones, seen_s = [], [], set()
    for r in judged:
        s = r["clause"]
        if norm(s) in seen_s:
            continue
        seen_s.add(norm(s))
        if r["label"] == "EDIT":
            if r.get("keep") and r["slang"].lower() in BANK:
                picks.append((s, r["slang"].lower(), r["edit"], r.get("also_ok", []), r["judge"]["score"], r["scene_id"]))
            for ak in ("alt", "alt2"):
                a = r.get(ak)
                if a and a.get("keep") and a["slang"].lower() in BANK:
                    picks.append((s, a["slang"].lower(), a["edit"], [r["slang"]] + r.get("also_ok", []),
                                  a["judge"]["score"], r["scene_id"]))
        elif r["label"] == "NONE" and not has_sentiment(s) and len(s.split()) >= 8:
            nones.append((s, r["scene_id"]))
    rng.shuffle(nones)
    print(f"judged: {len(picks)} pick rows, {len(nones)} confirmed NONE")

    # --- diversity controls: cap per slang (keep best-scored), cap fillers ---
    picks.sort(key=lambda x: -x[4])
    per, kept = collections.Counter(), []
    for p in picks:
        sl = p[1].lower()
        if per[sl] >= args.per_slang_cap:
            continue
        per[sl] += 1
        kept.append(p)
    fill = [p for p in kept if p[1].lower() in FILLERS]
    max_fill = int(args.filler_share * len(kept))
    drop = set(id(p) for p in fill[max_fill:])
    kept = [p for p in kept if id(p) not in drop]
    # dedupe (sentence, slang)
    seen, final = set(), []
    for p in kept:
        k = (norm(p[0]), p[1].lower())
        if k not in seen:
            seen.add(k); final.append(p)
    picks = final
    print(f"after caps: {len(picks)} pick rows over {len({p[1].lower() for p in picks})} distinct slangs")

    # --- examples (group by source sentence so a sentence never straddles train/val) ---
    groups = collections.defaultdict(list)
    for s, sl, edit, also, score, sid in picks:
        groups[norm(s)].append(make_example(s, build_shortlist(s, sl, also, rng), edit))
    # opposite-tone abstain rows
    n_opp = int(args.opp_share * len(picks))
    cand = [p for p in picks if tone_of(p[1]) in "PN"]
    rng.shuffle(cand)
    for s, sl, edit, also, score, sid in cand[:n_opp]:
        groups[norm(s)].append(make_example(s, opposite_shortlist(s, tone_of(sl), rng), s))
    # content-aware (teacher+judge confirmed) NONE rows
    nones = nones[: int(args.none_share * len(picks))]
    for s, sid in nones:
        groups[norm(s)].append(make_example(s, none_shortlist(s, rng), s))
    n_none = len(nones)

    # --- old short gold rows (keeps the short-sentence skill; capped, bank slangs only) ---
    old, per_old = [], collections.Counter()
    with open("data/felix_dataset/final_dataset.csv", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    rng.shuffle(rows)
    for r in rows:
        n, o, sl = (r.get("neutral") or "").strip(), (r.get("output") or "").strip(), (r.get("slangs") or "").strip().lower()
        if sl not in BANK or per_old[sl] >= 8 or not (5 <= len(n.split()) <= 16):
            continue
        if not (newly_inserted(n, o, sl) and keeps_meaning(n, o)):
            continue
        per_old[sl] += 1
        old.append(make_example(n, build_shortlist(n, sl, [], rng), o))
        if len(old) >= args.old_rows:
            break
    for i, ex in enumerate(old):
        groups[f"old{i}"].append(ex)

    keys = sorted(groups)
    rng.shuffle(keys)
    n_val = max(1, int(args.val_frac * len(keys)))
    val = [ex for k in keys[:n_val] for ex in groups[k]]
    train = [ex for k in keys[n_val:] for ex in groups[k]]
    rng.shuffle(train)
    write_jsonl(D + "lora_train.jsonl", train)
    write_jsonl(D + "lora_val.jsonl", val)

    def is_abs(ex):
        u = ex["messages"][1]["content"]
        return norm(ex["messages"][2]["content"]) == norm(u.split("\nLine: ", 1)[1])
    n_abs = sum(is_abs(e) for e in train)

    # --- evaluation sets ---
    def eval_rows(src, tag, need_keep=True):
        out = []
        for r in (read_jsonl(src) if isinstance(src, str) else src):
            s = r["clause"]
            if r["label"] == "EDIT" and (r.get("keep") or not need_keep) and r["slang"].lower() in BANK:
                sl = build_shortlist(s, r["slang"].lower(), [a for a in r.get("also_ok", []) if a.lower() in BANK], rng)
                out.append({"id": f"{tag}{len(out)}", "src": tag, "story": r.get("scene_id"), "line": s,
                            "shortlist": sl, "expect": "edit", "teacher_slang": r["slang"],
                            "teacher_edit": r["edit"]})
            elif r["label"] == "NONE" and not has_sentiment(s) and len(s.split()) >= 8:
                out.append({"id": f"{tag}{len(out)}", "src": tag, "story": r.get("scene_id"), "line": s,
                            "shortlist": none_shortlist(s, rng), "expect": "abstain"})
        return out
    ev_s = eval_rows(holdout_rows, "synth", need_keep=True)
    ev_r = (eval_rows(D + args.real_file, "real", need_keep=False)
            if os.path.exists(D + args.real_file) else [])
    write_jsonl(D + "eval_synth.jsonl", ev_s)
    write_jsonl(D + "eval_real.jsonl", ev_r)

    # --- sanity: every training prompt must be the v2 prompt ---
    assert all(e["messages"][0]["content"] == SYSTEM_V2 for e in train[:200])
    lens = [len(e["messages"][1]["content"].split("\nLine: ", 1)[1].split()) for e in train]
    slang_use = collections.Counter()
    for e in train:
        if not is_abs(e):
            out = e["messages"][2]["content"].lower()
            sl = e["messages"][1]["content"].split("\nSlangs: ", 1)[1].split("\nLine:", 1)[0].split(", ")
            hit = [s for s in sl if s.lower() in out]
            if hit:
                slang_use[hit[0].lower()] += 1
    rep = {"train": len(train), "val": len(val), "abstain_rows": n_abs,
           "abstain_share": round(n_abs / max(len(train), 1), 3), "content_aware_none": n_none,
           "old_gold_rows": len(old), "mean_words": round(sum(lens) / len(lens), 1),
           "share_ge14_words": round(sum(l >= 14 for l in lens) / len(lens), 3),
           "distinct_slangs_used": len(slang_use), "top5_slangs": slang_use.most_common(5),
           "eval_synth": len(ev_s), "eval_real": len(ev_r)}
    json.dump(rep, open(D + "build_report.json", "w"), indent=2)
    print(json.dumps(rep, indent=2))
    print("wrote", D + "lora_train.jsonl", "and", D + "lora_val.jsonl")


if __name__ == "__main__":
    main()
