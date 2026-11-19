"""
data_prep/v7_assemble.py  -  turn the accepted teacher edits + the "keep as is" cases into the v7 training / dev / test files.

    python -m data_prep.v7_assemble [--keep-share 0.4] [--hint-prob 0.6] [--max-word-share 0.06]

Rows (chat format, see data_prep.v7_format):
  edit          the teacher's edit passed the deterministic filter AND the cross-family judge
  keep_teacher  the teacher looked at the line and returned it unchanged (valid: "nothing fits naturally")
  keep_forced   the situation forbids slang (elder / respectful / traditional / grief / thanks ...): the answer is the line unchanged, no teacher call needed
Teacher edits that FAILED a check are dropped, never turned into "keep" rows.

Balance: the share of "keep" rows is capped (default 40%) so the model does not learn to always abstain; every added slang word is capped to a share of the edits (default 6%) so one
word ("mast") cannot dominate; filler-only edits ("yaar", "arre") are capped at 15% of the edits. Whole STORIES go to dev / test (about 10% each): the benchmark measures generalisation to
unseen stories, never memorisation. dev / test keep every row (no balancing) and test.jsonl carries the full context for the benchmark.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.getcwd())
from data_prep import v7_filter, v7_teacher
from data_prep.v7_format import make_chat

D = "data/v7"


def _read(p):
    return [json.loads(x) for x in open(p, encoding="utf-8")] if os.path.exists(p) else []


def split_of(story: str, dev: float = 0.1, test: float = 0.1) -> str:
    """Deterministic split by story id."""
    h = int(hashlib.md5(story.encode()).hexdigest(), 16) % 1000 / 1000
    return "test" if h < test else ("dev" if h < test + dev else "train")


def added_key(line: str, edited: str) -> str:
    """The new words of an edit, e.g. 'zabardast' or 'hawa mein' (used to cap how often one slang item appears)."""
    lt = set(v7_filter._toks(line))
    new = [w for w in v7_filter._toks(edited) if w not in lt]
    return " ".join(new[:3]) or "(reorder)"


def with_hints(c: dict, p: float) -> dict:
    """Optional hints: shown for a random p-share of the rows, 4 sampled from everything the pool offers for the line (the model must not depend on them)."""
    pool = c.get("hints_all") or []
    r = random.Random("h" + c["id"])
    return dict(c, hints=(r.sample(pool, min(4, len(pool))) if pool and r.random() < p else []))


def assemble(keep_share: float = 0.4, hint_prob: float = 0.6, max_word_share: float = 0.06, filler_share: float = 0.15, seed: int = 7) -> dict:
    ctx = {c["id"]: c for c in _read(f"{D}/contexts.jsonl")}
    raw = {r["id"]: r for r in _read(f"{D}/teacher_raw.jsonl")}
    acc = v7_teacher.accepted()
    rng = random.Random(seed)
    rows = {"train": [], "dev": [], "test": []}
    pools = {"edit": [], "keep_teacher": [], "keep_forced": []}
    for cid, c in ctx.items():
        sp = split_of(c["story"])
        if cid in raw:
            r = raw[cid]
            cc = dict(c, max_edits=r["max_edits"])
            v = v7_filter.check(cc, r["edited"])
            if not v["changed"]:
                kind, target = "keep_teacher", c["line"]
            elif cid in acc:
                kind, target = "edit", r["edited"].strip()
            else:
                continue
        elif not c.get("allows_edit", True):
            cc = dict(c, max_edits=rng.choice([1, 2, 2, 3]))
            kind, target = "keep_forced", c["line"]
        else:
            continue
        cc = with_hints(cc, hint_prob)
        row = {"id": cid, "story": c["story"], "kind": kind, "line": c["line"], "target": target, "max_edits": cc["max_edits"], **make_chat(cc, target)}
        if sp == "train":
            pools[kind].append((row, cc))
        else:
            rows[sp].append(row)
            if sp == "test":
                row["ctx"] = cc
    # ---- balance the training set
    edits = pools["edit"]
    rng.shuffle(edits)
    n_total_edits = len(edits)
    cap_word = max(4, int(max_word_share * max(n_total_edits, 1)))
    cap_fill = int(filler_share * max(n_total_edits, 1))
    seen, fill, kept = Counter(), 0, []
    for row, cc in edits:
        k = added_key(row["line"], row["target"])
        fo = v7_filter.is_filler_only(cc, row["target"])
        if seen[k] >= cap_word or (fo and fill >= cap_fill):
            continue
        seen[k] += 1
        fill += fo
        kept.append(row)
    n_keep = int(round(len(kept) * keep_share / (1 - keep_share)))
    kt, kf = pools["keep_teacher"], pools["keep_forced"]
    rng.shuffle(kt)
    rng.shuffle(kf)
    half = n_keep // 2
    keeps = [r for r, _ in kf[:half]] + [r for r, _ in kt[:n_keep - min(half, len(kf))]]
    train = kept + keeps[:n_keep]
    rng.shuffle(train)
    rows["train"] = train
    os.makedirs(D, exist_ok=True)
    for sp, rs in rows.items():
        with open(f"{D}/{sp}.jsonl", "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rep = {sp: {"rows": len(rs), **dict(Counter(r["kind"] for r in rs))} for sp, rs in rows.items()}
    rep["edits_before_caps"] = n_total_edits
    rep["distinct_added"] = len(seen)
    rep["top_added"] = seen.most_common(12)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-share", type=float, default=0.4)
    ap.add_argument("--hint-prob", type=float, default=0.6)
    ap.add_argument("--max-word-share", type=float, default=0.06)
    a = ap.parse_args()
    rep = assemble(a.keep_share, a.hint_prob, a.max_word_share)
    for k, v in rep.items():
        print(k, v)


if __name__ == "__main__":
    main()
