"""
evaluation/calibration.py  -  the SMALL human calibration set that validates the automatic metrics ONCE, so no human has to rate the system again.

    python -m evaluation.calibration create  --out outputs/calibration --raters 3 [--n-real 90 --n-gold 25 --n-control 25]
    python -m evaluation.calibration analyze --dir outputs/calibration            (after the raters returned their CSVs)

WHY: an automatic metric is only evidence if it is shown to agree with people.  Instead of rating whole systems, humans rate ~140 (neutral, edited) line pairs
that mix three kinds of items, shuffled and blinded (opaque ids):
    real     edits the system actually produced on many stories (all ablation arms + fresh runs), de-duplicated      -> ecological validity
    gold     held-out gold edits (known good, written + gate-checked before)                                      \\ attention / reliability controls: a careful rater
    control  the same kind of edits deliberately corrupted (negation, wrong word, hallucinated clause, wrong mood)  /  accepts the gold ones and rejects the corrupted ones
Raters give two 1-5 ratings and a yes / no:  meaning_grammar (meaning, tense and grammar are right), naturalness_fit (natural AND the slang suits the situation),
acceptable (would you keep this line in a screenplay?).

ANALYZE reports: rater reliability (Krippendorff's alpha, control accuracy per rater), the human consensus, and for every automatic metric (the judge of each
model family, their ensemble, the rule audit) its agreement with the human consensus: Cohen's kappa with bootstrap confidence intervals, true-positive and
true-negative rates, split by item kind.  This is the "meta-evaluation" the panel asks for: it says which metric to trust, and how much.
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

load_dotenv()

FIELDS = ("meaning_grammar", "naturalness_fit", "acceptable")
COLS = ["item_id", "rater_id", "neutral_line", "edited_line", "meaning_grammar", "naturalness_fit", "acceptable", "comment"]

INSTRUCTIONS = """# Rating instructions (about 45 minutes)

You see a Hinglish line (ORIGINAL) and the same line after a program changed it (EDITED). The program may only swap one word / phrase for a slang or add a casual
filler like "yaar". Judge the EDITED line only against the ORIGINAL. Do not try to guess which system produced it; ratings are blind.

Fill three columns for every row (leave `comment` empty unless something is strange):

* **meaning_grammar** (1-5): are the meaning, tense, gender agreement and grammar of the EDITED line still right?
  5 = identical meaning and perfectly correct, 3 = small slip, 1 = meaning changed or ungrammatical.
* **naturalness_fit** (1-5): would a young Indian really say this, and does the slang suit what the speaker is talking about?
  5 = completely natural and fitting, 3 = understandable but a little forced, 1 = nonsense / forced / trivialises something serious.
* **acceptable** (Y / N): would you keep this edited line in a screenplay? If the edited line is identical to the original, answer Y.

Rules: rate alone, do not discuss with other raters until you have returned the sheet, do not look anything up from the system. If you cannot decide between
two numbers, pick the lower one. Some rows are deliberately broken or deliberately good - that is intended.
Return the same CSV file (keep `item_id` and `rater_id`; save as CSV UTF-8).
"""


def _real_pairs(max_per_story: int, rng):
    """Distinct changed (neutral, edited) line pairs of the system's own outputs, with the story they came from."""
    by_story = {}
    files = glob.glob("outputs/ablation_multi/*_trace_*.json") + glob.glob("outputs/ablation_final/*_trace_*.json") + glob.glob("outputs/val_*_trace.json") \
        + glob.glob("outputs/final_check*_trace.json")
    for f in files:
        if any(x in os.path.basename(f) for x in ("metrics_", "manifest", "summary")):
            continue
        try:
            t = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        story = re.sub(r"_(full|no_[a-z0-9_]+)\.json$", "", os.path.basename(f)).replace(".json", "")
        nm = {s["scene_id"]: s for s in t.get("neutral_script", [])}
        for sc in t.get("slang_script", []):
            from evaluation.compare import _parse_turns        # empty "Narrator:" turns dropped on both sides, so pairs line up
            n = [t["text"] for t in _parse_turns(nm.get(sc["scene_id"], {}).get("dialogue", ""))]
            s = [t["text"] for t in _parse_turns(sc["dialogue"])]
            for a, b in zip(n, s):
                if a != b:
                    by_story.setdefault(story, {})[(a, b)] = True
    out = []
    for story, d in by_story.items():
        items = list(d)
        rng.shuffle(items)
        out += [{"neutral": a, "edited": b, "kind": "real", "story": story} for a, b in items[:max_per_story]]
    return out


def create(out_dir: str, raters: int, n_real: int, n_gold: int, n_control: int, seed: int):
    from evaluation.eval_judge import build_sets
    rng = random.Random(seed)
    real = _real_pairs(max_per_story=max(6, n_real // 5), rng=rng)
    rng.shuffle(real)
    real = real[:n_real]
    sets = build_sets(400)
    rng.shuffle(sets)
    gold = [{"neutral": d["line"], "edited": d["good"], "kind": "gold", "story": "eval_slots"} for d in sets[:n_gold]]
    ctrl = []
    for d in sets[n_gold:n_gold + 4 * n_control]:
        if len(ctrl) >= n_control:
            break
        if d["corrupt"]:
            k = rng.choice([c for c in d["corrupt"] if c != "unrelated_line"] or list(d["corrupt"]))
            ctrl.append({"neutral": d["line"], "edited": d["corrupt"][k], "kind": "control", "story": "eval_slots", "corruption": k})
    items = real + gold + ctrl
    ids = rng.sample(range(1, 10 * len(items) + 1), len(items))
    for it, n in zip(items, ids):
        it["item_id"] = f"c{n:05d}"
    os.makedirs(out_dir, exist_ok=True)
    for r in range(1, raters + 1):
        rows = [dict(it) for it in items]
        rng.shuffle(rows)
        with open(os.path.join(out_dir, f"rater_{r:02d}.csv"), "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            w.writeheader()
            for it in rows:
                w.writerow({"item_id": it["item_id"], "rater_id": f"R{r:02d}", "neutral_line": it["neutral"], "edited_line": it["edited"],
                            "meaning_grammar": "", "naturalness_fit": "", "acceptable": "", "comment": ""})
    key = {it["item_id"]: {k: v for k, v in it.items() if k not in ("item_id",)} for it in items}
    json.dump(key, open(os.path.join(out_dir, "_key_DO_NOT_SHARE.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    open(os.path.join(out_dir, "RATER_INSTRUCTIONS.md"), "w", encoding="utf-8").write(INSTRUCTIONS)
    from collections import Counter
    print(f"{len(items)} items ({Counter(i['kind'] for i in items)}) x {raters} raters -> {out_dir}")
    print("Stories in the real items:", dict(Counter(i["story"] for i in real)))
    print("Give each rater ONLY their rater_XX.csv + RATER_INSTRUCTIONS.md. Keep _key_DO_NOT_SHARE.json private.")


def _read(out_dir):
    key = json.load(open(os.path.join(out_dir, "_key_DO_NOT_SHARE.json"), encoding="utf-8"))
    per = {}
    for f in sorted(glob.glob(os.path.join(out_dir, "rater_*.csv"))):
        with open(f, encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                if r["item_id"] not in key:
                    continue
                try:
                    mg, nf = int(r["meaning_grammar"]), int(r["naturalness_fit"])
                except ValueError:
                    continue
                acc = (r["acceptable"] or "").strip().upper()[:1]
                if acc not in ("Y", "N") or not (1 <= mg <= 5 and 1 <= nf <= 5):
                    continue
                per.setdefault(r["item_id"], {})[r["rater_id"]] = {"meaning_grammar": mg, "naturalness_fit": nf, "acceptable": acc == "Y"}
    return key, per


def analyze(out_dir: str, models=None):
    from evaluation import edit_audit, judge
    from evaluation.human_eval import _ordinal_alpha
    from evaluation.stats import cohen_kappa, fmt_rate, kappa_ci
    key, per = _read(out_dir)
    raters = sorted({r for v in per.values() for r in v})
    complete = {i: v for i, v in per.items() if len(v) >= 1}
    if not complete:
        raise SystemExit("no rated items found: put the returned rater_XX.csv files next to _key_DO_NOT_SHARE.json")
    rep = {"n_raters": len(raters), "raters": raters, "n_items_rated": len(complete)}
    if len(raters) < 2:
        rep["WARNING"] = ("single rater: no inter-rater agreement can be computed, and the 'human consensus' is this one person's judgement, so every "
                          "metric-vs-human number below is agreement with ONE rater. Report it as such.")
    # reliability among humans (needs at least two raters on the same items)
    multi = {i: v for i, v in complete.items() if len(v) >= 2}
    if multi:
        rep["krippendorff_alpha_ordinal"] = {f: _ordinal_alpha([{r: x[f] for r, x in v.items()} for v in multi.values()]) for f in ("meaning_grammar", "naturalness_fit")}
        ab = [list(v.values()) for v in multi.values()]
        rep["acceptable_pairwise_raw_agreement"] = round(sum(len({x["acceptable"] for x in v}) == 1 for v in ab) / len(ab), 3)
    # rater quality on the controls
    q = {}
    for r in raters:
        g = [per[i][r]["acceptable"] for i in per if r in per[i] and key[i]["kind"] == "gold"]
        c = [not per[i][r]["acceptable"] for i in per if r in per[i] and key[i]["kind"] == "control"]
        q[r] = {"gold_accepted": fmt_rate(sum(g), len(g)), "corrupted_rejected": fmt_rate(sum(c), len(c))}
    rep["rater_control_accuracy"] = q
    # consensus: majority "acceptable"; the continuous score is the mean of both ratings
    ids = sorted(complete)
    human = {i: (sum(x["acceptable"] for x in complete[i].values()) / len(complete[i])) > 0.5 for i in ids}
    mean_score = {i: sum((x["meaning_grammar"] + x["naturalness_fit"]) / 2 for x in complete[i].values()) / len(complete[i]) for i in ids}
    rep["human_consensus_accept_rate"] = {k: fmt_rate(sum(human[i] for i in ids if key[i]["kind"] == k), sum(1 for i in ids if key[i]["kind"] == k)) for k in ("real", "gold", "control")}
    # automatic metrics on the SAME items
    models = tuple(models or judge.DEFAULT_MODELS)
    pairs = [(key[i]["neutral"], key[i]["edited"]) for i in ids]
    jr = judge.judge_pairs(pairs, models=models)
    metrics = {f"judge:{m}": [judge._accept(r["models"].get(m)) for r in jr] for m in models}
    metrics["judge:ENSEMBLE(all accept)"] = [r["accept"] for r in jr]
    metrics["judge:ANY accepts"] = [r["accept_any"] for r in jr]
    metrics["rule audit (edit_audit)"] = [edit_audit.audit_edit(a, b, None)["ok"] for a, b in pairs]
    metrics["audit AND judge ensemble"] = [x and y for x, y in zip(metrics["rule audit (edit_audit)"], metrics["judge:ENSEMBLE(all accept)"])]
    h = [human[i] for i in ids]
    rep["metric_vs_human_consensus"] = {}
    for name, pred in metrics.items():
        k, lo, hi = kappa_ci(pred, h)
        tp = sum(1 for p, y in zip(pred, h) if p and y)
        tn = sum(1 for p, y in zip(pred, h) if (not p) and (not y))
        row = {"kappa": f"{k:.2f} [{lo:.2f}, {hi:.2f}]", "agreement": f"{sum(p == y for p, y in zip(pred, h)) / len(h):.0%}",
               "human_acceptable_also_accepted (TPR)": fmt_rate(tp, sum(h)), "human_rejected_also_rejected (TNR)": fmt_rate(tn, len(h) - sum(h))}
        for kind in ("real", "gold", "control"):
            idx = [n for n, i in enumerate(ids) if key[i]["kind"] == kind]
            row[f"accept_rate_on_{kind}"] = fmt_rate(sum(pred[n] for n in idx), len(idx))
        rep["metric_vs_human_consensus"][name] = row
    # --- the same comparison under several HUMAN bars, and with the audit split into QUALITY checks and safety POLICY (reported together, never one cherry-picked view)
    POLICY = {"serious-context", "respect-context", "elder-speaker"}
    qual_ok = []
    for a, b in pairs:
        au = edit_audit.audit_edit(a, b, None)
        qual_ok.append(not any(f["blocking"] and f["code"] not in POLICY for f in au["flags"]))
    views = dict(metrics)
    views["rule audit - QUALITY checks only"] = qual_ok
    views["rule audit - safety POLICY only"] = [not any(f["blocking"] and f["code"] in POLICY for f in edit_audit.audit_edit(a, b, None)["flags"]) for a, b in pairs]
    mg = {i: sum(x["meaning_grammar"] for x in complete[i].values()) / len(complete[i]) for i in ids}
    nf = {i: sum(x["naturalness_fit"] for x in complete[i].values()) / len(complete[i]) for i in ids}
    bars = {"human bar A: 'keep' = Y": [human[i] for i in ids],
            "human bar B: both ratings >= 4": [mg[i] >= 4 and nf[i] >= 4 for i in ids],
            "human bar C: both ratings >= 3": [mg[i] >= 3 and nf[i] >= 3 for i in ids]}
    by_bar = {}
    for bname, hb in bars.items():
        rows_b = {"human_accepts": f"{sum(hb)}/{len(hb)}"}
        for name, pred in views.items():
            k, lo, hi = kappa_ci(pred, hb)
            tp = sum(1 for p, y in zip(pred, hb) if p and y)
            tn = sum(1 for p, y in zip(pred, hb) if (not p) and (not y))
            rows_b[name] = {"kappa": f"{k:.2f} [{lo:.2f}, {hi:.2f}]", "TPR": fmt_rate(tp, sum(hb)), "TNR": fmt_rate(tn, len(hb) - sum(hb))}
        by_bar[bname] = rows_b
    rep["metric_vs_human_under_different_bars"] = by_bar
    # continuous: does the judge's naturalness track the humans' mean score?
    from scipy.stats import spearmanr
    rho = spearmanr([r["natural_mean"] for r in jr], [mean_score[i] for i in ids])
    rep["spearman_judge_natural_vs_human_mean"] = round(float(rho.correlation), 3)
    # who is right where the validators DISAGREE? (the small set labels each real edit with its audit x judge quadrant)
    aud_ok, jud_ok = metrics["rule audit (edit_audit)"], metrics["judge:ENSEMBLE(all accept)"]
    quad = {}
    for n, i in enumerate(ids):
        qd = key[i].get("quadrant")
        if qd and qd not in ("gold", "control"):
            d = quad.setdefault(qd, {"n": 0, "human_would_keep": 0, "audit_accepts": 0, "judges_accept": 0})
            d["n"] += 1
            d["human_would_keep"] += int(human[i])
            d["audit_accepts"] += int(aud_ok[n])
            d["judges_accept"] += int(jud_ok[n])
    if quad:
        rep["real_edits_by_quadrant"] = quad
    rep["judge_version"] = judge.JUDGE_VERSION
    json.dump(rep, open(os.path.join(out_dir, "calibration_report.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    return rep


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("--out", default="outputs/calibration")
    c.add_argument("--raters", type=int, default=3)
    c.add_argument("--n-real", type=int, default=90)
    c.add_argument("--n-gold", type=int, default=25)
    c.add_argument("--n-control", type=int, default=25)
    c.add_argument("--seed", type=int, default=7)
    a = sub.add_parser("analyze")
    a.add_argument("--dir", default="outputs/calibration")
    args = ap.parse_args()
    if args.cmd == "create":
        create(args.out, args.raters, args.n_real, args.n_gold, args.n_control, args.seed)
    else:
        analyze(args.dir)


if __name__ == "__main__":
    main()
