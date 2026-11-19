"""
evaluation/eval_validator.py  -  VALIDATING THE VALIDATOR: can its signals tell good edits from bad ones, and meaning-preserving edits from drifted ones?

    python -m evaluation.eval_validator [--fluency N] [--write-thresholds]

E1  real judged edits (data/felix_dataset/v3/judged*.jsonl: edits written by the teacher model, labelled keep / not-keep by a strict judge)
    -> how well does each signal separate keep from not-keep?  (AUC, and precision / recall for the pass-fail checks)
E2  meaning-drift detection: GOOD edits (gold edits of the held-out kaggle_v5/eval_slots sentences) vs the same edits CORRUPTED in five ways the
    pipeline could really produce (hallucinated clause, content-word swap, negation flip, unrelated line, wrong-mood slang).  Every threshold is set from
    the GOOD edits only (so that 95% of them pass); the corruptions are never used to tune anything.  -> detection rate per corruption.
E3  event coverage: scene dialogue vs its own key events (true) and vs another scene's events (shuffled).  Old keyword matching vs semantic MuRIL.

Signals compared:  raw MuRIL cosine (the old validator) | centred MuRIL cosine | edit audit | rule gate | LM fluency delta (E1 only, --fluency N).
Writes outputs/validator_eval.json and (with --write-thresholds) data/validator/thresholds.json read by the validator.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.metrics import roc_auc_score

from agents import edit_gate as eg
from evaluation import edit_audit


def load_judged(limit=None):
    rows = []
    for f in ("judged", "judged2", "judged_hand"):
        for l in open(f"data/felix_dataset/v3/{f}.jsonl", encoding="utf-8"):
            r = json.loads(l)
            if r.get("keep") is None or not r.get("edit") or r.get("label") != "EDIT":
                continue
            rows.append(r)
    random.Random(5).shuffle(rows)
    return rows[:limit] if limit else rows


def auc(y, score):
    y = np.asarray(y, dtype=int)
    return float(roc_auc_score(y, np.asarray(score, dtype=float))) if 0 < y.sum() < len(y) else None


def e1(v, n_fluency):
    rows = load_judged()
    y = [1 if r["keep"] else 0 for r in rows]
    raw = [v.raw_similarity(r["line"], r["edit"]) for r in rows]
    cen = [v.compute_similarity(r["line"], r["edit"]) for r in rows]
    aud = [edit_audit.audit_edit(r["line"], r["edit"], r["slang"]) for r in rows]
    gate = [eg.rule_gate(r["line"], r["edit"], r["slang"])[0] for r in rows]

    def prf(pred):
        pred = np.asarray(pred, dtype=bool)
        yy = np.asarray(y, dtype=bool)
        tp, fp, tn, fn = (pred & yy).sum(), (pred & ~yy).sum(), (~pred & ~yy).sum(), (~pred & yy).sum()
        return {"precision_of_passed": round(float(tp / max(tp + fp, 1)), 3), "good_kept": round(float(tp / max(tp + fn, 1)), 3),
                "bad_rejected": round(float(tn / max(tn + fp, 1)), 3), "n": int(len(y))}
    out = {"n": len(rows), "n_keep": int(sum(y)), "n_bad": int(len(y) - sum(y)),
           "auc_raw_cosine": auc(y, raw), "auc_centred_cosine": auc(y, cen),
           "audit": prf([a["ok"] for a in aud]), "rule_gate": prf(gate),
           "audit_flag_counts_on_bad": {}}
    from collections import Counter
    c = Counter(f["code"] for a, yy in zip(aud, y) if not yy for f in a["flags"] if f["blocking"])
    out["audit_flag_counts_on_bad"] = dict(c.most_common(10))
    if n_fluency:
        from evaluation.fluency import FluencyScorer
        fs = FluencyScorer()
        sub = rows[:n_fluency]
        ys = [1 if r["keep"] else 0 for r in sub]
        d = [fs.delta(r["line"], r["edit"]) for r in sub]
        out["fluency"] = {"n": len(sub), "auc_neg_delta": auc(ys, [-x for x in d]),
                          "mean_delta_keep": round(float(np.mean([x for x, yy in zip(d, ys) if yy])), 3),
                          "mean_delta_bad": round(float(np.mean([x for x, yy in zip(d, ys) if not yy])), 3)}
        # among the edits the audit already passes: does fluency still separate keep from bad?
        pa = [i for i, r in enumerate(sub) if edit_audit.audit_edit(r["line"], r["edit"], r["slang"])["ok"]]
        if len(pa) > 20:
            out["fluency"]["auc_neg_delta_among_audit_passed"] = auc([ys[i] for i in pa], [-d[i] for i in pa])
            out["fluency"]["n_audit_passed"] = len(pa)
        out["_fluency_deltas"] = {"keep": [round(x, 3) for x, yy in zip(d, ys) if yy], "bad": [round(x, 3) for x, yy in zip(d, ys) if not yy]}
    return out


def _content_words(s):
    return [w for w in re.findall(r"[a-zA-Z]{5,}", s) if w.lower() not in eg.EXISTING_SLANG and w.lower() not in eg.FILLERS]


def corruptions(rng, line, edit, slang, pool):
    """-> {name: corrupted edited line} (only the ones that can be built for this pair)."""
    out = {}
    other = rng.choice([p for p in pool if p != line])
    clause = " ".join(other.replace(",", " ,").split()[:6]).replace(" ,", ",")
    out["hallucinated_clause"] = edit.rstrip(".!? ") + ", " + clause + "."
    cw = _content_words(edit)
    ocw = _content_words(other)
    if cw and ocw:
        w = rng.choice(cw)
        out["content_swap"] = re.sub(r"\b" + re.escape(w) + r"\b", rng.choice(ocw), edit, count=1)
    toks = edit.split()
    if any(t.lower().strip(",.!?") in eg.NEGATORS for t in toks):
        out["negation_flip"] = " ".join(t for t in toks if t.lower().strip(",.!?") not in eg.NEGATORS)
    elif len(toks) > 3:
        out["negation_flip"] = " ".join(toks[:-1] + ["nahi"] + toks[-1:])
    out["unrelated_line"] = other
    sl = (slang or "").lower()
    if sl in edit_audit.POS_SIGN or sl in edit_audit.NEG_SIGN:
        opp = rng.choice(sorted(edit_audit.NEG_SIGN - eg.FEEL_SLANG) if sl in edit_audit.POS_SIGN else sorted(edit_audit.POS_SIGN))
        out["wrong_mood_slang"] = re.sub(re.escape(sl), opp, edit, count=1, flags=re.I)
    return out


def e2(v):
    rows = [json.loads(l) for l in open("kaggle_v5/eval_slots.jsonl", encoding="utf-8")]
    good = [r for r in rows if r.get("expect") == "edit" and r.get("gold") and r["gold"] != r["line"]]
    pool = [r["line"] for r in rows]
    rng = random.Random(11)
    # gold edits: thresholds come from these only
    g_raw = np.array([v.raw_similarity(r["line"], r["gold"]) for r in good])
    g_cen = np.array([v.compute_similarity(r["line"], r["gold"]) for r in good])
    thr_raw, thr_cen = float(np.quantile(g_raw, 0.05)), float(np.quantile(g_cen, 0.05))     # 95% of good edits pass
    g_aud_ok = np.mean([edit_audit.audit_edit(r["line"], r["gold"], None)["ok"] for r in good])
    res = {"n_good": len(good), "threshold_raw_cosine": round(thr_raw, 4), "threshold_centred_cosine": round(thr_cen, 4),
           "good_edits_passing_audit": round(float(g_aud_ok), 3), "by_corruption": {}}
    cnt = {}
    for r in good:
        slang = edit_audit._infer_slang(edit_audit.diff_edit(r["line"], r["gold"])["added"])
        for name, bad in corruptions(rng, r["line"], r["gold"], slang, pool).items():
            if not bad or bad.strip() == r["gold"].strip():
                continue
            raw_s, cen_s = v.raw_similarity(r["line"], bad), v.compute_similarity(r["line"], bad)
            a = edit_audit.audit_edit(r["line"], bad, None)
            d = cnt.setdefault(name, {"n": 0, "raw": 0, "centred": 0, "audit": 0, "centred_or_audit": 0})
            d["n"] += 1
            d["raw"] += raw_s < thr_raw
            d["centred"] += cen_s < thr_cen
            d["audit"] += (not a["ok"])
            d["centred_or_audit"] += (cen_s < thr_cen) or (not a["ok"])
    for name, d in cnt.items():
        res["by_corruption"][name] = {"n": d["n"], **{k: round(d[k] / d["n"], 3) for k in ("raw", "centred", "audit", "centred_or_audit")}}
    tot = {k: sum(d[k] for d in cnt.values()) / max(sum(d["n"] for d in cnt.values()), 1) for k in ("raw", "centred", "audit", "centred_or_audit")}
    res["overall_detection"] = {k: round(float(x), 3) for k, x in tot.items()}
    return res


def e3(v):
    scenes = []
    for p in sorted(glob.glob("outputs/*_trace.json")):
        try:
            t = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        for sc in t.get("neutral_script") or []:
            ev = (sc.get("events") or {}).get("key_events") or []
            if ev and sc.get("dialogue") and "[generation failed" not in sc["dialogue"].lower():
                scenes.append((sc["dialogue"], ev))
    rng = random.Random(2)
    y, kw, sem = [], [], []
    for i, (dlg, evs) in enumerate(scenes):
        j = rng.choice([k for k in range(len(scenes)) if k != i])
        for lab, events in ((1, evs), (0, scenes[j][1])):
            for e in events:
                y.append(lab)
                kw.append(v._keyword_hit(dlg, e))
                sem.append(v.event_similarity(dlg, e))
    return {"n_scenes": len(scenes), "n_pairs": len(y), "auc_keyword": auc(y, kw), "auc_semantic_centred": auc(y, sem),
            "_scores": {"true": [round(s, 3) for s, yy in zip(sem, y) if yy], "shuffled": [round(s, 3) for s, yy in zip(sem, y) if not yy]}}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--fluency", type=int, default=0, help="also score N judged edits with the LM fluency check (slow: ~0.4 s per edit)")
    ap.add_argument("--write-thresholds", action="store_true")
    ap.add_argument("--only", default="", help="comma list of e1,e2,e3 to (re)compute; the other results are kept from outputs/validator_eval.json")
    a = ap.parse_args()
    from evaluation.muril_validator import MuRILValidator
    v = MuRILValidator()
    only = {x.strip() for x in a.only.split(",") if x.strip()} or {"e1", "e2", "e3"}
    prev = json.load(open("outputs/validator_eval.json", encoding="utf-8")) if os.path.exists("outputs/validator_eval.json") else {}
    out = dict(prev)
    if "e1" in only:
        out["e1_judged_edits"] = e1(v, a.fluency)
    if "e2" in only:
        out["e2_meaning_drift"] = e2(v)
    if "e3" in only:
        out["e3_event_coverage"] = e3(v)
    os.makedirs("outputs", exist_ok=True)
    flu = out.get("e1_judged_edits", {}).pop("_fluency_deltas", None)
    ev = out.get("e3_event_coverage", {}).pop("_scores", None)
    json.dump(out, open("outputs/validator_eval.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if a.write_thresholds and ev:
        e2r = out["e2_meaning_drift"]
        t = {"line_sim_min": e2r["threshold_centred_cosine"],
             "event_sim_min": round(float(np.quantile(ev["true"], 0.25)), 3),           # 75% of true event/scene pairs pass
             "event_sim_chance": round(float(np.quantile(ev["shuffled"], 0.9)), 3)}
        if flu:
            t["fluency_delta_max"] = round(float(np.quantile(flu["keep"], 0.9)), 3)       # 90% of kept edits are below this
        os.makedirs("data/validator", exist_ok=True)
        json.dump(t, open("data/validator/thresholds.json", "w"), indent=1)
        print("thresholds ->", t)


if __name__ == "__main__":
    main()
