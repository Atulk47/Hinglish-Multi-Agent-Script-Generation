"""
evaluation/aggregate_ablation.py  -  the multi-story ablation table: per story, per arm, judged by the validated metrics, with 95% bootstrap intervals over STORIES.

    python -m evaluation.aggregate_ablation --dir outputs/ablation_multi --stories hostel=ui_20261028_151140_trace,cricket=cricket6_trace,...

For every (story, arm) it takes the dialogue lines the arm CHANGED and measures
    edit_rate        changed lines / dialogue lines                                    (coverage)
    unsafe_rate      share of the edits that fail a blocking check of the rule audit    (safety / structure, deterministic)
    judge_accept     share of the edits both judge families accept                       (meaning, grammar, naturalness, fit; evaluation.judge)
    useful_per_100   edits that pass BOTH the audit and the judges, per 100 dialogue lines   (the net benefit)
Across stories each metric is the mean over stories with a bootstrap interval (resampling STORIES - the independent unit - not lines); differences between arms
are PAIRED over stories.  The extra arm `full+judge_filter` is the full system with every judge-rejected edit reverted (what a runtime judge would do), computed
offline from the same edits.  With few stories the intervals are wide: that is the honest result.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

load_dotenv()
from evaluation import edit_audit, judge
from evaluation.stats import bootstrap_ci, paired_bootstrap_diff

ARMS = ["no_editor", "no_validator_repair", "full", "full+judge_filter", "no_gate2", "no_acceptance_rule_gate", "director", "agentic", "agentic_no_repair", "agentic_review", "agentic_review_no_repair"]


def _pairs(trace: dict):
    nm = {s["scene_id"]: s for s in trace.get("neutral_script", [])}
    lines, changed = 0, []
    from evaluation.compare import _parse_turns            # the validator's own turn parser: empty "Narrator:" turns dropped on both sides, so pairs line up
    for sc in trace.get("slang_script", []):
        n = [f"{t['speaker']}: {t['text']}" for t in _parse_turns(nm.get(sc["scene_id"], {}).get("dialogue", ""))]
        s = [f"{t['speaker']}: {t['text']}" for t in _parse_turns(sc["dialogue"])]
        lines += len(s)
        for a, b in zip(n, s):
            if a.strip() != b.strip():
                changed.append((a.split(":", 1)[1].strip(), b.split(":", 1)[1].strip()))
    return lines, changed


def measure(d: str, stem: str, arm: str, profiles=None):
    src = "full" if arm == "full+judge_filter" else arm
    t = json.load(open(os.path.join(d, f"{stem}_{src}.json"), encoding="utf-8"))
    lines, changed = _pairs(t)
    if not changed:
        return {"lines": lines, "edits": 0, "unsafe": 0, "judge_ok": 0, "useful": 0, "judge_any": 0, "useful_any": 0}
    audit_ok = [edit_audit.audit_edit(a, b, None)["ok"] for a, b in changed]
    jr = judge.judge_pairs(changed)
    j_ok = [r["accept"] for r in jr]                       # STRICT: both judge families accept
    j_any = [r["accept_any"] for r in jr]                  # LENIENT: at least one accepts (closest to a human rater, docs/EVALUATION.md)
    if arm == "full+judge_filter":
        keep = [i for i, ok in enumerate(j_ok) if ok]
        audit_ok, j_ok, j_any = [audit_ok[i] for i in keep], [True] * len(keep), [True] * len(keep)
        changed = [changed[i] for i in keep]
    return {"lines": lines, "edits": len(changed), "unsafe": sum(not a for a in audit_ok), "judge_ok": sum(j_ok), "judge_any": sum(j_any),
            "useful": sum(1 for a, j in zip(audit_ok, j_ok) if a and j), "useful_any": sum(1 for a, j in zip(audit_ok, j_any) if a and j)}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/ablation_multi")
    ap.add_argument("--stories", required=True, help="label=trace_stem,label=trace_stem,...")
    ap.add_argument("--arms", default=",".join(ARMS))
    a = ap.parse_args()
    stories = dict(x.split("=", 1) for x in a.stories.split(","))
    arms = a.arms.split(",")
    raw = {arm: {} for arm in arms}
    for label, stem in stories.items():
        for arm in arms:
            try:
                raw[arm][label] = measure(a.dir, stem, arm)
            except FileNotFoundError:
                print("missing:", stem, arm)
    per_story = {}
    for arm in arms:
        for label, m in raw[arm].items():
            per_story.setdefault(arm, {})[label] = {
                "edit_rate": m["edits"] / m["lines"] if m["lines"] else float("nan"),
                "unsafe_rate": (m["unsafe"] / m["edits"]) if m["edits"] else float("nan"),
                "judge_accept": (m["judge_ok"] / m["edits"]) if m["edits"] else float("nan"),
                "judge_accept_any": (m["judge_any"] / m["edits"]) if m["edits"] else float("nan"),
                "useful_per_100": 100 * m["useful"] / m["lines"] if m["lines"] else float("nan"),
                "useful_any_per_100": 100 * m["useful_any"] / m["lines"] if m["lines"] else float("nan")}
    out = {"judge_version": judge.JUDGE_VERSION, "stories": list(stories), "n_stories": len(stories), "arms": {}, "paired_vs_editor_alone": {}, "raw": raw}
    rows = ["| Arm | edit rate | unsafe edits | judge-accepted (both) | judge-accepted (either) | useful / 100 lines (both) | useful / 100 lines (either) |", "|---|---|---|---|---|---|---|"]
    for arm in arms:
        ps = per_story.get(arm, {})
        cell = {}
        for met in ("edit_rate", "unsafe_rate", "judge_accept", "judge_accept_any", "useful_per_100", "useful_any_per_100"):
            vals = [v[met] for v in ps.values() if v[met] == v[met]]
            est, lo, hi = bootstrap_ci(vals, lambda s: sum(s) / len(s)) if vals else (float("nan"),) * 3
            cell[met] = {"mean": est, "ci": [lo, hi], "n_stories": len(vals)}
        out["arms"][arm] = cell
        f = lambda m, pct=True: (f"{cell[m]['mean']:.0%} [{cell[m]['ci'][0]:.0%}-{cell[m]['ci'][1]:.0%}]" if pct else f"{cell[m]['mean']:.1f} [{cell[m]['ci'][0]:.1f}-{cell[m]['ci'][1]:.1f}]")
        rows.append(f"| {arm} | {f('edit_rate')} | {f('unsafe_rate')} | {f('judge_accept')} | {f('judge_accept_any')} | {f('useful_per_100', False)} | {f('useful_any_per_100', False)} |")
    base = per_story.get("no_validator_repair", {})
    for arm in arms:
        if arm in ("no_editor", "no_validator_repair") or arm not in per_story:
            continue
        out["paired_vs_editor_alone"][arm] = {}
        for met in ("unsafe_rate", "judge_accept", "judge_accept_any", "useful_per_100", "useful_any_per_100"):
            d, lo, hi, p = paired_bootstrap_diff({k: v[met] for k, v in per_story[arm].items()}, {k: v[met] for k, v in base.items()})
            out["paired_vs_editor_alone"][arm][met] = {"mean_diff": d, "ci": [lo, hi], "share_resamples_not_better": p}
    if "director" in per_story and "full" in per_story:      # the question this arm answers: does the scene-level Director beat the line-by-line gates?
        out["director_vs_full"] = {}
        for met in ("edit_rate", "unsafe_rate", "judge_accept", "judge_accept_any", "useful_per_100", "useful_any_per_100"):
            d, lo, hi, p = paired_bootstrap_diff({k: v[met] for k, v in per_story["director"].items()}, {k: v[met] for k, v in per_story["full"].items()})
            out["director_vs_full"][met] = {"mean_diff": d, "ci": [lo, hi], "share_resamples_not_better": p}
    out["table_markdown"] = "\n".join(rows)
    out["judge_cost_usd"] = round(judge.STATS["cost"], 4)
    json.dump(out, open(os.path.join(a.dir, "multistory_summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(out["table_markdown"])
    print("\nper-story:", json.dumps(per_story, indent=1, default=float)[:3000])
    print("\npaired vs editor alone:", json.dumps(out["paired_vs_editor_alone"], indent=1, default=float))
    if "director_vs_full" in out:
        print("\nDIRECTOR vs FULL (paired over stories; positive = director higher):", json.dumps(out["director_vs_full"], indent=1, default=float))


if __name__ == "__main__":
    main()
