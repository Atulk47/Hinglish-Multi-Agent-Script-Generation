"""
review_script.py - human-in-the-loop review of a pipeline run.

The pipeline PROPOSES slang (rules decide where, the local LoRA editor / Groq fallback
propose, guards + an LLM judge filter). A human then approves each slanged line. This gives:
  * a FINAL screenplay in which every slang line has been seen and approved by a person, and
  * the human-evaluation numbers for the paper: edit precision (accepted %), per-engine
    precision (local editor vs Groq fallback), and what reviewers changed.

Usage (PowerShell):
    python review_script.py outputs/final_demo_trace.json
    python review_script.py outputs/final_demo_trace.json --reviewer Anika
    python review_script.py outputs/final_demo_trace.json --auto-accept     # no prompts (testing)

For each slanged line:  [Enter] keep it   r = revert to the neutral line   e = type your own   q = stop
Writes:  outputs/<session>_reviewed_script.txt   and   outputs/<session>_review.json
Several people can review the same trace: use --reviewer and compare the *_review.json files.
"""
import argparse
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def ask(prompt):
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--reviewer", default="reviewer")
    ap.add_argument("--auto-accept", action="store_true")
    args = ap.parse_args()

    t = json.load(open(args.trace, encoding="utf-8"))
    session = os.path.basename(args.trace).replace("_trace.json", "")
    neutral = {s["scene_id"]: s for s in t["neutral_script"]}

    records, scenes_out = [], []
    quit_now = False
    n_total = sum(1 for sc in t["slang_script"] for d in sc.get("decisions", []) if d.get("applied"))
    print(f"\nReviewing '{session}': {n_total} slanged line(s). "
          "[Enter]=keep  r=revert to neutral  e=edit  q=stop (rest kept, marked unreviewed)\n")

    seen = 0
    for sc in t["slang_script"]:
        sid = sc["scene_id"]
        decisions = sc.get("decisions", [])
        n_turns = [l for l in neutral[sid]["dialogue"].split("\n")
                   if ":" in l and l.split(":", 1)[1].strip()]
        out_lines, di = [], 0
        for line in sc["dialogue"].split("\n"):
            if ":" not in line:
                if line.strip():
                    out_lines.append(line)
                continue
            d = decisions[di] if di < len(decisions) else {}
            n_line = n_turns[di] if di < len(n_turns) else line
            di += 1
            if not d.get("applied"):
                out_lines.append(line)
                continue
            seen += 1
            engine = d.get("engine") or "unknown"
            verdict, final = "accepted", line
            if args.auto_accept or quit_now:
                verdict = "unreviewed" if quit_now else "accepted"
            else:
                print(f"[{sid}]  ({seen}/{n_total})  engine: {engine}")
                print(f"   NEUTRAL: {n_line.strip()}")
                print(f"   SLANG  : {line.strip()}")
                ans = ask("   keep? [Enter]=keep  r=revert  e=edit  q=stop > ").lower()
                if ans == "r":
                    verdict, final = "reverted", n_line
                elif ans == "e":
                    speaker = line.split(":", 1)[0].strip()
                    new = ask(f"   your version of {speaker}'s line > ")
                    if new:
                        verdict, final = "edited", f"{speaker}: {new}"
                elif ans == "q":
                    quit_now = True
                    verdict = "unreviewed"
                print()
            out_lines.append(final)
            records.append({"scene": sid, "engine": engine, "neutral": n_line.strip(),
                            "proposed": line.strip(), "final": final.strip(), "verdict": verdict,
                            "slang": d.get("slang"), "reviewer": args.reviewer})
        loc = (neutral[sid].get("events") or {}).get("location", "unknown")
        scenes_out.append((sid, loc, "\n".join(out_lines)))

    lines = ["# HINGLISH SCREENPLAY (human-reviewed)", ""]
    for sid, loc, dlg in scenes_out:
        lines += [f"## {sid.upper()}", f"*Location: {loc}*", "", dlg, "\n---\n"]
    os.makedirs("outputs", exist_ok=True)
    script_path = f"outputs/{session}_reviewed_script.txt"
    review_path = f"outputs/{session}_review_{args.reviewer}.json"
    open(script_path, "w", encoding="utf-8").write("\n".join(lines))

    def rate(rs):
        n = len(rs)
        return {"n": n,
                "accepted": sum(r["verdict"] == "accepted" for r in rs),
                "reverted": sum(r["verdict"] == "reverted" for r in rs),
                "edited": sum(r["verdict"] == "edited" for r in rs),
                "unreviewed": sum(r["verdict"] == "unreviewed" for r in rs),
                "precision": round(sum(r["verdict"] == "accepted" for r in rs) /
                                   max(sum(r["verdict"] != "unreviewed" for r in rs), 1), 3)}
    engines = sorted({r["engine"] for r in records})
    summary = {"session": session, "reviewer": args.reviewer, "overall": rate(records),
               "by_engine": {e: rate([r for r in records if r["engine"] == e]) for e in engines}}
    json.dump({"summary": summary, "lines": records}, open(review_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"Final screenplay : {script_path}")
    print(f"Review record    : {review_path}")
    o = summary["overall"]
    print(f"Slanged lines {o['n']}: accepted {o['accepted']}, reverted {o['reverted']}, "
          f"edited {o['edited']}, unreviewed {o['unreviewed']}  ->  edit precision {o['precision']:.0%}")
    for e, r in summary["by_engine"].items():
        print(f"   {e:14s}: n={r['n']}  precision {r['precision']:.0%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
