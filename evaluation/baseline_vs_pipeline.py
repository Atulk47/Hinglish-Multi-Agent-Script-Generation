"""
evaluation/baseline_vs_pipeline.py  -  does the pipeline beat ONE call to a frontier model?  (the question every reviewer will ask)

    python -m evaluation.baseline_vs_pipeline --pipeline-dirs outputs/stress_full_v2 outputs/stress_full_v3 --out outputs/baseline_vs_pipeline.json

Arms, on the same stories:
    pipeline      the saved output of the whole pipeline (slang_script of a --full stress run)
    opus_simple   ONE call to Claude Opus, plain request: "turn this Devanagari story into Hinglish dialogue with natural slang"
    opus_rules    ONE call to Claude Opus, the same request plus the pipeline's rules written out (no narrator, no invented facts, slang only between peers ...)
Every arm is measured the same way:
    invariants    the structural checks of evaluation/invariants.py that do not depend on the slang stage (meta speaker, ungrounded speaker, Devanagari, duplicates ...)
    register      urban slang used where the story setting is traditional (edit_gate.register_ok), slang density (lines carrying slang / filler)
    judge         a blind pairwise LLM judge from a DIFFERENT model family (gpt-5.4): invented items, dropped events, naturalness, slang appropriateness, preference.
                  An LLM judge is a proxy, not a native speaker: it tells us where to look, it does not settle the question.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SIMPLE = """Convert the following Hindi (Devanagari) story into screenplay-style dialogue in Hinglish (Hindi written in Roman script). Make it sound natural, with appropriate casual slang.
Write every line as  Name: line  . Output only the dialogue.

STORY:
{story}"""

RULES = """Convert the following Hindi (Devanagari) story into screenplay-style dialogue in Hinglish (Hindi written in Roman script). Make it sound natural, with appropriate casual slang.
Write every line as  Name: line  . Output only the dialogue.

Rules:
- Only people of the story speak, labelled with the name the story uses (an unnamed person by role, e.g. Dukaandaar). No "Narrator" lines, no stage directions, no notes.
- Do not invent events, places, objects, secrets or people that are not in the story. Words the story quotes stay (translated into Hinglish); narration may become short dialogue only where the story implies the people say it.
- Use slang only between peers in a casual modern setting. No slang when talking to an elder, teacher, boss or stranger, none in a traditional, rural or historical story, none in serious moments (grief, apology, thanks).
- Roman script only; no Devanagari.

STORY:
{story}"""

JUDGE_SYSTEM = """You are a careful native speaker of Hindi and Hinglish reviewing two screenplay scripts (X and Y) written from the same Hindi story. Read the STORY first.
The task was: plain Roman-script Hinglish dialogue, one line per turn written as  Name: line  , only people of the story speaking. A script that is written in Devanagari,
is mostly English, is prose or a summary, contains headings or long stage directions, or repeats the same exchange over and over FAILS the task: give it naturalness 1 and slang_fit 1.
For EACH script report:
  invented: things in the script that the story does NOT support (events, places, objects, secrets, speakers). List them briefly and give the count.
  dropped: important events or quoted speech of the story that the script leaves out (count).
  naturalness: 1-5, how natural the Hinglish sounds spoken aloud by these people.
  slang_fit: 1-5, whether slang (or its absence) fits the relationships, the setting and the mood. 5 = never jarring.
Then give a preference per criterion: "X", "Y" or "tie" for fidelity (fewer invented + dropped), naturalness, slang_fit, and overall.
Answer with JSON only: {"X": {"invented": [...], "n_invented": 0, "dropped": [...], "n_dropped": 0, "naturalness": 3, "slang_fit": 3},
 "Y": {...}, "fidelity": "X|Y|tie", "naturalness": "X|Y|tie", "slang_fit": "X|Y|tie", "overall": "X|Y|tie", "why": "<two sentences>"}"""

URBAN_EXTRA = {"bro", "dope", "cringe", "chill", "vibe", "lit", "awesome", "cool", "epic", "fantastic", "legit", "savage", "slay", "flop", "trash", "pathetic"}
FILLERS = {"yaar", "bro", "arre", "abey", "abe", "bhai", "boss"}


def parse_dialogue(raw: str):
    """-> (dialogue 'Name: line' lines, number of non-dialogue lines the model emitted: headings, stage directions, notes)"""
    raw = re.sub(r"```[a-z]*", "", raw or "")
    keep, noise = [], 0
    for ln in raw.split("\n"):
        t = re.sub(r"^[\s>*#\-]+", "", ln.replace("**", "")).strip()
        if not t:
            continue
        m = re.match(r"^([A-Za-z][A-Za-z .'-]{0,30}?)\s*:\s*(.+)$", t)
        if m and not t.startswith("("):
            keep.append(f"{m.group(1).strip()}: {m.group(2).strip()}")
        else:
            noise += 1
    return keep, noise


def slang_metrics(lines, setting):
    from agents import edit_gate as eg
    pool = json.load(open(os.path.join(ROOT, "data", "slang", "slang_pool.json"), encoding="utf-8"))["entries"]
    eg.set_setting(setting)
    urban_phrases = {e["phrase"].lower() for e in pool if e.get("register") == "urban_youth"} | URBAN_EXTRA
    all_phrases = {e["phrase"].lower() for e in pool} | FILLERS | URBAN_EXTRA
    trad = bool(setting and setting.get("traditional"))
    with_slang, urban_in_trad = 0, []
    for ln in lines:
        body = ln.split(":", 1)[1].lower() if ":" in ln else ln.lower()
        words = " " + re.sub(r"[^a-z' -]", " ", body) + " "
        hit = [p for p in all_phrases if f" {p} " in words]
        if hit:
            with_slang += 1
        if trad:
            bad = [p for p in urban_phrases if f" {p} " in words]
            if bad:
                urban_in_trad.append((ln[:70], bad))
    eg.set_setting(None)
    n = max(1, len(lines))
    return {"lines": len(lines), "slang_density": round(with_slang / n, 3), "urban_in_traditional": len(urban_in_trad), "urban_examples": urban_in_trad[:3]}


def invariants(text_deva, roman, dialogue_lines):
    from evaluation import invariants as inv
    dlg = "\n".join(dialogue_lines)
    res = {"event_chain": [], "persona": {}, "neutral": [{"scene_id": "scene_01", "dialogue": dlg}]}
    V = inv.check_pass1([text_deva], [roman], res, max_lines=10 ** 6)
    V = [v for v in V if v["code"] not in ("slang-leaked-into-neutral", "empty-scene-with-people")]
    return V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline-dirs", nargs="+", default=["outputs/stress_full_v2", "outputs/stress_full_v3"])
    ap.add_argument("--out", default="outputs/baseline_vs_pipeline.json")
    ap.add_argument("--baseline-model", default="anthropic/claude-opus-5.5")
    ap.add_argument("--judge-model", default="openai/gpt-5.4")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--label", default="opus", help="name of the baseline model in the report (arms: <label>_simple, <label>_rules)")
    ap.add_argument("--reuse", default=None, help="a previous report JSON: its baseline texts are reused instead of calling the baseline model again")
    ap.add_argument("--judge-arms", nargs="*", default=None, help="baseline arms to judge against the pipeline (default: both)")
    a = ap.parse_args()
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    import config
    import llm_client
    from preprocessing.transliterate import HindiTransliterator
    config.LLM_COST_CAP_USD = 3.0
    config.OPENROUTER_MODELS["baseline"] = a.baseline_model
    config.OPENROUTER_MODELS["judge_ext"] = a.judge_model
    ARMS = [f"{a.label}_simple", f"{a.label}_rules"]
    JUDGE_ARMS = a.judge_arms or ARMS
    old = json.load(open(os.path.join(ROOT, a.reuse), encoding="utf-8")) if a.reuse else {}
    tr = HindiTransliterator()

    stories = {}
    for d in a.pipeline_dirs:
        for f in sorted(os.listdir(os.path.join(ROOT, d))):
            if f.endswith(".json") and f != "report.json":
                sid = f[:-5]
                if a.only and not any(o in sid for o in a.only):
                    continue
                run = json.load(open(os.path.join(ROOT, d, f), encoding="utf-8"))
                src = next((p for p in (os.path.join(ROOT, "data", "stress", sid + ".txt"), os.path.join(ROOT, "data", "raw", sid + ".txt")) if os.path.exists(p)), None)
                if run.get("slang_script") and src:
                    stories[sid] = {"run": run, "text": open(src, encoding="utf-8").read()}
    print(f"{len(stories)} stories: {sorted(stories)}", flush=True)

    def gen(job):
        sid, arm = job
        if sid in old and arm in old[sid]:                       # same prompt, same story, same model: reuse the saved text (no new call)
            return sid, arm, old[sid][arm].get("raw") or old[sid][arm]["text"], old[sid][arm].get("seconds")
        prompt = (SIMPLE if arm.endswith("_simple") else RULES).format(story=stories[sid]["text"].strip())
        t0 = time.time()
        text, _ = llm_client.complete("baseline", [{"role": "user", "content": prompt}], temperature=0.5, max_tokens=6000, only="openrouter")
        return sid, arm, text, round(time.time() - t0, 1)

    jobs = [(s, arm) for s in stories for arm in ARMS]
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        outs = list(ex.map(gen, jobs))
    arms = {}
    for sid, st in stories.items():
        run = st["run"]
        pipe = [l.strip() for sc in run["slang_script"] for l in sc["dialogue"].split("\n") if ":" in l and l.strip()]
        arms[sid] = {"pipeline": {"lines": pipe, "noise": 0, "seconds": run.get("seconds")}}
    for sid, arm, text, secs in outs:
        lines, noise = parse_dialogue(text)
        arms[sid][arm] = {"lines": lines, "noise": noise, "seconds": secs, "raw_chars": len(text), "raw": re.sub(r"```[a-z]*", "", text or "").strip()}

    rec = {}
    for sid, st in stories.items():
        setting = (st["run"]["result"]["persona"] or {}).get("setting")
        roman = tr.transliterate_sentence(st["text"])
        rec[sid] = {"setting": setting and {k: setting.get(k) for k in ("era", "milieu", "formality", "traditional")}}
        for arm, d in arms[sid].items():
            V = invariants(st["text"], roman, d["lines"])
            m = slang_metrics(d["lines"], setting)
            rec[sid][arm] = {**m, "non_dialogue_lines": d["noise"], "seconds": d["seconds"], "errors": sorted({v["code"] for v in V if v["level"] == "error"}),
                             "n_errors": sum(v["level"] == "error" for v in V), "n_warns": sum(v["level"] == "warn" for v in V), "text": "\n".join(d["lines"]),
                             "raw": d.get("raw", "")}

    # blind pairwise judge: pipeline vs each baseline arm, random side
    rnd = random.Random(7)

    def judge(job):
        sid, arm = job
        flip = rnd.random() < 0.5
        shown = rec[sid][arm].get("raw") or rec[sid][arm]["text"]               # the judge reads what the model actually wrote (wrong script, loops and all), not our parse of it
        shown = shown if len(shown) <= 7000 else shown[:7000] + "\n[... the script continues in the same way ...]"      # a runaway repetition loop: a reader would stop here too
        X, Y = (shown, rec[sid]["pipeline"]["text"]) if flip else (rec[sid]["pipeline"]["text"], shown)
        user = f"STORY:\n{stories[sid]['text'].strip()}\n\nSCRIPT X:\n{X}\n\nSCRIPT Y:\n{Y}"
        try:
            text, _ = llm_client.complete("judge_ext", [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}], temperature=0.0, max_tokens=4000, only="openrouter")
            a_, b_ = text.find("{"), text.rfind("}")
            v = json.loads(text[a_:b_ + 1])
        except Exception as e:
            return sid, arm, {"error": str(e)[:100]}
        who = {"X": arm if flip else "pipeline", "Y": "pipeline" if flip else arm, "tie": "tie"}
        out = {k: who.get(v.get(k), "?") for k in ("fidelity", "naturalness", "slang_fit", "overall")}
        ent = {who["X"]: v.get("X", {}), who["Y"]: v.get("Y", {})}
        out["scores"] = {n: {k: ent[n].get(k) for k in ("n_invented", "n_dropped", "naturalness", "slang_fit")} for n in ent}
        out["invented"] = {n: ent[n].get("invented", [])[:4] for n in ent}
        out["why"] = v.get("why", "")
        return sid, arm, out

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        verdicts = list(ex.map(judge, [(s, arm) for s in stories for arm in JUDGE_ARMS]))
    for sid, arm, v in verdicts:
        rec[sid].setdefault("judge", {})[arm] = v
    json.dump(rec, open(os.path.join(ROOT, a.out), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # summary
    print("\nPER STORY (errors / urban-in-traditional / slang density / lines / seconds):")
    for sid in rec:
        for arm in ["pipeline"] + ARMS:
            r = rec[sid][arm]
            print(f"  {sid:24s} {arm:12s} err={r['n_errors']} {r['errors']} urban_trad={r['urban_in_traditional']} dens={r['slang_density']} lines={r['lines']} noise={r['non_dialogue_lines']} {r['seconds']}s")
    print("\nJUDGE (wins for the pipeline vs each baseline; criteria fidelity / naturalness / slang_fit / overall):")
    for arm in JUDGE_ARMS:
        tally = {k: {"pipeline": 0, arm: 0, "tie": 0} for k in ("fidelity", "naturalness", "slang_fit", "overall")}
        for sid in rec:
            v = rec[sid].get("judge", {}).get(arm, {})
            for k in tally:
                if v.get(k) in tally[k]:
                    tally[k][v[k]] += 1
        print(" ", arm, json.dumps(tally))
    for arm in ["pipeline"] + ARMS:
        sc = [rec[s]["judge"][b]["scores"][arm] for s in rec for b in JUDGE_ARMS if "scores" in rec[s].get("judge", {}).get(b, {}) and arm in rec[s]["judge"][b]["scores"]]
        num = lambda k: [x[k] for x in sc if isinstance(x.get(k), (int, float))]
        mean = lambda l: round(sum(l) / len(l), 2) if l else None
        print(f"  mean judge scores {arm:12s} invented={mean(num('n_invented'))} dropped={mean(num('n_dropped'))} naturalness={mean(num('naturalness'))} slang_fit={mean(num('slang_fit'))}")
    print("\n" + llm_client.cost_summary())


if __name__ == "__main__":
    main()
