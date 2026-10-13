"""
data_prep/gen_filler_rows.py
Targeted training rows for the case the v3 adapter handles worst IN THE PIPELINE: a sentence whose tone is
neutral/unknown, where the shortlist contains only mood-neutral fillers (yaar, bhai, arre, bro, boss, chill,
abey, kasam se). The correct edit is a short, well-placed filler: a vocative after a name, an opener after
"Haan / Theek hai", or a filler closing a "Chinta mat karo"-style clause. On such shortlists v3 sometimes
hallucinated a new clause ("kuch exciting work kar raha hai!").

Positions are the ones agents/edit_gate.py accepts, so training targets == what the runtime will accept.
Formal sentences (aap / chaliye / ji ...) get an ABSTAIN row (line returned unchanged).

Writes data/felix_dataset/v4/filler_train.jsonl (+ filler_eval.jsonl built from REAL pipeline sentences).
  python -m data_prep.gen_filler_rows
"""
from __future__ import annotations
import json, random, re, collections
from data_prep.build_lora_dataset import make_example
from data_prep.v3_common import read_jsonl, write_jsonl, norm, keeps_meaning, newly_inserted
from agents.edit_gate import rule_gate, FORMAL

D = "data/felix_dataset/"
FEM = {"Meera", "Sneha", "Ananya", "Pooja", "Isha", "Riya", "Tanvi", "Divya", "Neha", "Kavya", "Simran", "Mehak",
       "Aditi", "Sakshi", "Naina", "Payal", "Diya", "Riy"}
NAMES = {"Aarav", "Meera", "Rohan", "Sneha", "Kabir", "Ananya", "Vikram", "Pooja", "Aditya", "Isha", "Nikhil", "Riya",
         "Siddharth", "Tanvi", "Karan", "Divya", "Manav", "Neha", "Yash", "Kavya", "Harsh", "Simran", "Dev", "Ishaan",
         "Pranav", "Mehak", "Rahul", "Aditi", "Varun", "Sakshi", "Tejas", "Naina", "Arjun", "Payal", "Samar", "Diya",
         "Arav", "Riy", "Amit", "Anika", "Rohit", "Sana"}
FILLERS = ["yaar", "bhai", "arre", "bro", "boss", "chill", "abey", "kasam se"]
FIRST_PERSON = re.compile(r"^(Mujhe|Main|Hum|Humne|Maine|Hamari|Humein|Mera|Meri|Hamara|Hume)\b")
OPENERS = re.compile(r"^(Haan|Theek hai|Sahi kaha|Bilkul|Accha|Achha)(,|\s)\s*(.*)$")
REASSURE = re.compile(r"^(Chinta mat karo|Tension mat lo|Ghabrao mat|Dhairya rakho)(,)?\s*(.*)$")


def lc(s):
    return s[:1].lower() + s[1:]


def make_edit(sent, rng):
    """-> (filler, edited) or None. Pure placement; no other word changes."""
    s = sent.strip()
    m = re.match(r"^([A-Z][a-z]+),\s+(.*)$", s)
    if m and m.group(1) in NAMES:
        name, rest = m.group(1), m.group(2)
        f = rng.choice(["yaar", "yaar", "bro"] if name in FEM else ["yaar", "yaar", "bhai", "bro", "boss"])
        return f, f"{name} {f}, {rest}"
    m = re.match(r"^(Haan|Theek hai|Sahi kaha|Bilkul)\s+([A-Z][a-z]+),\s+(.*)$", s)
    if m and m.group(2) in NAMES:
        f = rng.choice(["yaar", "yaar", "bro"] if m.group(2) in FEM else ["yaar", "yaar", "bhai", "bro"])
        return f, f"{m.group(1)} {m.group(2)} {f}, {m.group(3)}"
    m = REASSURE.match(s)
    if m:
        head, rest = m.group(1), m.group(3)
        if rng.random() < 0.5:
            f = rng.choice(["yaar", "bro", "bhai"])
            name = re.match(r"^([A-Z][a-z]+)[,.]", rest)
            if name and name.group(1) in NAMES:
                f = "yaar" if name.group(1) in FEM else f
            return f, f"{head} {f}, {rest}" if rest else f"{head} {f}."
        return "arre", f"Arre {lc(head)}, {rest}" if rest else f"Arre {lc(head)}."
    m = OPENERS.match(s)
    if m and m.group(3):
        f = rng.choice(["arre", "yaar"])
        if f == "arre":
            return f, f"Arre {lc(m.group(1))}, {m.group(3)}"
        return f, f"{m.group(1)} {f}, {m.group(3)}"
    if FIRST_PERSON.match(s) and len(s.split()) >= 6:
        f = rng.choice(["yaar", "arre", "bhai", "bro"])
        return f, f"{f.capitalize()}, {lc(s)}"
    return None


def shortlist(true, rng):
    k = rng.choice([3, 4, 4, 5, 6])
    others = [f for f in FILLERS if f != true]
    rng.shuffle(others)
    sl = [true] + others[: k - 1]
    rng.shuffle(sl)
    return sl


def main(seed=17, cap=520, n_abstain=110):
    rng = random.Random(seed)
    # sentences from TRAIN scenes only, never the held-out / real evaluation sentences
    banned = set()
    for fn in ("v3/eval_real.jsonl", "v3/eval_synth.jsonl"):
        banned |= {norm(r["line"]) for r in read_jsonl(D + fn)}
    sents = []
    for r in read_jsonl(D + "v3/lines.jsonl"):
        sid = int(r["scene_id"].replace("v3s", ""))
        if sid >= 240:
            continue
        for u in re.split(r"(?<=[.!?])\s+", r["line"].strip()):
            if 5 <= len(u.split()) <= 24 and not u.endswith("?") and norm(u) not in banned:
                sents.append(u)
    sents = list(dict.fromkeys(sents))
    rng.shuffle(sents)

    picks, abst = [], []
    for u in sents:
        if FORMAL.search(u.lower()):
            if len(abst) < n_abstain:
                abst.append(u)
            continue
        if len(picks) >= cap:
            continue
        r = make_edit(u, rng)
        if not r:
            continue
        f, e = r
        if (newly_inserted(u, e, f) and keeps_meaning(u, e) and rule_gate(u, e, f)[0]):
            picks.append((u, f, e))
    rows = [make_example(u, shortlist(f, rng), e) for u, f, e in picks]
    rows += [make_example(u, shortlist(rng.choice(FILLERS), rng), u) for u in abst]
    rng.shuffle(rows)
    write_jsonl(D + "v4/filler_train.jsonl", rows)

    # REAL-sentence evaluation rows (never trained on): where a rule-placed filler exists
    ev = []
    for r in read_jsonl(D + "v3/real_lines.jsonl"):
        for u in re.split(r"(?<=[.!?])\s+", r["line"].strip().strip('"')):
            if 5 <= len(u.split()) <= 24 and not u.endswith("?") and not FORMAL.search(u.lower()):
                x = make_edit(u, random.Random(5))
                if x and rule_gate(u, x[1], x[0])[0]:
                    ev.append({"id": f"fill{len(ev)}", "src": "filler", "story": r["scene_id"], "line": u,
                               "shortlist": shortlist(x[0], random.Random(len(ev))), "expect": "edit",
                               "teacher_slang": x[0], "teacher_edit": x[1]})
    write_jsonl(D + "v4/filler_eval.jsonl", ev)
    c = collections.Counter(f for _, f, _ in picks)
    print(f"filler rows: {len(picks)} pick + {len(abst)} abstain | by filler {dict(c)} | real eval rows {len(ev)}")
    for u, f, e in picks[:8]:
        print("  ", u, "\n   ->", e)


if __name__ == "__main__":
    main()
