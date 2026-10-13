"""
data_prep/gen_v3_lines.py
Step 1 of the v3 (on-distribution) slang-editor dataset.

WHY: the editor was trained on short casual texting lines (mean 8.9 words, 0% stiff vocabulary) but
Agent 3 feeds it long, textbook-stiff lines (mean 18.6 words, 73% >= 14 words, 15% Sanskritised
words). This script writes synthetic Agent-3-STYLE neutral dialogue across many NEW scenes (none
of the benchmark stories' topics) so the editor can be trained on the distribution it really sees.
A teacher LLM is used OFFLINE only; nothing here runs at inference time.

Usage:
  python -m data_prep.gen_v3_lines --scenes 3  --out data/felix_dataset/v3/lines_pilot.jsonl   # pilot
  python -m data_prep.gen_v3_lines --scenes 260 --out data/felix_dataset/v3/lines.jsonl
"""
from __future__ import annotations
import argparse, json, os, random, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from data_prep.v3_common import Teacher, parse_json, DEVANAGARI, read_jsonl, write_jsonl

# NOTE: deliberately excludes the topics of the benchmark narratives (science fair / robot,
# old library, Himalaya village drought, hackathon, exhibition) so evaluation stays clean.
SETTINGS = [
    "rehearsal for the school annual-day play", "cricket team practice before a district match",
    "college fest preparation committee", "hostel mess food complaint", "wedding preparations at home",
    "a long train journey with a delay", "pitching a small startup idea to friends",
    "a cooking competition at college", "band audition for the music club", "kite flying on Makar Sankranti",
    "a monsoon trek that goes wrong", "waiting for board exam results", "a cycling trip to a nearby town",
    "neighbourhood cleaning drive", "Diwali shopping in the market", "school bus breaks down on the way to a trip",
    "drama club selecting the lead role", "a painting competition at school", "football tournament final",
    "a mobile-game tournament in the hostel", "first day of an internship", "planning a farewell party for seniors",
    "searching for a lost wallet", "planning a surprise birthday party", "repairing a broken bike",
    "group project deadline tomorrow", "inter-school quiz competition", "charity bake sale for an orphanage",
    "a village fair (mela)", "volunteering for flood relief", "college canteen conversation after a bad exam",
    "moving to a new city for studies", "preparing for a job interview", "power cut during exam preparation",
    "a roommate fight and making up", "treasure hunt organised by the cultural club",
    "visiting a relative's farm", "a tuition class before the unit test", "swimming race at the sports day",
    "a road trip to Goa with friends", "a wedding band (baraat) rehearsal", "planning Holi celebrations in the society",
    "Eid get-together planning", "Ganesh visarjan preparations", "student council election campaign",
    "starting a YouTube channel together", "a school debate competition", "a photography walk in the old city",
    "volunteering at a blood-donation camp", "preparing for the yearly dance competition",
    "a misunderstanding about a borrowed notebook", "buying a second-hand laptop", "a lost-and-found at the railway station",
    "teaching a younger sibling for exams", "the college placement day", "organising a cricket match in the colony",
    "setting up a tea-stall stall at the fest", "a failed attempt at baking a cake", "a late-night study session",
    "the first rain and a cancelled picnic", "a bus-stand conversation about a missed bus", "a kabaddi match at the village",
    "making a short film for a college contest", "a robotics-free coding contest at a coaching class",
    "a gym membership argument", "a clash over the shared hostel room cupboard", "a phone gets stolen in the market",
    "preparing for a classical dance debut", "a surprise result in the state-level race",
]
EMOTIONS = ["excited", "disappointed", "nervous", "frustrated", "relieved", "proud", "worried", "amused",
            "determined", "sad", "hopeful", "embarrassed", "angry", "grateful", "confused", "enthusiastic"]
NAMES = ["Aarav", "Meera", "Rohan", "Sneha", "Kabir", "Ananya", "Vikram", "Pooja", "Aditya", "Isha",
         "Nikhil", "Riya", "Siddharth", "Tanvi", "Karan", "Divya", "Manav", "Neha", "Yash", "Kavya",
         "Harsh", "Simran", "Dev", "Ishaan", "Pranav", "Mehak", "Rahul", "Aditi", "Varun", "Sakshi",
         "Tejas", "Naina", "Arjun", "Payal", "Samar", "Diya"]

SYSTEM = ("You generate SYNTHETIC screenplay dialogue that faithfully imitates the output of a weaker "
          "LLM dialogue-writer used in a Hindi-story pipeline ('Agent 3'). That writer produces "
          "grammatical but STIFF, textbook Hinglish: long explanatory sentences, bookish Sanskritised "
          "words, no slang, no casual fillers. Imitate it faithfully; do NOT make the lines sound like "
          "natural youth speech - the stiffness is the point (a later step will casualise them).")

EXEMPLARS = [
    "Humne socha tha ki is baar annual day par ek natak karenge, lekin rehearsal ke dauran sabhi log alag alag vichar de rahe hain, isse mujhe thodi chinta ho rahi hai.",
    "Yeh bahut accha prayaas hai, Meera. Main tumhari mehnat ki prashansa karta hoon, aur mujhe vishwas hai ki hum zaroor safal honge.",
    "Mujhe lagta hai ki hume apni galtiyon se seekhna chahiye, aur agli baar zyada taiyari ke saath aana chahiye.",
    "Aaj ka match bahut rochak tha, lekin hamari team ne antim over mein kuch galat faisle liye, jiski wajah se hum haar gaye.",
    "Chinta mat karo, Rohan. Yeh sirf ek chhoti si samasya hai, aur hum milkar ise aasaani se hal kar lenge.",
    "Mera maksad sirf itna hai ki is project ko samay par poora kar sakoon, kyunki mujhe is par bahut garv mehsoos hota hai.",
]

PROMPT = """Write {n} consecutive dialogue lines for this scene, imitating the stiff 'Agent 3' writer.

Scene: {setting}
Speakers (young peers, same age group): {cast}
Overall mood of the scene: {emotion} (individual lines may vary; include a mix of positive remarks, negative / worried remarks and plain informative ones)

Style of the writer you are imitating (examples of its typical lines):
{ex}

Rules:
- Romanized Hinglish in Roman letters ONLY (no Devanagari). Hindi grammar with English loanwords (project, idea, plan, team, problem, ...).
- Neutral register: NO slang, NO fillers such as yaar / bhai / bro / arre / boss.
- Length mix over the {n} lines (count the words!): 3 lines of 7-11 words, 5 lines of 12-19 words, 2 lines of 20-25 words. The average must be about 16 words. Longer lines have a comma-joined second clause.
- Mostly everyday Hindi/English words (accha, bahut, problem, plan, kaam, theek hai, lagta hai, chinta, mehnat). Use ONE bookish / Sanskritised word in only about one line out of six (e.g. maksad, prashansa, dhairya, spasht, avashyak, uttam, sahayata, prayaas, parishram, vishwas, samasya, sthiti, nirnay, lakshya, saflata, nirasha) - do not overdo it.
- Move the scene forward (events progress); speakers address each other by name sometimes.
- At most ONE question in the whole scene.
- Do not mention science fairs, robots, libraries, hackathons, exhibitions or mountain villages.

Return ONLY a JSON array of {n} objects: [{{"speaker":"NAME","line":"..."}}]"""


def one_scene(teacher: Teacher, sid: int, rng: random.Random, n_lines: int):
    setting = rng.choice(SETTINGS)
    cast = rng.sample(NAMES, rng.choice([2, 2, 3]))
    emotion = rng.choice(EMOTIONS)
    ex = "\n".join("- " + e for e in rng.sample(EXEMPLARS, 4))
    txt = teacher.ask(SYSTEM, PROMPT.format(n=n_lines, setting=setting, cast=", ".join(cast),
                                            emotion=emotion, ex=ex), 3000)
    arr = parse_json(txt)
    out = []
    if isinstance(arr, list):
        for i, o in enumerate(arr):
            if not isinstance(o, dict):
                continue
            line = str(o.get("line", "")).strip()
            spk = str(o.get("speaker", "")).strip().upper() or cast[0].upper()
            if (not line or DEVANAGARI.search(line) or len(line.split()) < 5 or len(line.split()) > 40):
                continue
            out.append({"scene_id": f"v3s{sid:04d}", "setting": setting, "emotion": emotion,
                        "speaker": spk, "line": line})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=3)
    ap.add_argument("--lines", type=int, default=10)
    ap.add_argument("--out", default="data/felix_dataset/v3/lines.jsonl")
    ap.add_argument("--model", default=os.getenv("V3_LINES_MODEL", "claude-haiku-5-5"))
    ap.add_argument("--max-usd", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    have = read_jsonl(args.out)
    done = {r["scene_id"] for r in have}
    start = len(done)
    teacher = Teacher(args.model, args.max_usd)
    rows = list(have)
    print(f"model={args.model} | have {len(have)} lines / {len(done)} scenes | target {args.scenes} scenes")
    todo = list(range(start, args.scenes))
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(one_scene, teacher, sid, random.Random(args.seed * 1000 + sid), args.lines): sid
                for sid in todo}
        for k, fu in enumerate(as_completed(futs), 1):
            try:
                rows += fu.result()
            except Exception as e:
                print("  scene failed:", str(e)[:100])
            if k % 20 == 0:
                write_jsonl(args.out, rows)
                print(f"  {k}/{len(todo)} scenes | {len(rows)} lines | ~${teacher.cost:.3f}", flush=True)
    write_jsonl(args.out, rows)
    ws = [len(r["line"].split()) for r in rows]
    print(f"\nDONE {len(rows)} lines | mean words {sum(ws)/max(len(ws),1):.1f} | "
          f">=14 words: {100*sum(w>=14 for w in ws)/max(len(ws),1):.0f}% | ~${teacher.cost:.3f}")
    print("saved:", args.out)


if __name__ == "__main__":
    main()
