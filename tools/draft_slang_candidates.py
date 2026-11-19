"""
tools/draft_slang_candidates.py  -  draft NEW slang-pool candidates with a strong LLM, for a NATIVE SPEAKER to filter.

    python -m tools.draft_slang_candidates                # asks the model, writes data/slang/slang_candidates.json and outputs/slang_pool_review/review.html
    python -m tools.draft_slang_candidates --html-only    # rebuild the review form from the existing candidates file

Nothing drafted here is used at runtime: entries have status "candidate" until the review answers are applied (tools/apply_slang_review.py).
The model proposes phrase + slot + intensity + gender + grammatical frame + a correct example; this script dedupes against the current pool and bank, and
runs every example through the deterministic rule gate (info only: it tells us whether the editor could ever apply the phrase in that slot).
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

load_dotenv()

SLOTS = {                                   # slot -> (what the editor replaces, sign, how many to ask for)
    "praise": ("a plain positive evaluative word such as accha / achha / badhiya / shandar / behtareen in 'bahut accha laga', 'yeh idea bahut accha hai'", +1, 22),
    "criticism": ("a plain negative evaluative word such as kharab / bekaar / bura / ghatiya / boring in 'bahut kharab laga', 'yeh film bekaar thi'", -1, 12),
    "worry": ("a worry / fear word such as chinta / dar / ghabrahat / pareshani in 'mujhe bahut chinta ho rahi hai', 'mujhe bahut dar lag raha hai'", -1, 12),
    "trouble": ("a trouble noun such as problem / dikkat / samasya / 'mushkil situation' in 'yeh badi problem hai', 'bahut mushkil situation hai'", -1, 10),
    "ease": ("the ease phrase 'aasaani se' in 'yeh kaam aasaani se ho jayega'", +1, 8),
}

SYSTEM = """You are a native speaker of everyday Hinglish (Hindi written in Roman letters, mixed with English) and a careful lexicographer. You help curate a slang list for
a screenplay editor that swaps ONE plain word or short phrase in a line for a slang that a young Indian (about 18 to 30) would really say today, to a friend.

Rules for every candidate you propose:
- It must be in real, current, widely understood use. No dated, very regional or niche slang, nothing crude, abusive, sexual, religious or caste related.
- It must REPLACE the slot's plain word/phrase with minimal change to the sentence, and the example must be grammatically correct (gender, number, verb agreement).
- Subject-bound idioms (body part or feeling idioms) need their possessive subject IN THE EXAMPLE: "Mera dil baith raha hai", "Meri jaan nikal gayi", never "Mujhe dil baith raha hai".
  Put the noun that must follow the possessive in "frame_noun" (e.g. "dil", "jaan", "fatti"), otherwise null.
- Nouns used as predicates or modifiers need their grammatical gender ("m" or "f"); adjectives and exclamations: null.
- "intensity": 1 mild, 2 medium, 3 strong. "risk": "safe", "trendy" (may sound dated soon), "regional", "slightly_crude" or "situational" (only fits some topics: say which in do_not_use_when).
- Give "do_not_use_when": one short note on where it would sound wrong (for example "serious or sad moments", "talking to elders", "no money context").
- Do NOT repeat any phrase from the EXISTING list. Do not propose near-duplicates of each other.

Answer with a JSON array only, no prose. Each item exactly:
{"phrase": "...", "kind": "slang_word" | "idiom", "intensity": 1-3, "gender": "m" | "f" | null, "frame_noun": null | "...", "risk": "...", "do_not_use_when": "...",
 "example_neutral": "<a natural neutral Hinglish sentence containing the slot's plain word>", "example_edited": "<the same sentence with ONLY that word/phrase replaced by the slang>"}"""


def _existing_phrases() -> set:
    from agents import edit_gate as eg
    have = {e["phrase"].lower() for e in eg.POOL_ENTRIES}
    return have


def _bank_rows() -> list:
    rows = list(csv.reader(open("data/felix_dataset/slang_bank_v3.csv", encoding="utf-8-sig")))[1:]
    out = {}
    for r in rows:
        for p in (x.strip() for x in re.split(r"[|]", r[2])):
            if p:
                out.setdefault(p.lower(), set()).add((r[1], r[0]))
    return out


def _json_array(text: str):
    t = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    a, b = t.find("["), t.rfind("]")
    if a < 0 or b <= a:
        return None
    try:
        return json.loads(t[a:b + 1])
    except Exception:
        return None


def _ask(slot: str, what: str, n: int, existing: list, extra: str = "", system: str = None) -> list:
    import llm_client
    user = (f"SLOT: {slot}. The editor replaces {what}.\nPropose {n} NEW candidates for this slot.\n"
            f"EXISTING phrases (do not repeat): {', '.join(sorted(existing))}\n{extra}")
    text, _ = llm_client.complete("director", [{"role": "system", "content": system or SYSTEM}, {"role": "user", "content": user}],
                                  temperature=0.5, max_tokens=8000, only="openrouter")
    arr = _json_array(text)
    if not arr:
        print(f"  [draft] {slot}: unusable answer, skipped", flush=True)
        return []
    return arr


def _classify_bank_only(existing: set, bank: dict) -> list:
    """The bank has phrases that never made it into a slot list; ask the model to place them (same schema + a 'slot' field)."""
    import llm_client
    todo = sorted(p for p in bank if p not in existing and p not in {"abey", "arre", "bhai", "boss", "bro", "chill", "kasam se", "yaar"})
    if not todo:
        return []
    slots = "; ".join(f"{k}: {v[0]}" for k, v in SLOTS.items())
    user = (f"These phrases are already in our bank but unused. For EACH one decide the best slot ({slots}) or \"skip\" if it does not fit any slot as a clean replacement "
            f"(then give slot \"skip\" and a reason in do_not_use_when). Use the same JSON schema plus a \"slot\" field.\nPHRASES: {', '.join(todo)}")
    text, _ = llm_client.complete("director", [{"role": "system", "content": SYSTEM.replace("Do NOT repeat any phrase from the EXISTING list. ", "")},
                                               {"role": "user", "content": user}], temperature=0.2, max_tokens=8000, only="openrouter")
    return _json_array(text) or []


SLOTS2 = {**{k: (v[0], v[1], n) for (k, v), n in zip(SLOTS.items(), (30, 16, 12, 12, 10))},
          "joy": ("a joy word such as khushi / maza in 'mujhe bahut khushi ho rahi hai', 'mujhe bahut maza aa raha hai' (idioms such as 'dil garden-garden ho gaya' need their possessive subject in the example)", +1, 16),
          "pride": ("a pride word such as garv / gurur in 'mujhe tum par bahut garv hai', 'mujhe apne aap par garv hai'", +1, 10),
          "anger": ("an anger word such as gussa in 'mujhe bahut gussa aa raha hai', 'usne mujhe bahut gussa dilaya'", -1, 12),
          "like": ("the word pasand in 'mujhe yeh bahut pasand aaya' / 'mujhe yeh bahut pasand hai' (the replacement may restructure only that phrase)", +1, 8)}

ROUND2 = """

ROUND 2 GUIDANCE from the native speaker who reviewed round 1 (they kept 62 of 68 and then asked for MORE, BETTER and MORE VARIED HINDI slang):
- Round 1 was dominated by English loanword slang (killer, dope, epic, awesome, cool, legendary, insane, wholesome ...). We have plenty of those. Now prefer HINDI / URDU / HINDUSTANI-origin
  slang and desi idioms (muhavare) that young Indians really use: colourful metaphors and exclamations with VARIED imagery (food, cricket, Bollywood, street talk, weather, body, money).
  An English word is acceptable only if it is genuinely everyday Hinglish; at most 3 per slot.
- The speaker DROPPED these as unnatural, so do not propose their cousins: bawaal, "default mein", "setting kharab".
- The speaker confirmed these rules: idioms about the body or feelings take a possessive subject ("mera dimaag fry ho raha hai", "meri jaan nikal gayi"); "on point" is fine as "on point hai" but never "on point laga".
- Vary the register: some mild and everyday, some punchy. Every phrase must still work as a clean replacement of the slot word, with a correct example.
"""


def draft_round2() -> list:
    from agents import edit_gate as eg
    pool = json.load(open("data/slang/slang_pool.json", encoding="utf-8"))
    cand_file = json.load(open("data/slang/slang_candidates.json", encoding="utf-8"))
    existing = {e["phrase"].lower() for e in pool["entries"]} | {c["phrase"].lower() for c in cand_file["entries"]}
    kept_new = [c["phrase"] for c in cand_file["entries"] if c.get("status") == "promoted"]
    out = []
    for slot, (what, sign, n) in SLOTS2.items():
        print(f"  round 2: drafting {slot} ({n}) ...", flush=True)
        extra = f"Phrases the speaker already APPROVED in round 1 (for taste; do not repeat): {', '.join(kept_new[:60])}\n"
        for it in _ask(slot, what, n, sorted(existing), extra, SYSTEM + ROUND2):
            it["slot"], it["sign"], it["source"] = slot, sign, "new-r2"
            out.append(it)
    cands, seen = [], set(existing)
    for it in out:
        p = str(it.get("phrase", "")).strip()
        if not p or p.lower() in seen:
            continue
        seen.add(p.lower())
        en, ee = str(it.get("example_neutral", "")), str(it.get("example_edited", ""))
        cands.append({"phrase": p, "slots": [it["slot"]], "sign": it["sign"], "intensity": it.get("intensity"), "kind": it.get("kind", "slang_word"),
                      "gender": it.get("gender"), "frame": ({"possessive_before": it["frame_noun"], "gender": "f" if it.get("gender") == "f" else "m", "number": "sg"}
                                                            if it.get("frame_noun") else None),
                      "context": None, "controls": [], "status": "candidate", "note": "", "risk": it.get("risk", ""), "do_not_use_when": it.get("do_not_use_when", ""),
                      "example_neutral": en, "example_edited": ee, "source": "new-r2", "round": 2, "gate_ok": False, "gate_why": "unchecked"})
    return cands


def draft() -> list:
    from agents import edit_gate as eg
    existing = _existing_phrases()
    bank = _bank_rows()
    cands, seen = [], set(existing)
    for slot, (what, sign, n) in SLOTS.items():
        print(f"  drafting {slot} ...", flush=True)
        for it in _ask(slot, what, n, sorted(existing)):
            it["slot"], it["sign"], it["source"] = slot, sign, "new"
            cands.append(it)
    print("  classifying unused bank phrases ...", flush=True)
    for it in _classify_bank_only(existing, bank):
        if it.get("slot") in SLOTS:
            it["sign"], it["source"] = SLOTS[it["slot"]][1], "bank"
            cands.append(it)
    out = []
    for it in cands:
        p = str(it.get("phrase", "")).strip()
        if not p or p.lower() in seen:
            continue
        seen.add(p.lower())
        en, ee = str(it.get("example_neutral", "")), str(it.get("example_edited", ""))
        ok, why = eg.rule_gate(en, ee, p) if en and ee else (False, "no-example")
        out.append({"phrase": p, "slots": [it["slot"]], "sign": it["sign"], "intensity": it.get("intensity"), "kind": it.get("kind", "slang_word"),
                    "gender": it.get("gender"), "frame": ({"possessive_before": it["frame_noun"], "gender": "f" if it.get("gender") == "f" else "m", "number": "sg"}
                                                          if it.get("frame_noun") else None),
                    "context": None, "controls": [], "status": "candidate", "note": "", "risk": it.get("risk", ""), "do_not_use_when": it.get("do_not_use_when", ""),
                    "example_neutral": en, "example_edited": ee, "source": it.get("source", "new"), "gate_ok": ok, "gate_why": why})
    return out


HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
body{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:940px;margin:18px auto;padding:0 14px;color:#111;background:#fff}
.card{border:1px solid #ddd;border-radius:10px;padding:10px 14px;margin:12px 0}.card.keep{border-color:#86efac;background:#f0fdf4}.card.drop{border-color:#fca5a5;background:#fef2f2;opacity:.75}.card.fix{border-color:#fcd34d;background:#fffbeb}
.ph{font-size:20px;font-weight:700}.tag{display:inline-block;background:#eef;border-radius:6px;padding:1px 7px;margin:0 4px 0 0;font-size:12px;color:#334}
.ex{margin:6px 0 2px}.neu{color:#555}.edt{font-size:17px}b.hl{background:#bbf7d0;border-radius:3px;padding:0 2px}small{color:#666}.warn{color:#b45309}
#bar{position:sticky;top:0;background:#fff;padding:8px 0;border-bottom:1px solid #ddd;z-index:5}
label{margin-right:16px}input[type=text]{width:60%;font-size:15px;padding:3px}button{font-size:16px;padding:8px 16px;border-radius:8px;border:1px solid #333;background:#111;color:#fff;cursor:pointer}
h3{margin-top:28px}
</style></head><body>
<h2>__TITLE__</h2>
<p>You are the native-speaker filter. For each phrase decide: <b>Keep</b> (natural, current, and the example sentence is correct), <b>Fix</b> (right idea but the phrase, gender or example needs a change: write it in the box), or <b>Drop</b> (unnatural, dated, too crude/regional, or wrong in that slot).
The green words are what replaces the plain word. <span class="warn">Orange note</span> = where the drafter says it does not fit. Nothing is used until you finish and send me the downloaded file.</p>
<div id="bar"><span id="count"></span> &nbsp; <button onclick="dl()">Download my answers</button></div>
<div id="root"></div>
<script>
const DATA = __DATA__;
let ans = {}; try { ans = JSON.parse(localStorage.getItem("__KEY__") || "{}"); } catch(e) {}
function save(){ try { localStorage.setItem("__KEY__", JSON.stringify(ans)); } catch(e) {} }
function esc(s){ return String(s==null?"":s).replace(/[&<>"]/g, c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }
function hl(sent, phrase){ const i = sent.toLowerCase().indexOf(phrase.toLowerCase()); if(i<0) return esc(sent);
  return esc(sent.slice(0,i)) + "<b class='hl'>" + esc(sent.slice(i, i+phrase.length)) + "</b>" + esc(sent.slice(i+phrase.length)); }
function render(){
  const root = document.getElementById("root"); root.innerHTML = ""; let last = "";
  DATA.forEach((c, i) => {
    const sec = c.section; if(sec !== last){ last = sec; root.insertAdjacentHTML("beforeend", "<h3>" + esc(sec) + "</h3>"); }
    const a = ans[i] || {}; const cls = a.d || "";
    const tags = [c.slots.join("/"), c.kind, "intensity " + c.intensity, c.gender ? "gender " + c.gender : "", c.frame ? "needs possessive + " + c.frame.possessive_before : "", c.risk, c.source].filter(Boolean).map(t => "<span class='tag'>" + esc(t) + "</span>").join("");
    const ex = c.example_neutral ? "<div class='ex neu'>plain: " + esc(c.example_neutral) + "</div><div class='ex edt'>with slang: " + hl(c.example_edited, c.phrase) + "</div>" : "";
    const dn = c.do_not_use_when ? "<div class='warn'><small>not for: " + esc(c.do_not_use_when) + "</small></div>" : "";
    root.insertAdjacentHTML("beforeend",
      "<div class='card " + cls + "' id='c" + i + "'><div><span class='ph'>" + esc(c.phrase) + "</span> " + tags + "</div>" + ex + dn +
      "<div style='margin-top:6px'>" + ["keep","fix","drop"].map(d => "<label><input type='radio' name='r" + i + "' value='" + d + "' " + (a.d===d?"checked":"") + " onchange='setd(" + i + ",this.value)'> " + d.charAt(0).toUpperCase()+d.slice(1) + "</label>").join("") +
      "<input type='text' placeholder='fix / note (optional)' value='" + esc(a.n||"") + "' oninput='setn(" + i + ",this.value)'></div></div>");
  });
  count();
}
function setd(i, d){ (ans[i] = ans[i] || {}).d = d; save(); document.getElementById("c"+i).className = "card " + d; count(); }
function setn(i, n){ (ans[i] = ans[i] || {}).n = n; save(); }
function count(){ const done = Object.values(ans).filter(a => a.d).length; document.getElementById("count").textContent = done + " of " + DATA.length + " reviewed"; }
function dl(){
  const out = DATA.map((c, i) => ({phrase: c.phrase, section: c.section, slots: c.slots, decision: (ans[i]||{}).d || "", note: (ans[i]||{}).n || ""}));
  const blob = new Blob([JSON.stringify(out, null, 1)], {type: "application/json"}); const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "__FILE__"; a.click();
}
render();
</script></body></html>"""


def build_html(cands: list, path: str, title: str = "Slang pool review", key: str = "slang_review_v1", fname: str = "slang_review_answers.json", with_existing: bool = True) -> None:
    from agents import edit_gate as eg
    cards = []
    check = {"dimaag fry": "Is 'dimaag fry ho raha hai' natural with 'mujhe' or does it need 'mera'?",
             "fatti padi hai": "Is 'meri fatti padi hai' natural/acceptable, or too crude? Right frame?",
             "pasine chhoot gaye": "'mere pasine chhoot gaye': right frame and wording?",
             "haath-pair phool gaye": "'mere haath-pair phool gaye': right frame and wording?",
             "full tension": "'mujhe full tension ho rahi hai' vs 'full tension hai': is the gender (feminine 'ho rahi') right?",
             "paisa vasool": "Only for value-for-money topics (film, food, show, purchase): is that rule right?",
             "kamaal": "'ekdum kamaal laga' vs 'kamaal ka laga': which frame is natural for 'bahut accha laga'?",
             "on point": "'ekdum on point laga' sounded unnatural to you before: should it be only 'on point hai' (predicate), never after 'laga'?",
             "lajawaab": "'lajawaab laga' / 'lajawaab hai': natural?"}
    for e in (eg.POOL_ENTRIES if with_existing else []):
        if e["phrase"] in check:
            cards.append({**e, "section": "A. Existing phrases to check (answer Keep / Fix / Drop)", "example_neutral": "", "example_edited": "",
                          "risk": "", "do_not_use_when": check[e["phrase"]], "source": "existing"})
    for c in cands:
        cards.append({**c, "section": "B. New candidates, slot: " + c["slots"][0] + (" (from your unused bank)" if c["source"] == "bank" else "")})
    cards.sort(key=lambda c: c["section"])
    data = json.dumps(cards, ensure_ascii=False).replace("</", "<\\/")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w", encoding="utf-8").write(HTML.replace("__DATA__", data).replace("__TITLE__", title).replace("__KEY__", key).replace("__FILE__", fname))


def regate(cands: list) -> list:
    """Drop malformed drafts (non-ASCII phrase, phrase missing from its own example) and re-run the rule gate on every example with ALL candidates temporarily promoted
    into the pool, so worry / trouble / ease slots (which only accept listed phrases) are judged fairly. Restores the real pool afterwards."""
    from agents import edit_gate as eg
    keep = []
    for c in cands:
        p, ee = c["phrase"], c.get("example_edited", "")
        if not p.isascii() or (ee and p.lower() not in ee.lower()):
            continue
        keep.append(c)
    core = eg.load_pool() or []
    try:
        eg._apply_pool(core + [{**c, "status": "core"} for c in keep])
        for c in keep:
            en, ee = c.get("example_neutral", ""), c.get("example_edited", "")
            c["gate_ok"], c["gate_why"] = eg.rule_gate(en, ee, c["phrase"]) if en and ee else (False, "no-example")
    finally:
        eg._apply_pool(core)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--html-only", action="store_true")
    ap.add_argument("--out", default="data/slang/slang_candidates.json")
    ap.add_argument("--html", default="outputs/slang_pool_review/review.html")
    ap.add_argument("--round", type=int, default=1)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if a.round == 2:
        new = regate(draft_round2())
        keep = [c for c in new if c["gate_ok"]]
        print(f"  round 2: {len(new)} drafted after dedupe; {len(keep)} pass the rule gate (the other {len(new) - len(keep)} would never be applicable and are NOT shown)")
        allc = json.load(open(a.out, encoding="utf-8"))
        allc["entries"] += keep
        json.dump(allc, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        import llm_client
        print(llm_client.cost_summary())
        build_html(keep, "outputs/slang_pool_review/review_round2.html", "Slang pool review - round 2 (Hindi-first)", "slang_review_v2", "slang_review_answers_r2.json", False)
        by = {}
        for c in keep:
            by[c["slots"][0]] = by.get(c["slots"][0], 0) + 1
        print(f"round 2 review form: outputs/slang_pool_review/review_round2.html  {by}")
        return
    if a.html_only:
        cands = json.load(open(a.out, encoding="utf-8"))["entries"]
    else:
        cands = draft()
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump({"version": 1, "note": "DRAFT candidates (status candidate): NOT used at runtime until reviewed by a native speaker.", "entries": cands},
                  open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        import llm_client
        print(llm_client.cost_summary())
    n0 = len(cands)
    cands = regate(cands)
    if len(cands) != n0:
        print(f"  dropped {n0 - len(cands)} malformed drafts")
    json.dump({"version": 1, "note": "DRAFT candidates (status candidate): NOT used at runtime until reviewed by a native speaker.", "entries": cands},
              open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    build_html(cands, a.html)
    by = {}
    for c in cands:
        by[c["slots"][0]] = by.get(c["slots"][0], 0) + 1
    print(f"{len(cands)} candidates {by}; gate would accept the example for {sum(1 for c in cands if c.get('gate_ok'))}; review form: {a.html}")


if __name__ == "__main__":
    main()
