"""
evaluation/calibration_small.py  -  a SMALL, single-rater calibration set (about 50 items) with a self-contained HTML rating form.

    python -m evaluation.calibration_small create  [--out outputs/calibration_small] [--per-quadrant 8 --n-gold 9 --n-control 9]
    ... open <out>/rate_me.html in a browser, rate, press "Download my ratings", save the file as rater_01.csv in <out> ...
    python -m evaluation.calibration analyze --dir outputs/calibration_small

The real edits are NOT sampled at random: they come from the four quadrants of (does the rule audit accept?) x (do both AI judges accept?), so the cases where the
validators DISAGREE are rated too - those are the ones that show which validator a person agrees with. Gold edits (known good) and corrupted controls (known
broken) are mixed in as attention checks. With ONE rater there is no inter-rater agreement: the report says so, and the numbers are "agreement with this one person".
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

load_dotenv()
from evaluation.calibration import COLS, INSTRUCTIONS, _real_pairs

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Rate the edited lines</title>
<style>
body{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:920px;margin:18px auto;padding:0 14px;color:#111;background:#fff}
.card{border:1px solid #ddd;border-radius:10px;padding:10px 14px;margin:12px 0}.card.done{border-color:#86efac;background:#f0fdf4}
.o{color:#555}.e{font-size:18px;margin:4px 0 10px}ins{background:#bbf7d0;text-decoration:none;border-radius:3px;padding:0 2px}del{background:#fecaca;border-radius:3px;padding:0 2px}
select,input[type=text]{font-size:15px;padding:3px}label{margin-right:18px;display:inline-block;margin-bottom:4px}
#bar{position:sticky;top:0;background:#fff;padding:8px 0;border-bottom:1px solid #ddd;z-index:5}
button{font-size:16px;padding:8px 16px;border-radius:8px;border:1px solid #333;background:#111;color:#fff;cursor:pointer}small{color:#666}
</style></head><body>
<h2>Rate the edited lines</h2>
<p>For each pair, judge the <b>EDITED</b> line against the <b>ORIGINAL</b> (a program may only swap one word or phrase for a slang, or add a casual filler like "yaar").
Green = inserted, red = removed.</p>
<ul><li><b>Meaning &amp; grammar (1-5)</b>: are the meaning, tense, gender agreement and grammar still right? 5 = perfect, 3 = small slip, 1 = meaning changed or ungrammatical.</li>
<li><b>Natural &amp; fits (1-5)</b>: would a young Indian really say it, and does the slang suit what is being talked about? 5 = natural and fitting, 1 = nonsense / forced / trivialises something serious.</li>
<li><b>Keep?</b> would you keep the edited line in a screenplay? (identical line = Y)</li></ul>
<p>Some items are deliberately broken or deliberately good. If torn between two numbers pick the lower one. Your progress is saved in this browser.</p>
<div id="bar"><b id="prog"></b> &nbsp; <button onclick="dl()">Download my ratings (CSV)</button> &nbsp; <small>save as <b>__FILE__</b> in this folder</small></div>
<div id="list"></div>
<script>
const ITEMS=__ITEMS__, RATER="__RATER__", KEY="cal_"+RATER+"_"+ITEMS.length+"_"+ITEMS[0].id;
let S={}; try{S=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
const esc=t=>t.replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
function diff(a,b){
  const A=a.split(/\\s+/),B=b.split(/\\s+/),m=A.length,n=B.length,L=Array.from({length:m+1},()=>Array(n+1).fill(0));
  for(let i=m-1;i>=0;i--)for(let j=n-1;j>=0;j--)L[i][j]=A[i]===B[j]?L[i+1][j+1]+1:Math.max(L[i+1][j],L[i][j+1]);
  let i=0,j=0,o=[],e=[];
  while(i<m&&j<n){ if(A[i]===B[j]){o.push(esc(A[i]));e.push(esc(B[j]));i++;j++} else if(L[i+1][j]>=L[i][j+1]){o.push("<del>"+esc(A[i])+"</del>");i++} else {e.push("<ins>"+esc(B[j])+"</ins>");j++} }
  while(i<m)o.push("<del>"+esc(A[i++])+"</del>"); while(j<n)e.push("<ins>"+esc(B[j++])+"</ins>");
  return [o.join(" "),e.join(" ")];
}
function sel(id,f){const v=(S[id]||{})[f]||"";return '<select onchange="set(\\''+id+'\\',\\''+f+'\\',this.value)"><option value="">-</option>'+[1,2,3,4,5].map(x=>'<option'+(String(x)===v?' selected':'')+'>'+x+'</option>').join("")+'</select>';}
function render(){
  document.getElementById("list").innerHTML=ITEMS.map((it,k)=>{
    const d=diff(it.neutral,it.edited), r=S[it.id]||{}, ok=r.meaning_grammar&&r.naturalness_fit&&r.acceptable;
    return '<div class="card'+(ok?' done':'')+'" id="c'+it.id+'"><small>#'+(k+1)+'</small><div class="o">ORIGINAL: '+d[0]+'</div><div class="e">EDITED: '+d[1]+'</div>'
      +'<label>Meaning &amp; grammar '+sel(it.id,"meaning_grammar")+'</label><label>Natural &amp; fits '+sel(it.id,"naturalness_fit")+'</label>'
      +'<label>Keep? <input type="radio" name="a'+it.id+'" '+(r.acceptable==="Y"?"checked":"")+' onchange="set(\\''+it.id+'\\',\\'acceptable\\',\\'Y\\')"> Y '
      +'<input type="radio" name="a'+it.id+'" '+(r.acceptable==="N"?"checked":"")+' onchange="set(\\''+it.id+'\\',\\'acceptable\\',\\'N\\')"> N</label>'
      +'<label>Note <input type="text" size="26" value="'+esc(r.comment||"")+'" onchange="set(\\''+it.id+'\\',\\'comment\\',this.value)"></label></div>';
  }).join(""); prog();
}
function prog(){const n=ITEMS.filter(it=>{const r=S[it.id]||{};return r.meaning_grammar&&r.naturalness_fit&&r.acceptable}).length;document.getElementById("prog").textContent=n+" / "+ITEMS.length+" rated";}
function set(id,f,v){S[id]=S[id]||{};S[id][f]=v;localStorage.setItem(KEY,JSON.stringify(S));const c=document.getElementById("c"+id),r=S[id];
  if(c){c.className="card"+(r.meaning_grammar&&r.naturalness_fit&&r.acceptable?" done":"");} prog();}
function q(t){return '"'+String(t||"").replace(/"/g,'""')+'"';}
function dl(){
  const rows=[["item_id","rater_id","neutral_line","edited_line","meaning_grammar","naturalness_fit","acceptable","comment"].join(",")];
  ITEMS.forEach(it=>{const r=S[it.id]||{};rows.push([it.id,RATER,q(it.neutral),q(it.edited),r.meaning_grammar||"",r.naturalness_fit||"",r.acceptable||"",q(r.comment)].join(","));});
  const a=document.createElement("a");a.href=URL.createObjectURL(new Blob(["\\ufeff"+rows.join("\\r\\n")],{type:"text/csv;charset=utf-8"}));a.download="__FILE__";a.click();
}
render();
</script></body></html>"""


def write_html(out_dir: str, items: list, rater: str, fname: str) -> str:
    payload = [{"id": r["item_id"], "neutral": r["neutral"], "edited": r["edited"]} for r in items]
    page = PAGE.replace("__ITEMS__", json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")).replace("__RATER__", rater).replace("__FILE__", fname)
    path = os.path.join(out_dir, "rate_me.html")
    open(path, "w", encoding="utf-8").write(page)
    return path


def create(out_dir: str, per_quadrant: int, n_gold: int, n_control: int, seed: int):
    from evaluation import edit_audit, judge
    from evaluation.eval_judge import build_sets
    rng = random.Random(seed)
    pool, seen = [], set()
    for it in _real_pairs(max_per_story=10 ** 6, rng=rng):
        k = (it["neutral"], it["edited"])
        if k not in seen:
            seen.add(k)
            pool.append(it)
    aud = [edit_audit.audit_edit(i["neutral"], i["edited"], None)["ok"] for i in pool]
    jr = judge.judge_pairs([(i["neutral"], i["edited"]) for i in pool])
    quads = {}
    for it, a, r in zip(pool, aud, jr):
        it["quadrant"] = ("audit_ok" if a else "audit_block") + "+" + ("judges_ok" if r["accept"] else "judge_reject")
        quads.setdefault(it["quadrant"], []).append(it)
    real = []
    for qd in sorted(quads):
        items = quads[qd]
        rng.shuffle(items)
        by_story = {}
        for it in items:                                   # round-robin over stories so one story cannot dominate a quadrant
            by_story.setdefault(it["story"], []).append(it)
        picked = []
        while len(picked) < per_quadrant and any(by_story.values()):
            for st in list(by_story):
                if by_story[st] and len(picked) < per_quadrant:
                    picked.append(by_story[st].pop())
        real += picked
    sets = build_sets(400)
    rng.shuffle(sets)
    gold = [{"neutral": d["line"], "edited": d["good"], "kind": "gold", "story": "eval_slots", "quadrant": "gold"} for d in sets[:n_gold]]
    ctrl = []
    for d in sets[n_gold:n_gold + 4 * n_control]:
        if len(ctrl) >= n_control:
            break
        if d["corrupt"]:
            k = rng.choice([c for c in d["corrupt"] if c != "unrelated_line"] or list(d["corrupt"]))
            ctrl.append({"neutral": d["line"], "edited": d["corrupt"][k], "kind": "control", "story": "eval_slots", "corruption": k, "quadrant": "control"})
    items = real + gold + ctrl
    ids = rng.sample(range(1, 10 * len(items) + 1), len(items))
    for it, n in zip(items, ids):
        it["item_id"] = f"c{n:05d}"
    rng.shuffle(items)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "rater_01.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for it in items:
            w.writerow({"item_id": it["item_id"], "rater_id": "R01", "neutral_line": it["neutral"], "edited_line": it["edited"],
                        "meaning_grammar": "", "naturalness_fit": "", "acceptable": "", "comment": ""})
    key = {it["item_id"]: {k: v for k, v in it.items() if k != "item_id"} for it in items}
    json.dump(key, open(os.path.join(out_dir, "_key_DO_NOT_SHARE.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    open(os.path.join(out_dir, "RATER_INSTRUCTIONS.md"), "w", encoding="utf-8").write(INSTRUCTIONS)
    page = write_html(out_dir, items, "R01", "rater_01.csv")
    print(f"{len(items)} items -> {out_dir} | real edits by quadrant: {dict(Counter(i['quadrant'] for i in real))} | gold {len(gold)} | controls {len(ctrl)}")
    print("Open", page, "in a browser, rate, press 'Download my ratings', save as rater_01.csv in that folder, then:")
    print("  python -m evaluation.calibration analyze --dir", out_dir)
    return items


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("--out", default="outputs/calibration_small")
    c.add_argument("--per-quadrant", type=int, default=8)
    c.add_argument("--n-gold", type=int, default=9)
    c.add_argument("--n-control", type=int, default=9)
    c.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    create(a.out, a.per_quadrant, a.n_gold, a.n_control, a.seed)


if __name__ == "__main__":
    main()
