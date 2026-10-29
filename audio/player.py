"""
audio/player.py  -  a single self-contained HTML page for the demo: play the audio drama while the transcript scrolls and the current
line lights up, with speaker colours, emotion badges, click-to-seek, and a switch between the NEUTRAL (Pass 1) and SLANG (Pass 2)
versions so the panel can hear the difference.
"""
from __future__ import annotations
import html
import json
import os

PALETTE = ["#3b82f6", "#ec4899", "#10b981", "#f59e0b", "#8b5cf6", "#ef4444", "#14b8a6", "#f97316", "#6366f1", "#84cc16"]

TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hinglish Audio Screenplay</title>
<style>
:root{--bg:#f7f7f8;--card:#fff;--ink:#16181d;--mut:#6b7280;--line:#e5e7eb;--acc:#2563eb}
@media (prefers-color-scheme:dark){:root{--bg:#0f1115;--card:#171a21;--ink:#e8eaee;--mut:#9aa1ad;--line:#272b35;--acc:#60a5fa}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,Segoe UI,Roboto,sans-serif}
header{position:sticky;top:0;background:var(--card);border-bottom:1px solid var(--line);padding:12px 16px;z-index:5}
h1{font-size:18px;margin:0 0 8px}.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
audio{width:100%;max-width:640px}button{border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:8px;padding:6px 12px;cursor:pointer}
button.on{background:var(--acc);color:#fff;border-color:var(--acc)}
main{max-width:760px;margin:0 auto;padding:12px 16px 80px}.scene{margin:22px 0 6px;font-weight:600;color:var(--mut);text-transform:uppercase;font-size:12px;letter-spacing:.06em}
.line{display:flex;gap:10px;padding:8px 10px;border-radius:10px;cursor:pointer;border:1px solid transparent}
.line:hover{background:var(--card)}.line.active{background:var(--card);border-color:var(--acc);box-shadow:0 0 0 2px color-mix(in srgb,var(--acc) 25%,transparent)}
.who{min-width:96px;font-weight:600;font-size:14px}.who small{display:block;font-weight:400;color:var(--mut);font-size:11px}
.txt{flex:1}.narr .txt{color:var(--mut);font-style:italic}.dot{width:10px;height:10px;border-radius:50%;margin-top:7px;flex:none}
.legend{display:flex;gap:12px;flex-wrap:wrap;margin-top:8px;font-size:12px;color:var(--mut)}.legend i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:4px}
</style></head><body>
<header><h1>Hinglish Audio Screenplay</h1>
<div class="row"><audio id="au" controls preload="auto"></audio>
<span id="switch"></span></div><div class="legend" id="legend"></div></header>
<main id="main"></main>
<script>
const DATA=__DATA__; const COLORS=__COLORS__; let variant=DATA.variants[0];
const au=document.getElementById('au'), main=document.getElementById('main');
function render(){
  main.innerHTML=''; let lastScene=null;
  DATA.manifests[variant].timeline.forEach((it,i)=>{
    if(it.scene_id!==lastScene){lastScene=it.scene_id;const s=document.createElement('div');s.className='scene';s.textContent=it.scene_id.replace('_',' ');main.appendChild(s)}
    const d=document.createElement('div');d.className='line'+(it.kind==='narration'?' narr':'');d.id='l'+i;
    const c=COLORS[it.speaker]||'#9ca3af';
    d.innerHTML='<span class="dot" style="background:'+c+'"></span><div class="who" style="color:'+c+'">'+it.speaker+(it.kind==='dialogue'?'<small>'+it.emotion+' · '+it.intensity+'</small>':'')+'</div><div class="txt"></div>';
    d.querySelector('.txt').textContent=it.text;
    d.onclick=()=>{au.currentTime=it.start;au.play()};main.appendChild(d)});
}
function setVariant(v){const t=au.currentTime,p=!au.paused;variant=v;au.src=DATA.manifests[v].files.mp3||DATA.manifests[v].files.wav;
  au.onloadedmetadata=()=>{au.currentTime=Math.min(t,au.duration||t);if(p)au.play();au.onloadedmetadata=null};render();sw()}
function sw(){const e=document.getElementById('switch');e.innerHTML='';DATA.variants.forEach(v=>{const b=document.createElement('button');
  b.textContent=v==='slang'?'With slang (Pass 2)':'Neutral (Pass 1)';b.className=v===variant?'on':'';b.onclick=()=>setVariant(v);e.appendChild(b)})}
document.getElementById('legend').innerHTML=Object.entries(COLORS).map(([n,c])=>'<span><i style="background:'+c+'"></i>'+n+'</span>').join('');
au.addEventListener('timeupdate',()=>{const tl=DATA.manifests[variant].timeline,t=au.currentTime;let k=-1;
  for(let i=0;i<tl.length;i++){if(t>=tl[i].start-0.05&&t<tl[i].end+0.15){k=i;break}}
  document.querySelectorAll('.line.active').forEach(x=>x.classList.remove('active'));
  if(k>=0){const el=document.getElementById('l'+k);el.classList.add('active');if(!au.paused)el.scrollIntoView({block:'center',behavior:'smooth'})}});
au.src=DATA.manifests[variant].files.mp3||DATA.manifests[variant].files.wav; render(); sw();
</script></body></html>
"""


def write_player(out_dir: str, results: dict, cast: dict) -> str:
    variants = [v for v in ("slang", "neutral") if v in results]
    speakers = []
    for v in variants:
        for it in results[v]["timeline"]:
            if it["speaker"] not in speakers:
                speakers.append(it["speaker"])
    colors = {s: ("#9ca3af" if s == "Narrator" else PALETTE[i % len(PALETTE)]) for i, s in enumerate(speakers)}
    data = {"variants": variants, "manifests": {v: {"files": results[v]["files"], "timeline": results[v]["timeline"]} for v in variants}}
    page = TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False)).replace("__COLORS__", json.dumps(colors))
    path = os.path.join(out_dir, "player.html")
    open(path, "w", encoding="utf-8").write(page)
    return path
