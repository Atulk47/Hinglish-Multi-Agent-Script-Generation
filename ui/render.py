"""ui/render.py - HTML / table views of a finished run (scenes, characters, emotion arcs, neutral-vs-slang screenplay, audio timeline)."""
from __future__ import annotations
import difflib
import html
import re

import pandas as pd

PALETTE = ["#3b82f6", "#ec4899", "#10b981", "#f59e0b", "#8b5cf6", "#ef4444", "#14b8a6", "#f97316", "#6366f1", "#84cc16"]
EMO_COLOR = {"neutral": "#9ca3af", "happy": "#facc15", "excited": "#fb923c", "proud": "#a3e635", "relieved": "#5eead4",
             "affectionate": "#f9a8d4", "amused": "#fde047", "sad": "#60a5fa", "anxious": "#c4b5fd", "scared": "#a78bfa",
             "angry": "#f87171", "frustrated": "#fb7185", "surprised": "#67e8f9", "ashamed": "#94a3b8", "determined": "#34d399",
             "jealous": "#d8b4fe"}

CSS = """<style>
.card{border:1px solid var(--border-color-primary,#e5e7eb);border-radius:12px;padding:12px 14px;margin:10px 0;background:var(--background-fill-secondary,#fafafa)}
.card h4{margin:0 0 8px 0;font-size:14px;letter-spacing:.04em;text-transform:uppercase;opacity:.75}
.row{display:flex;gap:10px;padding:6px 0;align-items:flex-start}.who{min-width:104px;font-weight:600;font-size:14px}
.who small{display:block;font-weight:400;opacity:.65;font-size:11px}
.neu{opacity:.55;font-size:13px}.slg{font-size:15px}.slg mark{background:#bbf7d0;color:#14532d;border-radius:4px;padding:0 3px}
.neu del{background:#fee2e2;color:#7f1d1d;text-decoration:line-through;border-radius:4px;padding:0 2px}
.chip{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;margin:2px 4px 2px 0;border:1px solid rgba(128,128,128,.3)}
.kpis{display:flex;flex-wrap:wrap;gap:10px;margin:6px 0 4px}.kpi{border:1px solid rgba(128,128,128,.25);border-radius:12px;padding:8px 14px;min-width:120px}
.kpi b{display:block;font-size:20px}.kpi span{font-size:12px;opacity:.7}
table.arc{border-collapse:collapse;font-size:12px;width:100%}table.arc td,table.arc th{border:1px solid rgba(128,128,128,.25);padding:4px 6px;text-align:center}
table.arc td.name{text-align:left;font-weight:600}.badge{border-radius:6px;padding:1px 6px;color:#111}
</style>"""


def _color_map(trace):
    names = list((trace.get("character_profiles") or {}).keys())
    return {n.lower(): PALETTE[i % len(PALETTE)] for i, n in enumerate(names)}


def _emotion(trace, speaker, scene_id):
    from agents.agent2b_persona import emotion_at
    from agents.name_utils import match_profile
    prof = match_profile(trace.get("character_profiles") or {}, speaker)
    return emotion_at(prof, scene_id) if prof else {"emotion": "neutral", "intensity": 1}


def diff_html(neutral: str, slang: str) -> tuple:
    a, b = neutral.split(), slang.split()
    sm = difflib.SequenceMatcher(None, [w.lower() for w in a], [w.lower() for w in b])
    n_html, s_html = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        old, new = " ".join(a[i1:i2]), " ".join(b[j1:j2])
        if tag == "equal":
            n_html.append(html.escape(old))
            s_html.append(html.escape(new))
        else:
            if old:
                n_html.append(f"<del>{html.escape(old)}</del>")
            if new:
                s_html.append(f"<mark>{html.escape(new)}</mark>")
    return " ".join(n_html), " ".join(s_html)


def summary_html(trace) -> str:
    comp = trace.get("comparison") or []
    lines = [l for sc in comp for l in sc["lines"]]
    edited = [l for l in lines if l.get("changed")]
    slangs = sorted({(l.get("slang_added") or "") for l in edited} - {""})
    seg = trace.get("segmentation") or {}
    k = [("Scenes", len(comp)), ("Characters", len(trace.get("character_profiles") or {})), ("Dialogue lines", len(lines)),
         ("Lines with slang", len(edited)), ("Distinct slangs", len(slangs)),
         ("Text pipeline", f"{trace.get('seconds_pipeline', 0):.0f}s"), ("Cloud cost", f"${trace.get('llm_cost_usd', 0):.3f}")]
    if "audio" in trace:
        k.insert(5, ("Audio", f"{trace['audio']['slang']['duration_s'] / 60:.1f} min"))
    kp = "".join(f'<div class="kpi"><b>{v}</b><span>{html.escape(str(n))}</span></div>' for n, v in k)
    meth = f'<div style="opacity:.7;font-size:13px">Scene segmenter: <b>{html.escape(str(seg.get("method", "?")))}</b>' \
           f' ({seg.get("sentences", "?")} sentences)</div>' if seg else ""
    err = f'<div class="card" style="border-color:#f87171">Audio step failed: {html.escape(trace["audio_error"])}</div>' if trace.get("audio_error") else ""
    if trace.get("audio_pending"):
        err += '<div class="card" style="border-color:#60a5fa">Screenplay ready. The audio drama is still rendering and will appear in the <b>Audio drama</b> tab...</div>'
    return CSS + f'<div class="kpis">{kp}</div>{meth}{err}'


def scenes_html(trace) -> str:
    chunks = {c["scene_id"]: c for c in (trace.get("scene_chunks") or [])}
    deva = trace.get("deva_scenes") or []
    cols = _color_map(trace)
    cards = []
    for i, ev in enumerate(trace.get("event_chain") or []):
        sid = ev["scene_id"]
        chars = "".join(f'<span class="chip" style="background:{cols.get(c.lower(), "#e5e7eb")}33">{html.escape(c)}</span>'
                        for c in ev.get("characters") or [])
        events = "".join(f"<li>{html.escape(e)}</li>" for e in (ev.get("key_events") or [])[:4])
        src = html.escape((deva[i] if i < len(deva) else chunks.get(sid, {}).get("text", ""))[:330])
        cards.append(f'<div class="card"><h4>{sid.replace("_", " ")} &middot; {html.escape(ev.get("location") or "")}</h4>{chars}'
                     f'<div style="font-size:13px;opacity:.8;margin:6px 0">{html.escape(ev.get("scene_goal") or "")}</div>'
                     f'<ul style="margin:4px 0 8px 18px;font-size:13px">{events}</ul>'
                     f'<div style="font-size:13px;opacity:.6;border-left:3px solid rgba(128,128,128,.3);padding-left:8px">{src}...</div></div>')
    return CSS + "".join(cards)


def profiles_df(trace) -> pd.DataFrame:
    rows = []
    for n, p in (trace.get("character_profiles") or {}).items():
        rel = "; ".join(f"{o}: {r.get('relation', '')}" for o, r in list((p.get("relationship") or {}).items())[:3])
        rows.append({"Character": n, "Gender": p.get("gender", "?"), "Age": p.get("age_group", "?"), "Role": p.get("social_role", "?"),
                     "Slang allowed": "no" if p.get("slang_level", 5) == 0 else "yes", "Style": p.get("speech_style", ""),
                     "Traits": ", ".join(p.get("traits") or []), "Relationships": rel,
                     "Peak stress": round(max([e.get("stress", 0) for e in p.get("emotion_arc") or [{}]] or [0]), 2)})
    return pd.DataFrame(rows)


def arcs_html(trace) -> str:
    scenes = [e["scene_id"] for e in (trace.get("event_chain") or [])]
    head = "".join(f"<th>{s.replace('scene_', 'S')}</th>" for s in scenes)
    body = []
    for n, p in (trace.get("character_profiles") or {}).items():
        by = {e["scene_id"]: e for e in p.get("emotion_arc") or []}
        cells = []
        for s in scenes:
            e = by.get(s)
            cells.append("<td>&ndash;</td>" if not e else
                         f'<td><span class="badge" style="background:{EMO_COLOR.get(e["emotion"], "#ddd")}" '
                         f'title="intensity {e["intensity"]}, stress {e.get("stress", 0)}">{e["emotion"][:9]}{"!" * (e["intensity"] - 1)}</span></td>')
        body.append(f'<tr><td class="name">{html.escape(n)}</td>{"".join(cells)}</tr>')
    return CSS + f'<table class="arc"><tr><th></th>{head}</tr>{"".join(body)}</table>' \
                 '<div style="font-size:12px;opacity:.6;margin-top:6px">Each cell: the emotion the Persona Agent assigned to that character in that scene (! = stronger).</div>'


def comparison_html(trace) -> str:
    cols = _color_map(trace)
    cards = []
    for sc in trace.get("comparison") or []:
        rows = []
        for l in sc["lines"]:
            emo = _emotion(trace, l["speaker"], sc["scene_id"])
            n_h, s_h = diff_html(l["neutral"], l["slang"]) if l.get("changed") else (html.escape(l["neutral"]), html.escape(l["slang"]))
            c = cols.get(l["speaker"].lower(), "#9ca3af")
            reason = ""
            if not l.get("changed"):
                reason = f' <span style="opacity:.45;font-size:11px">(kept neutral{": " + str(l["gate_stopped"]) if l.get("gate_stopped") else ""})</span>'
            show_neutral = f'<div class="neu">{n_h}</div>' if l.get("changed") else ""
            rows.append(f'<div class="row"><div class="who" style="color:{c}">{html.escape(l["speaker"].title())}'
                        f'<small>{emo["emotion"]} &middot; {emo["intensity"]}</small></div>'
                        f'<div style="flex:1">{show_neutral}<div class="slg">{s_h}{reason}</div></div></div>')
        cards.append(f'<div class="card"><h4>{sc["scene_id"].replace("_", " ")} &middot; {html.escape(sc.get("location") or "")}</h4>{"".join(rows)}</div>')
    return CSS + "".join(cards)


def timeline_df(trace, variant="slang") -> pd.DataFrame:
    tl = ((trace.get("audio") or {}).get(variant) or {}).get("timeline") or []
    return pd.DataFrame([{"Start": f"{int(it['start'] // 60)}:{it['start'] % 60:04.1f}", "Speaker": it["speaker"], "Emotion": it["emotion"],
                          "Text": it["text"]} for it in tl])
