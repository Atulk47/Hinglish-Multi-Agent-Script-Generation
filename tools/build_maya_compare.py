"""
tools/build_maya_compare.py  -  side-by-side listening page for the Maya 2 test.

Put the zip downloaded from Kaggle (maya_test_out.zip) anywhere, then:
    python -m tools.build_maya_compare path/to/maya_test_out.zip
-> outputs/maya_test/compare.html : one row per line, with Edge (free) | Studio (gpt-audio) | Maya plain | Maya tagged | Maya other voice.
Open the file in a browser; it has a rating box per clip (saved in your browser) to pick winners.
"""
from __future__ import annotations
import html
import json
import os
import sys
import zipfile

D = "outputs/maya_test"
COLS = [("Edge (free)", "edge/{id}.wav"), ("Studio (gpt-audio)", "studio/{id}.wav"), ("Maya", "maya/{id}__plain.wav"),
        ("Maya + emotion tag", "maya/{id}__tagged.wav"), ("Maya, other voice", "maya/{id}__altvoice.wav"), ("Maya, romanized text", "maya/{id}__roman.wav")]


def main():
    if len(sys.argv) > 1:
        with zipfile.ZipFile(sys.argv[1]) as z:
            z.extractall(f"{D}/maya")
    lines = json.load(open(f"{D}/lines.json", encoding="utf-8"))
    rows = []
    for l in lines:
        cells = []
        for name, pat in COLS:
            p = pat.format(id=l["id"])
            if os.path.exists(f"{D}/{p}"):
                key = f"{l['id']}|{name}"
                cells.append(f'<td><audio controls preload="metadata" src="{p}"></audio><br><select data-k="{html.escape(key)}"><option value="">-</option>'
                             + "".join(f"<option>{n}</option>" for n in range(1, 6)) + "</select></td>")
            else:
                cells.append("<td class=na>-</td>")
        rows.append(f'<tr><td class=t><b>{l["speaker"]}</b> <i>({l["emotion"]}, voice {l["voice"]})</i><br>{html.escape(l["text_deva"])}</td>{"".join(cells)}</tr>')
    head = "".join(f"<th>{n}</th>" for n, _ in COLS)
    page = f"""<!doctype html><meta charset=utf-8><title>Maya 2 listening test</title>
<style>body{{font-family:system-ui;margin:16px}}table{{border-collapse:collapse}}td,th{{border:1px solid #bbb;padding:6px;vertical-align:top}}
.t{{max-width:340px}}.na{{color:#aaa;text-align:center}}audio{{width:210px}}</style>
<h2>Maya 2 Native vs Edge vs Studio - rate each clip 1-5 (naturalness + fits the emotion)</h2>
<table><tr><th>line</th>{head}</tr>{"".join(rows)}</table><p id=sum></p>
<script>
const s=document.querySelectorAll('select');
s.forEach(e=>{{try{{e.value=localStorage.getItem('maya|'+e.dataset.k)||''}}catch(_){{}};e.onchange=()=>{{try{{localStorage.setItem('maya|'+e.dataset.k,e.value)}}catch(_){{}};sum()}}}});
function sum(){{const a={{}};s.forEach(e=>{{if(e.value){{const c=e.dataset.k.split('|')[1];(a[c]=a[c]||[]).push(+e.value)}}}});
document.getElementById('sum').innerText='Average rating: '+Object.entries(a).map(([k,v])=>k+' '+(v.reduce((x,y)=>x+y,0)/v.length).toFixed(2)+' (n='+v.length+')').join('  |  ')}}sum();
</script>"""
    open(f"{D}/compare.html", "w", encoding="utf-8").write(page)
    print(f"-> {D}/compare.html")


if __name__ == "__main__":
    main()
