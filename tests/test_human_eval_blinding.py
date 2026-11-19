"""The blinded rating sheets must not reveal which system produced a line, and must not ask for the same line pair twice."""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation import human_eval


def _trace(path, variant, dialogue):
    json.dump({"ablation": {"variant": variant},
               "neutral_script": [{"scene_id": "scene_01", "dialogue": "Arav: Yeh accha hai.\nPriya: Haan, theek hai."}],
               "slang_script": [{"scene_id": "scene_01", "dialogue": dialogue}]}, open(path, "w", encoding="utf-8"))


def test_rater_sheets_are_blinded_deduplicated_and_summarise_per_variant(tmp_path):
    _trace(tmp_path / "orch_full_trace_full.json", "full", "Arav: Yeh zabardast hai.\nPriya: Haan, theek hai.")
    _trace(tmp_path / "orch_full_trace_no_editor.json", "no_editor", "Arav: Yeh accha hai.\nPriya: Haan, theek hai.")
    _trace(tmp_path / "orch_full_trace_no_gate1.json", "no_gate1", "Arav: Yeh zabardast hai.\nPriya: Haan, theek hai.")
    out = tmp_path / "he"
    human_eval.create_sheets(tmp_path, out, raters=2, seed=3)
    sheet = (out / "rater_01.csv").read_text(encoding="utf-8-sig")
    for leaked in ("full", "no_editor", "no_gate", "orch_full_trace", "variant"):
        assert leaked not in sheet.lower().replace("neutral_dialogue", ""), leaked
    rows = list(csv.DictReader((out / "rater_01.csv").open(encoding="utf-8-sig")))
    assert len(rows) == 3                                   # 6 arm-turns -> 3 distinct (scene, turn, neutral, edited) pairs
    key = json.loads((out / "condition_key.json").read_text(encoding="utf-8"))
    zab = next(k for k, v in key.items() if sorted(v["variants"]) == ["full", "no_gate1"])      # the identical edit is ONE item, two arms
    assert all(r["item_id"].startswith("it") and ":" not in r["item_id"] for r in rows)
    for n in (1, 2):                                        # two raters give every item a 4
        path = out / f"rater_0{n}.csv"
        rs = list(csv.DictReader(path.open(encoding="utf-8-sig")))
        for r in rs:
            r.update({f: "4" for f in human_eval.RATING_FIELDS})
            r["naturalness"] = "5" if r["item_id"] == zab else "3"
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rs[0].keys()))
            w.writeheader()
            w.writerows(rs)
    rep = human_eval.summarize(out)
    assert rep["by_variant"]["full"]["naturalness"]["mean"] > rep["by_variant"]["no_editor"]["naturalness"]["mean"]
    assert rep["by_variant"]["no_gate1"]["naturalness"]["n_ratings"] == rep["by_variant"]["full"]["naturalness"]["n_ratings"]
