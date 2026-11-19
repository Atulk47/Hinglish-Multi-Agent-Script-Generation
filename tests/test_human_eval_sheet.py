import csv

from evaluation import human_eval_sheet as hes


def test_sheet_is_blind_balanced_and_scoring_maps_back_to_the_systems(tmp_path):
    import json
    base = {"s01_office_dash": {a: {"text": f"X: line from {a}"} for a in ("opus_rules", "opus_simple")},
            "s02_playground_quotes": {a: {"text": f"X: line from {a}"} for a in ("opus_rules", "opus_simple")}}
    bj = tmp_path / "base.json"
    bj.write_text(json.dumps(base), encoding="utf-8")
    pdir = tmp_path / "pipe"
    pdir.mkdir()
    for sid in base:
        (pdir / (sid + ".json")).write_text(json.dumps({"slang_script": [{"dialogue": "A: pipeline line one\nB: pipeline line two"}]}), encoding="utf-8")
    import os
    # the story text is read from data/stress: the real corpus is in the repo
    n = hes.build(os.path.relpath(pdir, hes.ROOT) if str(pdir).startswith(hes.ROOT) else str(pdir), str(bj), str(tmp_path / "out"))
    assert n == 4
    rows = list(csv.DictReader(open(tmp_path / "out" / "sheet.csv", encoding="utf-8-sig")))
    key = {r["id"]: r for r in csv.DictReader(open(tmp_path / "out" / "key.csv", encoding="utf-8-sig"))}
    assert all("pipeline" not in (r["script_A"] + r["script_B"]).split("\n")[0].split(":")[0] for r in rows)          # no system name appears in the sheet
    assert sum(1 for k in key.values() if k["A"] == "pipeline") in (1, 2, 3)                                         # sides are randomised, not all one way
    # a rater who always prefers the pipeline: A where A is the pipeline, B otherwise
    for r in rows:
        k = key[r["id"]]
        r.update(more_faithful="A" if k["A"] == "pipeline" else "B", more_natural="tie", slang_fit="A" if k["A"] == "pipeline" else "B", overall="A" if k["A"] == "pipeline" else "B")
    filled = tmp_path / "filled.csv"
    with open(filled, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hes.COLUMNS)
        w.writeheader()
        w.writerows(rows)
    res = hes.score(str(tmp_path / "out" / "key.csv"), [str(filled)])
    assert res["pipeline vs opus_rules"]["overall"] == {"pipeline": 2, "other": 0, "tie": 0}
    assert res["pipeline vs opus_simple"]["more_natural"] == {"pipeline": 0, "other": 0, "tie": 2}
