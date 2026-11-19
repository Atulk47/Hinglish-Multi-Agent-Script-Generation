"""Offline tests: statistics helpers, the cached two-family judge (network stubbed), and the human-calibration create / analyze round trip."""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest

from evaluation import judge
from evaluation.stats import bootstrap_ci, cohen_kappa, paired_bootstrap_diff, wilson


def test_wilson_interval_and_kappa():
    p, lo, hi = wilson(8, 10)
    assert p == 0.8 and 0.49 < lo < 0.5 and 0.94 < hi < 0.96
    assert wilson(0, 0)[0] != wilson(0, 0)[0]                                   # nan for an empty sample
    assert cohen_kappa([True, True, False, False], [True, True, False, False]) == 1.0
    assert abs(cohen_kappa([True, False, True, False], [True, True, False, False])) < 1e-9      # chance-level agreement
    assert cohen_kappa([True] * 6, [True] * 6) == 1.0                              # degenerate: no variance, perfect agreement


def test_bootstrap_resamples_units_and_paired_diff_detects_a_shift():
    est, lo, hi = bootstrap_ci([0.2, 0.4, 0.3, 0.5, 0.35], lambda s: sum(s) / len(s), n_boot=500)
    assert lo < est < hi and 0.2 <= lo and hi <= 0.5
    a = {f"s{i}": 0.6 + 0.01 * i for i in range(7)}
    b = {f"s{i}": 0.4 + 0.01 * i for i in range(7)}
    d, lo, hi, p = paired_bootstrap_diff(a, b, n_boot=500)
    assert abs(d - 0.2) < 1e-9 and lo > 0.19 and p == 0.0


def _stub_chat(monkeypatch, answers, calls):
    def fake(model, prompt, max_tokens=250):
        calls.append(model)
        return answers[model]
    monkeypatch.setattr(judge, "_chat", fake)
    monkeypatch.setattr(judge, "_cache", {})
    monkeypatch.setattr(judge, "CACHE_PATH", "outputs/_judge_cache_test.json")
    judge.STATS.update({"calls": 0, "cached": 0, "unparsed": 0, "cost": 0.0})


GOOD = '{"meaning_preserved": true, "grammar_ok": true, "natural": 4, "fits_situation": true, "problem": "none"}'
BAD = '{"meaning_preserved": true, "grammar_ok": false, "natural": 2, "fits_situation": true, "problem": "gender agreement"}'


def test_judge_needs_both_families_to_accept_caches_and_counts_unparsed(monkeypatch, tmp_path):
    calls = []
    _stub_chat(monkeypatch, {"m1": GOOD, "m2": BAD}, calls)
    monkeypatch.setattr(judge, "CACHE_PATH", str(tmp_path / "c.json"))
    r = judge.judge_pair("Yeh accha hai.", "Yeh zabardast hai.", models=("m1", "m2"))
    assert r["accept"] is False and r["accept_any"] is True and r["agree"] is False and r["problems"] == ["gender agreement"]
    judge.judge_pair("Yeh accha hai.", "Yeh zabardast hai.", models=("m1", "m2"))
    assert calls == ["m1", "m2"]                                                # the second request came from the cache
    _stub_chat(monkeypatch, {"m1": "I cannot answer", "m2": GOOD}, [])
    monkeypatch.setattr(judge, "CACHE_PATH", str(tmp_path / "c2.json"))
    r = judge.judge_pair("a", "b", models=("m1", "m2"))
    assert r["accept"] is False and judge.STATS["unparsed"] == 1                 # an unparseable answer is never an acceptance


def test_judge_rubric_requires_meaning_grammar_fit_and_natural_enough():
    ok = {"meaning_preserved": True, "grammar_ok": True, "natural": 4, "fits_situation": True}
    assert judge._accept(ok)
    for bad in ({"meaning_preserved": False}, {"grammar_ok": False}, {"fits_situation": False}, {"natural": 3}):
        assert not judge._accept({**ok, **bad})
    assert not judge._accept(None)


def test_calibration_create_then_analyze_round_trip(monkeypatch, tmp_path):
    from evaluation import calibration
    out = tmp_path / "cal"
    calibration.create(str(out), raters=3, n_real=6, n_gold=6, n_control=6, seed=1)
    key = json.load(open(out / "_key_DO_NOT_SHARE.json", encoding="utf-8"))
    kinds = [v["kind"] for v in key.values()]
    assert kinds.count("gold") == 6 and kinds.count("control") == 6 and all(not k.startswith("c") or True for k in key)
    sheet = (out / "rater_01.csv").read_text(encoding="utf-8-sig")
    assert "gold" not in sheet and "control" not in sheet and "corruption" not in sheet          # the sheet carries no label
    for r in (1, 2, 3):                                  # raters accept gold, reject controls, rate real items acceptable
        path = out / f"rater_0{r}.csv"
        rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
        for row in rows:
            kind = key[row["item_id"]]["kind"]
            good = kind != "control"
            row.update({"meaning_grammar": "5" if good else "1", "naturalness_fit": "5" if good else "2", "acceptable": "Y" if good else "N"})
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    def fake_judge_pairs(pairs, models=judge.DEFAULT_MODELS, workers=8, contexts=None):
        ctrl = {(v["neutral"], v["edited"]) for v in key.values() if v["kind"] == "control"}
        res = []
        for p in pairs:
            ok = tuple(p) not in ctrl
            v = {"meaning_preserved": ok, "grammar_ok": ok, "natural": 5 if ok else 1, "fits_situation": ok, "problem": "none"}
            res.append({"accept": ok, "accept_any": ok, "agree": True, "models": {m: v for m in models}, "natural_mean": v["natural"], "problems": []})
        return res
    monkeypatch.setattr(judge, "judge_pairs", fake_judge_pairs)
    rep = calibration.analyze(str(out))
    assert rep["n_raters"] == 3 and rep["n_items_rated"] == 18
    assert rep["rater_control_accuracy"]["R01"]["corrupted_rejected"].startswith("100.0%")
    ens = rep["metric_vs_human_consensus"]["judge:ENSEMBLE(all accept)"]
    assert ens["kappa"].startswith("1.00") and rep["spearman_judge_natural_vs_human_mean"] > 0.9


def test_small_single_rater_set_html_and_analysis(monkeypatch, tmp_path):
    from evaluation import calibration, calibration_small

    def fake_judge_pairs(pairs, models=judge.DEFAULT_MODELS, workers=8, contexts=None):
        out = []
        for a, b in pairs:
            ok = (len(b) % 2 == 0)                                   # arbitrary but deterministic: some accepted, some rejected
            v = {"meaning_preserved": ok, "grammar_ok": ok, "natural": 5 if ok else 1, "fits_situation": ok, "problem": "none"}
            out.append({"accept": ok, "accept_any": ok, "agree": True, "models": {m: v for m in models}, "natural_mean": v["natural"], "problems": []})
        return out
    monkeypatch.setattr(judge, "judge_pairs", fake_judge_pairs)
    out = tmp_path / "small"
    items = calibration_small.create(str(out), per_quadrant=2, n_gold=3, n_control=3, seed=3)
    html = (out / "rate_me.html").read_text(encoding="utf-8")
    assert "gold" not in html.split("const ITEMS=")[1].split(";")[0] and "quadrant" not in html and "control" not in html.split("const ITEMS=")[1].split(";")[0]
    assert len([i for i in items if i["kind"] == "gold"]) == 3 and len([i for i in items if i["kind"] == "control"]) == 3
    key = json.load(open(out / "_key_DO_NOT_SHARE.json", encoding="utf-8"))
    path = out / "rater_01.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    for r in rows:
        good = key[r["item_id"]]["kind"] != "control"
        r.update({"meaning_grammar": "5" if good else "1", "naturalness_fit": "4" if good else "1", "acceptable": "Y" if good else "N"})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    rep = calibration.analyze(str(out))
    assert rep["n_raters"] == 1 and "single rater" in rep["WARNING"] and "krippendorff_alpha_ordinal" not in rep
    assert rep["rater_control_accuracy"]["R01"]["corrupted_rejected"].startswith("100.0%")
    assert "metric_vs_human_consensus" in rep
