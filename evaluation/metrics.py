"""
evaluation/metrics.py
Reference-free metrics for the Hinglish slang screenplay pipeline.

The pipeline has no gold Hinglish references, so evaluation compares Pass-2
(slang) against Pass-1 (neutral) and the scene's key events. These functions are
pure (no model loading); MuRIL semantic similarity lives in the runner, which
reuses evaluation/muril_validator.py.

Metrics (all per-scene, then aggregated):
  slang_coverage   fraction of dialogue lines that received >=1 real slang term
  slang_density    real-slang insertions per dialogue line
  slang_validity   of all words INSERTED vs the neutral line, the fraction that
                   are genuine slang (in the lexicon) rather than arbitrary words
  slang_diversity  script-level: unique slang / total slang insertions (1.0 = never
                   repeats; low = the monotony problem)
  event_recall     key events still detectable in the slang dialogue (fidelity)
  semantic_sim     MuRIL cosine(neutral, slang)  [computed in the runner]
"""
from __future__ import annotations
import csv
import os
import re
from collections import Counter

import config

_WORD = re.compile(r"[a-z]+")


def words(s: str) -> list:
    return _WORD.findall((s or "").lower())


# Compact English lexicon for an APPROXIMATE code-mixing index. No language-id
# model is installed, so a token is called English if it's here, else romanized
# Hindi. Covers common function words + the loanword nouns that show up in this
# domain's Hinglish (library, project, monument, ...). Approximate by design.
ENGLISH_WORDS = set("""
the a an and or but if of to in on at for with by from as is are was were be been
being do does did have has had will would can could should may might must not no yes
this that these those it its he she they we you i me my your his her our their them him
here there where when why how what who which all any some more most much many few
library project material monument record records history historical diary note notes
chest lock locked key door city council public private book books page pages research
researcher student students paper old ancient dust dusty leather cover clue clues map
maps merchant trader poor family families help helped truth quest search document
documents proof monument statue memory memories time future past present story stories
people person good bad great nice cool awesome unique special secret hidden lost found
discover discovered evidence archive council decided honor honored public society
building room shelf cupboard morning evening night day light shadow silence quiet
""".split())


def code_mixing_index(text: str):
    """Gambäck & Das (2014) CMI, approximate. Returns (cmi, en_ratio, hi_ratio).
    CMI = 100*(N - max_lang)/N over language tokens; 0 = monolingual, ~50 = balanced."""
    toks = words(text)
    if not toks:
        return 0.0, 0.0, 0.0
    en = sum(1 for w in toks if w in ENGLISH_WORDS)
    hi = len(toks) - en
    n = len(toks)
    cmi = 100.0 * (n - max(en, hi)) / n
    return round(cmi, 2), round(en / n, 3), round(hi / n, 3)


def load_slang_lexicon(path: str = None):
    """Return (phrases, single_words) from the dataset's `slangs` column —
    the same source Agent 4 selects from. Phrases keep multi-word slang like
    'scene hai' / 'waat lagi padi hai'; single_words holds their tokens."""
    path = path or config.FELIX_DATASET_CLEAN
    phrases = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                for s in str(row.get("slangs") or "").split("|"):
                    s = s.strip().lower()
                    if s:
                        phrases.add(s)
    single = {w for p in phrases for w in words(p)}
    return phrases, single


def _lines(dialogue: str) -> list:
    return [ln for ln in (dialogue or "").splitlines() if ln.strip()]


def _strip_speaker(line: str) -> str:
    # "Arav: text" -> "text"; leaves lines without a speaker prefix untouched
    return line.split(":", 1)[1] if ":" in line[:20] else line


def detect_slang(neutral: str, slang: str, phrases: set, single: set) -> dict:
    """Find slang the Pass-2 line added over Pass-1, at the scene level.

    inserted   = words present in slang but not neutral (multiset difference)
    valid      = inserted words that are genuine slang (in the lexicon)
    phrase_hits= multi-word slang phrases present in slang but not neutral
    """
    n_words = Counter(words(neutral))
    s_words = Counter(words(slang))
    inserted = list((s_words - n_words).elements())
    valid = [w for w in inserted if w in single]

    nl = (neutral or "").lower()
    sl = (slang or "").lower()
    phrase_hits = [p for p in phrases if " " in p and p in sl and p not in nl]

    # slang instances = distinct multi-word phrases + valid single words not part
    # of a counted phrase (avoid double-counting a phrase's own words)
    phrase_words = {w for p in phrase_hits for w in words(p)}
    single_hits = [w for w in valid if w not in phrase_words]
    detected = phrase_hits + single_hits
    return {
        "inserted": inserted,
        "valid": valid,
        "detected": detected,
        "n_inserted": len(inserted),
        "n_valid_single": len(valid),
    }


def scene_slang_metrics(neutral: str, slang: str, phrases: set, single: set) -> dict:
    d = detect_slang(neutral, slang, phrases, single)
    lines = _lines(slang)
    n_lines = max(len(lines), 1)

    # per-line coverage: a line "has slang" if it contains a detected term
    detected_lc = [t.lower() for t in d["detected"]]
    lines_with = 0
    for ln in lines:
        low = _strip_speaker(ln).lower()
        if any(t in low for t in detected_lc):
            lines_with += 1

    validity = (d["n_valid_single"] / d["n_inserted"]) if d["n_inserted"] else 1.0
    cmi, en_ratio, hi_ratio = code_mixing_index(slang)
    return {
        "n_lines": len(lines),
        "n_slang": len(d["detected"]),
        "detected": d["detected"],
        "coverage": lines_with / n_lines,
        "density": len(d["detected"]) / n_lines,
        "validity": validity,
        "cmi": cmi,
        "en_ratio": en_ratio,
        "hi_ratio": hi_ratio,
    }


def aggregate(scene_metrics: list) -> dict:
    """Roll per-scene metrics up to a script-level report."""
    if not scene_metrics:
        return {}
    n = len(scene_metrics)
    all_slang = [t.lower() for m in scene_metrics for t in m["detected"]]
    total = len(all_slang)
    unique = len(set(all_slang))

    def mean(key):
        return round(sum(m[key] for m in scene_metrics) / n, 4)

    return {
        "scenes": n,
        "slang_coverage": mean("coverage"),        # want high
        "slang_density": mean("density"),          # insertions per line
        "slang_validity": mean("validity"),        # want ~1.0 (real slang)
        "slang_total": total,
        "slang_unique": unique,
        "slang_diversity": round(unique / total, 4) if total else 0.0,  # want high
        "cmi_mean": mean("cmi"),               # code-mixing index (~0 mono, ~50 balanced)
        "en_ratio_mean": mean("en_ratio"),
        "hi_ratio_mean": mean("hi_ratio"),
        "top_slang": Counter(all_slang).most_common(8),
    }
