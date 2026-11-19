"""
evaluation/judge.py  -  the LLM JUDGE: a versioned, cached, two-family rubric judge for slang edits and event coverage.

Why a judge: rules (edit_audit) guarantee SAFETY and STRUCTURE but cannot see whether a slang is natural or fits ("mujhe ekdum jhand lag raha hai"), and
embeddings cannot see it either (docs/VALIDATOR.md). A native-speaker-style LLM rubric can; it is only trustworthy if it is VALIDATED, so this module is built
to be validated:

  * the rubric is fixed and versioned (JUDGE_VERSION); every verdict is cached on disk by (version, model, texts), so a number in the paper can be reproduced
    exactly and re-run for free;
  * decomposed boolean / ordinal criteria (meaning, grammar, naturalness, fits the situation), not one holistic score;
  * two judges from DIFFERENT model families (default Claude Haiku 5.5 and Gemini 2.5 Flash), neither of which is the editor (Qwen): no self-preference;
    `accept` requires BOTH to accept (conservative); disagreement is reported;
  * temperature 0; unparseable answers are counted, never silently accepted.

    from evaluation.judge import judge_pairs
    res = judge_pairs([(neutral_line, edited_line), ...])      # -> list of {accept, accept_any, models: {...}, agree}

The judge is validated by evaluation/eval_judge.py (known-answer perturbations with confidence intervals) and evaluation/calibration.py (agreement with a small
human-labelled set).
"""
from __future__ import annotations
import hashlib
import json
import os
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

JUDGE_VERSION = "edit-judge-v1"
DEFAULT_MODELS = ("anthropic/claude-haiku-5.5", "google/gemini-2.5-flash")
CACHE_PATH = "outputs/_judge_cache.json"
NATURAL_MIN = 3.5
URL = "https://openrouter.ai/api/v1/chat/completions"

EDIT_PROMPT = """You are a strict native speaker of Hindi and Hinglish (romanized Hindi-English) reviewing dialogue lines from a screenplay.
A system EDITED the ORIGINAL line into the EDITED line. It is only allowed to swap one evaluative word or phrase for a slang that fits, or to add a casual filler.
{context}
ORIGINAL: {a}
EDITED:   {b}

Judge the EDITED line against the ORIGINAL. Answer ONLY a JSON object:
{{"meaning_preserved": true/false,   // same facts, same polarity, same events; nothing added, lost or changed
  "grammar_ok": true/false,          // tense, gender agreement, number, word order are all correct Hindi
  "natural": 1-5,                    // would a young Indian really say this? 5 = completely natural, 1 = nonsense
  "fits_situation": true/false,      // the slang's mood and register suit what the speaker is talking about (not trivialising something serious, not a forced collocation)
  "problem": "<=12 words, or none"}}"""

EVENT_PROMPT = """You check whether a scene of a screenplay contains a story event.
SCENE DIALOGUE:
{d}

EVENT: {e}

Is this event clearly expressed or directly implied by what the characters say? Answer ONLY JSON: {{"covered": true/false}}"""

_lock = threading.Lock()
_cache: Optional[Dict[str, dict]] = None
STATS = {"calls": 0, "cached": 0, "unparsed": 0, "cost": 0.0}


def _load_cache() -> Dict[str, dict]:
    global _cache
    if _cache is None:
        try:
            _cache = json.load(open(CACHE_PATH, encoding="utf-8"))
        except Exception:
            _cache = {}
    return _cache


def save_cache() -> None:
    with _lock:
        if _cache is not None:
            os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
            json.dump(_cache, open(CACHE_PATH, "w", encoding="utf-8"), ensure_ascii=False)


def _key(kind: str, model: str, *parts: str) -> str:
    return hashlib.sha1("|".join([JUDGE_VERSION, kind, model, *parts]).encode("utf-8")).hexdigest()[:24]


def _chat(model: str, prompt: str, max_tokens: int = 250) -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    body = {"model": model, "temperature": 0.0, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}],
            "usage": {"include": True}}
    if model.startswith("google/gemini-2.5"):
        body["reasoning"] = {"max_tokens": 0}                       # no hidden reasoning tokens: cheaper, deterministic length
    elif model.startswith("anthropic/"):
        body["reasoning"] = {"enabled": False}                      # else Claude spends the token budget on hidden reasoning and answers with ''
    last = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                out = json.loads(r.read().decode("utf-8"))
            STATS["calls"] += 1
            STATS["cost"] += float((out.get("usage") or {}).get("cost") or 0.0)
            return out["choices"][0]["message"]["content"] or ""
        except Exception as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"judge call failed: {str(last)[:120]}")


def _parse(txt: str) -> Optional[dict]:
    m = re.search(r"\{.*\}", txt or "", re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(re.sub(r"//[^\n]*", "", m.group()))
    except Exception:
        return None


def _accept(v: Optional[dict]) -> bool:
    return bool(v) and bool(v.get("meaning_preserved")) and bool(v.get("grammar_ok")) and bool(v.get("fits_situation", True)) \
        and float(v.get("natural", 0)) >= NATURAL_MIN


def judge_one(neutral: str, edited: str, model: str, context: str = "") -> Optional[dict]:
    """One model's verdict on one (neutral, edited) pair; cached. None when the model's answer could not be parsed (counted in STATS)."""
    cache = _load_cache()
    k = _key("edit", model, neutral, edited, context)
    if k in cache:
        STATS["cached"] += 1
        return cache[k]
    ctx = f"CONTEXT: {context}\n" if context else ""
    v = _parse(_chat(model, EDIT_PROMPT.format(a=neutral, b=edited, context=ctx)))
    if v is None:
        STATS["unparsed"] += 1
        return None
    v = {"meaning_preserved": bool(v.get("meaning_preserved")), "grammar_ok": bool(v.get("grammar_ok")), "natural": float(v.get("natural", 0) or 0),
         "fits_situation": bool(v.get("fits_situation", True)), "problem": str(v.get("problem", ""))[:160]}
    with _lock:
        cache[k] = v
    return v


def judge_pair(neutral: str, edited: str, models: Sequence[str] = DEFAULT_MODELS, context: str = "") -> dict:
    """-> {accept (ALL judges accept), accept_any, agree, models: {model: verdict|None}, problems}. Unparsed = not accepted."""
    per = {}
    for m in models:
        try:
            per[m] = judge_one(neutral, edited, m, context)
        except Exception as e:
            per[m] = None
            STATS["unparsed"] += 1
    acc = [_accept(v) for v in per.values()]
    return {"accept": all(acc), "accept_any": any(acc), "agree": len(set(acc)) == 1, "models": per,
            "natural_mean": sum((v or {}).get("natural", 0) for v in per.values()) / max(len(per), 1),
            "problems": [v["problem"] for v in per.values() if v and v.get("problem") and v["problem"].lower() != "none"]}


def judge_pairs(pairs: Sequence[Tuple[str, str]], models: Sequence[str] = DEFAULT_MODELS, workers: int = 8, contexts: Optional[Sequence[str]] = None) -> List[dict]:
    ctx = list(contexts) if contexts else [""] * len(pairs)
    with ThreadPoolExecutor(workers) as ex:
        out = list(ex.map(lambda t: judge_pair(t[0][0], t[0][1], models, t[1]), zip(pairs, ctx)))
    save_cache()
    return out


def event_covered(dialogue: str, event: str, model: str = DEFAULT_MODELS[0]) -> Optional[bool]:
    cache = _load_cache()
    k = _key("event", model, dialogue, event)
    if k in cache:
        STATS["cached"] += 1
        return cache[k]["covered"]
    v = _parse(_chat(model, EVENT_PROMPT.format(d=dialogue, e=event), max_tokens=60))
    if v is None or "covered" not in v:
        STATS["unparsed"] += 1
        return None
    with _lock:
        cache[k] = {"covered": bool(v["covered"])}
    return bool(v["covered"])


def event_recall(dialogue: str, events: Sequence[str], model: str = DEFAULT_MODELS[0]) -> Optional[float]:
    """Share of the scene's key events the judge finds in its dialogue (None when no event could be judged)."""
    res = [event_covered(dialogue, e, model) for e in events]
    res = [r for r in res if r is not None]
    return (sum(res) / len(res)) if res else None
