"""
audio/script_restore.py  -  "Script Restorer": romanized Hinglish -> the mixed-script text a Hindi voice reads best.

A Hindi neural voice pronounces Devanagari words correctly but reads romanized Hindi with English letter-to-sound rules
("kya" -> "kyaa"? "hai" -> "hay"), while English words are best left in Latin letters (it says "tournament" properly). This step writes
every Hindi word of a line in Devanagari and leaves English words, names of English places and brand words in Latin letters:

    "Mujhe bahut chinta ho rahi hai, kal ka match hai."  ->  "मुझे बहुत चिंता हो रही है, कल का match है।"

One cheap LLM call handles a whole scene; answers are validated (same number of lines, no emptied lines, punctuation kept) and
cached on disk. If the LLM is unavailable or an answer is invalid, that line is read in its original romanized form.
"""
from __future__ import annotations
import hashlib
import json
import os
import re
from typing import List

CACHE = "outputs/_tts_cache/script_restore.json"
PROMPT = """Convert each romanized Hinglish line into Devanagari text for a Hindi text-to-speech voice, the way a Hindi speaker would
pronounce it:
- Write EVERY word in Devanagari, including English loanwords as they are said in Hindi (match -> मैच, tournament -> टूर्नामेंट,
  practice -> प्रैक्टिस, robot -> रोबोट, final -> फाइनल, team -> टीम, sir -> सर, coach -> कोच, project -> प्रोजेक्ट).
- Names of people in Devanagari (रोहन, मीरा, कबीर, आर्यन, दादाजी).
- Slang is Hindi: write it phonetically (झक्कास, पंगा, बकवास, कमाल, फट्टू, फुल टेंशन, एकदम).
- Only acronyms and brand names that are spelled out stay in Latin letters (OTT, WhatsApp).
- Keep the punctuation (! ? ,) and use "।" instead of a final "." for statements. Do not translate, add or remove anything.
Return ONLY a JSON array with exactly {n} strings, in the same order.

{lines}"""


def _load():
    try:
        return json.load(open(CACHE, encoding="utf-8"))
    except Exception:
        return {}


def _save(d):
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    json.dump(d, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False)


def _ok(src: str, out: str) -> bool:
    if not out or not out.strip() or len(out) > 3 * len(src) + 20:
        return False
    # the sentence-final punctuation intent must survive
    if src.rstrip().endswith("?") and "?" not in out:
        return False
    if src.rstrip().endswith("!") and "!" not in out:
        return False
    return bool(re.search(r"[ऀ-ॿ]", out))


def restore(lines: List[str], client=None, batch: int = 14) -> List[str]:
    """-> list of mixed-script lines (same length as `lines`; falls back to the original text per line)."""
    cache = _load()
    out = [None] * len(lines)
    todo = []
    for i, ln in enumerate(lines):
        k = hashlib.sha1(ln.encode("utf-8")).hexdigest()[:16]
        if k in cache:
            out[i] = cache[k]
        else:
            todo.append(i)
    if todo and client is None:
        try:
            from llm_client import make_client
            client = make_client("dialogue")
        except Exception:
            client = None
    if todo and client is not None:
        def work(s):
            idx = todo[s:s + batch]
            src = [lines[i] for i in idx]
            try:
                resp = client.chat.completions.create(
                    model="x", temperature=0.0, max_tokens=1800,
                    messages=[{"role": "user", "content": PROMPT.format(
                        n=len(src), lines="\n".join(f"{j + 1}. {t}" for j, t in enumerate(src)))}])
                txt = resp.choices[0].message.content or ""
                arr = json.loads(re.search(r"\[.*\]", txt, re.DOTALL).group())
                if len(arr) != len(src):
                    raise ValueError("wrong count")
                for i, a, t in zip(idx, arr, src):
                    a = str(a).strip()
                    if _ok(t, a):
                        out[i] = a
                        cache[hashlib.sha1(t.encode("utf-8")).hexdigest()[:16]] = a
            except Exception as e:
                print(f"  [script-restore] batch failed ({str(e)[:70]}); those lines stay romanized")
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as ex:                    # the batches are independent LLM calls
            list(ex.map(work, range(0, len(todo), batch)))
        _save(cache)
    return [o if o else l for o, l in zip(out, lines)]
