"""
preprocessing/segmenter.py  -  the SCENE SEGMENTER.

The old chunker trusted blank lines and, without them, relied on a short list of romanized "time words". A story typed without paragraph
breaks, or a story told ALOUD (where the only trace of a scene change is a pause and a change of topic), was cut in the wrong places.

This segmenter works on the original Devanagari (or romanized) text and combines independent signals per sentence gap:
  * CUE      the next sentence opens with a time / place shift ("अगले दिन", "उसी रात", "कुछ दिनों बाद", "agle din" ...),
  * COHESION lexical overlap between the sentences before and after the gap collapses when the topic / set of characters changes
             (a TextTiling-style measure on content words, Hindi stop-words removed),
  * PAUSE    (voice input) a long silence in the recording between the two sentences,
  * DIALOGUE two consecutive sentences that both contain quoted speech belong to the same exchange (a penalty),
  * AUTHOR   blank lines written by the author are trusted (and only split further if a paragraph is far too long).
Boundaries are picked from the combined score under size limits (a scene is 2-8 sentences), then (optionally) a single LLM call acts as
a referee: it sees the numbered sentences, the long-pause marks and the signal-based proposal and returns the scene starts; its answer is
validated and silently replaced by the signal-based proposal if it is malformed. So the LLM can improve the cut, never break the run.
"""
from __future__ import annotations
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional

DANDA = "।"
_SPLIT = re.compile(r"(?<=[।!?\.])\s+|(?<=[।!?\.][\"'”’)])\s+")      # the closing quote stays WITH its sentence
_WORD = re.compile(r"[ऀ-ॿ]+|[A-Za-z]+")

STOP = set("""है हैं था थी थे हो हूँ हूं रहा रही रहे गया गयी गई गए गये और का की के को में से पर ने यह वह ये वे इस उस इन उन एक भी तो ही कि जो कर
कर के लिए साथ बहुत अब फिर जब तब हम तुम मैं मुझे हमें तुम्हें आप कुछ सब क्या नहीं न कहा बोला बोली कहती कहता कहने बोले पूछा पूछी लेकिन मगर
क्योंकि इसलिए तक या अगर तो भी सा सी से हुआ हुई हुए वाला वाली वाले लगा लगी लगे दिया दी दिए ली लिया लिए जा जाता जाती जाते आया आई आए
है। hai hain tha thi the aur ka ki ke ko mein se par ne yeh woh ek bhi to hi ki jo kar ke liye saath bahut ab phir jab tab hum tum main
mujhe humein aap kuch sab kya nahi kaha bola boli lekin magar kyunki isliye""".split())

STRONG_CUES = ["अगले दिन", "अगली सुबह", "अगली शाम", "अगली रात", "उसी रात", "उस रात", "उसी दिन", "उस दिन", "एक दिन", "एक सुबह", "एक शाम",
               "कुछ दिन", "कुछ ही दिनों", "कई दिन", "कुछ हफ्ते", "कुछ महीने", "अगले हफ्ते", "अगले महीने", "कुछ साल", "सेमीफ़ाइनल से", "फ़ाइनल",
               "फाइनल", "पहले मैच", "अगले मैच", "उसी शाम", "उस शाम", "उसी सुबह", "कुछ देर बाद", "थोड़ी देर बाद", "कुछ समय बाद",
               "agle din", "agli subah", "agli shaam", "usi raat", "us raat", "ek din", "kuch din", "kuch hafte", "kuch mahine"]
WEAK_CUES = ["सुबह", "शाम", "रात", "दोपहर", "तभी", "उसके बाद", "इसके बाद", "आखिरकार", "आख़िरकार", "अंत में", "आखिर", "फिर", "लेकिन",
             "subah", "shaam", "raat", "dopahar", "tabhi", "uske baad", "iske baad", "akhirkar"]
# the time cue only means "new scene" when it OPENS the sentence
CUE_POS = 3


@dataclass
class Segmentation:
    scenes: List[str]
    sentences: List[str]
    starts: List[int]                      # index of the first sentence of every scene (starts[0] == 0)
    method: str
    scores: List[float] = field(default_factory=list)
    notes: str = ""


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    return [s.strip() for s in _SPLIT.split(text) if s.strip()]


def _tokens(s: str) -> List[str]:
    out = []
    for w in _WORD.findall(s.lower()):
        if w in STOP or len(w) < 2:
            continue
        out.append(w[:4] if re.match(r"[ऀ-ॿ]", w) and len(w) > 4 else w)     # crude stem: first 4 code points
    return out


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    num = sum(a[k] * b[k] for k in a if k in b)
    den = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return num / den if den else 0.0


def _cue_score(sentence: str) -> float:
    head = " ".join(re.findall(r"[ऀ-ॿ]+|[A-Za-z]+", sentence.lower())[:CUE_POS])
    for c in STRONG_CUES:
        cl = c.lower()
        if head.startswith(cl) or (cl in head and head.find(cl) <= 8):
            return 1.0
    for c in WEAK_CUES:
        if head.startswith(c.lower()):
            return 0.45
    return 0.0


def _has_quote(s: str) -> bool:
    return bool(re.search(r"[\"“”]", s))


def gap_scores(sents: List[str], pauses: Optional[List[float]] = None, win: int = 2) -> List[float]:
    """score[i] = how likely a new scene starts at sentence i+1 (i.e. a boundary in the gap between sentence i and i+1)."""
    toks = [Counter(_tokens(s)) for s in sents]
    out = []
    for i in range(len(sents) - 1):
        left, right = Counter(), Counter()
        for j in range(max(0, i - win + 1), i + 1):
            left.update(toks[j])
        for j in range(i + 1, min(len(sents), i + 1 + win)):
            right.update(toks[j])
        coh = _cosine(left, right)
        cue = _cue_score(sents[i + 1])
        dlg = 1.0 if (_has_quote(sents[i]) and _has_quote(sents[i + 1])) else 0.0
        s = 0.5 * cue + 0.32 * (1 - coh) - 0.28 * dlg
        if pauses is not None and i < len(pauses):
            p = max(0.0, min(1.0, (pauses[i] - 0.55) / 1.3))
            s += 0.55 * p
        out.append(s)
    return out


def pick_starts(n: int, scores: List[float], min_len: int = 2, max_len: int = 8, target_len: float = 5.4,
                threshold: float = 0.32) -> List[int]:
    """Choose boundaries greedily by score under the size limits. Returns the scene start indices (always includes 0)."""
    if n <= max_len:
        want = max(1, round(n / target_len)) if n >= 2 * min_len else 1
    else:
        want = max(2, round(n / target_len))
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    starts = {0}

    def ok(b):                               # b = first sentence of a new scene
        ss = sorted(starts | {b})
        sizes = [y - x for x, y in zip(ss, ss[1:] + [n])]
        return all(sz >= min_len for sz in sizes)
    for i in order:
        b = i + 1
        if len(starts) >= want:
            break
        if scores[i] < threshold and len(starts) >= 1 and n <= max_len:
            break
        if scores[i] < threshold * 0.6:
            break
        if ok(b):
            starts.add(b)
    # enforce the maximum scene length by splitting the longest scene at its best internal gap
    changed = True
    while changed:
        changed = False
        ss = sorted(starts)
        for a, b in zip(ss, ss[1:] + [n]):
            if b - a > max_len:
                inner = [(scores[i], i + 1) for i in range(a + min_len - 1, b - min_len) if a < i + 1 < b]
                if inner:
                    starts.add(max(inner)[1])
                    changed = True
                    break
    return sorted(starts)


LLM_PROMPT = """You split a Hindi story into SCENES for a screenplay. Below are the story's sentences, numbered. A scene is a stretch of the story
in one place and time with the same group of characters. A NEW scene starts when the place or time changes (the next day, that night, later at
the hospital), when a new set of characters takes over, or when a new beat of the story begins. Dialogue exchanges between the same people stay
in ONE scene. A scene should usually have {lo}-{hi} sentences, never fewer than 2.
{pause_note}
Signal-based proposal (may be imperfect): scene starts at sentences {proposal}.

{sentences}

Return ONLY a JSON object: {{"scene_starts": [1, ...]}} - the numbers of the sentences that BEGIN a scene, in increasing order, starting with 1."""


def llm_starts(sents: List[str], proposal: List[int], pauses: Optional[List[float]], client, lo: int = 3, hi: int = 6,
               why: Optional[list] = None) -> Optional[List[int]]:
    lines = []
    for i, s in enumerate(sents):
        lines.append(f"{i + 1}. {s}")
        if pauses is not None and i < len(pauses) and pauses[i] >= 1.1:
            lines.append(f"   [long pause of {pauses[i]:.1f}s in the recording]")
    pause_note = ("The story was told ALOUD; '[long pause]' marks silences in the recording - they often (not always) mark a scene change."
                  if pauses is not None else "")
    prompt = LLM_PROMPT.format(lo=lo, hi=hi, pause_note=pause_note, proposal=[i + 1 for i in proposal], sentences="\n".join(lines))
    n = len(sents)
    for attempt in range(2):                                    # one retry: the answer occasionally violates the size rules
        try:
            r = client.chat.completions.create(model="x", messages=[{"role": "user", "content": prompt}],
                                               temperature=0.0 if attempt == 0 else 0.4, max_tokens=900)
            txt = r.choices[0].message.content or ""
            arr = json.loads(re.search(r"\{.*\}", txt, re.DOTALL).group())["scene_starts"]
            arr = sorted({int(x) - 1 for x in arr})
            if not arr or arr[0] != 0:
                arr = [0] + [x for x in arr if x > 0]
            if any(x < 0 or x >= n for x in arr):
                reason = "index out of range"
            else:
                sizes = [b - a for a, b in zip(arr, arr[1:] + [n])]
                if max(sizes) > 14:
                    reason = f"a scene of {max(sizes)} sentences"
                elif sum(1 for z in sizes if z < 2) > max(1, len(arr) // 3):
                    reason = "too many one-sentence scenes"
                else:
                    return arr
            if why is not None:
                why.append(reason)
        except Exception as e:
            if why is not None:
                why.append("unparsable: " + str(e)[:40])
    return None


class SceneSegmenter:
    def __init__(self, client=None, use_llm: bool = True, max_scenes: int = 16):
        self.client, self.use_llm, self.max_scenes = client, use_llm, max_scenes
        if self.client is None and use_llm:
            try:
                from llm_client import make_client
                self.client = make_client("extract")
            except Exception:
                self.client = None

    def segment(self, text: str, pauses: Optional[List[float]] = None, trust_paragraphs: bool = True) -> Segmentation:
        """text: the story (Devanagari or romanized). pauses: seconds of silence AFTER each sentence (voice input), len == sentences-1."""
        paras = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
        if trust_paragraphs and len(paras) >= 2:
            return self._from_paragraphs(paras)
        sents = split_sentences(text)
        if not sents:
            return Segmentation([], [], [], "empty")
        return self._segment_sentences(sents, pauses)

    # ---------------------------------------------------------------------------------------------- author paragraphs
    def _from_paragraphs(self, paras: List[str]) -> Segmentation:
        scenes, sents_all, starts = [], [], []
        for p in paras:
            ss = split_sentences(p)
            if len(ss) > 9:                                     # a paragraph far too long for one scene: split it further
                sub = self._segment_sentences(ss, None, allow_llm=False)
                for a, b in zip(sub.starts, sub.starts[1:] + [len(ss)]):
                    starts.append(len(sents_all) + a)
                    scenes.append(" ".join(ss[a:b]))
            else:
                starts.append(len(sents_all))
                scenes.append(" ".join(ss))
            sents_all += ss
        return Segmentation(scenes, sents_all, starts, "author-paragraphs")

    # ---------------------------------------------------------------------------------------------- signals (+ LLM referee)
    def _segment_sentences(self, sents: List[str], pauses, allow_llm: bool = True) -> Segmentation:
        n = len(sents)
        pz = None
        if pauses is not None:
            pz = list(pauses) + [0.0] * max(0, n - 1 - len(pauses))
            pz = pz[:n - 1]
        scores = gap_scores(sents, pz)
        starts = pick_starts(n, scores)
        notes = ""
        method = "signals" + ("+pauses" if pz is not None else "")
        if allow_llm and self.use_llm and self.client is not None and n >= 5:
            why = []
            ls = llm_starts(sents, starts, pz, self.client, why=why)
            if ls is not None:
                starts, method = ls, method + "+llm"
            else:
                notes = "llm referee rejected: " + "; ".join(why)
        if len(starts) > self.max_scenes:                      # keep the strongest boundaries
            keep = sorted(sorted(starts[1:], key=lambda b: -scores[b - 1])[:self.max_scenes - 1])
            starts = [0] + keep
        scenes = [" ".join(sents[a:b]) for a, b in zip(starts, starts[1:] + [n])]
        return Segmentation(scenes, sents, starts, method, scores, notes)
