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
               "agle din", "agli subah", "agli shaam", "usi raat", "us raat", "ek din", "kuch din", "kuch hafte", "kuch mahine",
               # a change of PLACE: arriving, returning, being on the way
               "रास्ते में", "वापसी के रास्ते", "वापसी में", "घर पहुँचते", "घर पहुंचते", "घर पहुँचकर", "घर पहुंचकर", "वापस पहुँचा", "वापस पहुँची",
               "वापस पहुंचा", "वापस पहुंची", "उसी समय", "इतने में", "लौटकर", "लौटते समय", "पहुँचकर", "पहुंचकर", "पहुँचते ही", "पहुंचते ही",
               "raste mein", "vapsi ke raste", "ghar pahunchte", "ghar pahunchkar", "usi samay", "itne mein", "pahunchkar", "pahunchte hi"]
WEAK_CUES = ["सुबह", "शाम", "रात", "दोपहर", "तभी", "आखिरकार", "आख़िरकार", "अंत में", "जब वह", "जब वे", "वापस",
             "subah", "shaam", "raat", "dopahar", "tabhi", "akhirkar"]
# sequencing words inside a conversation ("फिर कबीर ने ...", "लेकिन ...", "उसके बाद ..."): barely evidence of a new scene
TIME_OF_DAY = {"सुबह", "शाम", "रात", "दोपहर", "सवेरे", "भोर", "तड़के", "subah", "shaam", "raat", "dopahar", "savere"}
CONNECTIVE_CUES = ["फिर", "लेकिन", "उसके बाद", "इसके बाद", "आखिर", "phir", "lekin", "uske baad", "iske baad"]
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
    for c in CONNECTIVE_CUES:
        if head.startswith(c.lower()):
            return 0.15
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


# ======================================================================================================================
# AUTHOR PARAGRAPHS AS ATOMIC UNITS.  Literary prose often puts every spoken line in its own paragraph; one paragraph per scene then gives 30+
# one-sentence "scenes" that have no context (no named person, no place). A scene is a stretch of the story with one place, one time and one group of
# people, so paragraphs are MERGED when the story does not change between them and kept apart when it does. There is NO target scene count:
# a short story gets few scenes, a long one more, and `max_scenes` (config.SEGMENT_MAX_SCENES, default 15) is only an upper limit.
# ======================================================================================================================
def _speech(text: str) -> bool:
    from agents.grounding import spoken_lines
    t = (text or "").lstrip()
    return bool(spoken_lines(text)) or _has_quote(text) or t.startswith(("—", "―", "-"))


def scene_size_ceiling(total_sentences: int, max_scenes: int) -> int:
    """How many sentences one scene may hold. It follows the story: about 16 (a normal scene is 3-8 sentences; only a clearly oversized one is cut), and larger only when the cap on the number of
    scenes forces it (a 300-sentence story cannot be cut into scenes of 8 without breaking the cap)."""
    return max(16, int(math.ceil(1.5 * total_sentences / max(1, max_scenes))))


def cue_cuts(sents: List[str], min_sents: int = 3) -> List[int]:
    """Indices inside a group of sentences where a sentence OPENS with a strong time / place cue ("घर पहुँचकर ...", "अगले दिन ...") and both sides keep at least
    `min_sents` sentences. A paragraph can change place in the middle; paragraph-level grouping cannot see that."""
    cuts, last = [], 0
    for i in range(min_sents, len(sents) - min_sents + 1):
        if _cue_score(sents[i]) >= 1.0 and i - last >= min_sents and len(sents) - i >= min_sents:
            cuts.append(i)
            last = i
    return cuts


def unit_gap_scores(units: List[List[str]]) -> List[float]:
    """score[i] = how likely the story CHANGES between paragraph i and i+1 (a new scene starts at i+1). Signals: a time / place cue opening the next
    paragraph, lexical cohesion of the surrounding paragraphs (topic and people), and spoken lines continuing across the break (an exchange stays one scene)."""
    toks = [Counter(_tokens(" ".join(u))) for u in units]
    out = []
    for i in range(len(units) - 1):
        left, right = Counter(), Counter()
        for j in range(max(0, i - 1), i + 1):
            left.update(toks[j])
        for j in range(i + 1, min(len(units), i + 3)):
            right.update(toks[j])
        coh = _cosine(left, right)
        first = units[i + 1][0]
        cue = _cue_score(first)
        opener = re.findall(r"[ऀ-ॿ]+|[A-Za-z]+", first.lower())[:1]
        if opener and opener[0] in TIME_OF_DAY:
            cue = 1.0                                           # a paragraph that OPENS with 'शाम होने तक', 'सुबह जब ...', 'रात को ...' is a time jump
        talk = 1.0 if (_speech(" ".join(units[i])) and _speech(" ".join(units[i + 1]))) else 0.0
        small = 1.0 if (len(units[i]) <= 2 or len(units[i + 1]) <= 2) else 0.0
        out.append(0.55 * cue + 0.45 * (1 - coh) - 0.40 * talk * (0.0 if cue >= 1.0 else 1.0) - 0.15 * small)      # a clear time / place shift outweighs 'speech continues'
    return out


def pick_unit_starts(units: List[List[str]], scores: List[float], max_scenes: int = 15, min_sents: int = 3, threshold: float = 0.33,
                     max_sents: Optional[int] = None, strong: float = 0.55) -> List[int]:
    """Scene starts as PARAGRAPH indices: boundary wherever the score reaches the threshold; scenes shorter than `min_sents` sentences are merged into a
    neighbour (the weaker of its two boundaries goes); never more than `max_scenes` scenes (the weakest boundaries go first)."""
    n = len(units)
    starts = [0] + [i + 1 for i, sc in enumerate(scores) if sc >= threshold]
    sizes = lambda ss: [sum(len(u) for u in units[a:b]) for a, b in zip(ss, ss[1:] + [n])]
    score_of = lambda b: scores[b - 1]
    while len(starts) > 1:
        sz = sizes(starts)
        mergeable = []                                 # a short scene is merged into a neighbour only across a WEAK boundary: a clear time / place
        for k, z in enumerate(sz):                     # shift ("the next day", "on reaching home") keeps even a two-sentence scene on its own
            if z < min_sents:
                cands = [b for b in (starts[k] if k > 0 else None, starts[k + 1] if k + 1 < len(starts) else None) if b and score_of(b) < strong]
                if cands:
                    mergeable.append((z, k, cands))
        if mergeable:
            _, _, cands = min(mergeable)
            starts.remove(min(cands, key=score_of))
            continue
        if len(starts) > max_scenes:                   # only the cap is violated: drop the weakest boundary
            starts.remove(min(starts[1:], key=score_of))
            continue
        break
    # SIZE CEILING: a scene is a bounded stretch of the story. It may only exceed `max_sents` sentences when the cap on the number of scenes forces it
    # (the ceiling is max(14, 2 * sentences / max_scenes)), so a long story gets longer scenes but never one scene swallowing several places.
    total = sum(len(u) for u in units)
    limit = max_sents or scene_size_ceiling(total, max_scenes)
    while len(starts) < max_scenes:
        sz = sizes(starts)
        best = None
        for k, (a, b) in enumerate(zip(starts, starts[1:] + [n])):
            if sz[k] <= limit or b - a < 2:
                continue
            for cut in range(a + 1, b):                 # candidate boundary between paragraphs inside this scene
                left, right = sum(len(u) for u in units[a:cut]), sum(len(u) for u in units[cut:b])
                if left >= min_sents and right >= min_sents:
                    key = (left <= limit and right <= limit, score_of(cut))      # one cut that leaves both halves within the ceiling beats a greedy fragment
                    if best is None or key > best[0]:
                        best = (key, cut)
        if best is None:
            break
        starts = sorted(starts + [best[1]])
    return starts


UNIT_LLM_PROMPT = """You group the paragraphs of a Hindi story into SCENES for a screenplay. A scene is ONE CONTINUOUS BEAT of the story: the same place, the same
moment and the same people doing one thing. Start a NEW scene when
  - the place changes (arriving somewhere, going home, being on the road),
  - time jumps (the next morning, some days later),
  - the activity changes materially (preparing to leave vs. walking there, buying something vs. being laughed at about it, the argument vs. its aftermath),
  - or people enter / leave so that a different group is talking.
Keep ONE conversation together even when each spoken line is its own paragraph. A scene usually covers 3 to 12 sentences and never more than {max_s}.
There is NO required number of scenes: a short story may be one scene or a few, a long story more; use at most {cap}.

Signal-based proposal (may be imperfect): scenes start at paragraphs {proposal}.

{paras}

Return ONLY a JSON object: {{"scene_starts": [1, ...]}} - the numbers of the paragraphs that BEGIN a scene, increasing, starting with 1."""


def llm_unit_starts(units: List[List[str]], proposal: List[int], client, cap: int, why: Optional[list] = None,
                    max_sents: int = 14) -> Optional[List[int]]:
    paras = "\n".join(f"{i + 1}. " + " ".join(u)[:230] for i, u in enumerate(units))
    prompt = UNIT_LLM_PROMPT.format(cap=cap, max_s=max_sents, proposal=[i + 1 for i in proposal], paras=paras)
    n = len(units)
    for attempt in range(2):
        try:
            r = client.chat.completions.create(model="x", messages=[{"role": "user", "content": prompt}],
                                               temperature=0.0 if attempt == 0 else 0.4, max_tokens=700)
            txt = r.choices[0].message.content or ""
            arr = sorted({int(x) - 1 for x in json.loads(re.search(r"\{.*\}", txt, re.DOTALL).group())["scene_starts"]})
            if not arr or arr[0] != 0:
                arr = [0] + [x for x in arr if x > 0]
            if any(x < 0 or x >= n for x in arr):
                reason = "index out of range"
            elif len(arr) > cap:
                reason = f"{len(arr)} scenes (cap {cap})"
            elif max(sum(len(u) for u in units[x:y]) for x, y in zip(arr, arr[1:] + [n])) > int(max_sents * 1.5):
                reason = "a scene far beyond the size ceiling"
            else:
                return arr
            if why is not None:
                why.append(reason)
        except Exception as e:
            if why is not None:
                why.append("unparsable: " + str(e)[:40])
    return None


class SceneSegmenter:
    def __init__(self, client=None, use_llm: bool = True, max_scenes: int = None):
        import config
        max_scenes = int(max_scenes if max_scenes is not None else getattr(config, "SEGMENT_MAX_SCENES", 15))
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
        """Author paragraphs are the atoms; scenes are runs of paragraphs in which the story does not change (see unit_gap_scores). A single paragraph far
        too long for one scene is still split further by the sentence segmenter."""
        units = [split_sentences(p) for p in paras]
        units = [u for u in units if u]
        sents_all = [x for u in units for x in u]
        n = len(units)
        if n == 0:
            return Segmentation([], [], [], "empty")
        scores = unit_gap_scores(units)
        starts_u = pick_unit_starts(units, scores, self.max_scenes)
        method, notes = "paragraphs+signals", ""
        if self.use_llm and self.client is not None and n >= 6:
            why = []
            total = sum(len(u) for u in units)
            ceiling = scene_size_ceiling(total, self.max_scenes)
            ls = llm_unit_starts(units, starts_u, self.client, self.max_scenes, why, ceiling)
            if ls is not None:
                lo, hi = max(1, int(0.7 * len(starts_u))), max(int(math.ceil(1.4 * len(starts_u))) + 1, len(starts_u) + 3)
                if not (lo <= len(ls) <= hi):                    # a referee answer far from the signal-based proposal made the scene count swing from run to run
                    why.append(f"{len(ls)} scenes vs the proposal's {len(starts_u)}: outside [{lo}, {hi}]")
                    notes = "llm referee rejected: " + "; ".join(why)
                    ls = None
            if ls is not None:
                starts_u, method = ls, "paragraphs+llm"
            else:
                notes = "llm referee rejected: " + "; ".join(why)
        first_sent = [0]
        for u in units:
            first_sent.append(first_sent[-1] + len(u))
        scenes, starts = [], []
        ceiling = scene_size_ceiling(len(sents_all), self.max_scenes)
        for a_, b_ in zip(starts_u, starts_u[1:] + [n]):
            ss = [x for u in units[a_:b_] for x in u]
            if len(ss) > ceiling:                          # too long for one scene (merged paragraphs, or one huge paragraph): cut it at its sentence-level
                method = method if "+sent" in method else method + "+sent"                                       # shifts (cues / cohesion), as for a story without paragraphs
                sub = self._segment_sentences(ss, None, allow_llm=False)
                for x, y in zip(sub.starts, sub.starts[1:] + [len(ss)]):
                    starts.append(first_sent[a_] + x)
                    scenes.append(" ".join(ss[x:y]))
            else:
                cuts = [0] + cue_cuts(ss) + [len(ss)]
                if len(cuts) > 2:
                    method = method if "+cue" in method else method + "+cue"
                for x, y in zip(cuts, cuts[1:]):
                    starts.append(first_sent[a_] + x)
                    scenes.append(" ".join(ss[x:y]))
        while len(scenes) > self.max_scenes:                # the huge-paragraph split can exceed the cap: merge the shortest neighbours
            k = min(range(len(scenes) - 1), key=lambda i: len(scenes[i]) + len(scenes[i + 1]))
            scenes[k] = scenes[k] + " " + scenes[k + 1]
            del scenes[k + 1], starts[k + 1]
        return Segmentation(scenes, sents_all, starts, method, scores, notes)

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
