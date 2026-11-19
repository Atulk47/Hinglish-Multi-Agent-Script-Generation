"""
agents/slang_director.py  -  the SLANG DIRECTOR: a scene-level agent that decides WHERE slang belongs.

Why: the old decision layer looked at one line at a time. Gate 2 classified a single utterance into one of 7 intents and Gate 3 was a token counter, so
the system never knew who a line was spoken TO, how the two people relate, how the speaker feels at that point of the story, or what slang the scene
already used. The Director reads a whole scene at once (cast, relationships and registers, each speaker's emotion in this scene, the scene's events,
slang used earlier in the story) and returns a plan per line: does slang fit, in which mood, how strongly, and why.

What it can and cannot do (the design rule is "the agent proposes, the rules dispose"):
  * It only NOMINATES lines. The local slang editor still writes the edit, the deterministic gates (Gate 1: elders / authority, the register guards,
    the rule gate) still veto, and the validator still audits and reverts unsafe edits. A bad plan can therefore cost coverage, never safety.
  * One call per scene (not per line), temperature 0, cached by prompt hash so ablation arms are paired and repeatable.
  * If the model is unreachable or answers with something unusable, plan_scene() returns None and the caller uses the classic line-by-line gates.
"""
from __future__ import annotations
import hashlib
import json
import os
import re
import threading
from typing import Callable, Dict, List, Optional, Tuple

import config

REGISTERS = ("hype", "banter", "frustration", "stress", "urgency")
REGISTER_TO_INTENT = {                       # the label space the editor / pool selection already understands
    "hype": "ASSERT_POSITIVE",
    "banter": "CASUAL_ASSERTION",
    "frustration": "ASSERT_FRUSTRATION",
    "stress": "EXPRESS_STRESS",
    "urgency": "REQUEST_URGENCY",
}

PROMPT_VERSION = "v6"   # v1 was too conservative on 7 stories (it refused feeling words inside plan lines, applied "avoid consecutive" as a hard rule, treated narrators as formal); see docs/DIRECTOR.md

SYSTEM = """You are the Slang Director of a Hinglish screenplay team. A writer has produced neutral Hinglish dialogue for one scene. A separate editor can swap
ONE plain evaluative word in a line for a youth slang word (for example "bahut accha" -> "ekdum kamaal", "bahut chinta" -> "full tension") or add a light
filler ("yaar"). You do NOT write any slang. You decide, line by line, whether slang would sound natural HERE, and in which mood.

Think about who is talking TO whom. The addressee is whoever the line is aimed at: usually the previous speaker, or the person named in the line. Use the
relationship register between the two people, not just the speaker's own personality.

NOMINATE a line when ALL of these hold:
  1. The speaker is casual with the addressee (friends, siblings, teammates, roommates, classmates, a peer-to-peer chat).
  2. The line contains a word or phrase of feeling or evaluation that a slang word could replace: "bahut accha", "bahut gussa", "chinta", "mazedaar",
     "kharab", "badhiya", "thak gaya". Joy, pride, anger and liking count too, not only praise and worry: "bahut khushi ho rahi hai", "bahut maza aa raha hai",
     "mujhe tum par garv hai", "bahut gussa aa raha hai", "mujhe yeh bahut pasand aaya" are all swappable (the editor has idioms such as "dil garden-garden ho gaya"
     or "chhaati phool gayi" for them). It is fine if the same line ALSO states a plan, a time or an instruction: the editor swaps only the feeling
     word and leaves the facts untouched. Judge the feeling expression, not the whole sentence.
  3. The moment can carry it. Excitement, relief, pride, teasing, praise between friends, annoyance, worry about a deadline or a match, and
     frustration are all good moments for slang: warm or sincere is NOT a reason to refuse when the speaker is casual with a friend.
  4. The slang would match the speaker's feeling in this scene (emotion and intensity are given): excitement or praise -> hype, teasing -> banter,
     annoyance -> frustration, worry -> stress, rushing someone -> urgency.

DO NOT nominate a line when ANY of these hold:
  a. The addressee is someone the speaker treats respectfully according to the relationships given (teacher, elder, boss, shopkeeper, official: aap, ji,
     sir, ma'am, chacha, dadi, nani), even if the speaker is young. A narrator, an unnamed voice, or a person with no relationship listed is NOT by itself
     a formal addressee: judge by the speaker's own register and the other people in the scene.
  b. The scene or line is genuinely serious: grief, illness, an accident, fear for someone's safety, a formal announcement or rule, an apology, thanks,
     a secret being confided, or a real argument with someone senior.
  c. The line has no feeling or evaluation word at all: it is only a fact, a number, a name, a time, an instruction or a real question that needs an answer
     (the one exception is a FILLER-ONLY line, see below).
  d. The line already sounds like slang or is crude, or the slang would change what is being said.

Density: a lively scene between friends of 4 or more lines should usually reach 2 or 3 slang lines (never more than 3): look hard for the lines that carry a feeling or an
opinion, in different moods and from different people. A formal, serious or one-sided scene may have none, and so may a casual one where nothing really fits: never nominate
a line only to reach the number. The ONLY limit on the number of nominations is the cap given above the dialogue, and the code enforces it (it keeps the strongest lines). So never
refuse a line that fits because another line, or the same speaker, was already nominated, because it comes right after a nominated line, or "for variety":
if more lines fit than the cap allows, nominate them all and give the best ones the higher strength. Do not stay below the cap when lines fit.
Slang the story already used is a mild reason to prefer a different mood, never a reason to refuse a line that fits.

FILLER-ONLY lines (optional): a line with NO feeling word may still get one light filler ("yaar", "arre") from the editor when the speaker is casual with the
addressee (rule 1), the moment is not serious (rule b does not apply), and the line is lively talk between friends: it starts with the other person's name,
starts with "Haan" / "Chalo", or ends with "!". Write "slang": false and "filler": true for such a line. Never for a fact, a number, an instruction, a question
that needs an answer, thanks, an apology, or anything aimed at someone the speaker treats respectfully. At most one filler-only line in four.

Every dialogue line ends with a note of what the EDITOR can swap in it: [swappable: accha=praise, ...] or [nothing the editor can swap]. The editor can only replace a
feeling or evaluation word that is listed there. So nominate a line for slang ONLY when it carries a [swappable ...] note and rules 1, 3 and 4 hold; a line marked
[nothing the editor can swap] may be nominated only as FILLER-ONLY (see below), never as a slang line. (A line to an elder is always marked [nothing the editor can swap].)

Answer with JSON only, no prose, exactly this shape (every line number once, in order):
{"scene_read": "<one sentence: mood and social setting>",
 "lines": [{"n": 1, "why": "<at most 12 words>", "slang": false, "filler": false, "register": "none", "strength": 0}, ...]}
"register" is one of: hype, banter, frustration, stress, urgency, none. "strength" is 1 (mild) to 3 (intense), 0 when slang is false.
"filler" is true only for a FILLER-ONLY line (then "slang" is false).
Write "why" before deciding "slang"."""


REVIEW_PROMPT_VERSION = "r1"

REVIEW_SYSTEM = """You are the Slang Reviewer of a Hinglish screenplay team. A planner chose lines for light slang and a small editor model then swapped ONE feeling word for a slang word
(or added a filler such as "yaar"). You see the finished scene. Lines the editor changed are marked EDITED with the neutral original beneath them. Judge ONLY the edited lines.

REVERT an edit only when a native Hinglish speaker reading the scene would clearly find it:
  1. wrong: the meaning, a name, a number, a negation or the polarity of the feeling changed (praise became insult, or the reverse);
  2. ungrammatical or garbled: wrong gender or verb agreement, a broken or doubled phrase, stray punctuation;
  3. out of place for THIS speaker talking to THIS person in THIS moment: too crude or too loud for the relationship, a joke at a serious or sad moment,
     slang aimed at someone the speaker treats respectfully, a feeling that does not match the speaker's emotion here;
  4. forced: slang stacked on slang, a filler that interrupts the sentence, or the same slang used again and again.
KEEP an edit when it is natural, even if you would have chosen a different word. Mild stylistic doubt is NOT a reason to revert. When unsure, keep.

Answer with JSON only, no prose, exactly this shape (one entry for every EDITED line number, in order):
{"verdicts": [{"n": 3, "why": "<at most 12 words>", "keep": true}, ...]}
Write "why" before deciding "keep"."""


def dialogue_items(dialogue: str) -> List[Tuple[str, str]]:
    """(speaker, utterance) for every spoken line, in the exact order the rewriter walks them (empty utterances skipped)."""
    out = []
    for line in (dialogue or "").split("\n"):
        if ":" in line:
            sp, utt = line.split(":", 1)
            if utt.strip():
                out.append((sp.strip(), utt.strip()))
    return out


def _profile_key(profiles: dict, speaker: str) -> Optional[str]:
    from agents.name_utils import match_profile
    hit = match_profile(profiles, speaker)
    if not hit:
        return None
    for k, v in profiles.items():
        if v is hit:
            return k
    return None


def swappable_note(utt: str) -> str:
    """What the slang editor can replace in this line, from the SAME slot detection the editor uses (agents/edit_gate.slot_candidates)."""
    try:
        from agents import edit_gate
        cands = edit_gate.slot_candidates(utt)
    except Exception:
        cands = []
    if not cands:
        return "  [nothing the editor can swap]"
    return "  [swappable: " + ", ".join(dict.fromkeys(f"{c['word']}={c['type']}" for c in cands)) + "]"


def build_context(scene_id: str, items: List[Tuple[str, str]], profiles: dict, events: Optional[dict],
                  used_so_far: List[str], max_lines: int) -> str:
    from agents.agent2b_persona import emotion_at
    speakers, seen = [], set()
    for sp, _ in items:
        if sp.upper() not in seen:
            seen.add(sp.upper())
            speakers.append(sp)
    keys = {sp: _profile_key(profiles, sp) for sp in speakers}
    cast = []
    for sp in speakers:
        k = keys[sp]
        p = profiles.get(k) if k else None
        if not p:
            cast.append(f"- {sp}: no profile (unknown person; treat as formal)")
            continue
        emo = emotion_at(p, scene_id)
        traits = ", ".join(p.get("traits") or [])
        cast.append(f"- {sp}: {p.get('gender', '?')}, {p.get('age_group', '?')}, social role {p.get('social_role', 'peer')}, "
                    f"speaks {p.get('speech_style', 'casual')}{(', ' + traits) if traits else ''}; "
                    f"feeling in this scene: {emo['emotion']} (intensity {emo['intensity']}/3)")
    rels = []
    for sp in speakers:
        k = keys[sp]
        p = profiles.get(k) if k else None
        for other, r in ((p or {}).get("relationship") or {}).items():
            if other in keys.values() and other != k:
                rels.append(f"- {k} -> {other}: {r.get('relation', '?')}; {k} talks to {other} in a {r.get('register', '?')} register "
                            f"(calls them \"{r.get('i_address_them', other)}\")")
    ev = events or {}
    scene = []
    if ev.get("location"):
        scene.append(f"Place: {ev['location']}")
    if ev.get("scene_goal"):
        scene.append(f"What the scene is about: {ev['scene_goal']}")
    for e in (ev.get("key_events") or [])[:6]:
        scene.append(f"- event: {e}")
    from agents import edit_gate as _eg
    lines = "\n".join(f"{i}. {sp}: {utt}{swappable_note(utt)}" for i, (sp, utt) in enumerate(items, 1))
    cap = max(1, min(int(getattr(config, "SLANG_DIRECTOR_MAX_PER_SCENE", 3)), int(config.SLANG_DIRECTOR_MAX_SHARE * len(items))))
    target = int(getattr(config, "SLANG_DIRECTOR_TARGET", 0))
    from agents import edit_gate
    if edit_gate.setting_is_traditional():          # only added for such stories, so the cached plans of ordinary modern stories keep their key
        st = edit_gate.get_setting()
        scene.append(f"STORY SETTING: {st.get('era')} / {st.get('milieu')} / {st.get('formality')} speech. Modern urban English slang (dope, cringe, flop, "
                     f"trash, panic, anxiety, cool, awesome ...) would be jarring here: nominate a line only if a plain Hindi idiom could carry the feeling, and "
                     f"never for people related as spouses, parent and child, in-laws or master and servant, nor with a filler like \"yaar\" between them. "
                     f"When in doubt, do not nominate.")
    return ("SCENE " + scene_id + "\n" + ("\n".join(scene) + "\n" if scene else "")
            + "\nCAST\n" + "\n".join(cast)
            + ("\n\nRELATIONSHIPS (register = how the first person talks to the second)\n" + "\n".join(rels) if rels else "")
            + ("\n\nSLANG ALREADY USED EARLIER IN THE STORY (prefer variety): " + ", ".join(used_so_far[-12:]) if used_so_far else "")
            + (f"\n\nTARGET: this is a casual scene of {len(items)} lines: aim for {min(target, cap)} slang line(s) if that many really fit (a feeling or opinion, a peer who is "
               f"casual with the listener, not serious). If fewer fit, nominate fewer." if target and len(items) >= 4 and not edit_gate.setting_is_traditional() else "")
            + f"\n\nDIALOGUE ({len(items)} lines; cap: {cap} slang line(s) - nominate the {cap} best if more fit, do not hold back below it)\n" + lines)


def _json_object(text: str) -> Optional[dict]:
    if not text:
        return None
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        return json.loads(t[a:b + 1])
    except Exception:
        return None


def parse_plan(text: str, n_lines: int, max_share: float, filler_share: Optional[float] = None, swappable: Optional[List[bool]] = None) -> Optional[dict]:
    """-> {'scene_read': str, 'lines': {i (0-based): {slang, register, strength, why}}} or None if the answer is unusable.
    Missing lines default to NO slang (conservative); more nominations than the cap keeps the strongest ones."""
    obj = _json_object(text)
    if not obj or not isinstance(obj.get("lines"), list):
        return None
    lines: Dict[int, dict] = {}
    for row in obj["lines"]:
        try:
            n = int(row.get("n")) - 1
        except Exception:
            continue
        if not 0 <= n < n_lines:
            continue
        reg = str(row.get("register", "none")).strip().lower()
        slang = bool(row.get("slang")) and reg in REGISTERS
        nothing = False
        if slang and swappable is not None and n < len(swappable) and not swappable[n]:
            slang, nothing = False, True                    # nothing the editor can swap: demoted BEFORE the per-scene cap, so it cannot use up a cap slot
        try:
            strength = max(1, min(3, int(row.get("strength") or 2))) if slang else 0
        except Exception:
            strength = 2 if slang else 0
        lines[n] = {"slang": slang, "register": reg if slang else "none", "strength": strength,
                    "why": "nothing the editor can swap in this line (was nominated)" if nothing else str(row.get("why", ""))[:120],
                    "filler": (not slang) and row.get("filler") is True}
    if not lines:
        return None
    for i in range(n_lines):
        lines.setdefault(i, {"slang": False, "register": "none", "strength": 0, "why": "not listed by the director", "filler": False})
    cap = max(1, min(int(getattr(config, "SLANG_DIRECTOR_MAX_PER_SCENE", 3)), int(max_share * n_lines)))
    nominated = sorted((i for i, v in lines.items() if v["slang"]), key=lambda i: (-lines[i]["strength"], i))
    for i in nominated[cap:]:
        lines[i] = {"slang": False, "register": "none", "strength": 0, "why": "over the per-scene cap", "filler": False}
    # filler-only lines: at most 1 in 4, and together with the slang lines never above the same cap
    fs = getattr(config, "SLANG_DIRECTOR_FILLER_SHARE", 0.25) if filler_share is None else filler_share
    room = max(0, min(int(fs * n_lines + 0.5), cap - sum(1 for v in lines.values() if v["slang"])))
    for i in [i for i in sorted(lines) if lines[i].get("filler")][room:]:
        lines[i]["filler"] = False
        lines[i]["why"] = "over the filler cap"
    return {"scene_read": str(obj.get("scene_read", ""))[:240], "lines": lines}


def parse_review(text: str, expected: set) -> Optional[Dict[int, dict]]:
    '''-> {line number (1-based): {'keep': bool, 'why': str}} for the expected EDITED line numbers, or None if unusable. Lines the reviewer does not mention are KEPT.'''
    obj = _json_object(text)
    if not obj or not isinstance(obj.get("verdicts"), list):
        return None
    out: Dict[int, dict] = {}
    for row in obj["verdicts"]:
        try:
            n = int(row.get("n"))
        except Exception:
            continue
        if n in expected:
            out[n] = {"keep": bool(row.get("keep", True)), "why": str(row.get("why", ""))[:120]}
    for n in expected:
        out.setdefault(n, {"keep": True, "why": "not listed by the reviewer"})
    return out


class SlangDirector:
    def __init__(self, scratchpad, task: str = "director", cache_path: Optional[str] = None,
                 caller: Optional[Callable[[list], str]] = None):
        self.scratchpad = scratchpad
        self.task = task
        self.cache_path = cache_path if cache_path is not None else getattr(config, "SLANG_DIRECTOR_CACHE", "")
        self._caller = caller
        self._cache: Dict[str, str] = {}
        self._lock = threading.Lock()
        self._loaded = False
        self.used: List[str] = []
        self.stats = {"calls": 0, "cache_hits": 0, "failures": 0}
        self.last_raw = ""                                         # last raw model answer (debugging an unusable plan)

    # -- cache -----------------------------------------------------------------------------------------------------
    def _load(self):
        if self._loaded:
            return
        self._loaded = True
        try:
            if self.cache_path and os.path.exists(self.cache_path):
                self._cache = json.load(open(self.cache_path, encoding="utf-8"))
        except Exception:
            self._cache = {}

    def _save(self):
        if not self.cache_path:
            return
        try:
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            json.dump(self._cache, open(self.cache_path, "w", encoding="utf-8"), ensure_ascii=False)
        except Exception:
            pass

    def _key(self, user: str) -> str:
        model = getattr(config, "OPENROUTER_MODELS", {}).get(self.task, "")
        return hashlib.sha1((model + "\n" + SYSTEM + "\n" + user).encode("utf-8")).hexdigest()

    def _ask(self, user: str) -> str:
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        if self._caller:
            return self._caller(messages)
        import llm_client
        text, _fin = llm_client.complete(self.task, messages, temperature=0.0, max_tokens=8000, only="openrouter")   # headroom: reasoning tokens count
        return text

    # -- public API ------------------------------------------------------------------------------------------------
    def reset(self):
        """New story: forget which slang was used (the plan cache stays: it is keyed by the full prompt)."""
        self.used = []

    def note_used(self, slangs: List[str]):
        self.used += [s for s in slangs if s]

    def plan_scene(self, scene_id: str, dialogue: str, events: Optional[dict] = None) -> Optional[dict]:
        items = dialogue_items(dialogue)
        if not items:
            return None
        profiles = self.scratchpad.read("character_profiles") or {}
        user = build_context(scene_id, items, profiles, events, self.used, max_lines=len(items))
        key = self._key(user)
        with self._lock:
            self._load()
            raw = self._cache.get(key)
        if raw is not None:
            self.stats["cache_hits"] += 1
        else:
            try:
                raw = self._ask(user)
                self.stats["calls"] += 1
            except Exception as e:                               # unreachable model -> the caller falls back to the classic gates
                self.stats["failures"] += 1
                print(f"  [director] {scene_id}: model call failed ({str(e)[:110]}); using the line-by-line gates for this scene", flush=True)
                return None
        self.last_raw = raw
        swappable = [not swappable_note(u).strip().startswith("[nothing") for _sp, u in items]
        plan = parse_plan(raw, len(items), float(getattr(config, "SLANG_DIRECTOR_MAX_SHARE", 0.5)), swappable=swappable)
        if plan is None:
            self.stats["failures"] += 1
            print(f"  [director] {scene_id}: unusable answer; using the line-by-line gates for this scene", flush=True)
            return None
        if raw is not None and key not in self._cache:
            with self._lock:
                self._cache[key] = raw
                self._save()
        self.stats["demoted"] = self.stats.get("demoted", 0) + sum(1 for v in plan["lines"].values() if str(v.get("why", "")).startswith("nothing the editor can swap"))
        plan["items"] = items
        return plan

    # -- review loop -----------------------------------------------------------------------------------------------
    def review_scene(self, scene_id: str, neutral_items: List[Tuple[str, str]], edited_items: List[Tuple[str, str]],
                     events: Optional[dict] = None) -> Optional[Dict[int, dict]]:
        '''A second agent (config 'reviewer' model) re-reads the edited scene. -> {line index (0-based): {'keep': bool, 'why': str}} for the edited lines,
        or None when the model is unreachable / unusable (FAIL OPEN: the caller keeps every edit; the deterministic validator still audits them).'''
        if not neutral_items or len(neutral_items) != len(edited_items):
            return None
        changed = [i for i, (a, b) in enumerate(zip(neutral_items, edited_items)) if a[1].strip() != b[1].strip()]
        if not changed:
            return {}
        profiles = self.scratchpad.read("character_profiles") or {}
        ctx = build_context(scene_id, edited_items, profiles, events, [], len(edited_items))
        head = ctx.split("\n\nDIALOGUE", 1)[0]                  # cast, relationships and scene facts; the numbered dialogue is rebuilt below with the EDITED marks
        rows = []
        for i, (sp, txt) in enumerate(edited_items):
            rows.append(f"{i + 1}. {sp}: {txt}" + (f"   [EDITED]\n     neutral original: {neutral_items[i][1]}" if i in changed else ""))
        user = head + f"\n\nSCENE AS EDITED ({len(edited_items)} lines; {len(changed)} edited)\n" + "\n".join(rows)
        model = getattr(config, "OPENROUTER_MODELS", {}).get("reviewer", "")
        key = hashlib.sha1((model + "\n" + REVIEW_SYSTEM + "\n" + user).encode("utf-8")).hexdigest()
        with self._lock:
            self._load()
            raw = self._cache.get(key)
        if raw is not None:
            self.stats["cache_hits"] += 1
        else:
            try:
                messages = [{"role": "system", "content": REVIEW_SYSTEM}, {"role": "user", "content": user}]
                if self._caller:
                    raw = self._caller(messages)
                else:
                    import llm_client
                    raw, _fin = llm_client.complete("reviewer", messages, temperature=0.0, max_tokens=8000, only="openrouter")
                self.stats["calls"] += 1
            except Exception as e:
                self.stats["failures"] += 1
                print(f"  [review] {scene_id}: reviewer call failed ({str(e)[:110]}); keeping every edit of this scene", flush=True)
                return None
        verdicts = parse_review(raw, {i + 1 for i in changed})
        if verdicts is None:
            self.stats["failures"] += 1
            print(f"  [review] {scene_id}: unusable answer; keeping every edit of this scene", flush=True)
            return None
        with self._lock:
            if key not in self._cache:
                self._cache[key] = raw
                self._save()
        return {n - 1: v for n, v in verdicts.items()}
